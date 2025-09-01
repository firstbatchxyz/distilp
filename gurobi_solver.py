"""
HALDA end-to-end (parameters + Algorithm 1 with Gurobi), using the same notation as the paper.

References to the paper (prima.cpp / Halda):
- LDA definition: Eqs. (1)-(5)                           [Sec. 3.2]
- Fixed-k ILP:    Eqs. (6)-(10)                          [Sec. 3.3]
- Vectorized a,b,c and selector matrices Pw, Pn          [App. A.3]
- RAM/VRAM bounds (z, z_gpu) and P^gpu_n                 [App. A.3]
- Case constraints (M1..M4) and GPU bounds: (28)-(37)    [App. A.3]
- Coefficients b', α_m, β_m, ξ_m and constants (Eq. 21)  [App. A.3]
- Algorithm 1 (HALDA), lines 1-17                        [Sec. 3.3]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple
import math

import gurobipy as gp
from gurobipy import GRB

from components.dataclasses import DeviceProfile, ModelProfile, QuantPerf

# --------------------------------------
# Utility: divisors used as k candidates
# --------------------------------------


def valid_factors_of_L(L: int) -> List[int]:
    """
    All positive factors k of L except L itself. (Algorithm 1, line 3; used to fix k in (6)-(10))
    """
    fs: List[int] = []
    for k in range(1, int(math.sqrt(L)) + 1):
        if L % k == 0:
            if k != L:
                fs.append(k)
            other = L // k
            if other != k and other != L:
                fs.append(other)
    return sorted(set(fs))


# ---------------------------------------------------------
# Appendix A.3: core constants b', α_m, β_m, ξ_m (Eq. 21)
# ---------------------------------------------------------


def b_prime(model: ModelProfile) -> int:
    """
    b' = b + 2(h_k e_k + h_v e_v) · n_kv.  [App. A.3, after Assumption 1]
    """
    return model.b_layer + 2 * (model.hk * model.ek + model.hv * model.ev) * model.n_kv


def _sum_f_over_S(f_by_q: QuantPerf, S_by_q: QuantPerf, Q: List[str]) -> float:
    """
    Helper: ∑_{q ∈ Q} f_q / S_q   (used in α_m and β_m)
    """
    return sum(f_by_q[q] / S_by_q[q] for q in Q if q in f_by_q and q in S_by_q)


def _gpu_table(dev: DeviceProfile) -> Optional[QuantPerf]:
    """
    Pick the GPU FLOPS table for this device (Metal preferred when present).
    """
    if dev.has_metal and dev.sgpu_metal:
        return dev.sgpu_metal
    if dev.has_cuda and dev.sgpu_cuda:
        return dev.sgpu_cuda
    return None


def _pick_T_gpu(dev: DeviceProfile) -> Optional[float]:
    """
    Pick the GPU register-loading throughput T^{gpu}_m (bytes/s).
    """
    if dev.has_metal and dev.T_metal:
        return dev.T_metal
    if dev.has_cuda and dev.T_cuda:
        return dev.T_cuda
    return None


def alpha_beta_xi(
    dev: DeviceProfile, model: ModelProfile
) -> Tuple[float, float, float]:
    """
    α_m, β_m, ξ_m exactly as defined under Assumption 1.  [App. A.3, Eq. 21 block]
      α_m =  Σ_q f_q/scpu_q  + t^{kv_cpy,cpu}_m + b'/T^{cpu}_m
      β_m =  Σ_q f_q/sgpu_q  - Σ_q f_q/scpu_q + (t^{kv_cpy,gpu}_m - t^{kv_cpy,cpu}_m)
             + (b'/T^{gpu}_m - b'/T^{cpu}_m)    (0 if no GPU path)
      ξ_m =  (t^{ram->vram}_m + t^{vram->ram}_m)·(1 - I_{UMA}) + t^{comm}_m
    """
    bprime = b_prime(model)
    # α_m (CPU path)
    comp_cpu = _sum_f_over_S(model.f_by_quant, dev.scpu, model.Q)
    alpha = comp_cpu + dev.t_kvcpy_cpu + (bprime / dev.T_cpu)

    # β_m (GPU minus CPU path), 0 if no GPU available
    S_gpu = _gpu_table(dev)
    T_gpu = _pick_T_gpu(dev)
    if S_gpu is not None and T_gpu is not None:
        comp_gpu_minus_cpu = _sum_f_over_S(model.f_by_quant, S_gpu, model.Q) - comp_cpu
        beta = (
            comp_gpu_minus_cpu
            + (dev.t_kvcpy_gpu - dev.t_kvcpy_cpu)
            + (bprime / T_gpu - bprime / dev.T_cpu)
        )
    else:
        beta = 0.0

    # ξ_m (traffic + comm)
    xi = (dev.t_ram2vram + dev.t_vram2ram) * (
        0 if dev.is_unified_mem else 1
    ) + dev.t_comm
    return alpha, beta, xi


# -------------------------------
# Cases (M1..M4) and assignment
# -------------------------------


def _b_cio(dev: DeviceProfile, model: ModelProfile) -> float:
    """
    b^cio_m = (b_i/V + b_o)·I_{m=1} + c^{cpu}.   [Eq. (34)]
    """
    return ((model.b_in / model.V) + model.b_out) * (
        1.0 if dev.is_head else 0.0
    ) + dev.c_cpu


def classify_device_case(
    dev: DeviceProfile,
    model: ModelProfile,
    w_m: int,
    n_m: int,
    k: int,
    sdisk_threshold: Optional[float] = None,
) -> int:
    """
    Decide Case 1..4 for device m given tentative (w_m, n_m, k),
    following the inequality structure in Eqs. (28)-(33).

    Case 1 (M1): macOS, Metal disabled AND insufficient RAM
    Case 2 (M2): macOS, Metal enabled   AND insufficient RAM
    Case 3 (M3): Linux/Android          AND insufficient RAM
    Case 4 (M4): sufficient RAM OR disk too slow (s_disk < threshold)
    """
    # Optional "slow-disk => M4" rule (paper includes "or low disk speed" in Case 4).
    if sdisk_threshold is not None and dev.s_disk < sdisk_threshold:
        return 4

    # W = L/k, and we will compare the per-case RAM thresholds
    W = model.L // max(1, k)
    bprime = b_prime(model)
    Lb = model.L * bprime  # integer-scaling factor used in constraints

    # Helper to test the lower/upper bound condition with ±1 byte slack
    def ge_strict(lhs, rhs):  # lhs >= rhs + 1
        return lhs >= rhs + 1.0

    def le_strict(lhs, rhs):  # lhs <= rhs - 1
        return lhs <= rhs - 1.0

    bcio = _b_cio(dev, model)
    dswap = (
        min(dev.d_bytes_can_swap, dev.d_swap_avail) if dev.os_type == "android" else 0
    )

    if dev.os_type == "mac_no_metal":
        # M1 (must overload): w_m * Lb >= W*(d_avail_ram - bcio) + 1  [Eq. (28)]
        overload = ge_strict(w_m * Lb, W * (dev.d_avail_ram - bcio))
        return 1 if overload else 4
    elif dev.os_type == "mac_metal":
        # M2 (must overload): w_m * Lb >= W*(d_avail_metal - bcio - c_gpu) + 1  [Eq. (29)]
        dav_metal = float(dev.d_avail_metal or 0)
        overload = ge_strict(w_m * Lb, W * (dav_metal - bcio - dev.c_gpu))
        return 2 if overload else 4
    else:
        # Linux/Android: (w_m - n_m) * Lb >= W*(d_avail_ram + dswap - bcio) + 1  [Eq. (30)]
        overload = ge_strict((w_m - n_m) * Lb, W * (dev.d_avail_ram + dswap - bcio))
        return 3 if overload else 4


def assign_sets(
    devs: List[DeviceProfile],
    model: ModelProfile,
    w: List[int],
    n: List[int],
    k: int,
    sdisk_threshold: Optional[float] = None,
) -> Dict[str, List[int]]:
    """
    Partition devices into M1..M4 by the most recent (w, n, k).  [Algorithm 1, line 6]
    """
    M1: List[int] = []
    M2: List[int] = []
    M3: List[int] = []
    M4: List[int] = []
    for i, d in enumerate(devs):
        case = classify_device_case(d, model, w[i], n[i], k, sdisk_threshold)
        (M1 if case == 1 else M2 if case == 2 else M3 if case == 3 else M4).append(i)
    return {"M1": M1, "M2": M2, "M3": M3, "M4": M4}


# -------------------------------------------------------
# Vectorized objective coefficients a, b, c  [App. A.3]
# -------------------------------------------------------


def objective_vectors(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sets: Dict[str, List[int]],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Build a, b, c as in the vectorized form (block right after ILFP).  [App. A.3]
       a_m = α_m + b'/s_disk   (m ∈ M1)
             α_m + b /s_disk   (m ∈ M2)
             α_m + b'/s_disk   (m ∈ M3)
             α_m              (m ∈ M4)
       b_m = 0                 (m ∈ M1)
             β_m               (m ∈ M2)
             β_m - b'/s_disk   (m ∈ M3)
             β_m               (m ∈ M4)
       c_m = ξ_m  (all)
    """
    a: List[float] = [0.0] * len(devs)
    b: List[float] = [0.0] * len(devs)
    c: List[float] = [0.0] * len(devs)

    bprime = b_prime(model)

    for i, d in enumerate(devs):
        alpha, beta, xi = alpha_beta_xi(d, model)
        print(f"alpha: {alpha}, beta: {beta}, xi: {xi}")
        c[i] = xi
        if i in sets["M1"]:
            a[i] = alpha + bprime / d.s_disk
            b[i] = 0.0
        elif i in sets["M2"]:
            a[i] = alpha + model.b_layer / d.s_disk
            b[i] = beta
        elif i in sets["M3"]:
            a[i] = alpha + bprime / d.s_disk
            b[i] = beta - bprime / d.s_disk
        else:  # M4
            a[i] = alpha
            b[i] = beta
    return a, b, c


# ---------------------------------------------------------
# κ constant term (independent of w,n for fixed sets) [A.3]
# ---------------------------------------------------------


def kappa_constant(
    devs: List[DeviceProfile], model: ModelProfile, sets: Dict[str, List[int]]
) -> float:
    """
    κ aggregates the constant parts in Eq. (21) that do not multiply l_m, n_m, or W_m.  [App. A.3]
    Specifically we include:
      - head-only constants: ∑_q f_{1,out}/s^{cpu}_{1,q} + (b_i/V + b_o)/T^{cpu}_1
                             + b_i/(V s^{disk}_1) + b_o/s^{disk}_1 · I_{1 ∉ M4}
      - per-device constants for m ∈ (M1 ∪ M3):
                             (c^{cpu} - d^{avail}_m - d^{swapout}_m·I_{Android}) / s^{disk}_m
    """
    # Identify head device index (first with is_head) or default to index 0.
    head_idx = next((i for i, d in enumerate(devs) if d.is_head), 0)
    head = devs[head_idx]

    head_compute = _sum_f_over_S(model.f_out_by_quant, head.scpu, model.Q)
    head_load_regs = (model.b_in / model.V + model.b_out) / head.T_cpu
    head_disk_in = model.b_in / (model.V * head.s_disk)
    head_disk_out = (
        (model.b_out / head.s_disk) if (head_idx not in sets.get("M4", [])) else 0.0
    )

    tail_const = 0.0
    for mi in sets.get("M1", []) + sets.get("M3", []):
        d = devs[mi]
        dswap = min(d.d_bytes_can_swap, d.d_swap_avail) if d.os_type == "android" else 0
        tail_const += (d.c_cpu - d.d_avail_ram - dswap) / d.s_disk

    return head_compute + head_load_regs + head_disk_in + head_disk_out + tail_const


# ---------------------------------------------------------
# Fixed-k ILP (Eqs. (6)-(10)) with per-case constraints
# ---------------------------------------------------------


@dataclass
class ILPResult:
    k: int
    w: List[int]
    n: List[int]
    obj_value: float  # reported objective value for comparison across k


def solve_fixed_k_ilp(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sets: Dict[str, List[int]],
    k: int,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = 1e-4,
) -> ILPResult:
    """
    Build and solve the fixed-k ILP:
      min k*(a^T w + b^T n + e^T c) + κ          [Eq. (6)]
      s.t. integrality & coupling                 [Eq. (7)]
           sum(w) = W = L/k                      [Eq. (8)]
           RAM bounds (per case)                  [Eqs. (28)-(33)]
           VRAM/shared-mem bounds                [Eqs. (35)-(36)]
           n_m = 0 if no GPU                     [Eq. (37)]
    """
    M = len(devs)
    # print(M)
    W = model.L // k
    bprime = b_prime(model)
    Lb = model.L * bprime

    # Coefficients [App. A.3]
    a, b, c = objective_vectors(devs, model, sets)
    kappa = kappa_constant(devs, model, sets)

    # Create model
    m = gp.Model("halda_fixed_k")
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    m.Params.OutputFlag = 0

    # Decision variables
    w = m.addVars(M, lb=1, vtype=GRB.INTEGER, name="w")
    n = m.addVars(M, lb=0, vtype=GRB.INTEGER, name="n")

    # Domain & coupling: n_m ≥ 0, w_m ≥ 1, n_m ≤ w_m ≤ L          [Eq. (7)]
    for i in range(M):
        m.addConstr(n[i] <= w[i], name=f"n_le_w[{i}]")

        # If no GPU backend, force n_m = 0                         [Eq. (37)]
        has_cuda = bool(devs[i].has_cuda and devs[i].d_avail_cuda is not None)
        has_metal = bool(devs[i].has_metal and devs[i].d_avail_metal is not None)
        if not (has_cuda or has_metal):
            n[i].UB = 0

    # Window sum: e^T w = W                                        [Eq. (8)]
    m.addConstr(gp.quicksum(w[i] for i in range(M)) == W, name="sum_w_eq_W")

    # --- RAM constraints per case (28)-(33); use integer-safe strictness via ±1 byte ---
    def bcio(i: int) -> float:
        return _b_cio(devs[i], model)

    # Case 1 (M1): macOS no Metal, must overload:                   [Eq. (28)]
    for i in sets["M1"]:
        rhs = W * (devs[i].d_avail_ram - bcio(i))
        print("M1 rhs: " + str(rhs) + "and Lb: " + str(Lb))
        m.addConstr(w[i] * Lb >= rhs, name=f"M1_lb[{i}]")

    # Case 2 (M2): macOS with Metal, must overload:                 [Eq. (29)]
    for i in sets["M2"]:
        dav_metal = float(devs[i].d_avail_metal or 0)
        rhs = W * (dav_metal - bcio(i) - devs[i].c_gpu)
        print("M2 rhs: " + str(rhs) + "and Lb: " + str(Lb))
        m.addConstr(w[i] * Lb >= rhs, name=f"M2_lb[{i}]")

    # Case 3 (M3): Linux/Android, must overload on CPU share:       [Eq. (30)]
    for i in sets["M3"]:
        d = devs[i]
        dswap = min(d.d_bytes_can_swap, d.d_swap_avail) if d.os_type == "android" else 0
        rhs = W * (d.d_avail_ram + dswap - bcio(i))
        print("M3 rhs: " + str(rhs) + "and Lb: " + str(Lb))
        m.addConstr((w[i] - n[i]) * Lb >= rhs, name=f"M3_lb[{i}]")

    # Case 4 (M4): sufficient RAM / slow disk ⇒ upper bounds        [Eqs. (31)-(33)]
    for i in sets["M4"]:
        d = devs[i]
        if d.os_type == "mac_no_metal":
            rhs = W * (d.d_avail_ram - bcio(i))
            print("M41 rhs: " + str(rhs) + "and Lb: " + str(Lb))
            m.addConstr(w[i] * Lb <= rhs, name=f"M4_mac_no_metal_ub[{i}]")
        elif d.os_type == "mac_metal":
            dav_metal = float(d.d_avail_metal or 0)
            rhs = W * (dav_metal - bcio(i) - d.c_gpu)
            print("M42 rhs: " + str(rhs) + "and Lb: " + str(Lb))
            m.addConstr(w[i] * Lb <= rhs, name=f"M4_mac_metal_ub[{i}]")
        else:  # linux or android
            dswap = (
                min(d.d_bytes_can_swap, d.d_swap_avail) if d.os_type == "android" else 0
            )
            rhs = W * (d.d_avail_ram + dswap - bcio(i))
            print("M43 rhs: " + str(rhs) + "and Lb: " + str(Lb))
            m.addConstr((w[i] - n[i]) * Lb <= rhs, name=f"M4_lin_and_ub[{i}]")

    # --- VRAM / shared memory bounds (Metal)                        [Eqs. (35)-(36)] ---
    for i in range(M):
        d = devs[i]
        # CUDA VRAM bound
        if d.has_cuda and d.d_avail_cuda is not None:
            rhs = W * (d.d_avail_cuda - d.c_gpu)
            m.addConstr(n[i] * Lb <= rhs, name=f"cuda_vram[{i}]")
        # Metal shared-memory bound (subtract b_o on head)          [Eq. (36)]
        if d.has_metal and d.d_avail_metal is not None:
            head = 1.0 if d.is_head else 0.0
            rhs = W * (d.d_avail_metal - d.c_gpu - model.b_out * head)
            m.addConstr(n[i] * Lb <= rhs, name=f"metal_shared[{i}]")

    # Objective: min k*(a^T w + b^T n + e^T c) + κ                  [Eq. (6)]
    obj_affine = k * (
        gp.quicksum(float(a[i]) * w[i] for i in range(M))
        + gp.quicksum(float(b[i]) * n[i] for i in range(M))
    )
    m.setObjective(obj_affine, GRB.MINIMIZE)
    m.ObjCon = (
        k * sum(float(x) for x in c) + kappa
    )  # add constants so cross-k is comparable

    m.optimize()
    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # m.write("model.lp")
        # input("Paused. Press Enter to continue...")
        raise RuntimeError(f"Gurobi status {m.status} on fixed-k solve.")

    w_sol = [int(round(w[i].X)) for i in range(M)]
    n_sol = [int(round(n[i].X)) for i in range(M)]
    for i in m.getVars():
        if i.X > 0:
            print(i.varName, i.X)
    return ILPResult(k=k, w=w_sol, n=n_sol, obj_value=float(m.ObjVal))


# ----------------------
# Algorithm 1 (HALDA)
# ----------------------


@dataclass
class HALDAResult:
    w: List[int]  # w* (layer windows per device)
    n: List[int]  # n* (GPU layers per device)
    k: int  # best k
    obj_value: float  # objective value for the best (w*,n*,k)
    sets: Dict[str, List[int]]  # final sets M1..M4
    iterations: int  # outer-loop iterations
    forced_M4: List[int]  # indices forced into M4 during calibration


def _print_sets(
    label: str, sets: Dict[str, List[int]], devs: List[DeviceProfile]
) -> None:
    """Pretty-print M1..M4 with device names."""

    def names(idxs: List[int]) -> List[str]:
        return [devs[i].name for i in sorted(idxs)]

    print(f"\n{label} sets:")
    print(f"  M1: {names(sets.get('M1', []))}")
    print(f"  M2: {names(sets.get('M2', []))}")
    print(f"  M3: {names(sets.get('M3', []))}")
    print(f"  M4: {names(sets.get('M4', []))}")


def halda_solve(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sdisk_threshold: Optional[float] = None,
    k_candidates: Optional[Iterable[int]] = None,
    time_limit_per_k: Optional[float] = None,
    mip_gap: Optional[float] = 1e-4,
    # strict_eps_bytes: float = 1.0,
    max_outer_iters: int = 50,
) -> HALDAResult:
    """
    Full Algorithm 1 (HALDA), lines 1-17.  [Sec. 3.3]
      - Initializes w by memory budgets; n ← 0.           (line 1)
      - Computes α,β,ξ implicitly when building a,b,c.     (line 2)
      - Enumerates valid k ∈ K_L (excluding L).           (line 3)
      - Outer loop: assign sets, solve fixed-k ILPs,       (lines 4-16)
        apply calibration (lines 13-15), stop on stability (lines 7-8).
    """
    # ----- line 3: k candidates -----
    Ks = sorted(set(k_candidates)) if k_candidates else valid_factors_of_L(model.L)

    # ----- line 1: initialize w proportional to memory budgets; n = 0 -----
    def mem_cap(d: DeviceProfile) -> float:
        if d.os_type == "mac_metal":
            return float(d.d_avail_metal or 0)
        if d.os_type == "android":
            return d.d_avail_ram + min(d.d_bytes_can_swap, d.d_swap_avail)
        return d.d_avail_ram

    caps = [max(mem_cap(d), 1.0) for d in devs]
    total = sum(caps)
    W0 = model.L // 1  # temporary W for initialization (k=1)
    w = [max(1, round(W0 * c / total)) for c in caps]
    # normalize sum to W0
    delta = W0 - sum(w)
    for i in range(abs(delta)):
        w[i % len(devs)] += 1 if delta > 0 else -1
    n = [0] * len(devs)
    # --- print initial sets (based on the initial w,n and k_init = L / sum(w)) ---
    k_init = model.L // max(1, sum(w))
    sets_init = assign_sets(devs, model, w, n, k_init, sdisk_threshold)
    _print_sets("Initial", sets_init, devs)

    forced_M4: List[int] = []  # Algorithm 1, line 14
    prev_sets: Optional[Dict[str, List[int]]] = None
    best: Optional[HALDAResult] = None

    for outer in range(1, max_outer_iters + 1):
        # ----- line 5: derive W and k from current w -----
        W = sum(w)
        if W == 0 or model.L % W != 0:
            # ensure W divides L; if not, reset using smallest k
            k_now = Ks[0]
            W = model.L // k_now
            w = [max(1, round(W * c / total)) for c in caps]
            delta = W - sum(w)
            for i in range(abs(delta)):
                w[i % len(devs)] += 1 if delta > 0 else -1
        k_now = model.L // W

        # ----- line 6: assign sets M1..M4 (respect forced_M4) -----
        sets = assign_sets(devs, model, w, n, k_now, sdisk_threshold)
        if forced_M4:
            for idx in forced_M4:
                for s in ("M1", "M2", "M3"):
                    if idx in sets[s]:
                        sets[s].remove(idx)
                if idx not in sets["M4"]:
                    sets["M4"].append(idx)
                # Print sets for this iteration
        _print_sets(f"Iter {outer}", sets, devs)
        # ----- lines 7-8: stop on stability of set assignment -----
        if prev_sets is not None and all(
            set(sets[k]) == set(prev_sets[k]) for k in ("M1", "M2", "M3", "M4")
        ):
            break
        prev_sets = {k: list(v) for k, v in sets.items()}

        # ----- lines 9-12: solve fixed-k ILPs for all k ∈ K_L and pick the best -----
        best_this_round: Optional[ILPResult] = None
        per_k_objs: List[Tuple[int, Optional[float]]] = (
            []
        )  # (k, obj or None if infeasible)

        print(f"\nIter {outer}: objectives by k")
        for kf in Ks:
            try:
                res = solve_fixed_k_ilp(
                    devs,
                    model,
                    sets,
                    kf,
                    time_limit=time_limit_per_k,
                    mip_gap=mip_gap,
                )
                per_k_objs.append((kf, res.obj_value))
                print(f"  k={kf:<4d}  obj={res.obj_value:.6f}")
                if (best_this_round is None) or (
                    res.obj_value < best_this_round.obj_value
                ):
                    best_this_round = res
            except RuntimeError:
                per_k_objs.append((kf, None))
                print(f"  k={kf:<4d}  obj=infeasible")
        if best_this_round is None:
            raise RuntimeError("No feasible ILP found for any k this round.")

        # ----- lines 13-15: calibration / forcing step -----
        # If any GPU has free VRAM but another device is overloaded (in M1∪M2∪M3), force
        # the slowest-disk overloaded device into M4, then continue outer loop.
        overloaded_any = len(sets["M1"] + sets["M2"] + sets["M3"]) > 0

        def vram_slack_bytes(i: int, n_i: int, k_fixed: int) -> float:
            W_fixed = model.L // k_fixed
            d = devs[i]
            bprime = b_prime(model)
            Lb = model.L * bprime
            slack = 0.0
            if d.has_cuda and d.d_avail_cuda is not None:
                slack = max(slack, W_fixed * (d.d_avail_cuda - d.c_gpu) - n_i * Lb)
            if d.has_metal and d.d_avail_metal is not None:
                head = 1.0 if d.is_head else 0.0
                slack = max(
                    slack,
                    W_fixed * (d.d_avail_metal - d.c_gpu - model.b_out * head)
                    - n_i * Lb,
                )
            return slack

        any_free_vram = any(
            vram_slack_bytes(i, best_this_round.n[i], best_this_round.k) > 1.0
            for i in range(len(devs))
        )
        if overloaded_any and any_free_vram:
            candidates = sets["M1"] + sets["M2"] + sets["M3"]
            if candidates:
                slowest = min(candidates, key=lambda idx: devs[idx].s_disk)
                if slowest not in forced_M4:
                    forced_M4.append(slowest)
            continue  # go re-assign sets with this forced M4 update

        # ----- line 16: accept the best (w*, n*) this round -----
        w = list(best_this_round.w)
        n = list(best_this_round.n)

        # track best overall
        if (best is None) or (best_this_round.obj_value < best.obj_value):
            best = HALDAResult(
                w=w,
                n=n,
                k=best_this_round.k,
                obj_value=best_this_round.obj_value,
                sets={k: list(v) for k, v in sets.items()},
                iterations=outer,
                forced_M4=list(forced_M4),
            )

    # Finalize
    if best is None:
        # fallback (should not happen if feasible): report current state
        best = HALDAResult(
            w=w,
            n=n,
            k=(model.L // max(1, sum(w))),
            obj_value=float("nan"),
            sets=prev_sets if prev_sets else {"M1": [], "M2": [], "M3": [], "M4": []},
            iterations=0,
            forced_M4=list(forced_M4),
        )
    return best
