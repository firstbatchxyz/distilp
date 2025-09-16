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
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple, Literal
import math

import gurobipy as gp
from gurobipy import GRB

try:
    # Package context
    from .components.dataclasses import DeviceProfile, ModelProfile, QuantPerf
    from .components.plotter import plot_k_curve, plot_batch_tpot
except Exception:
    # Script context fallback
    from .components.dataclasses import DeviceProfile, ModelProfile, QuantPerf
    from .components.plotter import plot_k_curve, plot_batch_tpot


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
    print(L, fs)
    return sorted(set(fs))


# ---------------------------------------------------------
# Appendix A.3: core constants b', α_m, β_m, ξ_m (Eq. 21)
# ---------------------------------------------------------


def b_prime(model: ModelProfile, batch_size: int = 1) -> int:
    """
    b' = b + 2(h_k e_k + h_v e_v) · n_kv.  [App. A.3, after Assumption 1]
    """
    return model.b_layer + 2 * (model.hk * model.ek + model.hv * model.ev) * model.n_kv * batch_size


def _sum_f_over_S(f_by_q: Dict[str, QuantPerf], S_by_q: Dict[str, QuantPerf], q: Literal["Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"], batch_size: int = 1) -> float:
    """
    Helper: ∑_{q ∈ Q} f_q / S_q   (used in α_m and β_m)
    Now handles batch sizes in S_by_q (device performance data)
    batch_size: integer batch size (e.g., 1, 2, 4) - will be converted to "b_1", "b_2", etc.
    """
    batch_key = f"b_{batch_size}"
    total = 0.0
    if batch_key in f_by_q and q in S_by_q:
        # Handle new format where S_by_q[q] is a dict with batch sizes
        if isinstance(S_by_q[q], dict):
            if batch_key not in S_by_q[q]:
                raise ValueError(f"Batch size {batch_size} (key '{batch_key}') not found in S_by_q[{q}]")
            s_val = S_by_q[q][batch_key]
        else:
            # Old format compatibility - direct float values
            # Get rid of this branch once all data is updated
            s_val = S_by_q[q]

        # Handle f_by_q which might also have batch sizes in future
        if isinstance(f_by_q, dict):
            if batch_key not in f_by_q:
                raise ValueError(f"Batch size {batch_size} (key '{batch_key}') not found in f_by_q[{q}]")
            f_val = f_by_q[batch_key]

        if s_val > 0:
            total += f_val / s_val
    return total


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
    dev: DeviceProfile, model: ModelProfile, batch_size: int = 1
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
    comp_cpu = _sum_f_over_S(model.f_q, dev.scpu, model.Q)
    alpha = comp_cpu + dev.t_kvcpy_cpu * batch_size + (bprime / dev.T_cpu)

    # β_m (GPU minus CPU path), 0 if no GPU available
    S_gpu = _gpu_table(dev)
    T_gpu = _pick_T_gpu(dev)
    if S_gpu is not None and T_gpu is not None:
        comp_gpu_minus_cpu = _sum_f_over_S(model.f_q, S_gpu, model.Q) - comp_cpu
        beta = (
            comp_gpu_minus_cpu
            + (dev.t_kvcpy_gpu - dev.t_kvcpy_cpu) * batch_size
            + (bprime / T_gpu - bprime / dev.T_cpu)
        )
    else:
        beta = 0.0

    # ξ_m (traffic + comm)
    # dev.t_ram2vram + dev.t_vram2ram is done once per round as it is done for sequence of layers within a window.
    xi = (dev.t_ram2vram + dev.t_vram2ram) * (
        0 if dev.is_unified_mem else 1
    ) + dev.t_comm * batch_size
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
) -> int:
    """
    Decide Case 1..4 for device m given tentative (w_m, n_m, k),
    following the inequality structure in Eqs. (28)-(33).

    Case 1 (M1): macOS, Metal disabled AND insufficient RAM
    Case 2 (M2): macOS, Metal enabled   AND insufficient RAM
    Case 3 (M3): Linux/Android          AND insufficient RAM
    Case 4 (M4): sufficient RAM OR disk too slow (s_disk < threshold)
    """
    if dev.os_type == "mac_no_metal":
        return 1
    elif dev.os_type == "mac_metal":
        return 2
    else:
        return 3


def assign_sets(
    devs: List[DeviceProfile],
) -> Dict[str, List[int]]:
    """
    Partition devices into M1..M4 by the most recent (w, n, k).  [Algorithm 1, line 6]
    """
    M1: List[int] = []
    M2: List[int] = []
    M3: List[int] = []
    for i, d in enumerate(devs):
        case = classify_device_case(d)
        if case == 1:
            M1.append(i)
        elif case == 2:
            M2.append(i)
        else:
            M3.append(i)

    return {"M1": M1, "M2": M2, "M3": M3}


# -------------------------------------------------------
# Vectorized objective coefficients a, b, c  [App. A.3]
# -------------------------------------------------------


def objective_vectors(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sets: Dict[str, List[int]],
    batch_size: int = 1,
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

    for i, d in enumerate(devs):
        alpha, beta, xi = alpha_beta_xi(d, model, batch_size)
        print(f"alpha: {alpha}, beta: {beta}, xi: {xi}")
        c[i] = xi
        if i in sets["M1"]:
            a[i] = alpha
            b[i] = 0.0
        elif i in sets["M2"]:
            a[i] = alpha
            b[i] = beta
        elif i in sets["M3"]:
            a[i] = alpha
            b[i] = beta
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

    head_compute = _sum_f_over_S(model.f_out, head.scpu, model.Q)
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
    batch_size: int = 1,
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
    bprime = b_prime(model, batch_size)
    Lb = model.L * bprime
    disk_size = 2000000000000
    disk_speed_threshold = 51446428  # removed one digit from the end (s_disk of M4), THIS HAS TO BE CHANGED
    # Coefficients [App. A.3]
    a, b, c = objective_vectors(devs, model, sets, batch_size)
    kappa = kappa_constant(devs, model, sets)

    # Create model
    m = gp.Model("halda_fixed_k")
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    m.Params.OutputFlag = 0

    # Decision variables
    # w = m.addVars(M, lb=0, vtype=GRB.INTEGER, name="w")     # lb=0, not every device has to be chosen
    w = m.addVars(
        M, lb=1, vtype=GRB.INTEGER, name="w"
    )  # lb=1, every device has to be chosen
    n = m.addVars(M, lb=0, vtype=GRB.INTEGER, name="n")
    y = m.addVars(M, vtype=GRB.BINARY, name="y")

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
        rhs = devs[i].d_avail_ram - bcio(i) + y[i] * disk_size
        # print("M1 rhs: " + str(rhs) + "and Lb: " + str(Lb))
        m.addConstr(w[i] * bprime <= rhs, name=f"M1_lb[{i}]")

    # Case 2 (M2): macOS with Metal, must overload:                 [Eq. (29)]
    for i in sets["M2"]:
        dav_metal = float(devs[i].d_avail_metal or 0)
        rhs = dav_metal - bcio(i) - devs[i].c_gpu + y[i] * disk_size
        # print("M2 rhs: " + str(rhs) + "and Lb: " + str(Lb))
        m.addConstr(w[i] * bprime <= rhs, name=f"M2_lb[{i}]")

    # Case 3 (M3): Linux/Android, must overload on CPU share:       [Eq. (30)]
    for i in sets["M3"]:
        d = devs[i]
        dswap = min(d.d_bytes_can_swap, d.d_swap_avail) if d.os_type == "android" else 0
        rhs = d.d_avail_ram + dswap - bcio(i) + y[i] * disk_size
        # print("M3 rhs: " + str(rhs) + "and Lb: " + str(Lb))
        m.addConstr((w[i] - n[i]) * bprime <= rhs, name=f"M3_lb[{i}]")

    # --- VRAM / shared memory bounds (Metal)                        [Eqs. (35)-(36)] ---
    for i in range(M):
        d = devs[i]
        # CUDA VRAM bound
        if d.has_cuda and d.d_avail_cuda is not None:
            rhs = d.d_avail_cuda - d.c_gpu
            m.addConstr(n[i] <= rhs, name=f"cuda_vram[{i}]")
        # Metal shared-memory bound (subtract b_o on head)          [Eq. (36)]
        if d.has_metal and d.d_avail_metal is not None:
            head = 1.0 if d.is_head else 0.0
            rhs = d.d_avail_metal - d.c_gpu - model.b_out * head
            m.addConstr(n[i] <= rhs, name=f"metal_shared[{i}]")

    # Objective: min k*(a^T w + b^T n + e^T c) + κ                  [Eq. (6)]
    obj_affine = k * (
        gp.quicksum(float(a[i]) * w[i] for i in range(M))
        + gp.quicksum(float(b[i]) * n[i] for i in range(M))
    )
    for i, d in enumerate(devs):
        if d.s_disk < disk_speed_threshold:
            m.addConstr(y[i] == 0, name=f"slow_disk[{i}]")
        if i in sets["M1"]:
            obj_affine += ((bprime / d.s_disk) * y[i] * w[i]) * k
        elif i in sets["M2"]:
            obj_affine += ((model.b_layer / d.s_disk) * y[i] * w[i]) * k

        elif i in sets["M3"]:
            obj_affine += ((bprime / d.s_disk) * y[i] * w[i]) * k
            obj_affine -= ((bprime / d.s_disk) * y[i] * n[i]) * k

    m.setObjective(obj_affine, GRB.MINIMIZE)
    m.ObjCon = (
        k * sum(float(x) for x in c) + kappa
    )  # add constants so cross-k is comparable

    m.optimize()
    # m.write("model.lp")
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
    batch_size: int  # batch
    tpot: float
    # iterations: int  # outer-loop iterations
    # forced_M4: List[int]  # indices forced into M4 during calibration


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


def halda_solve(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sdisk_threshold: Optional[float] = None,
    k_candidates: Optional[Iterable[int]] = None,
    time_limit_per_k: Optional[float] = None,
    mip_gap: Optional[float] = 1e-4,
    # strict_eps_bytes: float = 1.0,
    max_outer_iters: int = 50,
    plot: bool = True,

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
    best: Optional[HALDAResult] = None
    best_of_all_rounds = []
    sets = assign_sets(devs)
    # _print_sets(sets, devs)
    batch_list = [1, 2, 4, 8, 16]

    for b in batch_list:

        best_this_round: Optional[ILPResult] = None
        per_k_objs: List[Tuple[int, Optional[float]]] = []  # (k, obj or None if infeasible)
        print("Objectives by k")
        for kf in Ks:
            try:

                print("k: " + str(kf))
                res = solve_fixed_k_ilp(
                    devs,
                    model,
                    sets,
                    kf,
                    time_limit=3600,
                    mip_gap=mip_gap,
                    batch_size=b,
                )
                per_k_objs.append((kf, res.obj_value))
                print(f"  k={kf:<4d}  obj={res.obj_value:.6f}")
                if (best_this_round is None) or (res.obj_value < best_this_round.obj_value):
                    best_this_round = res
            except RuntimeError:
                per_k_objs.append((kf, None))
                print(f"  k={kf:<4d}  obj=infeasible")
        if best_this_round is None:
            raise RuntimeError("No feasible ILP found for any k this round.")

        # ----- line 16: accept the best (w*, n*) this round -----
        w = list(best_this_round.w)
        n = list(best_this_round.n)

        # track best overall

        best_of_this_round = HALDAResult(
            w=w,
            n=n,
            k=best_this_round.k,
            obj_value=best_this_round.obj_value,
            sets={k: list(v) for k, v in sets.items()},
            batch_size=b,
            tpot=round(best_this_round.obj_value/b, 6)
        )

        best_of_all_rounds.append(best_of_this_round)

        if plot:
            plot_k_curve(
                per_k_objs,
                k_star=best_of_this_round.k,
                title="HALDA: k vs objective (final sweep), b= " + str(b),
                # save_path="k_vs_objective.png",  # uncomment to save a PNG instead of only showing
            )

    tpots = []
    for candidate in best_of_all_rounds:
        tpots.append(candidate.tpot)
        if (best is None) or (candidate.tpot < best.tpot):
            best = candidate

    if plot:
        plot_batch_tpot(
            tpots,
            batch_list,
            # save_path="batch_vs_tpot.png",  # uncomment to save a PNG instead of only showing
        )
    return best
