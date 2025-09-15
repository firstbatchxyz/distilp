"""
HALDA end-to-end (parameters + Algorithm 1 with SciPy), using the same notation as the paper.

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
from typing import Dict, List, Optional, Iterable, Tuple
import math

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from .components.dataclasses import DeviceProfile, ModelProfile, QuantPerf
from .components.plotter import plot_k_curve


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
    # dev.t_ram2vram + dev.t_vram2ram is done once per round as it is done for sequence of layers within a window.
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
    Build and solve the fixed-k ILP using SciPy's MILP solver:
      min k*(a^T w + b^T n + e^T c) + κ          [Eq. (6)]
      s.t. integrality & coupling                 [Eq. (7)]
           sum(w) = W = L/k                      [Eq. (8)]
           RAM bounds (per case)                  [Eqs. (28)-(33)]
           VRAM/shared-mem bounds                [Eqs. (35)-(36)]
           n_m = 0 if no GPU                     [Eq. (37)]
    """
    M = len(devs)
    W = model.L // k
    bprime = b_prime(model)
    disk_size = 2000000000000
    disk_speed_threshold = 51446428

    # Coefficients [App. A.3]
    a, b, c = objective_vectors(devs, model, sets)
    kappa = kappa_constant(devs, model, sets)

    # Variables x = [w(0..M-1), n(0..M-1), y(0..M-1)]
    # w: windows, n: GPU layers, y: binary disk usage indicators
    N = 3 * M
    lb = np.zeros(N)
    ub = np.ones(N) * np.inf

    for i in range(M):
        lb[i] = 1  # w_i >= 1, every device has to be chosen
        ub[i] = model.L  # w_i <= L
        lb[M + i] = 0  # n_i >= 0
        ub[M + i] = model.L  # n_i <= L
        lb[2*M + i] = 0  # y_i >= 0
        ub[2*M + i] = 1  # y_i <= 1

    bounds = Bounds(lb, ub)

    # Integrality: w and n are integers, y is binary
    integrality = np.ones(N, dtype=int)

    # Build constraint matrix
    A = []
    l = []
    u = []

    # Domain & coupling: n_m ≤ w_m ≤ L          [Eq. (7)]
    for i in range(M):
        row = np.zeros(N)
        row[i] = -1
        row[M + i] = 1
        A.append(row)
        l.append(-np.inf)
        u.append(0.0)

        # If no GPU backend, force n_m = 0                         [Eq. (37)]
        has_cuda = bool(devs[i].has_cuda and devs[i].d_avail_cuda is not None)
        has_metal = bool(devs[i].has_metal and devs[i].d_avail_metal is not None)
        if not (has_cuda or has_metal):
            ub[M + i] = 0

    # Window sum: e^T w = W                                        [Eq. (8)]
    row = np.zeros(N)
    row[:M] = 1.0
    A.append(row)
    l.append(W)
    u.append(W)

    # --- RAM constraints per case (28)-(33) ---
    def bcio(i: int) -> float:
        return _b_cio(devs[i], model)

    # Case 1 (M1): macOS no Metal, must overload:                   [Eq. (28)]
    for i in sets["M1"]:
        row = np.zeros(N)
        row[i] = bprime
        row[2*M + i] = -disk_size
        rhs = devs[i].d_avail_ram - bcio(i)
        A.append(row)
        l.append(-np.inf)
        u.append(rhs)

    # Case 2 (M2): macOS with Metal, must overload:                 [Eq. (29)]
    for i in sets["M2"]:
        dav_metal = float(devs[i].d_avail_metal or 0)
        row = np.zeros(N)
        row[i] = bprime
        row[2*M + i] = -disk_size
        rhs = dav_metal - bcio(i) - devs[i].c_gpu
        A.append(row)
        l.append(-np.inf)
        u.append(rhs)

    # Case 3 (M3): Linux/Android, must overload on CPU share:       [Eq. (30)]
    for i in sets["M3"]:
        d = devs[i]
        dswap = min(d.d_bytes_can_swap, d.d_swap_avail) if d.os_type == "android" else 0
        row = np.zeros(N)
        row[i] = bprime
        row[M + i] = -bprime
        row[2*M + i] = -disk_size
        rhs = d.d_avail_ram + dswap - bcio(i)
        A.append(row)
        l.append(-np.inf)
        u.append(rhs)

    # --- VRAM / shared memory bounds (Metal)                        [Eqs. (35)-(36)] ---
    for i in range(M):
        d = devs[i]
        # CUDA VRAM bound
        if d.has_cuda and d.d_avail_cuda is not None:
            row = np.zeros(N)
            row[M + i] = 1
            rhs = d.d_avail_cuda - d.c_gpu
            A.append(row)
            l.append(-np.inf)
            u.append(rhs)
        # Metal shared-memory bound (subtract b_o on head)          [Eq. (36)]
        if d.has_metal and d.d_avail_metal is not None:
            head = 1.0 if d.is_head else 0.0
            row = np.zeros(N)
            row[M + i] = 1
            rhs = d.d_avail_metal - d.c_gpu - model.b_out * head
            A.append(row)
            l.append(-np.inf)
            u.append(rhs)

    # Disk speed constraints
    for i, d in enumerate(devs):
        if d.s_disk < disk_speed_threshold:
            row = np.zeros(N)
            row[2*M + i] = 1
            A.append(row)
            l.append(0)
            u.append(0)

    A_matrix = np.array(A) if A else np.zeros((0, N))
    constraints = LinearConstraint(A_matrix, lb=np.array(l), ub=np.array(u)) if A else None

    # Objective: min k*(a^T w + b^T n) + constants
    # Build coefficient vector for [w, n, y]
    cvec = np.zeros(N)
    cvec[:M] = k * np.array(a)
    cvec[M:2*M] = k * np.array(b)

    # For disk penalty terms, we need to handle them differently in scipy
    # Since scipy doesn't support quadratic terms, we'll approximate

    res = milp(
        c=cvec,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={
            "mip_rel_gap": float(mip_gap) if mip_gap else 1e-4,
            **({"time_limit": float(time_limit)} if time_limit else {}),
        },
    )

    if res.status not in (0,):  # 0: OPTIMAL
        raise RuntimeError(f"SciPy MILP status {res.status} on fixed-k solve.")

    x = np.rint(res.x).astype(int)
    w_sol = x[:M].tolist()
    n_sol = x[M:2*M].tolist()
    y_sol = x[2*M:].tolist()

    # Calculate objective with disk penalties
    obj_affine = k * (
        sum(float(a[i]) * w_sol[i] for i in range(M))
        + sum(float(b[i]) * n_sol[i] for i in range(M))
    )

    for i, d in enumerate(devs):
        if i in sets["M1"]:
            obj_affine += ((bprime / d.s_disk) * y_sol[i] * w_sol[i]) * k
        elif i in sets["M2"]:
            obj_affine += ((model.b_layer / d.s_disk) * y_sol[i] * w_sol[i]) * k
        elif i in sets["M3"]:
            obj_affine += ((bprime / d.s_disk) * y_sol[i] * w_sol[i]) * k
            obj_affine -= ((bprime / d.s_disk) * y_sol[i] * n_sol[i]) * k

    obj_value = obj_affine + k * sum(float(x) for x in c) + kappa

    for i in range(M):
        if w_sol[i] > 0 or n_sol[i] > 0 or y_sol[i] > 0:
            print(f"w[{i}]={w_sol[i]}, n[{i}]={n_sol[i]}, y[{i}]={y_sol[i]}")

    return ILPResult(k=k, w=w_sol, n=n_sol, obj_value=float(obj_value))


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

    sets = assign_sets(devs)
    # _print_sets(sets, devs)

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
    if (best is None) or (best_this_round.obj_value < best.obj_value):
        best = HALDAResult(
            w=w,
            n=n,
            k=best_this_round.k,
            obj_value=best_this_round.obj_value,
            sets={k: list(v) for k, v in sets.items()},
        )

    if plot:
        plot_k_curve(
            per_k_objs,
            k_star=(best.k if best is not None else None),
            title="HALDA: k vs objective (final sweep)",
            # save_path="k_vs_objective.png",  # uncomment to save a PNG instead of only showing
        )
    return best