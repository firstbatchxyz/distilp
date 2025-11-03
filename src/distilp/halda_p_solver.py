"""
HALDA with SciPy.

This is a slightly modified version of the HALDA solver from Prima.CPP implemented in Scipy
refining the mathematical model suggested by introducing cycle time which more accurately
accounts for inter-device overlap and fits to Apple UMA.

@misc{li2025primacppfast3070bllm,
      title={Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters},
      author={Zonghang Li and Tao Li and Wenjiao Feng and Rongxing Xiao and Jianshu She and Hong Huang and Mohsen Guizani and Hongfang Yu and Qirong Ho and Wei Xiang and Steve Liu},
      year={2025},
      eprint={2504.08791},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2504.08791},
}
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from .components.plotter import plot_k_curve
import logging

from .components.dense_common import (
    b_cio_b,
    b_prime,
    assign_sets,
    objective_vectors,
    kappa_constant,
    valid_factors_of_L,
    DeviceProfile,
    ModelProfile,
    ILPResult,
    HALDAResult,
)

logger = logging.getLogger(__name__)

def _kv_bits_to_factor(kv_bits: str) -> float:
    """
    Map kv_bits string to a scalar factor for KV size.

    Accepted values:
    - '4bit' -> 0.5
    - '8bit' -> 1.0
    - 'fp16' or 'bf16' -> 2.0
    Any other value raises ValueError.
    """
    s = kv_bits.strip().lower()
    if s == "4bit":
        return 0.5
    if s == "8bit":
        return 1.0
    if s in ("fp16", "bf16"):
        return 2.0
    raise ValueError(f"Unsupported kv_bits '{kv_bits}'. Use one of: 4bit, 8bit, fp16, bf16")


def solve_fixed_k_milp(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sets: Dict[str, List[int]],
    k: int,
    kv_factor: float,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = 1e-4,
) -> ILPResult:
    """
    SciPy >= 1.11 required (for scipy.optimize.milp).
    """
    M = len(devs)
    W = model.L // k
    bprime = b_prime(model, kv_bits_k=kv_factor)

    # Coefficients and constants
    a, b, c_vec = objective_vectors(devs, model, sets, kv_factor)
    kappa = kappa_constant(devs, model, sets)
    total_inter_comm_time_per_round = 0

    # x = [ w(0..M-1) | n(0..M-1) | s1(0..M-1) | s2(0..M-1) | s3(0..M-1) | t(0..M-1) ]
    def idx_w(i):
        return i

    def idx_n(i):
        return M + i

    def idx_s1(i):
        return 2 * M + i

    def idx_s2(i):
        return 3 * M + i

    def idx_s3(i):
        return 4 * M + i

    def idx_t(i):
        return 5 * M + i

    # NEW: stall variables z_i and global cycle time C
    def idx_z(i):
        return 6 * M + i

    idx_C = 7 * M  # scalar index for cycle time

    Nvars = 7 * M + 1

    # Bounds & integrality
    lb = np.zeros(Nvars)
    ub = np.zeros(Nvars)

    # n upper bounds: allow up to W layers unless the device has no GPU backend at all
    # slacks are in LAYERS (integers): [0, W]

    for i, d in enumerate(devs):
        # w: at least 1 layer on each active device as before, up to W
        lb[idx_w(i)] = 1
        ub[idx_w(i)] = W

        # n: if no CUDA and no Metal, n must be 0; otherwise allow up to W
        has_cuda = bool(d.has_cuda and d.d_avail_cuda is not None)
        has_metal = bool(d.has_metal and d.d_avail_metal is not None)
        if not (has_cuda or has_metal):
            lb[idx_n(i)] = 0
            ub[idx_n(i)] = 0
        else:
            lb[idx_n(i)] = 0
            ub[idx_n(i)] = W

        # Slacks default: [0, W] (in layers)
        # Also fix to 0 if the device is not in the corresponding set.
        in_M1 = i in sets.get("M1", [])
        in_M2 = i in sets.get("M2", [])
        in_M3 = i in sets.get("M3", [])

        # s1 (M1 overflow in layers)
        lb[idx_s1(i)] = 0
        ub[idx_s1(i)] = W if in_M1 else 0

        # s2 (M2 overflow in layers)
        lb[idx_s2(i)] = 0
        ub[idx_s2(i)] = W if in_M2 else 0

        # s3 (M3 overflow in layers)
        lb[idx_s3(i)] = 0
        ub[idx_s3(i)] = W if in_M3 else 0

        # t (VRAM overflow in layers) — always allowed up to W if any GPU backend exists,
        # otherwise fixed to 0.
        lb[idx_t(i)] = 0
        ub[idx_t(i)] = W if (has_cuda or has_metal) else 0

        total_inter_comm_time_per_round += d.t_comm

    # NEW: bounds for stall z_i and cycle time C
    for i in range(M):
        lb[idx_z(i)] = 0.0
        ub[idx_z(i)] = np.inf
    lb[idx_C] = 0.0
    ub[idx_C] = np.inf

    bounds = Bounds(lb, ub)

    integrality = np.ones(Nvars, dtype=int)  # integers by default
    # NEW: make z_i and C continuous
    for i in range(M):
        integrality[idx_z(i)] = 0
    integrality[idx_C] = 0

    
    # Constraints
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    # n_i - w_i <= 0 (can't put more layers on GPU than assigned to device i)
    for i in range(M):
        row = np.zeros(Nvars)
        row[idx_n(i)] = 1
        row[idx_w(i)] = -1
        A_ub.append(row)
        b_ub.append(0.0)

    # Sum w_i = W
    row = np.zeros(Nvars)
    for i in range(M):
        row[idx_w(i)] = 1
    A_eq.append(row)
    b_eq.append(float(W))

    def bcio(i: int) -> float:
        return b_cio_b(devs[i], model)

    # Helpers for cycle-time + prefetch modeling
    def busy_row_for(i: int) -> np.ndarray:
        s_disk_i = max(1.0, float(devs[i].s_disk))  # avoid divide-by-zero
        penM1 = bprime / s_disk_i
        penM2 = model.b_layer / s_disk_i
        penM3 = bprime / s_disk_i
        if i in sets.get("M2", []):
            penVRAM = penM2
        else:
            penVRAM = penM3

        row = np.zeros(Nvars)
        row += devs[i].t_comm
        row += c_vec[i]
        row[idx_w(i)] = float(a[i])   # sec/layer including comms
        row[idx_n(i)] = float(b[i])   # (can be negative) GPU delta
        row[idx_s1(i)] = float(penM1)
        row[idx_s2(i)] = float(penM2)
        row[idx_s3(i)] = float(penM3)
        row[idx_t(i)]  = float(penVRAM)
        return row

    def fetch_row_for(i: int) -> np.ndarray:
        # F_i = w_i * (bprime / s_disk_i)
        s_disk_i = max(1.0, float(devs[i].s_disk))
        coef = bprime / s_disk_i
        row = np.zeros(Nvars)
        row[idx_w(i)] = coef

        return row

    # M1: b' * w_i <= d_avail_ram - bcio + b' * s1_i
    for i in sets.get("M1", []):
        rhs_cap = float(devs[i].d_avail_ram) - float(bcio(i))
        row = np.zeros(Nvars)
        row[idx_w(i)] = bprime
        row[idx_s1(i)] = -bprime
        A_ub.append(row)
        b_ub.append(rhs_cap)

    # M2: b' * w_i <= d_avail_metal - bcio - c_gpu + b' * s2_i
    for i in sets.get("M2", []):
        dav_metal = float(devs[i].d_avail_metal)
        rhs_cap = dav_metal - float(bcio(i)) - float(devs[i].c_gpu)
        row = np.zeros(Nvars)
        row[idx_w(i)] = bprime
        row[idx_s2(i)] = -bprime
        A_ub.append(row)
        b_ub.append(rhs_cap)

    # M3: b' * (w_i - n_i) <= d_avail_ram + dswap - bcio + b' * s3_i
    for i in sets.get("M3", []):
        d = devs[i]
        dswap = (min(d.d_bytes_can_swap, d.d_swap_avail) if d.os_type == "android" else 0)
        rhs_cap = float(d.d_avail_ram + dswap) - float(bcio(i))
        row = np.zeros(Nvars)
        row[idx_w(i)] = bprime
        row[idx_n(i)] = -bprime
        row[idx_s3(i)] = -bprime
        A_ub.append(row)
        b_ub.append(rhs_cap)

    # VRAM/shared caps with single overflow t_i used in both inequalities (no double charge)
    for i, d in enumerate(devs):
        has_cuda = bool(d.has_cuda and d.d_avail_cuda is not None)
        has_metal = bool(d.has_metal and d.d_avail_metal is not None)
        if has_cuda:
            rhs = float(d.d_avail_cuda) - float(d.c_gpu)
            row = np.zeros(Nvars)
            row[idx_n(i)] = bprime
            row[idx_t(i)] = -bprime
            A_ub.append(row)
            b_ub.append(rhs)
        if has_metal:
            head = 1.0 if d.is_head else 0.0
            row = np.zeros(Nvars)
            rhs = float(d.d_avail_metal) - float(d.c_gpu) - float(model.b_out * head)
            row[idx_n(i)] = bprime
            row[idx_t(i)] = -bprime
            A_ub.append(row)
            b_ub.append(rhs)

    # Cycle time C and stall z_i constraints per device
    # For each i: (1) C >= B_i + z_i ; (2) z_i >= F_i - (C - B_i)
    for i in range(M):
        # (1) C >= B_i + z_i  ->  -B_i - z_i + C >= 0  -> add as <= 0 by multiplying -1
        row1 = -busy_row_for(i)
        row1[idx_z(i)] += -1.0
        row1[idx_C]    +=  1.0
        A_ub.append(-row1)
        b_ub.append(0.0)

        # (2) z_i >= F_i - (C - B_i) -> z_i - B_i + C - F_i >= 0 -> add as <= 0 with -1
        row2 = np.zeros(Nvars)
        row2[idx_z(i)] = 1.0
        row2[idx_C]    = 1.0
        row2 -= busy_row_for(i)
        row2 -= fetch_row_for(i)  # bring F_i to LHS
        A_ub.append(-row2)
        b_ub.append(0.0)

    constraints = []
    if A_ub:
        A_ub = np.vstack(A_ub)
        b_ub = np.asarray(b_ub, dtype=float)
        constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
    if A_eq:
        A_eq = np.vstack(A_eq)
        b_eq = np.asarray(b_eq, dtype=float)
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

    
    # Objective: minimize cycle time C + penalties
    c_obj = np.zeros(Nvars)
    # Minimize k * C (steady-state cycle time scaled by k)
    c_obj[idx_C] = float(k-1)

    for i in range(M):
        # No direct cost on w_i or n_i; they influence objective via C through constraints
        c_obj[idx_w(i)] = float(a[i])
        c_obj[idx_n(i)] = float(b[i])

        # per-layer overflow penalties for RAM / VRAM (unchanged)
        s_disk_i = max(1.0, float(devs[i].s_disk))  # avoid divide-by-zero
        penM1 = bprime / s_disk_i
        penM2 = model.b_layer / s_disk_i
        penM3 = bprime / s_disk_i
        if i in sets.get("M2", []):
            penVRAM = penM2
        else:
            penVRAM = penM3

        c_obj[idx_s1(i)] = penM1
        c_obj[idx_s2(i)] = penM2
        c_obj[idx_s3(i)] = penM3
        c_obj[idx_t(i)]  = penVRAM

    options = {}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    if mip_gap is not None:
        options["mip_rel_gap"] = float(mip_gap)

    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    if not res.success:
        raise RuntimeError("No feasible MILP found.")

    x = res.x

    w_sol = [int(round(x[idx_w(i)])) for i in range(M)]
    n_sol = [int(round(x[idx_n(i)])) for i in range(M)]

    # Full objective value with constants
    linear_val = float(c_obj.dot(x))
    obj_value = linear_val + total_inter_comm_time_per_round + sum(float(ci) for ci in c_vec) + kappa

    # Optional: print only non-zero decision variables
    #for i, (w_i, n_i) in enumerate(zip(w_sol, n_sol)):
    #    if w_i > 0:
    #        print(f"w[{i}] {float(w_i)}")
    #    if n_i > 0:
    #        print(f"n[{i}] {float(n_i)}")

    return ILPResult(k=k, w=w_sol, n=n_sol, obj_value=float(obj_value))


def halda_solve(
    devs: List[DeviceProfile],
    model: ModelProfile,
    k_candidates: Optional[Iterable[int]] = None,
    mip_gap: Optional[float] = 1e-4,
    plot: bool = True,
    debug: bool = False,
    kv_bits: str = "8bit",
) -> HALDAResult:
    """
    HALDA with SciPy MILP.
    """
    Ks = sorted(set(k_candidates)) if k_candidates else valid_factors_of_L(model.L)
    kv_factor = _kv_bits_to_factor(kv_bits)
    best: Optional[HALDAResult] = None

    sets = assign_sets(devs)

    best_this_round: Optional[ILPResult] = None
    per_k_objs: List[Tuple[int, Optional[float]]] = []  # (k, obj or None if infeasible)
    logger.debug("Objectives by k")
    for kf in Ks:
        try:
            logger.debug("k: %d", kf)
            res = solve_fixed_k_milp(
                devs,
                model,
                sets,
                kf,
                kv_factor,
                time_limit=3600,
                mip_gap=mip_gap,
            )
            per_k_objs.append((kf, res.obj_value))
            if debug:
                print(f"  k={kf:<4d}  obj={res.obj_value:.6f}")
            if (best_this_round is None) or (res.obj_value < best_this_round.obj_value):
                best_this_round = res
        except RuntimeError:
            per_k_objs.append((kf, None))
            if debug:
                print(f"  k={kf:<4d}  obj=infeasible")
    if best_this_round is None:
        raise RuntimeError("No feasible MILP found for any k this round.")

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
