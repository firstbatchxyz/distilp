"""
HALDA end-to-end (parameters + Algorithm 1 with Gurobi), using the same notation as the paper.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Dict
import itertools
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from .components.plotter import plot_k_curve

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


def solve_fixed_k_milp(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sets: Dict[str, List[int]],
    k: int,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = 1e-4,
) -> ILPResult:
    """
    SciPy >= 1.11 required (for scipy.optimize.milp).
    """
    M = len(devs)
    W = model.L // k
    bprime = b_prime(model)
    disk_speed_threshold = 51446428

    # Coefficients and constants (same as Gurobi version)
    a, b, c_vec = objective_vectors(devs, model, sets)
    kappa = kappa_constant(devs, model, sets)

    # Determine which devices have too-slow disks (overflow not allowed)
    forced_zero = [int(d.s_disk < disk_speed_threshold) for d in devs]

    # ----------------------------
    # Variable indexing
    # ----------------------------
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

    Nvars = 6 * M

    # ----------------------------
    # Bounds & integrality
    # ----------------------------
    lb = np.zeros(Nvars)
    ub = np.zeros(Nvars)

    # n upper bounds: allow up to W layers unless the device has no GPU backend at all
    # slacks are in LAYERS (integers): [0, W], but can be forced to 0 by slow-disk policy

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
        # If disk is too slow, forbid RAM-related overflows by fixing the slack to 0.
        # Also fix to 0 if the device is not in the corresponding set.
        in_M1 = i in sets.get("M1", [])
        in_M2 = i in sets.get("M2", [])
        in_M3 = i in sets.get("M3", [])

        # s1 (M1 overflow in layers)
        lb[idx_s1(i)] = 0
        ub[idx_s1(i)] = (W if (in_M1 and not forced_zero[i]) else 0)

        # s2 (M2 overflow in layers)
        lb[idx_s2(i)] = 0
        ub[idx_s2(i)] = (W if (in_M2 and not forced_zero[i]) else 0)

        # s3 (M3 overflow in layers)
        lb[idx_s3(i)] = 0
        ub[idx_s3(i)] = (W if (in_M3 and not forced_zero[i]) else 0)

        # t (VRAM overflow in layers) — always allowed up to W if any GPU backend exists,
        # otherwise fix to 0. Tune via objective penalty to discourage unless necessary.
        lb[idx_t(i)] = 0
        ub[idx_t(i)] = (W if (has_cuda or has_metal) else 0)

    bounds = Bounds(lb, ub)

    integrality = np.ones(Nvars, dtype=int)  # all integers: w, n, s1, s2, s3, t

    # ----------------------------
    # Constraints
    # ----------------------------
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

    # M1: b' * w_i <= d_avail_ram - bcio + b' * s1_i
    for i in sets.get("M1", []):
        rhs_cap = float(devs[i].d_avail_ram) - bcio(i)
        row = np.zeros(Nvars)
        row[idx_w(i)] = bprime
        row[idx_s1(i)] = -bprime
        A_ub.append(row)
        b_ub.append(rhs_cap)

    # M2: b' * w_i <= d_avail_metal - bcio - c_gpu + b' * s2_i
    for i in sets.get("M2", []):
        dav_metal = float(devs[i].d_avail_metal or 0)
        rhs_cap = dav_metal - bcio(i) - float(devs[i].c_gpu)
        row = np.zeros(Nvars)
        row[idx_w(i)] = bprime
        row[idx_s2(i)] = -bprime
        A_ub.append(row)
        b_ub.append(rhs_cap)

    # M3: b' * (w_i - n_i) <= d_avail_ram + dswap - bcio + b' * s3_i
    for i in sets.get("M3", []):
        d = devs[i]
        dswap = (min(d.d_bytes_can_swap, d.d_swap_avail) if d.os_type == "android" else 0)
        rhs_cap = float(d.d_avail_ram + dswap) - bcio(i)
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
            row[idx_n(i)] = 1
            row[idx_t(i)] = -bprime
            A_ub.append(row)
            b_ub.append(rhs)
        if has_metal:
            head = 1.0 if d.is_head else 0.0
            rhs = float(d.d_avail_metal) - float(d.c_gpu) - model.b_out * head
            row = np.zeros(Nvars)
            row[idx_n(i)] = 1
            row[idx_t(i)] = -bprime
            A_ub.append(row)
            b_ub.append(rhs)

    constraints = []
    if A_ub:
        A_ub = np.vstack(A_ub)
        b_ub = np.asarray(b_ub, dtype=float)
        constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
    if A_eq:
        A_eq = np.vstack(A_eq)
        b_eq = np.asarray(b_eq, dtype=float)
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

    # ----------------------------
    # Objective: base cost + penalties per overflowed layer
    # ----------------------------
    c_obj = np.zeros(Nvars)
    for i in range(M):
        # base terms
        c_obj[idx_w(i)] = k * float(a[i])
        c_obj[idx_n(i)] = k * float(b[i])

        # per-layer overflow penalties
        # Calibrate to prior logic: s1 and s3 ~ b' / s_disk, s2 ~ b_layer / s_disk
        s_disk_i = max(1.0, float(devs[i].s_disk))  # avoid divide-by-zero
        penM1 = bprime / s_disk_i
        penM2 = model.b_layer / s_disk_i
        penM3 = bprime / s_disk_i
        # VRAM oversubscription penalty per extra GPU layer (t_i).
        penVRAM = penM2

        c_obj[idx_s1(i)] = k * penM1
        c_obj[idx_s2(i)] = k * penM2
        c_obj[idx_s3(i)] = k * penM3
        c_obj[idx_t(i)]  = k * penVRAM

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
        raise RuntimeError("No feasible MINLP found.")

    x = res.x

    w_sol = [int(round(x[idx_w(i)])) for i in range(M)]
    n_sol = [int(round(x[idx_n(i)])) for i in range(M)]

    # Full objective value with constants
    linear_val = float(c_obj.dot(x))
    obj_value = linear_val + k * sum(float(ci) for ci in c_vec) + kappa

    # Optional: print only nonzero decisions
    for i, (w_i, n_i) in enumerate(zip(w_sol, n_sol)):
        if w_i > 0:
            print(f"w[{i}] {float(w_i)}")
        if n_i > 0:
            print(f"n[{i}] {float(n_i)}")

    return ILPResult(k=k, w=w_sol, n=n_sol, obj_value=float(obj_value))


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
    k_candidates: Optional[Iterable[int]] = None,
    mip_gap: Optional[float] = 1e-4,
    plot: bool = True,
) -> HALDAResult:
    """
    HALDA end-to-end (parameters + Algorithm 1 with Gurobi or SciPy MILP).
    """
    Ks = sorted(set(k_candidates)) if k_candidates else valid_factors_of_L(model.L)
    best: Optional[HALDAResult] = None

    sets = assign_sets(devs)

    best_this_round: Optional[ILPResult] = None
    per_k_objs: List[Tuple[int, Optional[float]]] = []  # (k, obj or None if infeasible)
    print("Objectives by k")
    for kf in Ks:
        try:
            print("k: " + str(kf))
            res = solve_fixed_k_milp(
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
