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


def solve_fixed_k_minlp(
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
    disk_size = 2000000000000
    disk_speed_threshold = 51446428

    # Coefficients and constants (same as Gurobi version)
    a, b, c_vec = objective_vectors(devs, model, sets)
    kappa = kappa_constant(devs, model, sets)

    # Build the set of feasible y-patterns (respect slow-disk => y=0)
    forced_zero = [int(d.s_disk < disk_speed_threshold) for d in devs]
    patterns = []
    for bits in itertools.product([0, 1], repeat=M):
        ok = True
        for i in range(M):
            if forced_zero[i] and bits[i] == 1:
                ok = False
                break
        if ok:
            patterns.append(tuple(bits))

    best_obj = None
    best_sol = None
    best_y = None

    # Precompute per-device caps for n (Nmax) from VRAM/shared mem constraints
    Nmax = [W] * M
    for i, d in enumerate(devs):
        has_cuda = bool(d.has_cuda and d.d_avail_cuda is not None)
        has_metal = bool(d.has_metal and d.d_avail_metal is not None)
        if not (has_cuda or has_metal):
            Nmax[i] = 0
        else:
            caps = [W]
            if has_cuda:
                caps.append(max(0, float(d.d_avail_cuda) - float(d.c_gpu)))
            if has_metal:
                head = 1.0 if d.is_head else 0.0
                caps.append(
                    max(0, float(d.d_avail_metal) - float(d.c_gpu) - model.b_out * head)
                )
            Nmax[i] = max(0, int(min(caps)))

    for y_pat in patterns:
        # Variables: x = [w | n] of length 2M
        def idx_w(i):
            return i

        def idx_n(i):
            return M + i

        Nvars = 2 * M

        # Bounds
        lb = np.zeros(Nvars)
        ub = np.zeros(Nvars)
        for i in range(M):
            lb[idx_w(i)] = 1
            ub[idx_w(i)] = W
            lb[idx_n(i)] = 0
            ub[idx_n(i)] = Nmax[i]
        bounds = Bounds(lb, ub)

        integrality = np.zeros(Nvars, dtype=int)
        for i in range(M):
            integrality[idx_w(i)] = 1
            integrality[idx_n(i)] = 1

        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []

        for i in range(M):
            row = np.zeros(Nvars)
            row[idx_n(i)] = 1
            row[idx_w(i)] = -1
            A_ub.append(row)
            b_ub.append(0.0)

        row = np.zeros(Nvars)
        for i in range(M):
            row[idx_w(i)] = 1
        A_eq.append(row)
        b_eq.append(float(W))

        def bcio(i: int) -> float:
            return b_cio_b(devs[i], model)

        # M1: b' w_i <= d_avail_ram - bcio + disk_size*y_i
        for i in sets.get("M1", []):
            rhs = float(devs[i].d_avail_ram) - bcio(i) + disk_size * y_pat[i]
            row = np.zeros(Nvars)
            row[idx_w(i)] = bprime
            A_ub.append(row)
            b_ub.append(rhs)

        # M2: b' w_i <= d_avail_metal - bcio - c_gpu + disk_size*y_i
        for i in sets.get("M2", []):
            dav_metal = float(devs[i].d_avail_metal or 0)
            rhs = dav_metal - bcio(i) - float(devs[i].c_gpu) + disk_size * y_pat[i]
            row = np.zeros(Nvars)
            row[idx_w(i)] = bprime
            A_ub.append(row)
            b_ub.append(rhs)

        # M3: b' w_i - b' n_i <= d_avail_ram + dswap - bcio + disk_size*y_i
        for i in sets.get("M3", []):
            d = devs[i]
            dswap = (
                min(d.d_bytes_can_swap, d.d_swap_avail) if d.os_type == "android" else 0
            )
            rhs = float(d.d_avail_ram + dswap) - bcio(i) + disk_size * y_pat[i]
            row = np.zeros(Nvars)
            row[idx_w(i)] = bprime
            row[idx_n(i)] = -bprime
            A_ub.append(row)
            b_ub.append(rhs)

        # (4) VRAM/shared bounds: n_i <= cap
        for i, d in enumerate(devs):
            # CUDA
            if d.has_cuda and d.d_avail_cuda is not None:
                rhs = float(d.d_avail_cuda) - float(d.c_gpu)
                row = np.zeros(Nvars)
                row[idx_n(i)] = 1
                A_ub.append(row)
                b_ub.append(rhs)
            # Metal
            if d.has_metal and d.d_avail_metal is not None:
                head = 1.0 if d.is_head else 0.0
                rhs = float(d.d_avail_metal) - float(d.c_gpu) - model.b_out * head
                row = np.zeros(Nvars)
                row[idx_n(i)] = 1
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

        c_obj = np.zeros(Nvars)
        for i in range(M):
            coeff_w = float(a[i])
            coeff_n = float(b[i])
            # overload terms depending on case
            if i in sets.get("M1", []):
                coeff_w += (bprime / float(devs[i].s_disk)) * y_pat[i]
            elif i in sets.get("M2", []):
                coeff_w += (model.b_layer / float(devs[i].s_disk)) * y_pat[i]
            elif i in sets.get("M3", []):
                coeff_w += (bprime / float(devs[i].s_disk)) * y_pat[i]
                coeff_n += -(bprime / float(devs[i].s_disk)) * y_pat[i]
            # M4: no extra term
            c_obj[idx_w(i)] = k * coeff_w
            c_obj[idx_n(i)] = k * coeff_n

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
            continue

        x = res.x
        w_sol = [int(round(x[idx_w(i)])) for i in range(M)]
        n_sol = [int(round(x[idx_n(i)])) for i in range(M)]

        # Full objective value with constants
        linear_val = float(c_obj.dot(x))
        obj_value = linear_val + k * sum(float(ci) for ci in c_vec) + kappa

        if (best_obj is None) or (obj_value < best_obj):
            best_obj = obj_value
            best_sol = (w_sol, n_sol)
            best_y = y_pat

    if best_sol is None:
        raise RuntimeError("No feasible MINLP found for any y pattern (SciPy).")

    w_sol, n_sol = best_sol
    for i, (w_i, n_i, y_i) in enumerate(zip(w_sol, n_sol, best_y)):
        if w_i > 0:
            print(f"w[{i}] {float(w_i)}")
        if n_i > 0:
            print(f"n[{i}] {float(n_i)}")
        if y_i > 0:
            print(f"y[{i}] {float(y_i)}")

    return ILPResult(k=k, w=w_sol, n=n_sol, obj_value=float(best_obj))


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
            res = solve_fixed_k_minlp(
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
