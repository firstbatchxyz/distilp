"""
HALDA with SciPy (HiGHS MILP).

This is the SciPy-based replacement for the Gurobi MILP implementation in this file.
It keeps the *same* variables/constraints structure you currently have (including the
modified scheduling constraints) and preserves the debug prints + Gantt plot behavior
for k == 2.

Solver backend: scipy.optimize.milp (HiGHS)

Requirements:
    scipy >= 1.11 recommended (milp + HiGHS options)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


from .components.plotter import plot_k_curve
from .components.dense_common import (
    b_cio_b,
    b_prime,
    b_prime_adj,
    assign_sets,
    objective_vectors,
    kappa_constant,
    valid_factors_of_L,
    DeviceProfile,
    ModelProfile,
    ILPResult,
    HALDAResult,
)


def plot_schedule_gantt_ordered(
    devs,
    Start,
    End,
    busy_times,
    fetch_times,
    k: int,
    title: str | None = None,
):
    """
    Gantt chart ordered by:
        round → compute start time → device index

    Shows for each round/device:
        - Prefetch (blue)
        - Compute (orange)
        - Communication t_comm (green)

    This function works with either:
      - Gurobi vars/exprs (uses .X and .getValue()), OR
      - numeric arrays/lists (SciPy path).
    """

    def _val(z) -> float:
        # Gurobi Var has .X; SciPy path uses floats
        if hasattr(z, "X"):
            return float(z.X)
        return float(z)

    def _expr_val(z) -> float:
        # Gurobi LinExpr has .getValue(); SciPy path uses floats
        if hasattr(z, "getValue"):
            return float(z.getValue())
        return float(z)

    M = len(devs)

    busy_val = [_expr_val(bt) for bt in busy_times]
    fetch_val = [_expr_val(ft) for ft in fetch_times]
    tcomm_val = [float(d.t_comm) for d in devs]

    fig, ax = plt.subplots(figsize=(14, max(4, 0.5 * M * k)))

    y = 0
    yticks = []
    ylabels = []

    # Legend flags
    added_prefetch = False
    added_compute = False
    added_comm = False

    # Loop by round first
    for r in range(k):
        # Sort devices in this round by actual compute start time
        order = sorted(range(M), key=lambda i: _val(Start[i, r]))

        for i in order:
            comp_start = _val(Start[i, r])
            comp_dur = busy_val[i]

            # Prefetch start logic
            if r == 0:
                prefetch_start = 0.0
            else:
                prefetch_start = _val(End[i, r - 1])
            prefetch_dur = fetch_val[i]

            # Communication interval
            comm_start = _val(End[i, r])
            comm_dur = tcomm_val[i]

            # Labels
            yticks.append(y)
            ylabels.append(f"Round {r} – Dev {i}")

            # === PREFETCH BAR (blue) ===
            ax.broken_barh(
                [(prefetch_start, prefetch_dur)],
                (y - 0.35, 0.25),
                facecolor="tab:blue",
                edgecolor="black",
                label=None if added_prefetch else "prefetch",
            )
            if not added_prefetch:
                added_prefetch = True

            # === COMPUTE BAR (orange) ===
            ax.broken_barh(
                [(comp_start, comp_dur)],
                (y - 0.05, 0.25),
                facecolor="tab:orange",
                edgecolor="black",
                label=None if added_compute else "compute",
            )
            if not added_compute:
                added_compute = True

            # === COMM BAR (green) ===
            ax.broken_barh(
                [(comm_start, comm_dur)],
                (y + 0.25, 0.20),
                facecolor="tab:green",
                edgecolor="black",
                label=None if added_comm else "comm (t_comm)",
            )
            if not added_comm:
                added_comm = True

            y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("time")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    if title is None:
        title = f"HALDA Gantt (k={k})"
    ax.set_title(title)

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


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
    Solve fixed-k HALDA MILP using SciPy (HiGHS via scipy.optimize.milp),
    matching the *current* model in this file.

    Variables:
      - Integer: w[i], n[i], s1[i], s2[i], s3[i], t[i]
      - Continuous: Start[i,r], End[i,r]
    Constraints:
      - n[i] <= w[i]
      - sum_i w[i] == W
      - memory constraints (M1/M2/M3) + GPU caps with overflow t[i]
      - scheduling constraints:
          End[i,r] == Start[i,r] + busy[i]
          Start[i,0] >= fetch[i]
          Start[i,0] >= End[i-1,0] + t_comm[i-1]  (for i>0)
          For r>=1:
            Start[0,r] >= End[M-1,r-1] + t_comm[M-1]
            Start[i,r] >= End[i-1,r] + t_comm[i-1] (for i>0)
            Start[i,r] >= End[i,r-1] + fetch[i]
    Objective:
      - minimize End[M-1,k-1] + devs[M-1].t_comm + kappa
    """
    M = len(devs)
    if model.L % k != 0:
        raise ValueError(f"model.L={model.L} must be divisible by k={k}")
    W = model.L // k

    bprime = float(b_prime(model, kv_bits_k=kv_factor))
    bprime_adj = float(b_prime_adj(model, kv_bits_k=kv_factor))

    a, b, c_vec = objective_vectors(devs, model, sets, kv_factor)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c_vec = np.asarray(c_vec, dtype=float)

    kappa = float(kappa_constant(devs, model, sets))

    M1 = set(sets.get("M1", []))
    M2 = set(sets.get("M2", []))
    M3 = set(sets.get("M3", []))

    # -------------------------
    # Variable indexing in x
    # -------------------------
    # Integers: w, n, s1, s2, s3, t (each length M)
    # Continuous: Start(M*k), End(M*k)
    off_w = 0
    off_n = off_w + M
    off_s1 = off_n + M
    off_s2 = off_s1 + M
    off_s3 = off_s2 + M
    off_t = off_s3 + M
    off_Start = off_t + M
    off_End = off_Start + M * k
    n_vars = off_End + M * k

    def idx_w(i: int) -> int: return off_w + i
    def idx_n(i: int) -> int: return off_n + i
    def idx_s1(i: int) -> int: return off_s1 + i
    def idx_s2(i: int) -> int: return off_s2 + i
    def idx_s3(i: int) -> int: return off_s3 + i
    def idx_t(i: int) -> int: return off_t + i
    def idx_Start(i: int, r: int) -> int: return off_Start + i * k + r
    def idx_End(i: int, r: int) -> int: return off_End + i * k + r

    # -------------------------
    # Bounds + integrality
    # -------------------------
    lb = np.zeros(n_vars, dtype=float)
    ub = np.full(n_vars, np.inf, dtype=float)
    integrality = np.zeros(n_vars, dtype=int)

    # Integer vars bounds
    for i, d in enumerate(devs):
        has_cuda = bool(getattr(d, "has_cuda", False) and getattr(d, "d_avail_cuda", None) is not None)
        has_metal = bool(getattr(d, "has_metal", False) and getattr(d, "d_avail_metal", None) is not None)
        has_gpu = has_cuda or has_metal

        # w[i] in [1, W]
        lb[idx_w(i)] = 1
        ub[idx_w(i)] = W
        integrality[idx_w(i)] = 1

        # n[i] in [0,W] if has_gpu else fixed 0
        lb[idx_n(i)] = 0
        ub[idx_n(i)] = W if has_gpu else 0
        integrality[idx_n(i)] = 1

        # s1/s2/s3 only allowed on their set
        lb[idx_s1(i)] = 0
        ub[idx_s1(i)] = W if i in M1 else 0
        integrality[idx_s1(i)] = 1

        lb[idx_s2(i)] = 0
        ub[idx_s2(i)] = W if i in M2 else 0
        integrality[idx_s2(i)] = 1

        lb[idx_s3(i)] = 0
        ub[idx_s3(i)] = W if i in M3 else 0
        integrality[idx_s3(i)] = 1

        # t[i] in [0,W] if has_gpu else fixed 0
        lb[idx_t(i)] = 0
        ub[idx_t(i)] = W if has_gpu else 0
        integrality[idx_t(i)] = 1

    # Start/End are continuous and already have lb=0, ub=inf
    bounds = Bounds(lb, ub)

    # -------------------------
    # Build linear constraints A x in [clb, cub]
    # -------------------------
    data: List[float] = []
    rows: List[int] = []
    cols: List[int] = []
    clb: List[float] = []
    cub: List[float] = []

    def add_row(coefs: Dict[int, float], row_lb: float, row_ub: float):
        r = len(clb)
        clb.append(float(row_lb))
        cub.append(float(row_ub))
        for j, v in coefs.items():
            v = float(v)
            if v != 0.0:
                rows.append(r)
                cols.append(int(j))
                data.append(v)

    # n[i] <= w[i]  ->  n[i] - w[i] <= 0
    for i in range(M):
        add_row({idx_n(i): 1.0, idx_w(i): -1.0}, -np.inf, 0.0)

    # sum_i w[i] == W
    add_row({idx_w(i): 1.0 for i in range(M)}, W, W)

    # Memory helper
    def bcio(i: int) -> float:
        return float(b_cio_b(devs[i], model))

    # M1: bprime*w - bprime_adj*s1 <= d_avail_ram - bcio
    for i in M1:
        rhs_cap = float(devs[i].d_avail_ram) - bcio(i)
        add_row({idx_w(i): bprime, idx_s1(i): -bprime_adj}, -np.inf, rhs_cap)

    # M2: bprime_adj*w - bprime_adj*s2 <= d_avail_metal - bcio - c_gpu
    for i in M2:
        if devs[i].d_avail_metal is None:
            continue
        rhs_cap = float(devs[i].d_avail_metal) - bcio(i) - float(devs[i].c_gpu)
        add_row({idx_w(i): bprime_adj, idx_s2(i): -bprime_adj}, -np.inf, rhs_cap)

    # M3: bprime_adj*w - bprime_adj*n - bprime_adj*s3 <= d_avail_ram + dswap - bcio
    for i in M3:
        d = devs[i]
        dswap = (
            float(min(d.d_bytes_can_swap, d.d_swap_avail))
            if getattr(d, "os_type", None) == "android"
            else 0.0
        )
        rhs_cap = float(d.d_avail_ram) + dswap - bcio(i)
        add_row(
            {idx_w(i): bprime_adj, idx_n(i): -bprime_adj, idx_s3(i): -bprime_adj},
            -np.inf,
            rhs_cap,
        )

    # VRAM / shared caps with single overflow t[i] used in both inequalities (no double charge)
    for i, d in enumerate(devs):
        if getattr(d, "has_cuda", False) and getattr(d, "d_avail_cuda", None) is not None:
            rhs = float(d.d_avail_cuda) - float(d.c_gpu)
            add_row({idx_n(i): bprime_adj, idx_t(i): -bprime_adj}, -np.inf, rhs)

        if getattr(d, "has_metal", False) and getattr(d, "d_avail_metal", None) is not None:
            head = 1.0 if getattr(d, "is_head", False) else 0.0
            rhs = float(d.d_avail_metal) - float(d.c_gpu) - float(model.b_out) * head
            add_row({idx_n(i): bprime_adj, idx_t(i): -bprime_adj}, -np.inf, rhs)

    # Scheduling model pieces
    # busy[i] = sum(coefs*intvars) + constB
    # fetch[i] = fetch_coef[i] * w[i]
    busy_int_coefs: List[Dict[int, float]] = []
    busy_const: List[float] = []
    fetch_coef: List[float] = []

    for i in range(M):
        s_disk_i = max(1.0, float(devs[i].s_disk))  # avoid divide-by-zero

        penM1 = bprime / s_disk_i
        penM2 = float(model.b_layer) / s_disk_i
        penM3 = bprime / s_disk_i
        penVRAM = penM2 if i in M2 else penM3

        busy_int_coefs.append(
            {
                idx_w(i): float(a[i]),
                idx_n(i): float(b[i]),
                idx_s1(i): penM1,
                idx_s2(i): penM2,
                idx_s3(i): penM3,
                idx_t(i): penVRAM,
            }
        )
        busy_const.append(float(c_vec[i]))
        fetch_coef.append(bprime / s_disk_i)

    def pred(i: int) -> int:
        return (i - 1) % M

    # End[i,r] == Start[i,r] + busy[i]
    # => End - Start - sum(intcoefs) == const
    for i in range(M):
        for r in range(k):
            coefs = {idx_End(i, r): 1.0, idx_Start(i, r): -1.0}
            for j, v in busy_int_coefs[i].items():
                coefs[j] = coefs.get(j, 0.0) - float(v)
            add_row(coefs, busy_const[i], busy_const[i])

    # Start[i,0] >= fetch[i]  => Start - fetch_coef*w >= 0
    for i in range(M):
        add_row({idx_Start(i, 0): 1.0, idx_w(i): -fetch_coef[i]}, 0.0, np.inf)

    # Round 0 comm: for i>0: Start[i,0] >= End[p,0] + t_comm[p]
    for i in range(1, M):
        p = pred(i)
        add_row({idx_Start(i, 0): 1.0, idx_End(p, 0): -1.0}, float(devs[p].t_comm), np.inf)

    # For rounds >=1
    if k > 1:
        for i in range(M):
            for r in range(1, k):
                p = pred(i)
                if i == 0:
                    # Start[0,r] >= End[M-1,r-1] + t_comm[M-1]
                    add_row(
                        {idx_Start(i, r): 1.0, idx_End(p, r - 1): -1.0},
                        float(devs[p].t_comm),
                        np.inf,
                    )
                else:
                    # Start[i,r] >= End[i-1,r] + t_comm[i-1]
                    add_row(
                        {idx_Start(i, r): 1.0, idx_End(p, r): -1.0},
                        float(devs[p].t_comm),
                        np.inf,
                    )

                # Start[i,r] >= End[i,r-1] + fetch[i]
                add_row(
                    {idx_Start(i, r): 1.0, idx_End(i, r - 1): -1.0, idx_w(i): -fetch_coef[i]},
                    0.0,
                    np.inf,
                )

    A = coo_matrix(
        (np.asarray(data, dtype=float), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(len(clb), n_vars),
    ).tocsr()

    constraints = LinearConstraint(A, np.asarray(clb, dtype=float), np.asarray(cub, dtype=float))

    # -------------------------
    # Objective: minimize End[M-1,k-1] (+ constants added after solve)
    # -------------------------
    c = np.zeros(n_vars, dtype=float)
    c[idx_End(M - 1, k - 1)] = 1.0

    options = {}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    if mip_gap is not None:
        options["mip_rel_gap"] = float(mip_gap)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options=options)

    if (not getattr(res, "success", False)) or res.x is None:
        raise RuntimeError(f"No feasible MILP found. HiGHS status: {getattr(res, 'message', res)}")

    x = np.asarray(res.x, dtype=float)

    # Extract integer decisions (rounded for safety)
    w_sol = [int(round(x[idx_w(i)])) for i in range(M)]
    n_sol = [int(round(x[idx_n(i)])) for i in range(M)]

    # Extract Start/End for debug/plot
    Start_val = np.zeros((M, k), dtype=float)
    End_val = np.zeros((M, k), dtype=float)
    for i in range(M):
        for r in range(k):
            Start_val[i, r] = float(x[idx_Start(i, r)])
            End_val[i, r] = float(x[idx_End(i, r)])

    # Compute busy/fetch values (numeric) for printing and Gantt
    busy_times = []
    fetch_times = []
    for i in range(M):
        # use x (not rounded) for consistency with solved model
        wi = float(x[idx_w(i)])
        ni = float(x[idx_n(i)])
        s1i = float(x[idx_s1(i)])
        s2i = float(x[idx_s2(i)])
        s3i = float(x[idx_s3(i)])
        ti = float(x[idx_t(i)])

        s_disk_i = max(1.0, float(devs[i].s_disk))
        penM1 = bprime / s_disk_i
        penM2 = float(model.b_layer) / s_disk_i
        penM3 = bprime / s_disk_i
        penVRAM = penM2 if i in M2 else penM3

        constB = float(c_vec[i])

        busy = (
            float(a[i]) * wi
            + float(b[i]) * ni
            + penM1 * s1i
            + penM2 * s2i
            + penM3 * s3i
            + penVRAM * ti
            + constB
        )
        fetch = (bprime / s_disk_i) * wi

        busy_times.append(float(busy))
        fetch_times.append(float(fetch))

    # Debug prints + Gantt plot (as in the original code)
    if k == 2:
        # for parity with old Gurobi code:
        for kprime in range(k):
            for i in range(M):
                print(
                    "Prefetch time of device "
                    + str(i)
                    + " at round "
                    + str(kprime)
                    + " : "
                    + str(fetch_times[i])
                )
                print(
                    "Start of device "
                    + str(i)
                    + " at round "
                    + str(kprime)
                    + " : "
                    + str(Start_val[i, kprime])
                )
                print(
                    "Compute time of device "
                    + str(i)
                    + " at round "
                    + str(kprime)
                    + " : "
                    + str(busy_times[i])
                )
                print(
                    "End of device "
                    + str(i)
                    + " at round "
                    + str(kprime)
                    + " : "
                    + str(End_val[i, kprime])
                )
                print("T_comm of device " + str(i) + " is: " + str(devs[i].t_comm))

        plot_schedule_gantt_ordered(
            devs=devs,
            Start=Start_val,
            End=End_val,
            busy_times=busy_times,
            fetch_times=fetch_times,
            k=k,
            title=f"HALDA timeline (k={k})",
        )

    # Objective value = End[M-1,k-1] + devs[M-1].t_comm + kappa
    linear_val = float(res.fun)
    obj_value = float(linear_val + float(devs[M - 1].t_comm) + kappa)

    return ILPResult(k=k, w=w_sol, n=n_sol, obj_value=obj_value)


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
    HALDA with SciPy MILP (HiGHS).
    """
    Ks = sorted(set(k_candidates)) if k_candidates else valid_factors_of_L(model.L)
    kv_factor = _kv_bits_to_factor(kv_bits)
    best: Optional[HALDAResult] = None

    sets = assign_sets(devs)

    best_this_round: Optional[ILPResult] = None
    per_k_objs: List[Tuple[int, Optional[float]]] = []

    if debug:
        print("Objectives by k")

    for kf in Ks:
        try:
            if debug:
                print("k: " + str(kf))
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

    if (best is None) or (best_this_round.obj_value < best.obj_value):
        best = HALDAResult(
            w=w,
            n=n,
            k=best_this_round.k,
            obj_value=best_this_round.obj_value,
            sets={kk: list(vv) for kk, vv in sets.items()},
        )

    if plot:
        plot_k_curve(
            per_k_objs,
            k_star=(best.k if best is not None else None),
            title="HALDA: k vs objective (final sweep)",
        )

    return best
