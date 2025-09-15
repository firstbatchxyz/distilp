# halda_scipy.py
# HALDA (full variables) with SciPy MILP — single script.
# Implements: αm, βm, ξm; vectorized a,b,c; Pw, Pn, P^gpu_n; z, z_gpu; cases (M1..M4);
# fixed‑k ILPs; Algorithm 1 outer loop with calibration. SciPy ≥ 1.11 (optimize.milp).

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple, Literal
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from .components.dataclasses import DeviceProfile, ModelProfile, QuantPerf


# -----------------------------
# Math helpers (A.3)
# -----------------------------
def valid_factors_of_L(L: int) -> List[int]:
    ans = []
    for k in range(1, int(math.sqrt(L)) + 1):
        if L % k == 0:
            if k != L:
                ans.append(k)
            o = L // k
            if o != k and o != L:
                ans.append(o)
    return sorted(set(ans))


def b_prime(model: ModelProfile) -> int:
    return model.b_layer + 2 * (model.hk * model.ek + model.hv * model.ev) * model.n_kv


def _sum_f_over_S(f_by_q: QuantPerf, S_by_q: QuantPerf, Q: List[str]) -> float:
    return sum(f_by_q[q] / S_by_q[q] for q in Q if q in f_by_q and q in S_by_q)


def _gpu_table(dev: DeviceProfile) -> Optional[QuantPerf]:
    if dev.has_metal and dev.sgpu_metal:
        return dev.sgpu_metal
    if dev.has_cuda and dev.sgpu_cuda:
        return dev.sgpu_cuda
    return None


def _pick_T_gpu(dev: DeviceProfile) -> Optional[float]:
    if dev.has_metal and dev.T_metal:
        return dev.T_metal
    if dev.has_cuda and dev.T_cuda:
        return dev.T_cuda
    return None


def alpha_beta_xi(
    dev: DeviceProfile, model: ModelProfile
) -> Tuple[float, float, float]:
    bp = b_prime(model)
    comp_cpu = _sum_f_over_S(model.f_by_quant, dev.scpu, model.Q)
    alpha = comp_cpu + dev.t_kvcpy_cpu + (bp / dev.T_cpu)
    Sg = _gpu_table(dev)
    Tg = _pick_T_gpu(dev)
    if Sg is not None and Tg is not None:
        comp_gpu_minus_cpu = _sum_f_over_S(model.f_by_quant, Sg, model.Q) - comp_cpu
        beta = (
            comp_gpu_minus_cpu
            + (dev.t_kvcpy_gpu - dev.t_kvcpy_cpu)
            + (bp / Tg - bp / dev.T_cpu)
        )
    else:
        beta = 0.0
    xi = (dev.t_ram2vram + dev.t_vram2ram) * (
        0 if dev.is_unified_mem else 1
    ) + dev.t_comm
    return alpha, beta, xi


def _b_cio(dev: DeviceProfile, model: ModelProfile) -> float:
    # Eq. (34)
    return ((model.b_in / model.V) + model.b_out) * (
        1.0 if dev.is_head else 0.0
    ) + dev.c_cpu


# -----------------------------
# Case classification (28) - (33)
# -----------------------------
def _dswap_cap(dev: DeviceProfile) -> int:
    return (
        min(dev.d_bytes_can_swap, dev.d_swap_avail) if dev.os_type == "android" else 0
    )


def classify_device_case(
    dev: DeviceProfile,
    model: ModelProfile,
    w_m: int,
    n_m: int,
    k: int,
    sdisk_threshold: Optional[float] = None,
) -> int:
    if sdisk_threshold is not None and dev.s_disk < sdisk_threshold:
        return 4
    W = model.L // max(1, k)
    Lb = model.L * b_prime(model)
    bcio = _b_cio(dev, model)
    dswap = _dswap_cap(dev)

    def overload_ge(lhs, rhs):
        return lhs >= rhs + 1.0  # emulate strict >

    if dev.os_type == "mac_no_metal":
        return 1 if overload_ge(w_m * Lb, W * (dev.d_avail_ram - bcio)) else 4
    elif dev.os_type == "mac_metal":
        dav = float(dev.d_avail_metal or 0)
        return 2 if overload_ge(w_m * Lb, W * (dav - bcio - dev.c_gpu)) else 4
    else:
        return (
            3
            if overload_ge((w_m - n_m) * Lb, W * (dev.d_avail_ram + dswap - bcio))
            else 4
        )


def assign_sets(
    devs: List[DeviceProfile],
    model: ModelProfile,
    w: List[int],
    n: List[int],
    k: int,
    sdisk_threshold: Optional[float] = None,
) -> dict:
    M1, M2, M3, M4 = [], [], [], []
    for i, d in enumerate(devs):
        c = classify_device_case(d, model, w[i], n[i], k, sdisk_threshold)
        (M1 if c == 1 else M2 if c == 2 else M3 if c == 3 else M4).append(i)
    return {"M1": M1, "M2": M2, "M3": M3, "M4": M4}


# -----------------------------
# a,b,c, κ, z, z_gpu (A.3)
# -----------------------------
def objective_vectors(
    devs: List[DeviceProfile], model: ModelProfile, sets: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = len(devs)
    a = np.zeros(M)
    b = np.zeros(M)
    c = np.zeros(M)
    bp = b_prime(model)
    for i, d in enumerate(devs):
        alpha, beta, xi = alpha_beta_xi(d, model)
        c[i] = xi
        if i in sets["M1"]:
            a[i] = alpha + bp / d.s_disk
            b[i] = 0.0
        elif i in sets["M2"]:
            a[i] = alpha + model.b_layer / d.s_disk
            b[i] = beta
        elif i in sets["M3"]:
            a[i] = alpha + bp / d.s_disk
            b[i] = beta - bp / d.s_disk
        else:
            a[i] = alpha
            b[i] = beta
    return a, b, c


def kappa_constant(devs: List[DeviceProfile], model: ModelProfile, sets: dict) -> float:
    head_idx = next((i for i, d in enumerate(devs) if d.is_head), 0)
    head = devs[head_idx]
    head_compute = _sum_f_over_S(model.f_out_by_quant, head.scpu, model.Q)
    head_load_regs = (model.b_in / model.V + model.b_out) / head.T_cpu
    head_disk_in = model.b_in / (model.V * head.s_disk)
    head_disk_out = (
        (model.b_out / head.s_disk) if (head_idx not in sets.get("M4", [])) else 0.0
    )

    tail = 0.0
    for mi in sets.get("M1", []) + sets.get("M3", []):
        d = devs[mi]
        dswap = _dswap_cap(d)
        tail += (d.c_cpu - d.d_avail_ram - dswap) / d.s_disk
    return head_compute + head_load_regs + head_disk_in + head_disk_out + tail


def build_vectorized_bounds(
    devs: List[DeviceProfile], model: ModelProfile, sets: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Pw, Pn (for reference/debug), z, z_gpu per A.3 vectorization (page 19)."""
    # We will also use z,z_gpu directly to assemble the MILP constraints.
    M = len(devs)
    Lb = model.L * b_prime(model)
    # z is concatenated blocks: M1 | M2 | M3 | M4(1) | M4(2) | M4(3)
    z_blocks = []
    for i in sets["M1"]:
        d = devs[i]
        bcio = _b_cio(d, model)
        z_blocks.append((d.d_avail_ram - bcio) / Lb)
    for i in sets["M2"]:
        d = devs[i]
        bcio = _b_cio(d, model)
        dav = float(d.d_avail_metal or 0)
        z_blocks.append((dav - bcio - d.c_gpu) / Lb)
    for i in sets["M3"]:
        d = devs[i]
        bcio = _b_cio(d, model)
        dswap = _dswap_cap(d)
        z_blocks.append((d.d_avail_ram + dswap - bcio) / Lb)
    # M4 blocks (sign-flipped as in App. A.3)
    for i in sets["M4"]:
        d = devs[i]
        bcio = _b_cio(d, model)
        z_blocks.append((-d.d_avail_ram + bcio) / Lb)
    for i in sets["M4"]:
        d = devs[i]
        bcio = _b_cio(d, model)
        dav = float(d.d_avail_metal or 0)
        z_blocks.append((-dav + bcio + d.c_gpu) / Lb)
    for i in sets["M4"]:
        d = devs[i]
        bcio = _b_cio(d, model)
        dswap = _dswap_cap(d)
        z_blocks.append((-d.d_avail_ram - dswap + bcio) / Lb)
    z = np.array(z_blocks, dtype=float)

    # z_gpu is length M; P^gpu_n is diagonal {0/1}
    z_gpu = np.zeros(M, dtype=float)
    Pgpu = np.zeros(M, dtype=int)
    for i, d in enumerate(devs):
        if d.has_cuda and d.d_avail_cuda is not None:
            z_gpu[i] = (d.d_avail_cuda - d.c_gpu) / Lb
            Pgpu[i] = 1
        elif d.has_metal and d.d_avail_metal is not None:
            head = 1.0 if d.is_head else 0.0
            z_gpu[i] = (d.d_avail_metal - d.c_gpu - model.b_out * head) / Lb
            Pgpu[i] = 1
        else:
            z_gpu[i] = 0.0
            Pgpu[i] = 0

    # For completeness, provide symbolic Pw,Pn (not used directly in MILP assembly here).
    # Shapes: block-diagonal according to M1|M2|M3|M4a|M4b|M4c.
    def diag_eye(n):
        return np.eye(n, dtype=float) if n > 0 else np.zeros((0, 0))

    I1, I2, I3 = len(sets["M1"]), len(sets["M2"]), len(sets["M3"])
    I4 = len(sets["M4"])
    Pw = np.block(
        [
            [
                -diag_eye(I1),
                np.zeros((I1, I2)),
                np.zeros((I1, I3)),
                np.zeros((I1, I4)),
                np.zeros((I1, I4)),
                np.zeros((I1, I4)),
            ],
            [
                np.zeros((I2, I1)),
                -diag_eye(I2),
                np.zeros((I2, I3)),
                np.zeros((I2, I4)),
                np.zeros((I2, I4)),
                np.zeros((I2, I4)),
            ],
            [
                np.zeros((I3, I1)),
                np.zeros((I3, I2)),
                -diag_eye(I3),
                np.zeros((I3, I4)),
                np.zeros((I3, I4)),
                np.zeros((I3, I4)),
            ],
            [
                np.zeros((I4, I1)),
                np.zeros((I4, I2)),
                np.zeros((I4, I3)),
                diag_eye(I4),
                np.zeros((I4, I4)),
                np.zeros((I4, I4)),
            ],
            [
                np.zeros((I4, I1)),
                np.zeros((I4, I2)),
                np.zeros((I4, I3)),
                np.zeros((I4, I4)),
                diag_eye(I4),
                np.zeros((I4, I4)),
            ],
            [
                np.zeros((I4, I1)),
                np.zeros((I4, I2)),
                np.zeros((I4, I3)),
                np.zeros((I4, I4)),
                np.zeros((I4, I4)),
                diag_eye(I4),
            ],
        ]
    )
    Pn = np.block(
        [
            [
                np.zeros((I1, I1)),
                np.zeros((I1, I2)),
                np.zeros((I1, I3)),
                np.zeros((I1, I4)),
                np.zeros((I1, I4)),
                np.zeros((I1, I4)),
            ],
            [
                np.zeros((I2, I1)),
                np.zeros((I2, I2)),
                np.zeros((I2, I3)),
                np.zeros((I2, I4)),
                np.zeros((I2, I4)),
                np.zeros((I2, I4)),
            ],
            [
                np.zeros((I3, I1)),
                np.zeros((I3, I2)),
                diag_eye(I3),
                np.zeros((I3, I4)),
                np.zeros((I3, I4)),
                np.zeros((I3, I4)),
            ],
            [
                np.zeros((I4, I1)),
                np.zeros((I4, I2)),
                np.zeros((I4, I3)),
                np.zeros((I4, I4)),
                np.zeros((I4, I4)),
                np.zeros((I4, I4)),
            ],
            [
                np.zeros((I4, I1)),
                np.zeros((I4, I2)),
                np.zeros((I4, I3)),
                np.zeros((I4, I4)),
                np.zeros((I4, I4)),
                np.zeros((I4, I4)),
            ],
            [
                np.zeros((I4, I1)),
                np.zeros((I4, I2)),
                np.zeros((I4, I3)),
                np.zeros((I4, I4)),
                np.zeros((I4, I4)),
                -diag_eye(I4),
            ],
        ]
    )
    return Pw, Pn, z, z_gpu, Pgpu


# -----------------------------
# ILP for fixed k (6)–(10)
# -----------------------------
@dataclass
class ILPResult:
    k: int
    w: List[int]
    n: List[int]
    obj_value: float


def solve_fixed_k_ilp(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sets: dict,
    k: int,
    time_limit: Optional[float] = None,
    mip_gap: float = 1e-4,
) -> ILPResult:
    M = len(devs)
    L = model.L
    W = L // k
    assert L % k == 0
    a, b, c = objective_vectors(devs, model, sets)
    kappa = kappa_constant(devs, model, sets)
    _, _, z, z_gpu, Pgpu = build_vectorized_bounds(devs, model, sets)
    Lb = L * b_prime(model)

    # Variables x = [w(0..M-1), n(0..M-1)]
    N = 2 * M
    lb = np.zeros(N)
    ub = np.full(N, W, dtype=float)
    for i in range(M):
        lb[i] = 1  # w_i >= 1
        lb[M + i] = 0  # n_i >= 0
    bounds = Bounds(lb, ub)

    # Integrality: all integers
    integrality = np.ones(N, dtype=int)

    A = []
    l = []
    u = []

    # (7) n_i <= w_i
    for i in range(M):
        row = np.zeros(N)
        row[i] *= -1
        row[M + i] += 1
        A.append(row)
        l.append(-np.inf)
        u.append(0.0)

    # (8) sum w == W
    row = np.zeros(N)
    row[:M] = 1.0
    Aeq = LinearConstraint(row, lb=W, ub=W)

    # RAM / case constraints via (9): we directly materialize device-wise inequalities.
    eps = 1e-9

    # Helper for b_cio and dswap
    def bcio(i):
        return _b_cio(devs[i], model)

    def dswap(i):
        return _dswap_cap(devs[i])

    # M1
    for i in sets["M1"]:
        row = np.zeros(N)
        row[i] += -1.0
        rhs = -W * ((devs[i].d_avail_ram - bcio(i)) / Lb) - eps
        A.append(row)
        l.append(-np.inf)
        u.append(rhs)
    # M2
    for i in sets["M2"]:
        dav = float(devs[i].d_avail_metal or 0)
        row = np.zeros(N)
        row[i] += -1.0
        rhs = -W * ((dav - bcio(i) - devs[i].c_gpu) / Lb) - eps
        A.append(row)
        l.append(-np.inf)
        u.append(rhs)
    # M3
    for i in sets["M3"]:
        row = np.zeros(N)
        row[i] += -1.0
        row[M + i] += 1.0
        rhs = -W * ((devs[i].d_avail_ram + dswap(i) - bcio(i)) / Lb) - eps
        A.append(row)
        l.append(-np.inf)
        u.append(rhs)
    # M4 upper bounds depend on OS
    for i in sets["M4"]:
        d = devs[i]
        if d.os_type == "mac_no_metal":
            row = np.zeros(N)
            row[i] += +1.0
            rhs = -W * ((-d.d_avail_ram + bcio(i)) / Lb) - eps
            A.append(row)
            l.append(-np.inf)
            u.append(rhs)
        elif d.os_type == "mac_metal":
            dav = float(d.d_avail_metal or 0)
            row = np.zeros(N)
            row[i] += +1.0
            rhs = -W * ((-dav + bcio(i) + d.c_gpu) / Lb) - eps
            A.append(row)
            l.append(-np.inf)
            u.append(rhs)
        else:  # linux/android: (w - n) <= W*(d_avail + dswap - bcio)/Lb
            row = np.zeros(N)
            row[i] += +1.0
            row[M + i] += -1.0
            rhs = -W * ((-d.d_avail_ram - dswap(i) + bcio(i)) / Lb) - eps
            A.append(row)
            l.append(-np.inf)
            u.append(rhs)

    # (10) VRAM / shared-mem: n_i <= W * z_gpu_i  (or 0 if no GPU)
    for i in range(M):
        row = np.zeros(N)
        row[M + i] = 1.0
        rhs = W * z_gpu[i] if Pgpu[i] == 1 else 0.0
        A.append(row)
        l.append(-np.inf)
        u.append(rhs)

    A_ineq = LinearConstraint(np.array(A), lb=np.array(l), ub=np.array(u))

    # Objective: min k*(a^T w + b^T n)  (constants handled after solve for comparability)
    cvec = np.concatenate([k * a, k * b])

    res = milp(
        c=cvec,
        constraints=[A_ineq, Aeq],
        integrality=integrality,
        bounds=bounds,
        options={
            "mip_rel_gap": float(mip_gap),
            **({"time_limit": float(time_limit)} if time_limit else {}),
        },
    )
    if res.status not in (0,):  # 0: OPTIMAL
        raise RuntimeError(f"HiGHS MILP status {res.status}")

    x = np.rint(res.x).astype(int)
    w = x[:M].tolist()
    n = x[M:].tolist()
    obj = float(k * (a @ x[:M] + b @ x[M:] + np.sum(c)) + kappa)
    return ILPResult(k=k, w=w, n=n, obj_value=obj)


# -----------------------------
# HALDA outer loop (Alg. 1)
# -----------------------------
@dataclass
class HALDAResult:
    w: List[int]
    n: List[int]
    k: int
    obj_value: float
    sets: dict
    iterations: int
    forced_M4: List[int]


def halda_solve(
    devs: List[DeviceProfile],
    model: ModelProfile,
    sdisk_threshold: Optional[float] = None,
    k_candidates: Optional[Iterable[int]] = None,
    time_limit_per_k: Optional[float] = None,
    mip_gap: float = 1e-4,
    max_outer_iters: int = 50,
) -> HALDAResult:
    Ks = sorted(set(k_candidates)) if k_candidates else valid_factors_of_L(model.L)

    # init w by memory budgets; n=0 (Alg.1 line 1)
    def mem_cap(d: DeviceProfile) -> float:
        if d.os_type == "mac_metal":
            return float(d.d_avail_metal or 0)
        if d.os_type == "android":
            return d.d_avail_ram + _dswap_cap(d)
        return d.d_avail_ram

    caps = [max(mem_cap(d), 1.0) for d in devs]
    total = sum(caps)
    W0 = model.L // 1
    w = [max(1, round(W0 * c / total)) for c in caps]
    delta = W0 - sum(w)
    for i in range(abs(delta)):
        w[i % len(devs)] += 1 if delta > 0 else -1
    n = [0] * len(devs)

    forced_M4 = []
    prev_sets = None
    best = None

    for outer in range(1, max_outer_iters + 1):
        W = sum(w)
        if W == 0 or model.L % W != 0:
            k_now = Ks[0]
            W = model.L // k_now
            w = [max(1, round(W * c / total)) for c in caps]
            delta = W - sum(w)
            for i in range(abs(delta)):
                w[i % len(devs)] += 1 if delta > 0 else -1
        k_now = model.L // W

        sets = assign_sets(devs, model, w, n, k_now, sdisk_threshold)
        if forced_M4:
            for idx in forced_M4:
                for s in ("M1", "M2", "M3"):
                    if idx in sets[s]:
                        sets[s].remove(idx)
                if idx not in sets["M4"]:
                    sets["M4"].append(idx)

        if prev_sets is not None and all(
            set(sets[k]) == set(prev_sets[k]) for k in ("M1", "M2", "M3", "M4")
        ):
            break
        prev_sets = {k: list(v) for k, v in sets.items()}

        best_round = None
        for kf in Ks:
            try:
                res = solve_fixed_k_ilp(
                    devs, model, sets, kf, time_limit_per_k, mip_gap
                )
                if best_round is None or res.obj_value < best_round.obj_value:
                    best_round = res
            except RuntimeError:
                continue
        if best_round is None:
            raise RuntimeError("No feasible ILP for any k.")

        # calibration (Alg.1 lines 13–15)
        overloaded = len(sets["M1"] + sets["M2"] + sets["M3"]) > 0

        def vram_slack_bytes(i: int, n_i: int, k_fixed: int) -> float:
            Wf = model.L // k_fixed
            d = devs[i]
            Lb = model.L * b_prime(model)
            slack = 0.0
            if d.has_cuda and d.d_avail_cuda is not None:
                slack = max(slack, Wf * (d.d_avail_cuda - d.c_gpu) - n_i * Lb)
            if d.has_metal and d.d_avail_metal is not None:
                head = 1.0 if d.is_head else 0.0
                slack = max(
                    slack,
                    Wf * (d.d_avail_metal - d.c_gpu - model.b_out * head) - n_i * Lb,
                )
            return slack

        any_free_vram = any(
            vram_slack_bytes(i, best_round.n[i], best_round.k) > 1.0
            for i in range(len(devs))
        )
        if overloaded and any_free_vram:
            cand = sets["M1"] + sets["M2"] + sets["M3"]
            if cand:
                slowest = min(cand, key=lambda idx: devs[idx].s_disk)
                if slowest not in forced_M4:
                    forced_M4.append(slowest)
            continue

        # accept best of this round
        w = list(best_round.w)
        n = list(best_round.n)
        if best is None or best_round.obj_value < best.obj_value:
            best = HALDAResult(
                w=w,
                n=n,
                k=best_round.k,
                obj_value=best_round.obj_value,
                sets={k: list(v) for k, v in sets.items()},
                iterations=outer,
                forced_M4=list(forced_M4),
            )

    if best is None:
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


# -----------------------------
# Minimal usage example
# -----------------------------
if __name__ == "__main__":
    MiB = 1024**2
    GiB = 1024**3
    model = ModelProfile(
        L=70,
        b_layer=1 * GiB,
        b_in=64 * MiB,
        b_out=128 * MiB,
        hk=32,
        ek=256,
        hv=32,
        ev=256,
        n_kv=0,
        e_embed=8192,
        V=32000,
        f_by_quant={"Q4_K": 1.2e12, "Q8_0": 1.8e12, "F16": 3.5e12},
        f_out_by_quant={"Q4_K": 2.0e12, "Q8_0": 3.0e12, "F16": 6.0e12},
    )
    devs = [
        DeviceProfile(
            name="MacBookPro-M2Max",
            os_type="mac_metal",
            is_head=True,
            is_unified_mem=True,
            has_cuda=False,
            has_metal=True,
            scpu={"Q4_K": 2.0e12, "Q8_0": 1.4e12, "F16": 0.8e12},
            T_cpu=30e9,
            t_kvcpy_cpu=0.0015,
            t_kvcpy_gpu=0.0010,
            t_ram2vram=0.0010,
            t_vram2ram=0.0010,
            t_comm=0.0008,
            s_disk=2.5e9,
            d_avail_ram=64 * GiB,
            sgpu_metal={"Q4_K": 15e12, "Q8_0": 11e12, "F16": 20e12},
            T_metal=220e9,
            d_avail_metal=48 * GiB,
            c_cpu=512 * MiB,
            c_gpu=256 * MiB,
        ),
        DeviceProfile(
            name="MacMini-M1",
            os_type="mac_metal",
            is_head=False,
            is_unified_mem=True,
            has_cuda=False,
            has_metal=True,
            scpu={"Q4_K": 1.2e12, "Q8_0": 0.9e12, "F16": 0.5e12},
            T_cpu=18e9,
            t_kvcpy_cpu=0.0022,
            t_kvcpy_gpu=0.0014,
            t_ram2vram=0.0015,
            t_vram2ram=0.0015,
            t_comm=0.0012,
            s_disk=1.8e9,
            d_avail_ram=24 * GiB,
            sgpu_metal={"Q4_K": 8e12, "Q8_0": 6e12, "F16": 10e12},
            T_metal=140e9,
            d_avail_metal=5 * GiB,
            c_cpu=384 * MiB,
            c_gpu=int(4.5 * GiB),
        ),
        DeviceProfile(
            name="WS-RTX4090",
            os_type="linux",
            is_head=False,
            is_unified_mem=False,
            has_cuda=True,
            has_metal=False,
            scpu={"Q4_K": 1.8e12, "Q8_0": 1.2e12, "F16": 0.7e12},
            T_cpu=22e9,
            t_kvcpy_cpu=0.0018,
            t_kvcpy_gpu=0.0010,
            t_ram2vram=0.0020,
            t_vram2ram=0.0020,
            t_comm=0.0009,
            s_disk=3.0e9,
            d_avail_ram=128 * GiB,
            sgpu_cuda={"Q4_K": 42e12, "Q8_0": 30e12, "F16": 80e12},
            T_cuda=520e9,
            d_avail_cuda=24 * GiB,
            c_cpu=512 * MiB,
            c_gpu=256 * MiB,
        ),
        DeviceProfile(
            name="Intel-iMac-Old",
            os_type="mac_no_metal",
            is_head=False,
            is_unified_mem=False,
            has_cuda=False,
            has_metal=False,
            scpu={"Q4_K": 1.0e12, "Q8_0": 0.7e12, "F16": 0.4e12},
            T_cpu=16e9,
            t_kvcpy_cpu=0.0025,
            t_kvcpy_gpu=0.0025,
            t_ram2vram=0.0030,
            t_vram2ram=0.0030,
            t_comm=0.0014,
            s_disk=1.0e9,
            d_avail_ram=6 * GiB,
            c_cpu=int(5.5 * GiB),
            c_gpu=0,
        ),
    ]
    res = halda_solve(
        devs, model, time_limit_per_k=5.0, mip_gap=1e-4, max_outer_iters=25
    )
    print("k*=", res.k, "\nw*=", res.w, "\nn*=", res.n, "\nobj=", res.obj_value)
    print(
        "sets:",
        {k: sorted(v) for k, v in res.sets.items()},
        "forced_M4:",
        res.forced_M4,
        "iters:",
        res.iterations,
    )
