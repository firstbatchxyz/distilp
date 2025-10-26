from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import math

from .dataclasses import DeviceProfile, ModelProfile, QuantPerf


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


def b_prime(model: ModelProfile) -> int:
    """
    b' = b + 2(h_k e_k + h_v e_v) · n_kv.  [App. A.3, after Assumption 1]
    """
    return model.b_layer + 2 * (model.hk * model.ek + model.hv * model.ev) * model.n_kv


def _sum_f_over_S(
    f_by_q: Dict[str, QuantPerf],
    S_by_q: Dict[str, QuantPerf],
    q: Literal["Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"],
    batch_size: int = 1,
) -> float:
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
                raise ValueError(
                    f"Batch size {batch_size} (key '{batch_key}') not found in S_by_q[{q}]"
                )
            s_val = S_by_q[q][batch_key]
        else:
            # Old format compatibility - direct float values
            # Get rid of this branch once all data is updated
            s_val = S_by_q[q]

        # Handle f_by_q which might also have batch sizes in future
        if isinstance(f_by_q, dict):
            if batch_key not in f_by_q:
                raise ValueError(
                    f"Batch size {batch_size} (key '{batch_key}') not found in f_by_q[{q}]"
                )
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
    comp_cpu = _sum_f_over_S(model.f_q, dev.scpu, model.Q)
    alpha = comp_cpu + dev.t_kvcpy_cpu + (bprime / dev.T_cpu)

    # β_m (GPU minus CPU path), 0 if no GPU available
    S_gpu = _gpu_table(dev)
    T_gpu = _pick_T_gpu(dev)
    if S_gpu is not None and T_gpu is not None:
        comp_gpu_minus_cpu = _sum_f_over_S(model.f_q, S_gpu, model.Q) - comp_cpu
        beta = (
            comp_gpu_minus_cpu
            + (dev.t_kvcpy_gpu - dev.t_kvcpy_cpu)
            + (bprime / T_gpu - bprime / dev.T_cpu)
        )
    else:
        beta = 0.0

    # ξ_m (traffic + comm)
    # dev.t_ram2vram + dev.t_vram2ram is done once per round as it is done for sequence of layers within a window.
    # xi = (dev.t_ram2vram + dev.t_vram2ram) * (
    #     0 if dev.is_unified_mem else 1
    # ) + dev.t_comm

    xi = (dev.t_ram2vram + dev.t_vram2ram) * (
        0 if dev.is_unified_mem else 1
    )
    return alpha, beta, xi


def b_cio_b(dev: DeviceProfile, model: ModelProfile) -> float:
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


@dataclass
class ILPResult:
    k: int
    w: List[int]
    n: List[int]
    obj_value: float  # reported objective value for comparison across k


@dataclass
class HALDAResult:
    w: List[int]  # w* (layer windows per device)
    n: List[int]  # n* (GPU layers per device)
    k: int  # best k
    obj_value: float  # objective value for the best (w*,n*,k)
    sets: Dict[str, List[int]]  # final sets M1..M4
