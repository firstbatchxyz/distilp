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


def b_prime(
    model: ModelProfile,
    kv_bits_k: float = 1.0,
    kv_bits_v: float | None = None,
    *,
    rho_w: float = 0.15,
    kv_group: int = 64,
) -> int:
    """
    b'_mlx = (1 + rho_w) * b_layer
             + (1 + 2/kv_group) * [ (h_k*e_k*kv_bits_k) + (h_v*e_v*kv_bits_v) ] * n_kv
    - kv_bits_* are bytes/elem (1.0 -> 8-bit, 0.5 -> 4-bit)
    - rho_w ~ 0.10..0.20; kv_group=64 -> +3.125% on KV for per-group scales
    """
    kv_bits_v = kv_bits_k if kv_bits_v is None else kv_bits_v
    elems_k = model.hk * model.ek * model.n_kv
    elems_v = model.hv * model.ev * model.n_kv
    kv_bytes_nominal = kv_bits_k * elems_k + kv_bits_v * elems_v
    scale_factor = 1.0 + (2.0 / float(max(1, kv_group)))  # 2 bytes per group
    kv_bytes = scale_factor * kv_bytes_nominal
    b_weights = (1.0 + float(rho_w)) * float(model.b_layer)
    return int(b_weights + kv_bytes)


def _sum_f_over_S(
    f_by_q: Dict[str, QuantPerf],
    S_by_q: Dict[str, QuantPerf],
    q: Literal["Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"],
    batch_size: int = 1,
) -> float:
    """
    Helper: SUM_{q ∈ Q} f_q / S_q   (used in α_m and β_m)
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

        # FIXME: error here
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
    dev: DeviceProfile, model: ModelProfile, kv_factor: float = 1.0
) -> Tuple[float, float, float]:
    """
    α_m, β_m, ξ_m exactly as defined under Assumption 1.  [App. A.3, Eq. 21 block]
    """
    bprime = b_prime(model, kv_bits_k=kv_factor)
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

    xi = (dev.t_ram2vram + dev.t_vram2ram) * (0 if dev.is_unified_mem else 1)
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
    kv_factor: float = 1.0,
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
        alpha, beta, xi = alpha_beta_xi(d, model, kv_factor)
        # print(f"alpha: {alpha}, beta: {beta}, xi: {xi}")
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
    obj_value: float


@dataclass
class HALDAResult:
    w: List[int]
    n: List[int]
    k: int
    obj_value: float
    sets: Dict[str, List[int]]

    def print_solution(self, devices: List[DeviceProfile]) -> None:
        """Print the HALDA solution in a formatted way w.r.t devices."""
        print(f"\n{'=' * 60}")
        print("HALDA Solution")
        print(f"{'=' * 60}")

        print(f"\nOptimal k: {self.k}")
        print(f"Objective value: {self.obj_value:.6f}")
        # print(f"Iterations: {result.iterations}")

        print("\nLayer distribution (w):")
        total_layers = sum(self.w)
        for dev, wi in zip(devices, self.w):
            percentage = (wi / total_layers) * 100
            print(f"  {dev.name:40s}: {wi:3d} layers ({percentage:5.1f}%)")

        print("\nGPU assignments (n):")
        for dev, ni in zip(devices, self.n):
            if ni > 0:
                print(f"  {dev.name:40s}: {ni:3d} layers on GPU")
            else:
                print(f"  {dev.name:40s}: CPU only")

        print("\nDevice sets:")
        for set_name in ["M1", "M2", "M3"]:
            if self.sets[set_name]:
                device_names = [devices[i].name for i in self.sets[set_name]]
                print(f"  {set_name}: {', '.join(device_names)}")

        # if result.forced_M4:
        #    print("\nDevices forced to M4 during calibration:")
        #    for idx in result.forced_M4:
        #        print(f"  - {devices[idx].name}")
