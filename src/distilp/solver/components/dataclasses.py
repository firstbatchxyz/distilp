"""
Data classes for HALDA solver profiles.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

QuantPerf = Dict[str, Dict[str, float]]


@dataclass
class DeviceProfile:
    """
    One device dm with measured/profiler data.
    Notation in comments matches the paper's symbols.
    """

    name: str
    os_type: str  # 'mac_no_metal' | 'mac_metal' | 'linux' | 'android'
    is_head: bool  # I_{m=1}  (True for the head device that holds input/output layers on CPU)
    is_unified_mem: bool  # I_UMA (Apple Silicon etc.)
    has_cuda: bool  # I_cuda
    has_metal: bool  # I_metal

    # Throughput tables (FLOPS) per quantization for CPU/GPU paths
    scpu: QuantPerf  # s^{cpu}_{m,q}
    T_cpu: float  # T^{cpu}_m (register loading throughput, bytes/s)

    # KV-copy times (sec) for a fixed kv_bits *(h_k e_k + h_v e_v) * n_kv byte payload
    t_kvcpy_cpu: float  # t^{kv_cpy,cpu}_m
    t_kvcpy_gpu: float  # t^{kv_cpy,gpu}_m

    # Host<->GPU staging + inter-device comm (sec)
    t_ram2vram: float  # t^{ram->vram}_m
    t_vram2ram: float  # t^{vram->ram}_m
    t_comm: float  # t^{comm}_m

    # Disk read throughput (bytes/s)
    s_disk: float  # s^{disk}_m

    # Available memories / working sets (bytes)
    d_avail_ram: int  # d^{avail}_m (RAM)
    # --- optional (come after required) ---
    sgpu_cuda: Optional[QuantPerf] = None  # s^{gpu}_{m,q} for CUDA
    sgpu_metal: Optional[QuantPerf] = None  # s^{gpu}_{m,q} for Metal
    T_cuda: Optional[float] = None  # T^{gpu}_m for CUDA (bytes/s)
    T_metal: Optional[float] = None  # T^{gpu}_m for Metal (bytes/s)
    d_avail_cuda: Optional[int] = None  # d^{avail}_{m,cuda} (VRAM)
    d_avail_metal: Optional[int] = None  # d^{avail}_{m,metal} (Metal working set)

    # --- small buffers and swap caps (bytes) ---
    c_cpu: int = 0  # c^{cpu} (CPU compute buffer)
    c_gpu: int = 0  # c^{gpu} (GPU compute buffer)

    # Android swap capacity (only used if os_type == "android")
    d_bytes_can_swap: int = 0  # potential bytes we allow swapping
    d_swap_avail: int = 0  # actually available swap bytes

    def print_summary(self) -> None:
        """Print a summary of loaded devices."""
        ram_gb = self.d_avail_ram / (1024**3)
        print(f"   OS Type: {self.os_type}")
        print(f"   RAM: {ram_gb:.1f} GB")
        print(f"   Is Head: {self.is_head}")
        print(f"   Unified Memory: {self.is_unified_mem}")

        if self.has_cuda and self.d_avail_cuda:
            cuda_gb = self.d_avail_cuda / (1024**3)
            print(f"   CUDA: {cuda_gb:.1f} GB")

        if self.has_metal and self.d_avail_metal:
            metal_gb = self.d_avail_metal / (1024**3)
            print(f"   Metal: {metal_gb:.1f} GB")

        print(f"   Disk Speed: {self.s_disk / (1024**2):.1f} MB/s")


@dataclass
class ModelProfile:
    """
    Model-global constants (bytes, sizes, FLOPs) from profiler.
    """

    L: int  # total layers, L
    b_layer: int  # bytes per layer,   b  (paper notation: b)
    b_in: int  # input bytes,       b_i
    b_out: int  # output bytes,      b_o
    hk: int  # heads for keys,    h_k
    ek: int  # emb per head (k),  e_k
    hv: int  # heads for values,  h_v
    ev: int  # emb per head (v),  e_v
    n_kv: int  # tokens in KV cache, n_{kv}
    e_embed: int  # embedding size,    e
    V: int  # vocabulary size,   V

    # FLOPs per layer per batch_size, and for the output layer:
    f_q: Dict[str, QuantPerf]  # (batch_size, f_q) (per "typical" layer), per batch size
    f_out: Dict[str, QuantPerf]  # (batch_size,f_{q, out})   (for output layer)
    Q: str  # Model quantization level (e.g., "Q4_K", "MXFP4", etc.)

    # Optional MoE-specific fields
    is_moe: bool = False  # Whether model uses Mixture of Experts
    n_routed_experts: int = 0  # Number of routed experts
    n_shared_experts: int = 0  # Number of shared experts
    experts_per_token: int = 0  # Number of experts activated per token
    moe_intermediate_size: int = 0  # MoE intermediate size
    moe_layer_freq: int = 1  # Frequency of MoE layers
    first_k_dense_replace: int = 0  # Number of dense layers before MoE
    total_moe_layers: int = 0  # Total number of MoE layers
    moe_layer_indices: Optional[list] = None  # Indices of MoE layers

    # Optional per-layer byte metrics (for MoE models)
    attn_bytes: Optional[list] = None  # Attention bytes per layer
    bytes_per_expert: Optional[Dict[str, int]] = None  # Bytes per expert per layer
    bytes_shared_experts: Optional[Dict[str, int]] = None  # Bytes for shared experts

    # Optional per-layer FLOP metrics (for MoE models)
    attn_flops: Optional[Dict[str, Dict[str, list]]] = (
        None  # Attention FLOPs (prefill/decode)
    )
    flops_per_expert: Optional[Dict[str, int]] = None  # FLOPs per expert per layer
    flops_shared_experts: Optional[Dict[str, int]] = None  # FLOPs for shared experts
    router_flops: Optional[Dict[str, int]] = None  # Router FLOPs per layer
    router_bytes: Optional[Dict[str, int]] = None  # Router bytes per layer
    flops_per_active_expert_per_token: Optional[Dict[str, int]] = (
        None  # Active expert FLOPs
    )

    def print_summary(self) -> None:
        """Print a summary of the loaded model."""
        print(f"\n{'=' * 60}")
        print("Model Profile:")
        print(f"{'=' * 60}")
        print(f"  Layers (L): {self.L}")
        print(f"  Bytes per layer: {self.b_layer / (1024**2):.1f} MB")
        print(f"  Input bytes: {self.b_in / (1024**2):.1f} MB")
        print(f"  Output bytes: {self.b_out / (1024**2):.1f} MB")
        print(f"  Attention heads (k/v): {self.hk}/{self.hv}")
        print(f"  Head dimensions (k/v): {self.ek}/{self.ev}")
        print(f"  KV cache tokens: {self.n_kv}")
        print(f"  Embedding dimension: {self.e_embed}")
        print(f"  Vocabulary size: {self.V}")
        print(f"  Quantizations: {', '.join(self.Q)}")
