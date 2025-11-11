"""
Data classes for HALDA solver profiles.
"""

from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel, Field


# TODO: this is used by solver
class DeviceProfile(BaseModel):
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
    scpu: Dict[str, Dict[str, float]]  # s^{cpu}_{m,q}
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
    sgpu_cuda: Optional[Dict[str, Dict[str, float]]] = None  # s^{gpu}_{m,q} for CUDA
    sgpu_metal: Optional[Dict[str, Dict[str, float]]] = None  # s^{gpu}_{m,q} for Metal
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


class DeviceProfileInfo(BaseModel):
    """
    One device dm with measured/profiler data.
    Notation in comments matches the paper's symbols.
    """

    # --- required (no defaults) ---
    name: str = ""  # Device name
    os_type: str = ""  # 'mac_no_metal' | 'mac_metal' | 'linux' | 'android'
    is_head: bool = True  # I_{m=1}  (True for the head device that holds input/output layers on CPU)
    is_unified_mem: bool = False  # I_UMA (Apple Silicon etc.)
    has_cuda: bool = False  # I_cuda
    has_metal: bool = False  # I_metal

    # Throughput tables (FLOPS) per quantization for CPU/GPU paths
    # TODO: ["f32", "fp16", "bf16"] as the first keys?
    scpu: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # s^{cpu}_{m,q}
    T_cpu: float = 0.0  # T^{cpu}_m (register loading throughput, bytes/s)

    # KV-copy times (sec) for a fixed 2*(h_k e_k + h_v e_v)·n_kv byte payload
    t_kvcpy_cpu: float = 0.0  # t^{kv_cpy,cpu}_m
    t_kvcpy_gpu: float = 0.0  # t^{kv_cpy,gpu}_m

    # Host<->GPU staging + inter-device comm (sec)
    t_ram2vram: float = 0.0  # t^{ram->vram}_m
    t_vram2ram: float = 0.0  # t^{vram->ram}_m
    t_comm: float = 0.0  # t^{comm}_m

    # Disk read throughput (bytes/s)
    s_disk: float = 0.0  # s^{disk}_m

    # Available memories / working sets (bytes)
    d_avail_ram: int = 0  # d^{avail}_m (RAM)

    # --- optional (come after required) ---
    sgpu_cuda: Optional[Dict[str, float]] = None  # s^{gpu}_{m,q} for CUDA
    sgpu_metal: Optional[Dict[str, float]] = None  # s^{gpu}_{m,q} for Metal
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

    def to_json_str(self):
        from json import dumps
        from dataclasses import asdict

        return dumps(asdict(self))
