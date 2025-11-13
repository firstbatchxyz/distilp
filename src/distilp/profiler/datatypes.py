from pydantic import BaseModel, Field
from typing import Literal


class CPUTopology(BaseModel):
    packages: int = 1
    cores: int = 0
    threads: int = 0


class CPUClock(BaseModel):
    base: float = 0.0  # MHz
    max: float = 0.0  # MHz


class CPUFeatures(BaseModel):
    AVX: bool = False
    FMA: bool = False
    BF16: bool = False
    SSE: bool = False


class CPUCache(BaseModel):
    l1d: int = 0
    l1i: int = 0
    l2: int = 0
    l3: int = 0


class Stat(BaseModel):
    samples: int = 0
    min: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    stddev: float = 0.0


class Batches(BaseModel):
    b_1: float = 0.0
    b_2: float = 0.0
    b_4: float = 0.0
    b_8: float = 0.0
    b_16: float = 0.0
    b_32: float = 0.0
    b_64: float = 0.0
    b_128: float = 0.0
    b_256: float = 0.0
    b_512: float = 0.0


class Benchmarks(BaseModel):
    f64: Batches = Field(default_factory=Batches)
    f32: Batches = Field(default_factory=Batches)
    tf32: Batches = Field(default_factory=Batches)
    fp16: Batches = Field(default_factory=Batches)
    bf16: Batches = Field(default_factory=Batches)
    u32: Batches = Field(default_factory=Batches)
    u16: Batches = Field(default_factory=Batches)
    u8: Batches = Field(default_factory=Batches)
    i32: Batches = Field(default_factory=Batches)
    i16: Batches = Field(default_factory=Batches)
    i8: Batches = Field(default_factory=Batches)


class SystemMemory(BaseModel):
    can_swap: int = 0
    total: float = 0.0
    available: float = 0.0
    total_swap: float = 0.0
    available_swap: float = 0.0
    cpu_read_cold_bw: float = 0.0
    cpu_read_warm_bw: float = 0.0
    cpu_write_cold_bw: float = 0.0
    cpu_write_warm_bw: float = 0.0
    memcpy_delay: float = 0.0


class DiskInfo(BaseModel):
    read: float = 0.0
    write: float = 0.0
    random: float = 0.0


class CPUInfo(BaseModel):
    vendor: str = ""
    model: str = ""
    arch: str = ""
    topology: CPUTopology = Field(default_factory=CPUTopology)
    clock: CPUClock = Field(default_factory=CPUClock)
    cache: CPUCache = Field(default_factory=CPUCache)
    features: CPUFeatures = Field(default_factory=CPUFeatures)
    benchmarks: Benchmarks = Field(default_factory=Benchmarks)
    memcpy_hot: float = 0.0
    memcpy_cold: float = 0.0


class GPUMemory(BaseModel):
    name: str = ""
    free: float = 0
    total: float = 0
    read_bw: float = 0.0
    write_bw: float = 0.0
    read_write_bw: float = 0.0
    two_read_one_write_bw: float = 0.0
    vram_to_compute: float = 0.0
    unified_memory: bool = False


class GPUInfo(BaseModel):
    name: Literal["cuda", "metal", ""] = ""  # "cuda" | "metal" | "" (none)
    memory: GPUMemory = Field(default_factory=GPUMemory)
    benchmarks: Benchmarks = Field(default_factory=Benchmarks)


class DeviceInfo(BaseModel):
    # empty string is also how `platform.system()` indicates unknown OS
    os: str = ""  # 'linux' | 'windows' | ... | "" (none)
    cpu: CPUInfo = Field(default_factory=CPUInfo)
    gpu: GPUInfo = Field(default_factory=GPUInfo)
    disk: DiskInfo = Field(default_factory=DiskInfo)
    memory: SystemMemory = Field(default_factory=SystemMemory)
