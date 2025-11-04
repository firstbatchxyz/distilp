import os
import uuid
import json
import psutil
import pathlib
import platform
import random
import gc
import time
import statistics as stats
from cpuinfo import get_cpu_info
import fcntl

from .parsers.mlx import in_profile_model
from .datatypes import DeviceInfo
from dataclasses import asdict, dataclass, field

import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Optional

QuantPerf = Dict[str, float]

# from src.utils.logger import logger

try:
    import cupy as cp # type: ignore | optional dep
    import ctypes as C # FIXME: do we need this? ctypes is built-in

    _has_cupy = True
except ImportError:
    # Be quiet by default; functions will print if debug >= 1
    cp = None
    _has_cupy = False


def get_os(device_info):
    device_info.os = platform.system()


def fill_cpu_info(di, debug):
    info = get_cpu_info()
    di.cpu.model = info["brand_raw"]
    di.cpu.arch = ["arch_string_raw"]

    # cpuid instruction only on x86
    if info["arch_string_raw"] in ["x86_64", "amd64"]:
        di.cpu.clock.base = info["hz_actual"][0]
        di.cpu.clock.max = info["hz_advertised"][0]
        di.cpu.topology.cores = (
            info["count"] // 2 if "ht" in info["flags"] else info["count"]
        )
        di.cpu.topology.threads = info["count"]
        di.cpu.vendor = info["vendor_id_raw"]
        if info["arch_string_raw"] in ["x86_64", "amd64", "aarch64", "arm64"]:
            di.cpu.arch = info["arch_string_raw"]
        else:
            raise TypeError(f"Unsupported CPU architecture {info.arch_string.raw}")
        di.cpu.features.AVX = True if "avx" in info["flags"] else False
        di.cpu.features.AVX2 = True if "avx2" in info["flags"] else False
        di.cpu.features.FMA = True if "fma" in info["flags"] else False
        di.cpu.features.BF16 = True if "bf16" in info["flags"] else False
        di.cpu.features.SSE = True if "sse" in info["flags"] else False
        if di.cpu.arch == "aarch64" or di.cpu.arch == "arm64":
            di.cpu.features.NEON = True if "neon" in info["flags"] else False
        di.cpu.cache.l3 = info["l3_cache_size"] * 1e-6
        di.cpu.cache.l2 = info["l2_cache_size"] * 1e-6
        di.cpu.cache.l1d = info["l1_data_cache_size"] * 1e-6
        di.cpu.cache.l1i = info["l1_instruction_cache_size"] * 1e-6

    # Apple
    else:
        di.cpu.topology.cores = psutil.cpu_count(logical=False)
        di.cpu.topology.threads = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq is not None:
            di.cpu.clock.base = cpu_freq.current
            di.cpu.clock.max = cpu_freq.max
        uname = platform.uname()
        di.cpu.vendor = uname.processor or uname.system or ""
        di.cpu.model = uname.machine or ""


# Only the A tensor is batched, B is shared. Emulating inference
# Note: take tensors as argument to cut the cost of initializing them, all tests are hot in memory anyway
def _mlx_batched_gemm_benchmark(
    device, B, N, M, K, warmup=3, iters=10, dtype=mx.float16, debug=0
):
    try:
        mx.set_default_device(device)
        a = mx.random.normal((B, M, K), dtype=dtype)
        b = mx.random.normal((K, N), dtype=dtype)

        for _ in range(warmup):
            c = mx.matmul(a, b)
            mx.eval(c)
        mx.synchronize()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            c = mx.matmul(a, b)
            mx.eval(c)
            mx.synchronize()
            times.append(time.perf_counter() - t0)

        median = stats.median(times)
        flop = 2.0 * B * N * M * K

        if debug >= 1:
            mean = stats.mean(times) * 1000
            std = stats.stdev(times) * 1000
            p50 = stats.median(times) * 1000
            p95 = stats.quantiles(times, n=iters)[iters - 5] * 1000
            p99 = stats.quantiles(times, n=iters)[iters - 2] * 1000
            mean_gflop = stats.mean([(flop / t) * 1e-9 for t in times])

            print(
                f"gemm {B}x{N}x{M} @ {K}x{N} ({flop * 1e-9:.2f} GFLOPs) ({dtype}, {device})"
            )
            print(
                f"    {iters} runs [ms]: avg {mean:<8.2f} ± {std:<5.2f}(std)  "
                f" p50={p50:<8.2f}  p95={p95:<8.2f}  p99={p99:<8.2f}"
            )
            print(f"    [GFLOP/s]: {mean_gflop:.3f}")

        result = flop / median  # Calculate FLOPS before cleanup

        # Clean up memory
        del a, b, c
        gc.collect()
        mx.clear_cache()

        return result
    except Exception as e:
        if debug >= 1:
            print(
                f"Error: Skipping gemm {B}x{N}x{M} @ {K}x{N} ({dtype}, {device}): {e}"
            )
        return 0.0


# MLX doesn't support multiprocessing for these ops, only separate streams with one op/stream
# Int datatype not supported on either device
def run_cpu_benchmarks(device_info, n_embd: int, max_batch_exp: int, debug):
    M = N = K = int(n_embd / 8 if n_embd >= 4096 else 4096 / 8)  # Smaller size on CPU
    dtypes = [mx.float32, mx.float16, mx.bfloat16, mx.uint32]
    tags = ["f32", "fp16", "bf16", "u32"]
    b_tags = [f"b_{2**n}" for n in range(9)]

    for dtype, tag in zip(dtypes, tags):
        for B in range(max_batch_exp):
            di_dtype_field = getattr(device_info.cpu.benchmarks, tag)
            setattr(
                di_dtype_field,
                b_tags[B],
                _mlx_batched_gemm_benchmark(
                    mx.cpu, 2**B, N, M, K, 10, 60, dtype, debug
                ),
            )


# consumer Nvidia GPUs don't support f64
def run_gpu_benchmarks(device_info, n_embd: int, max_batch_exp: int, debug):
    M = N = K = n_embd if n_embd >= 4096 else 4096
    dtypes = [mx.float32, mx.float16, mx.bfloat16, mx.uint32]
    tags = ["f32", "fp16", "bf16", "u32"]
    b_tags = [f"b_{2**n}" for n in range(9)]

    for dtype, tag in zip(dtypes, tags):
        for B in range(max_batch_exp):
            di_dtype_field = getattr(device_info.gpu.benchmarks, tag)
            setattr(
                di_dtype_field,
                b_tags[B],
                _mlx_batched_gemm_benchmark(mx.gpu, 2**B, N, M, K, 5, 10, dtype, debug),
            )


def bench(fn, stream, name="", warmup=3, iters=10, debug=0):
    times = []
    for _ in range(warmup):
        mx.synchronize()
        mx.eval(fn())
        mx.synchronize()
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - t0)
        mx.synchronize()

    if debug >= 1 and len(times) > 2:
        mean = stats.mean(times) * 1000
        std = stats.stdev(times) * 1000
        p50 = stats.median(times) * 1000
        p95 = stats.quantiles(times, n=iters)[iters - 5] * 1000
        p99 = stats.quantiles(times, n=iters)[iters - 2] * 1000

        print(
            f"{name:10}: {iters} runs [ms]: avg {mean:5.3f} ± {std:.3f}  "
            f" p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}"
        )

    return stats.median(times)


def bench_cpu_to_gpu_transfers(di, n_embd, debug):
    if _has_cupy:  # Benchmark VRAM <-> RAM through CUDA
        N = n_embd if n_embd >= 4096 else 4096
        bytes_total = N * N * cp.dtype(cp.float32).itemsize
        shape = N * N

        d = cp.empty(shape, dtype=cp.float32)
        h_in = cp.cuda.alloc_pinned_memory(bytes_total)
        h_out = cp.cuda.alloc_pinned_memory(bytes_total)
        h_in_ptr, h_out_ptr = h_in.ptr, h_out.ptr

        sec_cpu2gpu = cp.cuda.Stream(non_blocking=True)
        sec_gpu2cpu = cp.cuda.Stream(non_blocking=True)
        sec_rw = cp.cuda.Stream(non_blocking=True)

        def cpu_to_gpu():
            with sec_cpu2gpu:
                cp.cuda.runtime.memcpyAsync(
                    d.data.ptr,
                    h_in_ptr,
                    bytes_total,
                    cp.cuda.runtime.memcpyHostToDevice,
                    sec_cpu2gpu.ptr,
                )

        def gpu_to_cpu():
            with sec_gpu2cpu:
                cp.cuda.runtime.memcpyAsync(
                    h_out_ptr,
                    d.data.ptr,
                    bytes_total,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    sec_gpu2cpu.ptr,
                )

        def read_write():
            with sec_cpu2gpu:
                cp.cuda.runtime.memcpyAsync(
                    d.data.ptr,
                    h_in_ptr,
                    bytes_total,
                    cp.cuda.runtime.memcpyHostToDevice,
                    sec_cpu2gpu.ptr,
                )

            with sec_gpu2cpu:
                cp.cuda.runtime.memcpyAsync(
                    h_out_ptr,
                    d.data.ptr,
                    bytes_total,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    sec_gpu2cpu.ptr,
                )

        di.gpu.memory.read_bw = bytes_total / bench(
            cpu_to_gpu, sec_cpu2gpu, "cpu_to_gpu", debug=debug
        )  # bytes/s
        di.gpu.memory.write_bw = bytes_total / bench(
            gpu_to_cpu, sec_gpu2cpu, "gpu_to_cpu", debug=debug
        )  # bytes/s
        di.gpu.memory.read_write_bw = (
            2 * bytes_total / bench(read_write, sec_rw, "read_write", debug=debug)
        )  # bytes/s
    else:
        if debug >= 1:
            print("CuPy not available; skipping CUDA transfer benchmarks.")


def _bytes_per_weight_from_config(config) -> float:
    """
    Estimate stored bytes-per-weight for model weights based on config.
    Allows explicit override via env var DPERF_BYTES_PER_WEIGHT.
    """
    # Highest priority: explicit override
    try:
        _ovr = os.getenv("DPERF_BYTES_PER_WEIGHT", "").strip()
        if _ovr:
            v = float(_ovr)
            if v > 0:
                return v
    except Exception:
        pass

    # Try to infer from quantization config
    try:
        q = getattr(config, "quantization", None)
        if not isinstance(q, dict):
            q = getattr(config, "quantization_config", None)
        bits = 0
        if isinstance(q, dict):
            try:
                bits = int(q.get("bits", 0) or 0)
            except Exception:
                bits = 0
            quant_method = (
                q.get("quant_method") if isinstance(q.get("quant_method"), (str,)) else None
            )
            if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
                bits = 4
        if bits > 0:
            return max(0.125, bits / 8.0)
    except Exception:
        pass

    # Fallback: assume fp16 storage if nothing is specified
    return 2.0


def _estimate_layer_file_bytes_from_config(config, di) -> int:
    # Basic transformer layer param count approximation: attn (4 d^2) + MLP (3 d i)
    try:
        d = int(getattr(config, "hidden_size"))
    except Exception:
        d = 4096
    try:
        i = int(getattr(config, "intermediate_size") or (4 * d))
    except Exception:
        i = 4 * d
    params = 4 * d * d + 3 * d * i
    bpw = _bytes_per_weight_from_config(config)
    try:
        ov = float(os.getenv("DPERF_LAYER_OVERHEAD", "1.03"))
    except Exception:
        ov = 1.03
    est_bytes = int(params * bpw * ov)
    # Clamp to reasonable bounds to avoid excessive IO in the synthetic bench
    try:
        mn_mb = int(os.getenv("DPERF_LAYER_MIN_MB", "256"))
    except Exception:
        mn_mb = 256
    try:
        mx_mb = int(os.getenv("DPERF_LAYER_MAX_MB", "1024"))
    except Exception:
        mx_mb = 1024
    min_bytes = max(64, mn_mb) * 1024 * 1024
    max_bytes = max(min_bytes, mx_mb) * 1024 * 1024
    # Also cap by available RAM fraction to keep allocation safe
    try:
        avail = int(getattr(di.memory, "available", 0) or 0)
        ram_cap = max(min_bytes, avail // 8) if avail > 0 else max_bytes
        max_bytes = min(max_bytes, ram_cap)
    except Exception:
        pass
    return max(min_bytes, min(est_bytes, max_bytes))


def _m_from_file_bytes(target_bytes: int) -> int:
    M = int(round((target_bytes / 4.0) ** (1.0 / 3.0)))
    align = 16
    return max(128, (M + align - 1) // align * align)


def bench_disk_mainfs(di, iter=10, reads=200, debug=0, config=None):
    """
    Simplified disk benchmark: one sequential-read estimate using a single file.

    - Size the file roughly to one layer of weights (honoring quantization) when
      config is provided, clamped by env min/max. Otherwise default to a modest size.
    - Create the file with non-zero data in chunks to avoid APFS sparse optimizations.
    - Measure sequential read using configurable chunk size (default 4 MB).
    - Avoid creating many files and large temporary tensors.
    """
    # Determine target file size in bytes
    if config is not None:
        target_bytes = _estimate_layer_file_bytes_from_config(config, di)
    else:
        # default to 256 MB when config is unknown
        target_bytes = 256 * 1024 * 1024

    # Allow an explicit file size override (in MB) for the on-disk test
    try:
        _file_mb = int(os.getenv("DPERF_DISK_FILE_MB", "0"))
    except Exception:
        _file_mb = 0
    if _file_mb and _file_mb > 0:
        target_bytes = max(1, _file_mb) * 1024 * 1024

    # Helper: write a non-zero file, optionally avoiding cache on macOS
    def _create_file(pth: pathlib.Path, size: int) -> float:
        fd = os.open(str(pth), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
        # Avoid page cache population on write so read benefits from readahead later
        if platform.system() == "Darwin" and fcntl is not None:
            try:
                NC = getattr(fcntl, "F_NOCACHE", 48)
                fcntl.fcntl(fd, NC, 1)
            except Exception:
                pass
        chunk = os.urandom(4 * 1024 * 1024)
        remaining = size
        t0 = time.perf_counter()
        while remaining > 0:
            n = os.write(fd, chunk[: min(len(chunk), remaining)])
            if n <= 0:
                break
            remaining -= n
        os.fsync(fd)
        os.close(fd)
        return time.perf_counter() - t0

    # Default to 4MB sequential read chunk
    try:
        _mb = int(os.getenv("DPERF_DISK_CHUNK_MB", "4"))
    except Exception:
        _mb = 4
    chunk_bytes = max(1, _mb) * 1024 * 1024

    # Create a single file sized to target_bytes
    path = pathlib.Path(__file__).parent.resolve() / f"__tmp_{uuid.uuid4().hex}.bin"
    w_time = _create_file(path, target_bytes)

    # Measure sequential read throughput on that file (with OS readahead)
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        remaining = target_bytes
        while remaining > 0:
            b = f.read(min(chunk_bytes, remaining))
            if not b:
                break
            _ = b[0]
            remaining -= len(b)
    r_time = time.perf_counter() - t0

    # Cleanup
    try:
        os.remove(path)
    except Exception:
        pass

    # Populate device-info metrics
    di.disk.write = target_bytes / max(w_time, 1e-9)
    di.disk.read = target_bytes / max(r_time, 1e-9)
    # Single-method benchmark; set random ~= sequential for simplicity
    di.disk.random = di.disk.read


def get_sysmem_info(device_info: DeviceInfo, debug):
    import numpy as np

    sm = psutil.swap_memory()
    # mx_cpu = mx.Device(type=mx.cpu) # FIXME: use this for the op below?
    mx.set_default_device(mx.cpu)
    vm = psutil.virtual_memory()
    device_info.memory.total = vm.total  # bytes
    device_info.memory.available = vm.available  # bytes
    device_info.memory.total_swap = sm.total  # bytes
    device_info.memory.available_swap = sm.free  # bytes
    device_info.memory.can_swap = True if sm.total > 0 else False

    M = 2 << 8
    A = mx.random.normal((M, M, M), dtype=mx.float32)
    B = np.random.randn(M, M, M)
    bytes_A = M * M * M * 4

    device_info.memory.cpu_read_cold_bw = bytes_A / bench(
        lambda: mx.max(A), mx.cpu, "cpy_read_cold_bw", 0, 1, debug
    )  # bytes/s

    t = 4
    parts = mx.split(A, t)
    streams = [mx.new_stream(mx.cpu) for _ in range(t)]

    # def parallel_read_hot():
    #     return [mx.eval(mx.abs(p, stream=s)) for p, s in zip(parts, streams)]
    # device_info.memory.cpu_read_warm_bw = bytes_A/bench(lambda: parallel_read_hot())  # bytes/s

    device_info.memory.cpu_read_warm_bw = bytes_A / bench(
        lambda: mx.abs(A), mx.cpu, "cpu_read_warm_bw", 5, 10, debug
    )  # bytes/s

    device_info.memory.cpu_write_cold_bw = bytes_A / bench(
        lambda: mx.full((M * M * M), 23.4, dtype=mx.float32),
        mx.cpu,
        "cpu_write_cold_bw",
        0,
        1,
        debug,
    )

    device_info.memory.cpu_write_warm_bw = bytes_A / bench(
        lambda: mx.full((M * M * M), 351.23, dtype=mx.float32),
        mx.cpu,
        "cpu_write_warm_bw",
        5,
        10,
        debug,
    )

    device_info.memory.memcpy_delay = 1000 * bench(
        lambda: mx.eval(mx.array(B)), mx.cpu, "memcpy_delay", debug=debug
    )

    # Clean up memory
    del A, B, parts, streams
    gc.collect()
    mx.clear_cache()


# TODO: Maybee transfer this to the Metal package
def metal_get_memory_info(device_info, debug):
    unified_mem = platform.machine() == "arm64"
    vm = psutil.virtual_memory()
    if unified_mem:
        device_info.gpu.name = "metal"
        device_info.gpu.memory.unified_memory = True
        device_info.gpu.memory.total = vm.total  # bytes
        device_info.gpu.memory.free = vm.available  # bytes
        # bench_gpu_transfer_times(device_info)
    # Skip the intel macbooks for now


# Get memory information
def cuda_get_memory_info(di, debug):
    if _has_cupy:
        free, total = cp.cuda.runtime.memGetInfo()
        di.gpu.memory.total = total  # bytes
        di.gpu.memory.free = free  # bytes
    else:
        if debug >= 1:
            print("CuPy not available; skipping CUDA memory info.")


def cuda_bench_mem_to_compute(di, debug):
    if not _has_cupy:
        if debug >= 1:
            print("CuPy not available; skipping CUDA compute-memory benchmark.")
        return
    pass


# Best aproximation, still short of ~100GB/s expected
def metal_bench_mem_to_compute(di, debug):
    M = 512
    s_gpu = mx.new_stream(mx.gpu)

    # Randomize to escape caching
    A = mx.random.normal((8 * M, M, M), dtype=mx.float32, stream=s_gpu)
    idxs = list(range(8))

    # Estimate the copy from RAM to compute units
    def mem_load():
        i = random.choice(idxs)
        out = mx.sum(A[i * M : (i + 1) * M])
        mx.eval(out)

    sec = bench(mem_load, mx.gpu, "vram_to_compute", 30, 15, debug)
    bw_cpy = (2 * M * M * M * 4) / sec
    bw_ram_read = bw_cpy
    di.gpu.memory.vram_to_compute = bw_ram_read  # bytes/s

    # Clean up memory - release the large tensor
    del A
    gc.collect()
    mx.clear_cache()  # Clear MLX cache to release GPU memory


# Solver-facing API


# Aggregate info on the current system
def profile(config, max_batch_exp, debug) -> DeviceInfo:
    di = DeviceInfo()
    get_sysmem_info(di, debug)
    get_os(di)
    fill_cpu_info(di, debug)
    run_cpu_benchmarks(di, config.hidden_size, max_batch_exp, debug)
    run_gpu_benchmarks(di, config.hidden_size, max_batch_exp, debug)
    if platform.system() == "Darwin":
        metal_bench_mem_to_compute(di, debug)
        metal_get_memory_info(di, debug)
        di.gpu.name = "metal"
    else:
        cuda_bench_mem_to_compute(di, debug)
        cuda_get_memory_info(di, debug)
        di.gpu.name = "cuda"
    bench_cpu_to_gpu_transfers(di, config.hidden_size, debug)
    bench_disk_mainfs(di, debug=debug, config=config)
    return di


@dataclass
class ModelProfileInfo:
    """
    Model-global constants (bytes, sizes, FLOPs) from profiler.
    """

    # Per-layer metrics
    b: List[int] = []  # bytes per layer (weights)
    b_i: List[int] = []  # input bytes per layer (base batch)
    b_o: List[int] = []  # output bytes per layer (base batch)
    # FLOPs per layer for each batch size (e.g., {'b_1': [...], 'b_2': [...]})
    f_q: Dict[str, List[float]] = field(default_factory=dict)

    # Model-level metrics (new fields)
    L: int = 0  # total layers
    hk: int = 0  # heads for keys
    ek: int = 0  # emb per head (k)
    hv: int = 0  # heads for values
    ev: int = 0  # emb per head (v)
    n_kv: int = 0  # tokens in KV cache
    e_embed: int = 0  # embedding size
    V: int = 0  # vocabulary size

    # Quantization level label for this model/profile
    quantization: str = ""  # One of: Q4_K, Q5_K, Q6_K, Q8_0, BF16, F16, F32
    # Output-layer FLOPs per batch size (e.g., {'b_1': 123.0})
    f_out: Dict[str, float] = field(default_factory=dict)
    # Sequence length used for profiling
    seq_len: int = 0


@dataclass
class MoEModelProfileInfo(ModelProfileInfo):
    """
    MoE-specific model profile with component-level metrics for solver assignment.
    Inherits base metrics from ModelProfileInfo.
    """

    # MoE configuration
    n_routed_experts: int = 0  # Number of routed experts per MoE layer
    n_shared_experts: int = 0  # Number of always-active shared experts per MoE layer
    experts_per_token: int = 0  # Top-k experts selected per token
    moe_intermediate_size: int = 0  # FFN hidden dimension in each expert
    moe_layer_freq: int = 0  # Every N layers is MoE (1 = all MoE after first_k_dense)
    first_k_dense_replace: int = 0  # First K layers remain dense (no MoE)
    total_moe_layers: int = 0  # Total number of MoE layers in the model

    # Per-layer component metrics for solver assignment
    moe_layer_indices: List[int] = field(default_factory=list)  # Which layers are MoE

    # Attention component (same for MoE and dense, but tracked separately for assignment)
    attn_bytes: List[int] = field(
        default_factory=list
    )  # Attention weight bytes per layer
    attn_flops: Dict[str, List[float]] = field(
        default_factory=dict
    )  # Attention FLOPs per layer by batch size

    # MoE FFN component (per layer, indexed by layer number)
    bytes_per_expert: Dict[int, int] = field(
        default_factory=dict
    )  # Bytes per routed expert by layer
    bytes_shared_experts: Dict[int, int] = field(
        default_factory=dict
    )  # Total bytes for shared experts by layer
    flops_per_expert: Dict[int, float] = field(
        default_factory=dict
    )  # FLOPs per routed expert by layer
    flops_shared_experts: Dict[int, float] = field(
        default_factory=dict
    )  # Total shared experts FLOPs by layer
    router_flops: Dict[int, float] = field(
        default_factory=dict
    )  # Router/gate FLOPs by layer
    router_bytes: Dict[int, int] = field(
        default_factory=dict
    )  # Router/gate weight bytes by layer
    flops_per_active_expert_per_token: Dict[int, float] = field(
        default_factory=dict
    )  # Per-active-expert per-token FLOPs by layer


@dataclass
class ModelProfilePhased:
    prefill: ModelProfileInfo
    decode: ModelProfileInfo


@dataclass
class ModelProfileSplit:
    b: List[int]
    b_i: List[int]
    b_o: List[int]
    L: int
    hk: int
    hv: int
    ek: int
    ev: int
    n_kv: int
    e_embed: int
    V: int
    seq_len: int

    f_q: Dict[str, Dict[str, List[float]]]  # phase -> b_tag -> [FLOPs per layer]
    f_out: Dict[str, Dict[str, float]]  # phase -> b_tag -> output layer FLOPs
    quantization: str  # quantization label

    # MoE fields (optional, populated only for MoE models)
    is_moe: bool = False
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    moe_layer_freq: int = 0
    first_k_dense_replace: int = 0
    total_moe_layers: int = 0
    moe_layer_indices: List[int] = field(default_factory=list)

    # Component metrics for solver assignment
    attn_bytes: List[int] = field(default_factory=list)
    attn_flops: Dict[str, Dict[str, List[float]]] = field(
        default_factory=dict
    )  # phase -> b_tag -> [FLOPs]
    bytes_per_expert: Dict[int, int] = field(default_factory=dict)
    bytes_shared_experts: Dict[int, int] = field(default_factory=dict)
    flops_per_expert: Dict[int, float] = field(default_factory=dict)
    flops_shared_experts: Dict[int, float] = field(default_factory=dict)
    router_flops: Dict[int, float] = field(default_factory=dict)
    router_bytes: Dict[int, int] = field(default_factory=dict)
    flops_per_active_expert_per_token: Dict[int, float] = field(default_factory=dict)


@dataclass
class DeviceProfileInfo:
    """
    One device dm with measured/profiler data.
    Notation in comments matches the paper's symbols.
    """

    # --- required (no defaults) ---
    name: str = ""
    os_type: str = ""  # 'mac_no_metal' | 'mac_metal' | 'linux' | 'android'
    is_head: bool = True  # I_{m=1}  (True for the head device that holds input/output layers on CPU)
    is_unified_mem: bool = False  # I_UMA (Apple Silicon etc.)
    has_cuda: bool = False  # I_cuda
    has_metal: bool = False  # I_metal

    # Throughput tables (FLOPS) per quantization for CPU/GPU paths
    scpu: QuantPerf = None  # s^{cpu}_{m,q}
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

    def json(self):
        return json.dumps(asdict(self))


# Get device information in solver variable names
def profile_device(config, debug, max_batch_exp=6) -> DeviceProfileInfo:
    device_info = profile(config, max_batch_exp, debug)
    ret = DeviceProfileInfo()

    # Set device name (hostname or identifier)
    ret.name = platform.node() or "device"

    # Determine OS type with metal/no-metal distinction
    ret.has_metal = True if device_info.gpu.name == "metal" else False
    ret.has_cuda = True if device_info.gpu.name == "cuda" else False
    ret.is_unified_mem = ret.has_metal  # Apple Silicon has unified memory

    # Set OS type based on platform and GPU availability
    if platform.system() == "Darwin":
        if ret.has_metal:
            ret.os_type = "mac_metal"
        else:
            ret.os_type = "mac_no_metal"
    elif platform.system() == "Linux":
        # Check if Android
        try:
            with open("/proc/version", "r") as f:
                if "android" in f.read().lower():
                    ret.os_type = "android"
                else:
                    ret.os_type = "linux"
        except TypeError:
            ret.os_type = "linux"
    else:
        ret.os_type = "linux"  # Default fallback

    # Set is_head to True by default (single device scenario)
    ret.is_head = True

    # CPU throughput tables (FLOPS)
    scpu_dtypes = ["f32", "fp16", "bf16"]
    scpu_batches = [f"b_{2**n}" for n in range(max_batch_exp)]
    ret.scpu = {}
    for type in scpu_dtypes:
        ret.scpu.update({type: {}})
        di_type = getattr(device_info.cpu.benchmarks, type)
        for b in scpu_batches:
            _val = ret.scpu.get(type)
            _val.update({b: getattr(di_type, b)})
            ret.scpu.update({type: _val})

    # CPU register loading throughput (bytes/s) - use warm bandwidth
    ret.T_cpu = device_info.memory.cpu_read_warm_bw  # Already in bytes/s

    # GPU throughput tables (FLOPS) - separate for CUDA and Metal
    sgpu_dtypes = ["f32", "fp16", "bf16"]
    sgpu_batches = [f"b_{2**n}" for n in range(max_batch_exp)]
    _field = {}
    for type in sgpu_dtypes:
        _field.update({type: {}})
        di_type = getattr(device_info.gpu.benchmarks, type)
        for b in sgpu_batches:
            _val = _field.get(type)
            _val.update({b: getattr(di_type, b)})
            _field.update({type: _val})

    # CUDA memory throughput (bytes/s)
    if ret.has_cuda:
        ret.sgpu_cuda = _field
        ret.T_cuda = device_info.gpu.memory.vram_to_compute  # Already in bytes/s

    # Metal memory throughput (bytes/s)
    elif ret.has_metal:
        ret.sgpu_metal = _field
        ret.T_metal = device_info.gpu.memory.vram_to_compute  # Already in bytes/s

    # KV-copy times (sec) - time for a standard KV operation
    # Using a full layer KV I/O
    kv_payload_size = 0
    if hasattr(config, "num_attention_heads"):
        if hasattr(config, "num_key_value_heads"):
            head_dim = config.hidden_size // config.num_attention_heads
            kv_payload_size += (
                2 * head_dim * config.num_key_value_heads * mx.float16.size
            )
        else:
            kv_payload_size += 2 * config.hidden_size * mx.float16.size

    # Use cold CPU write bandwidth
    ret.t_kvcpy_cpu = kv_payload_size / device_info.memory.cpu_write_cold_bw  # s/layer

    if device_info.gpu.name == "cuda":
        ret.t_kvcpy_gpu = kv_payload_size / device_info.gpu.memory.write_bw * 1e3
    elif ret.has_metal:
        ret.t_kvcpy_gpu = (
            kv_payload_size / device_info.memory.cpu_write_cold_bw
        )  # s/layer

    # Host<->GPU staging times (sec) - time for 1MB transfer
    transfer_size = 1024 * 1024  # 1MB standard transfer
    if not ret.is_unified_mem:
        if device_info.gpu.memory.read_bw > 0:
            ret.t_ram2vram = transfer_size / device_info.gpu.memory.read_bw  # seconds
        if device_info.gpu.memory.write_bw > 0:
            ret.t_vram2ram = transfer_size / device_info.gpu.memory.write_bw  # seconds

    # Inter-device communication time (0 for single device)
    ret.t_comm = 0.0

    # Disk read throughput (bytes/s) - single-method estimate
    ret.s_disk = device_info.disk.read

    # Available memories (already in bytes)
    ret.d_avail_ram = int(device_info.memory.available)

    if ret.has_cuda:
        ret.d_avail_cuda = int(device_info.gpu.memory.free)
    elif ret.has_metal:
        ret.d_avail_metal = int(device_info.memory.available)  # Unified memory

    # Small buffers (bytes) - set to 0 for now
    ret.c_cpu = 0
    ret.c_gpu = 0

    # Swap capacity (already in bytes)
    ret.d_bytes_can_swap = int(device_info.memory.total_swap)
    ret.d_swap_avail = int(device_info.memory.available_swap)

    # Clean up any lingering memory from profiling operations
    gc.collect()
    mx.clear_cache()

    return ret


# Estimate FLOPs for Model
def profile_model(
    model: nn.Module,
    config,
    B: int = 1,
    L: int = 4096,
    config_dict: Dict = {},
    debug = 0,
    bs_list: List[int] = [],
    phase: str = "merged",
):
    dtype = None
    bits = 0
    group_size = 0

    # Prefer explicit quantization section for bit-width
    quant_method = None
    if isinstance(config_dict.get("quantization"), dict):
        q = config_dict["quantization"]
        bits = int(q.get("bits", 0) or 0)
        group_size = int(q.get("group_size", 0) or 0)
    elif isinstance(config_dict.get("quantization_config"), dict):
        q = config_dict["quantization_config"]
        bits = int(q.get("bits", 0) or 0)
        group_size = int(q.get("group_size", 0) or 0)
        quant_method = q.get("quant_method")
    # Fallback to dtype when no explicit quantization
    if bits == 0:
        # Try config_dict first, then the config object, then default to f16
        dtype = (
            config_dict.get("torch_dtype")
            or config_dict.get("dtype")
            or getattr(config, "torch_dtype", None)
            or getattr(config, "dtype", None)
        )
        # Map quant_method if present (e.g., mxfp4)
        if not dtype and isinstance(config_dict.get("quantization_config"), dict):
            quant_method = (
                config_dict["quantization_config"].get("quant_method") or quant_method
            )
        if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
            bits = 4
            # Default a group size if none provided
            if group_size == 0:
                group_size = 128
        if dtype:
            if dtype in ("bfloat16", "bf16"):
                bits = 16
            elif dtype in ("float16", "fp16"):
                bits = 16
            elif dtype in ("float32", "f32"):
                # Default to fp16 when quantization is not explicit
                bits = 16
        if bits == 0:
            # Default to fp16 if nothing explicit is set
            bits = 16
    # Determine fp_bits for exclusions (non-quantized modules)
    has_quant_cfg = isinstance(config_dict.get("quantization"), dict) or isinstance(
        config_dict.get("quantization_config"), dict
    )
    fp_bits = 16
    if has_quant_cfg:
        d_dtype = (
            config_dict.get("torch_dtype")
            or config_dict.get("dtype")
            or getattr(config, "torch_dtype", None)
            or getattr(config, "dtype", None)
        )
        if d_dtype in ("float32", "f32"):
            fp_bits = 32
        elif d_dtype in ("bfloat16", "bf16", "float16", "fp16") or d_dtype is None:
            fp_bits = 16

    # Quantization exclusions
    exclude_patterns = []
    if isinstance(config_dict.get("quantization_config"), dict):
        exclude_patterns = (
            config_dict["quantization_config"].get("modules_to_not_convert", []) or []
        )

    model_info = in_profile_model(
        model,
        config,
        B,
        L,
        16,
        bits,
        group_size,
        debug,
        phase,
        config_dict,
        exclude_patterns,
        fp_bits,
    )
    ret = ModelProfileInfo()

    # Per-layer metrics (base batch)
    ret.b = [int(x.weight_bytes) for x in model_info]
    ret.b_i = [int(x.input_bytes) for x in model_info]
    ret.b_o = [int(x.output_bytes) for x in model_info]
    ret.f_q[f"b_{B}"] = [float(x.flops) for x in model_info]
    ret.f_out[f"b_{B}"] = ret.f_q[f"b_{B}"][-1] if ret.f_q[f"b_{B}"] else 0.0
    ret.seq_len = int(L)

    # Use config_dict if available for more complete access, otherwise fall back to config object
    cfg = config_dict if config_dict else {}

    # Model-level metrics from config
    ret.L = cfg.get(
        "num_hidden_layers", getattr(config, "num_hidden_layers", len(model_info) - 1)
    )
    ret.e_embed = cfg.get("hidden_size", getattr(config, "hidden_size", 0))
    ret.V = cfg.get("vocab_size", getattr(config, "vocab_size", 0))

    # Attention head configuration
    num_attention_heads = cfg.get(
        "num_attention_heads", getattr(config, "num_attention_heads", 0)
    )
    ret.hk = cfg.get(
        "num_key_value_heads",
        getattr(config, "num_key_value_heads", num_attention_heads),
    )
    ret.hv = cfg.get(
        "num_key_value_heads",
        getattr(config, "num_key_value_heads", num_attention_heads),
    )

    # Calculate head dimension
    head_dim = cfg.get("head_dim", getattr(config, "head_dim", 0))
    if head_dim is None and ret.e_embed > 0 and num_attention_heads > 0:
        head_dim = ret.e_embed // num_attention_heads
    ret.ek = head_dim
    ret.ev = head_dim

    # KV cache tokens (using max position embeddings as proxy)
    ret.n_kv = cfg.get(
        "max_position_embeddings", getattr(config, "max_position_embeddings", L)
    )

    # Add quantization label
    # If no explicit quantization, default label to F16
    q_label = ""
    if isinstance(cfg.get("quantization"), dict) or isinstance(
        cfg.get("quantization_config"), dict
    ):
        qbits = None
        if isinstance(cfg.get("quantization"), dict):
            qbits = cfg["quantization"].get("bits")
        else:
            qbits = cfg["quantization_config"].get("bits")
            quant_method = cfg["quantization_config"].get("quant_method")
        try:
            qbits = int(qbits) if qbits is not None else None
        except Exception:
            qbits = None
        if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
            q_label = "MXFP4"
        mapping = {4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q8_0", 16: "F16", 32: "F32"}
        if qbits in mapping:
            q_label = mapping[qbits]
        if not q_label:
            d = cfg.get("torch_dtype") or cfg.get("dtype")
            if d in ("bfloat16", "bf16"):
                q_label = "BF16"
            elif d in ("float16", "fp16"):
                q_label = "F16"
            elif d in ("float32", "f32"):
                q_label = "F32"
    else:
        q_label = "F16"
    ret.quantization = q_label

    # Multi-batch-size profiles: only if provided via --batches

    for Bx in bs_list:
        tag = f"b_{Bx}"
        layers_bx = in_profile_model(
            model,
            config,
            Bx,
            L,
            16,
            bits,
            group_size,
            0,
            phase,
            config_dict,
            exclude_patterns,
            fp_bits,
        )
        ret.f_q[tag] = [float(x.flops) for x in layers_bx]
        ret.f_out[tag] = ret.f_q[tag][-1] if ret.f_q[tag] else 0.0

    return ret


def profile_moe_model(
    model: nn.Module,
    config,
    B: int = 1,
    L: int = 4096,
    config_dict: Dict = {},
    debug=0,
    bs_list: List[int] = [],
    phase: str = "merged",
):
    """
    Profile an MoE model with component-level metrics for solver assignment.
    Returns MoEModelProfileInfo if MoE is detected, otherwise returns ModelProfileInfo.
    """
    dtype = None
    bits = 0
    group_size = 0

    # Check if this is an MoE model - handle various naming conventions
    cfg = config_dict if config_dict else {}

    # Try different field names for number of experts
    n_routed_experts = cfg.get(
        "n_routed_experts",
        cfg.get(
            "num_experts",  # Qwen3, Mixtral
            cfg.get(
                "num_local_experts",  # Some Mixtral variants
                cfg.get(
                    "n_experts",  # Alternative naming
                    getattr(
                        config,
                        "n_routed_experts",
                        getattr(
                            config,
                            "num_experts",
                            getattr(
                                config,
                                "num_local_experts",
                                getattr(config, "n_experts", 0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    if not n_routed_experts or n_routed_experts == 0:
        # Not an MoE model, use regular profiling
        return profile_model(model, config, B, L, config_dict, debug, bs_list, phase)

    # Parse quantization info
    quant_method = None
    if isinstance(config_dict.get("quantization"), dict):
        q = config_dict["quantization"]
        bits = int(q.get("bits", 0) or 0)
        group_size = int(q.get("group_size", 0) or 0)
    elif isinstance(config_dict.get("quantization_config"), dict):
        q = config_dict["quantization_config"]
        bits = int(q.get("bits", 0) or 0)
        group_size = int(q.get("group_size", 0) or 0)
        quant_method = q.get("quant_method")
    if bits == 0:
        # Fallback: try config object then default to f16 or quant_method
        dtype = (
            config_dict.get("torch_dtype")
            or config_dict.get("dtype")
            or getattr(config, "torch_dtype", None)
            or getattr(config, "dtype", None)
        )
        if not dtype and isinstance(config_dict.get("quantization_config"), dict):
            quant_method = (
                config_dict["quantization_config"].get("quant_method") or quant_method
            )
        if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
            bits = 4
            if group_size == 0:
                group_size = 128
        if dtype:
            if dtype in ("bfloat16", "bf16"):
                bits = 16
            elif dtype in ("float16", "fp16"):
                bits = 16
            elif dtype in ("float32", "f32"):
                # Default to fp16 when quantization is not explicit
                bits = 16
        if bits == 0:
            # Default to fp16 if nothing explicit is set
            bits = 16

    # Prepare per-module quantization exclusions
    has_quant_cfg = isinstance(config_dict.get("quantization"), dict) or isinstance(
        config_dict.get("quantization_config"), dict
    )
    fp_bits = 16
    if has_quant_cfg:
        d_dtype = (
            config_dict.get("torch_dtype")
            or config_dict.get("dtype")
            or getattr(config, "torch_dtype", None)
            or getattr(config, "dtype", None)
        )
        if d_dtype in ("float32", "f32"):
            fp_bits = 32
        elif d_dtype in ("bfloat16", "bf16", "float16", "fp16") or d_dtype is None:
            # Default to fp16 when dtype is missing
            fp_bits = 16
    exclude_patterns = []
    if isinstance(config_dict.get("quantization_config"), dict):
        exclude_patterns = (
            config_dict["quantization_config"].get("modules_to_not_convert", []) or []
        )

    # Profile the model to get layer-level metrics
    model_info = in_profile_model(
        model,
        config,
        B,
        L,
        16,
        bits,
        group_size,
        debug,
        phase,
        config_dict,
        exclude_patterns,
        fp_bits,
    )

    # Create MoE profile
    ret = MoEModelProfileInfo()

    # Populate base metrics
    ret.b = [int(x.weight_bytes) for x in model_info]
    ret.b_i = [int(x.input_bytes) for x in model_info]
    ret.b_o = [int(x.output_bytes) for x in model_info]
    ret.f_q[f"b_{B}"] = [float(x.flops) for x in model_info]
    ret.f_out[f"b_{B}"] = ret.f_q[f"b_{B}"][-1] if ret.f_q[f"b_{B}"] else 0.0
    ret.seq_len = int(L)

    # Model-level metrics
    ret.L = cfg.get(
        "num_hidden_layers", getattr(config, "num_hidden_layers", len(model_info) - 1)
    )
    ret.e_embed = cfg.get("hidden_size", getattr(config, "hidden_size", 0))
    ret.V = cfg.get("vocab_size", getattr(config, "vocab_size", 0))

    # Attention head configuration
    num_attention_heads = cfg.get(
        "num_attention_heads", getattr(config, "num_attention_heads", 0)
    )
    ret.hk = cfg.get(
        "num_key_value_heads",
        getattr(config, "num_key_value_heads", num_attention_heads),
    )
    ret.hv = cfg.get(
        "num_key_value_heads",
        getattr(config, "num_key_value_heads", num_attention_heads),
    )

    # Head dimension
    head_dim = cfg.get("head_dim", getattr(config, "head_dim", 0))
    if head_dim == 0 and ret.e_embed > 0 and num_attention_heads > 0:
        head_dim = ret.e_embed // num_attention_heads
    ret.ek = head_dim
    ret.ev = head_dim
    ret.n_kv = cfg.get(
        "max_position_embeddings", getattr(config, "max_position_embeddings", L)
    )

    # MoE configuration - handle various naming conventions
    ret.n_routed_experts = n_routed_experts

    # Shared experts (Qwen3 uses shared_expert_intermediate_size to indicate shared experts)
    shared_expert_size = cfg.get(
        "shared_expert_intermediate_size",
        getattr(config, "shared_expert_intermediate_size", 0),
    )
    ret.n_shared_experts = cfg.get(
        "n_shared_experts",
        cfg.get(
            "num_shared_experts",
            getattr(
                config,
                "n_shared_experts",
                getattr(
                    config, "num_shared_experts", 1 if shared_expert_size > 0 else 0
                ),
            ),
        ),
    )  # Infer from shared_expert_intermediate_size

    # Experts per token (top-k selection)
    ret.experts_per_token = cfg.get(
        "num_experts_per_tok",  # Standard naming
        cfg.get(
            "num_experts_per_token",  # Alternative
            cfg.get(
                "experts_per_token",  # GPT-OSS naming
                cfg.get(
                    "num_selected_experts",  # Alternative
                    cfg.get(
                        "top_k",  # Some models use top_k
                        getattr(
                            config,
                            "num_experts_per_tok",
                            getattr(
                                config,
                                "num_experts_per_token",
                                getattr(
                                    config,
                                    "experts_per_token",
                                    getattr(
                                        config,
                                        "num_selected_experts",
                                        getattr(config, "top_k", None),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    if ret.experts_per_token is None or ret.experts_per_token == 0:
        raise ValueError(
            "MoE model detected but experts_per_token not found or is 0. "
            "Config must have one of: num_experts_per_tok, experts_per_token, "
            "num_selected_experts, or top_k"
        )

    # MoE FFN hidden size - use intermediate_size for MoE models if no explicit MoE size
    ret.moe_intermediate_size = cfg.get(
        "moe_intermediate_size",  # Explicit MoE intermediate size (Qwen3)
        cfg.get(
            "expert_intermediate_size",  # Alternative naming
            cfg.get(
                "intermediate_size",  # Standard intermediate_size (GPT-OSS, Mixtral)
                cfg.get(
                    "ffn_dim",  # Some models
                    getattr(
                        config,
                        "moe_intermediate_size",
                        getattr(
                            config,
                            "expert_intermediate_size",
                            getattr(
                                config,
                                "intermediate_size",
                                getattr(config, "ffn_dim", None),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    if ret.moe_intermediate_size is None or ret.moe_intermediate_size == 0:
        raise ValueError(
            "MoE model detected but no valid intermediate/FFN size found. "
            "Config must have one of: moe_intermediate_size, expert_intermediate_size, "
            "intermediate_size, or ffn_dim"
        )

    # MoE layer frequency (which layers have MoE)
    ret.moe_layer_freq = cfg.get(
        "moe_layer_freq",
        cfg.get(
            "decoder_sparse_step",  # Qwen3 uses this
            cfg.get(
                "expert_interval",  # Alternative
                getattr(
                    config,
                    "moe_layer_freq",
                    getattr(
                        config,
                        "decoder_sparse_step",
                        getattr(config, "expert_interval", 1),
                    ),
                ),
            ),
        ),
    )

    # First K dense layers (before MoE starts)
    ret.first_k_dense_replace = cfg.get(
        "first_k_dense_replace",
        cfg.get(
            "num_dense_layers",  # Alternative
            getattr(
                config, "first_k_dense_replace", getattr(config, "num_dense_layers", 0)
            ),
        ),
    )

    # Determine MoE layer indices from parsed layers for accuracy
    moe_indices = [
        i
        for i, layer in enumerate(model_info[1:], 1)
        if getattr(layer, "is_moe_layer", False)
    ]
    # Fallback to heuristic if parser didn't tag MoE layers
    if not moe_indices and ret.L:
        for layer_idx in range(1, ret.L + 1):
            if (
                layer_idx > ret.first_k_dense_replace
                and (ret.moe_layer_freq or 1) > 0
                and layer_idx % max(ret.moe_layer_freq, 1) == 0
            ):
                moe_indices.append(layer_idx)
    ret.moe_layer_indices = moe_indices
    ret.total_moe_layers = len(moe_indices)

    # Extract component metrics from LayerMeta
    ret.attn_bytes = []
    ret.attn_flops[f"b_{B}"] = []

    for idx, layer in enumerate(model_info[1:], 1):  # Skip prefill layer
        # Attention metrics (all layers have attention)
        ret.attn_bytes.append(layer.attn_bytes)
        ret.attn_flops[f"b_{B}"].append(layer.attn_flops)

        # MoE metrics (only for MoE layers)
        if (getattr(layer, "is_moe_layer", False)) or (idx in moe_indices):
            ret.bytes_per_expert[idx] = layer.moe_expert_bytes
            ret.bytes_shared_experts[idx] = layer.moe_shared_bytes
            ret.flops_per_expert[idx] = layer.moe_expert_flops
            ret.flops_shared_experts[idx] = layer.moe_shared_flops
            ret.router_flops[idx] = layer.moe_router_flops
            ret.router_bytes[idx] = layer.moe_router_bytes
            if hasattr(layer, "moe_expert_flops_per_token"):
                ret.flops_per_active_expert_per_token[idx] = (
                    layer.moe_expert_flops_per_token
                )

    # Quantization label
    if isinstance(cfg.get("quantization"), dict) or isinstance(
        cfg.get("quantization_config"), dict
    ):
        q_label = ""
        qbits = None
        if isinstance(cfg.get("quantization"), dict):
            qbits = cfg["quantization"].get("bits")
        else:
            qbits = cfg["quantization_config"].get("bits")
            quant_method = cfg["quantization_config"].get("quant_method")
        try:
            qbits = int(qbits) if qbits is not None else None
        except Exception:
            qbits = None
        if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
            q_label = "MXFP4"
        mapping = {4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q8_0", 16: "F16", 32: "F32"}
        if qbits in mapping:
            q_label = mapping[qbits]
        if not q_label:
            d = cfg.get("torch_dtype") or cfg.get("dtype")
            if d in ("bfloat16", "bf16"):
                q_label = "BF16"
            elif d in ("float16", "fp16"):
                q_label = "F16"
            elif d in ("float32", "f32"):
                q_label = "F32"
    else:
        q_label = "F16"
    ret.quantization = q_label

    # Multi-batch profiles
    for Bx in bs_list:
        tag = f"b_{Bx}"
        layers_bx = in_profile_model(
            model,
            config,
            Bx,
            L,
            16,
            bits,
            group_size,
            0,
            phase,
            config_dict,
            exclude_patterns,
            fp_bits,
        )
        ret.f_q[tag] = [float(x.flops) for x in layers_bx]
        ret.f_out[tag] = ret.f_q[tag][-1] if ret.f_q[tag] else 0.0
        ret.attn_flops[tag] = [
            float(x.attn_flops) for x in layers_bx[1:]
        ]  # Skip prefill

    return ret


def profile_model_phased(
    model: nn.Module,
    config,
    B: int,
    L: int,
    config_dict: Dict,
    debug=0,
    bs_list: List[int] = [],
):
    # Use profile_moe_model which auto-detects MoE models
    prefill = profile_moe_model(
        model,
        config,
        B=B,
        L=L,
        config_dict=config_dict,
        debug=debug,
        bs_list=bs_list,
        phase="prefill",
    )
    decode = profile_moe_model(
        model,
        config,
        B=B,
        L=L,
        config_dict=config_dict,
        debug=debug,
        bs_list=bs_list,
        phase="decode",
    )
    return ModelProfilePhased(prefill=prefill, decode=decode)


def profile_model_split(
    model: nn.Module,
    config,
    B: int,
    L: int,
    config_dict: Dict,
    debug = 0,
    bs_list: List[int] = [],
):
    phased = profile_model_phased(
        model,
        config,
        B=B,
        L=L,
        config_dict=config_dict,
        debug=debug,
        bs_list=bs_list,
    )

    pre = phased.prefill
    dec = phased.decode

    # Create base split result
    result = ModelProfileSplit(
        b=pre.b,
        b_i=pre.b_i,
        b_o=pre.b_o,
        L=pre.L,
        hk=pre.hk,
        hv=pre.hv,
        ek=pre.ek,
        ev=pre.ev,
        n_kv=pre.n_kv,
        e_embed=pre.e_embed,
        V=pre.V,
        seq_len=pre.seq_len,
        f_q={
            "prefill": pre.f_q,
            "decode": dec.f_q,
        },
        f_out={
            "prefill": pre.f_out,
            "decode": dec.f_out,
        },
        quantization=pre.quantization,
    )

    # If this is an MoE model, populate MoE fields
    if isinstance(pre, MoEModelProfileInfo):
        result.is_moe = True
        result.n_routed_experts = pre.n_routed_experts
        result.n_shared_experts = pre.n_shared_experts
        result.experts_per_token = pre.experts_per_token
        result.moe_intermediate_size = pre.moe_intermediate_size
        result.moe_layer_freq = pre.moe_layer_freq
        result.first_k_dense_replace = pre.first_k_dense_replace
        result.total_moe_layers = pre.total_moe_layers
        result.moe_layer_indices = pre.moe_layer_indices
        result.attn_bytes = pre.attn_bytes
        result.attn_flops = {
            "prefill": pre.attn_flops,
            "decode": dec.attn_flops if isinstance(dec, MoEModelProfileInfo) else {},
        }
        result.bytes_per_expert = pre.bytes_per_expert
        result.bytes_shared_experts = pre.bytes_shared_experts
        result.flops_per_expert = pre.flops_per_expert
        result.flops_shared_experts = pre.flops_shared_experts
        result.router_flops = pre.router_flops
        result.router_bytes = pre.router_bytes
        result.flops_per_active_expert_per_token = pre.flops_per_active_expert_per_token

    return result
