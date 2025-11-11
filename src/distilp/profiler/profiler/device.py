import os
import uuid
import psutil
import pathlib
import platform
import random
import gc
import time
import statistics as stats
from cpuinfo import get_cpu_info
import fcntl
import mlx.core as mx
from typing import Any, Callable

from distilp.profiler.models import MLX_ModelArgs

from ..datatypes import DeviceInfo
from ...common import (
    DeviceProfileInfo,
)

try:
    import cupy as cp  # type: ignore | optional dep
    # import ctypes as C  # FIXME: do we need this? ctypes is built-in

    _has_cupy = True
except ImportError:
    # Be quiet by default; functions will print if debug >= 1
    cp: Any = None  # casted to Any to suppress errors
    _has_cupy = False


def get_os(device_info):
    device_info.os = platform.system()


def fill_cpu_info(di, debug):
    info = get_cpu_info()
    di.cpu.model = info["brand_raw"]
    di.cpu.arch = info["arch_string_raw"]

    # cpuid instruction only on x86
    if info["arch_string_raw"] in ["x86_64", "amd64"]:
        di.cpu.clock.base = info["hz_actual"][0]
        di.cpu.clock.max = info["hz_advertised"][0]
        di.cpu.topology.cores = info["count"] // 2 if "ht" in info["flags"] else info["count"]
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
    device,
    B: int,
    N: int,
    M: int,
    K: int,
    warmup: int = 3,
    iters: int = 10,
    dtype: mx.Dtype = mx.float16,
    debug: int = 0,
):
    try:
        mx.set_default_device(device)
        a = mx.random.normal((B, M, K), dtype=dtype)
        b = mx.random.normal((K, N), dtype=dtype)

        for _ in range(warmup):
            c = mx.matmul(a, b)
            mx.eval(c)
        mx.synchronize()

        times: list[float] = []
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

            print(f"gemm {B}x{N}x{M} @ {K}x{N} ({flop * 1e-9:.2f} GFLOPs) ({dtype}, {device})")
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
            print(f"Error: Skipping gemm {B}x{N}x{M} @ {K}x{N} ({dtype}, {device}): {e}")
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
                _mlx_batched_gemm_benchmark(mx.cpu, 2**B, N, M, K, 10, 60, dtype, debug),
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


def bench(fn: Callable, stream, name="", warmup=3, iters=10, debug=0):
    times: list[float] = []
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
            f"{name:10}: {iters} runs [ms]: avg {mean:5.3f} ± {std:.3f}   p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}"
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

        di.gpu.memory.read_bw = bytes_total / bench(cpu_to_gpu, sec_cpu2gpu, "cpu_to_gpu", debug=debug)  # bytes/s
        di.gpu.memory.write_bw = bytes_total / bench(gpu_to_cpu, sec_gpu2cpu, "gpu_to_cpu", debug=debug)  # bytes/s
        di.gpu.memory.read_write_bw = 2 * bytes_total / bench(read_write, sec_rw, "read_write", debug=debug)  # bytes/s
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
            quant_method = q.get("quant_method") if isinstance(q.get("quant_method"), (str,)) else None
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
    mx.set_default_device(mx.Device(type=mx.cpu))
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

    def __cpu_read_cold_bw():
        nonlocal A
        return mx.max(A)

    device_info.memory.cpu_read_cold_bw = bytes_A / bench(
        __cpu_read_cold_bw, mx.cpu, "cpy_read_cold_bw", 0, 1, debug
    )  # bytes/s

    t = 4
    parts = mx.split(A, t)
    streams = [mx.new_stream(mx.Device(type=mx.cpu)) for _ in range(t)]

    def __cpu_read_warm_bw():
        nonlocal A
        return mx.abs(A)

    device_info.memory.cpu_read_warm_bw = bytes_A / bench(
        __cpu_read_warm_bw, mx.cpu, "cpu_read_warm_bw", 5, 10, debug
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

    def __memcpy_delay():
        nonlocal B
        return mx.array(B)

    device_info.memory.memcpy_delay = 1000 * bench(__memcpy_delay, mx.cpu, "memcpy_delay", debug=debug)

    # Clean up memory
    del A, B, parts, streams
    gc.collect()
    mx.clear_cache()


# TODO: Maybee transfer this to the Metal package
def metal_get_memory_info(device_info: DeviceInfo, debug):
    unified_mem = platform.machine() == "arm64"
    vm = psutil.virtual_memory()
    if unified_mem:
        device_info.gpu.name = "metal"
        device_info.gpu.memory.unified_memory = True
        device_info.gpu.memory.total = vm.total  # bytes
        device_info.gpu.memory.free = vm.available  # bytes
        # bench_gpu_transfer_times(device_info)

    # TODO: skipping the intel macbooks for now


# Get memory information
def cuda_get_memory_info(device_info: DeviceInfo, debug):
    if _has_cupy:
        free, total = cp.cuda.runtime.memGetInfo()
        device_info.gpu.memory.total = total  # bytes
        device_info.gpu.memory.free = free  # bytes
    else:
        if debug >= 1:
            print("CuPy not available; skipping CUDA memory info.")


def cuda_bench_mem_to_compute(device_info: DeviceInfo, debug):
    if not _has_cupy:
        if debug >= 1:
            print("CuPy not available; skipping CUDA compute-memory benchmark.")
        return
    pass  # TODO: !!!


# Best aproximation, still short of ~100GB/s expected
def metal_bench_mem_to_compute(device_info: DeviceInfo, debug):
    M = 512
    s_gpu = mx.new_stream(device=mx.Device(type=mx.gpu))

    # Randomize to escape caching
    A = mx.random.normal((8 * M, M, M), dtype=mx.float32, stream=s_gpu)
    idxs = list(range(8))

    # Estimate the copy from RAM to compute units
    def mem_load():
        nonlocal A

        i = random.choice(idxs)
        out = mx.sum(A[i * M : (i + 1) * M])
        mx.eval(out)

    sec = bench(mem_load, mx.gpu, "vram_to_compute", 30, 15, debug)
    bw_cpy = (2 * M * M * M * 4) / sec
    bw_ram_read = bw_cpy
    device_info.gpu.memory.vram_to_compute = bw_ram_read  # bytes/s

    # Clean up memory - release the large tensor
    del A
    gc.collect()
    mx.clear_cache()  # Clear MLX cache to release GPU memory


# Solver-facing API


# Aggregate info on the current system
def profile(config: MLX_ModelArgs, max_batch_exp, debug) -> DeviceInfo:
    di = DeviceInfo()
    get_sysmem_info(di, debug)
    get_os(di)
    fill_cpu_info(di, debug)

    run_cpu_benchmarks(di, config.hidden_size(), max_batch_exp, debug)
    run_gpu_benchmarks(di, config.hidden_size(), max_batch_exp, debug)
    if platform.system() == "Darwin":
        metal_bench_mem_to_compute(di, debug)
        metal_get_memory_info(di, debug)
        di.gpu.name = "metal"
    else:
        cuda_bench_mem_to_compute(di, debug)
        cuda_get_memory_info(di, debug)
        di.gpu.name = "cuda"
    bench_cpu_to_gpu_transfers(di, config.hidden_size(), debug)
    bench_disk_mainfs(di, debug=debug, config=config.raw)
    return di


# Get device information in solver variable names
def profile_device(config: MLX_ModelArgs, debug, max_batch_exp=6, is_head=True) -> DeviceProfileInfo:
    """
    Profile the device and return device-specific information.

    Args:
        config: Model configuration object with attributes like hidden_size.
        debug (int): Debug level for verbose output.
        max_batch_exp (int): Maximum batch exponent for throughput tables.
        is_head (bool): Whether this device is the head node (has the first layer)

    `is_head` defaults to True for single-device profiling.
    """
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
    ret.is_head = is_head

    # CPU throughput tables (FLOPS)
    scpu_dtypes = ["f32", "fp16", "bf16"]
    scpu_batches = [f"b_{2**n}" for n in range(max_batch_exp)]
    ret.scpu = {}
    for dtype in scpu_dtypes:
        ret.scpu.update({dtype: {}})
        di_type = getattr(device_info.cpu.benchmarks, dtype)
        for b in scpu_batches:
            _val = ret.scpu.get(dtype, {})
            _val.update({b: getattr(di_type, b)})
            ret.scpu.update({dtype: _val})

    # CPU register loading throughput (bytes/s) - use warm bandwidth
    ret.T_cpu = device_info.memory.cpu_read_warm_bw  # Already in bytes/s

    # GPU throughput tables (FLOPS) - separate for CUDA and Metal
    sgpu_dtypes = ["f32", "fp16", "bf16"]
    sgpu_batches = [f"b_{2**n}" for n in range(max_batch_exp)]
    _field = {}
    for dtype in sgpu_dtypes:
        _field.update({dtype: {}})
        di_type = getattr(device_info.gpu.benchmarks, dtype)
        for b in sgpu_batches:
            _val = _field.get(dtype, {})
            _val.update({b: getattr(di_type, b)})
            _field.update({dtype: _val})

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
    kv_payload_size = 2 * config.head_dim() * config.num_key_value_heads() * mx.float16.size

    # Use cold CPU write bandwidth
    ret.t_kvcpy_cpu = kv_payload_size / device_info.memory.cpu_write_cold_bw  # s/layer

    if device_info.gpu.name == "cuda":
        ret.t_kvcpy_gpu = kv_payload_size / device_info.gpu.memory.write_bw * 1e3
    elif ret.has_metal:
        ret.t_kvcpy_gpu = kv_payload_size / device_info.memory.cpu_write_cold_bw  # s/layer

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
