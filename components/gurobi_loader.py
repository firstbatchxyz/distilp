#!/usr/bin/env python3
"""
Loader for new profiler format that works directly with gurobi_solver.
This loader reads device and model profiles in the new format and converts them
to gurobi_solver's DeviceProfile and ModelProfile classes.
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from components.dataclasses import DeviceProfile, ModelProfile, QuantPerf
from gurobi_solver import halda_solve


def load_device_profile(device_path: str) -> DeviceProfile:
    """
    Load a device profile from the new JSON format directly to DeviceProfile.
    The new format already has all required fields.
    """
    with open(device_path, "r") as f:
        data = json.load(f)

    # Convert scpu format - map from f32/fp16/bf16 to quantization types
    scpu = {}
    if "scpu" in data and data["scpu"]:
        cpu_data = data["scpu"]
        # Map to quantization types used by gurobi solver
        scpu = {
            "Q4_K": cpu_data.get("f32", 0) * 0.25,  # Q4_K typically ~25% of f32 FLOPS
            "Q5_K": cpu_data.get("f32", 0) * 0.31,  # Q5_K typically ~31% of f32 FLOPS
            "Q6_K": cpu_data.get("f32", 0) * 0.37,  # Q6_K typically ~37% of f32 FLOPS
            "Q8_0": cpu_data.get("f32", 0) * 0.5,  # Q8_0 typically ~50% of f32 FLOPS
            "F16": cpu_data.get("fp16", cpu_data.get("f32", 0) * 0.75),
            "F32": cpu_data.get("f32", 0),
        }

    # Convert sgpu_metal format if present
    sgpu_metal = None
    if data.get("has_metal") and data.get("sgpu_metal"):
        gpu_data = data["sgpu_metal"]
        sgpu_metal = {
            "Q4_K": gpu_data.get("f32", 0) * 0.25,
            "Q5_K": gpu_data.get("f32", 0) * 0.31,
            "Q6_K": gpu_data.get("f32", 0) * 0.37,
            "Q8_0": gpu_data.get("f32", 0) * 0.5,
            "F16": gpu_data.get("fp16", gpu_data.get("f32", 0) * 0.75),
            "F32": gpu_data.get("f32", 0),
        }

    # Convert sgpu_cuda format if present
    sgpu_cuda = None
    if data.get("has_cuda") and data.get("sgpu_cuda"):
        gpu_data = data["sgpu_cuda"]
        sgpu_cuda = {
            "Q4_K": gpu_data.get("f32", 0) * 0.25,
            "Q5_K": gpu_data.get("f32", 0) * 0.31,
            "Q6_K": gpu_data.get("f32", 0) * 0.37,
            "Q8_0": gpu_data.get("f32", 0) * 0.5,
            "F16": gpu_data.get("fp16", gpu_data.get("f32", 0) * 0.75),
            "F32": gpu_data.get("f32", 0),
        }

    # Create DeviceProfile with all fields from the new format
    return DeviceProfile(
        name=data.get("name", "unknown_device"),
        os_type=data.get("os_type", "linux"),
        is_head=data.get("is_head", False),
        is_unified_mem=data.get("is_unified_mem", False),
        has_cuda=data.get("has_cuda", False),
        has_metal=data.get("has_metal", False),
        scpu=scpu,
        T_cpu=data.get("T_cpu", 1e10),
        t_kvcpy_cpu=data.get("t_kvcpy_cpu", 0.001),
        t_kvcpy_gpu=data.get("t_kvcpy_gpu", 0.001),
        t_ram2vram=data.get("t_ram2vram", 0.001),
        t_vram2ram=data.get("t_vram2ram", 0.001),
        t_comm=data.get("t_comm", 0.001),
        s_disk=data.get("s_disk", 1e9),
        d_avail_ram=int(data.get("d_avail_ram", 8 * 1024**3)),
        sgpu_cuda=sgpu_cuda,
        sgpu_metal=sgpu_metal,
        T_cuda=data.get("T_cuda"),
        T_metal=data.get("T_metal"),
        d_avail_cuda=data.get("d_avail_cuda"),
        d_avail_metal=data.get("d_avail_metal"),
        c_cpu=data.get("c_cpu", 0),
        c_gpu=data.get("c_gpu", 0),
        d_bytes_can_swap=data.get("d_bytes_can_swap", 0),
        d_swap_avail=data.get("d_swap_avail", 0),
    )


def load_model_profile(model_path: str) -> ModelProfile:
    """
    Load a model profile from the new JSON format directly to ModelProfile.
    The new format now includes all required fields including L, hk, ek, etc.
    """
    with open(model_path, "r") as f:
        data = json.load(f)

    # Get layer count
    L = data.get("L", len(data.get("b", [])) - 1)

    # Get single values for per-layer metrics (use first non-zero value)
    b_layer = (
        data["b"][1] if len(data.get("b", [])) > 1 else data.get("b_layer", 74711040)
    )
    b_in = (
        data["b_i"][1] if len(data.get("b_i", [])) > 1 else data.get("b_in", 28672000)
    )
    b_out = (
        data["b_o"][1] if len(data.get("b_o", [])) > 1 else data.get("b_out", 28672000)
    )

    # Use f_by_quant if available, otherwise create from f_q
    if "f_by_quant" in data:
        f_by_quant = data["f_by_quant"]
    else:
        # Fallback: create from f_q values
        f_base = data["f_q"][1] if len(data.get("f_q", [])) > 1 else 2972385280
        f_by_quant = {
            "Q4_K": f_base * 0.125,  # ~1/8 of F32
            "Q5_K": f_base * 0.156,  # ~1/6.4 of F32
            "Q6_K": f_base * 0.187,  # ~1/5.3 of F32
            "Q8_0": f_base * 0.25,  # 1/4 of F32
            "F16": f_base * 0.5,  # 1/2 of F32
            "F32": f_base,
        }

    # Use f_out_by_quant if available, otherwise use same as f_by_quant
    f_out_by_quant = data.get("f_out_by_quant", f_by_quant)

    # Get quantization list
    Q = data.get("Q", ["Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "F32"])

    return ModelProfile(
        L=L,
        b_layer=b_layer,
        b_in=b_in,
        b_out=b_out,
        hk=data.get("hk", 8),
        ek=data.get("ek", 128),
        hv=data.get("hv", 8),
        ev=data.get("ev", 128),
        n_kv=data.get("n_kv", 40960),
        e_embed=data.get("e_embed", 2560),
        V=data.get("V", 151936),
        f_by_quant=f_by_quant,
        f_out_by_quant=f_out_by_quant,
        Q=Q,
    )


def load_devices_and_model(
    device_files: List[str], model_file: str
) -> Tuple[List[DeviceProfile], ModelProfile]:
    """
    Load multiple device profiles and a model profile.

    Args:
        device_files: List of paths to device profile JSON files
        model_file: Path to model profile JSON file

    Returns:
        Tuple of (devices list, model profile)
    """
    devices = []
    for i, device_file in enumerate(device_files):
        device = load_device_profile(device_file)
        # Ensure first device is marked as head if not already set
        if i == 0 and not any(d.is_head for d in devices):
            device.is_head = True
        devices.append(device)

    model = load_model_profile(model_file)

    return devices, model


def load_from_profile_folder(
    profile_path: str,
) -> Tuple[List[DeviceProfile], ModelProfile]:
    """
    Load devices and model from a profile folder.

    The folder should contain:
    - model_profile.json: The model profile
    - Any other .json files: Device profiles

    Args:
        profile_path: Path to the profile folder (e.g., "profiles/hermes_70b")

    Returns:
        Tuple of (devices list, model profile)
    """
    import os
    from pathlib import Path

    profile_dir = Path(profile_path)
    if not profile_dir.exists():
        # Try with 'profiles' prefix if not found
        profile_dir = Path("profiles") / profile_path
        if not profile_dir.exists():
            raise FileNotFoundError(f"Profile folder not found: {profile_path}")

    # Find model profile
    model_file = profile_dir / "model_profile.json"
    if not model_file.exists():
        raise FileNotFoundError(f"model_profile.json not found in {profile_dir}")

    # Load model
    model = load_model_profile(str(model_file))

    # Find and load all device profiles (any .json file except model_profile.json)
    device_files = [
        f for f in profile_dir.glob("*.json") if f.name != "model_profile.json"
    ]

    if not device_files:
        raise ValueError(f"No device profiles found in {profile_dir}")

    # Sort device files for consistent ordering
    device_files.sort()

    # Load devices
    devices = []
    for i, device_file in enumerate(device_files):
        device = load_device_profile(str(device_file))
        # Ensure first device is marked as head if not already set
        if i == 0 and not any(d.is_head for d in devices):
            device.is_head = True
        devices.append(device)

    return devices, model


def load_from_combined_json(
    combined_file: str,
) -> Tuple[List[DeviceProfile], ModelProfile]:
    """
    Load devices and model from a combined JSON file (output of generate_devices.py).

    Args:
        combined_file: Path to combined JSON with 'devices' and optionally 'model' keys

    Returns:
        Tuple of (devices list, model profile)
    """
    with open(combined_file, "r") as f:
        data = json.load(f)

    devices_data = data.get("devices", [])
    model_data = data.get("model", {})

    # If model data is embedded in first device, extract it
    if not model_data and devices_data:
        first_device = devices_data[0]
        if "L" in first_device or "b" in first_device:
            model_data = {
                "L": first_device.get("L"),
                "b": first_device.get("b"),
                "b_i": first_device.get("b_i"),
                "b_o": first_device.get("b_o"),
                "f_q": first_device.get("f_q"),
                "hk": first_device.get("hk"),
                "ek": first_device.get("ek"),
                "hv": first_device.get("hv"),
                "ev": first_device.get("ev"),
                "n_kv": first_device.get("n_kv"),
                "e_embed": first_device.get("e_embed"),
                "V": first_device.get("V"),
                "f_by_quant": first_device.get("f_by_quant"),
                "f_out_by_quant": first_device.get("f_out_by_quant"),
                "Q": first_device.get("Q"),
            }

    # Convert devices
    devices = []
    for i, device_data in enumerate(devices_data):
        # Create a temporary file-like structure for the loader
        device = load_device_profile_from_dict(device_data)
        if i == 0:
            device.is_head = True
        devices.append(device)

    # Convert model
    model = load_model_profile_from_dict(model_data)

    return devices, model


def load_device_profile_from_dict(data: Dict[str, Any]) -> DeviceProfile:
    """Load DeviceProfile from dictionary (similar to load_device_profile but from dict)."""

    # Convert scpu format
    scpu = {}
    if "scpu" in data and data["scpu"]:
        cpu_data = data["scpu"]
        scpu = {
            "Q4_K": cpu_data.get("f32", 0) * 0.25,
            "Q5_K": cpu_data.get("f32", 0) * 0.31,
            "Q6_K": cpu_data.get("f32", 0) * 0.37,
            "Q8_0": cpu_data.get("f32", 0) * 0.5,
            "F16": cpu_data.get("fp16", cpu_data.get("f32", 0) * 0.75),
            "F32": cpu_data.get("f32", 0),
        }

    # Convert GPU formats
    sgpu_metal = None
    if data.get("has_metal") and data.get("sgpu_metal"):
        gpu_data = data["sgpu_metal"]
        sgpu_metal = {
            "Q4_K": gpu_data.get("f32", 0) * 0.25,
            "Q5_K": gpu_data.get("f32", 0) * 0.31,
            "Q6_K": gpu_data.get("f32", 0) * 0.37,
            "Q8_0": gpu_data.get("f32", 0) * 0.5,
            "F16": gpu_data.get("fp16", gpu_data.get("f32", 0) * 0.75),
            "F32": gpu_data.get("f32", 0),
        }

    sgpu_cuda = None
    if data.get("has_cuda") and data.get("sgpu_cuda"):
        gpu_data = data["sgpu_cuda"]
        sgpu_cuda = {
            "Q4_K": gpu_data.get("f32", 0) * 0.25,
            "Q5_K": gpu_data.get("f32", 0) * 0.31,
            "Q6_K": gpu_data.get("f32", 0) * 0.37,
            "Q8_0": gpu_data.get("f32", 0) * 0.5,
            "F16": gpu_data.get("fp16", gpu_data.get("f32", 0) * 0.75),
            "F32": gpu_data.get("f32", 0),
        }

    return DeviceProfile(
        name=data.get("name", data.get("device_id", "unknown_device")),
        os_type=data.get("os_type", "linux"),
        is_head=data.get("is_head", False),
        is_unified_mem=data.get("is_unified_mem", False),
        has_cuda=data.get("has_cuda", False),
        has_metal=data.get("has_metal", False),
        scpu=scpu,
        T_cpu=data.get("T_cpu", 1e10),
        t_kvcpy_cpu=data.get("t_kvcpy_cpu", 0.001),
        t_kvcpy_gpu=data.get("t_kvcpy_gpu", 0.001),
        t_ram2vram=data.get("t_ram2vram", 0.001),
        t_vram2ram=data.get("t_vram2ram", 0.001),
        t_comm=data.get("t_comm", 0.001),
        s_disk=data.get("s_disk", 1e9),
        d_avail_ram=int(data.get("d_avail_ram", 8 * 1024**3)),
        sgpu_cuda=sgpu_cuda,
        sgpu_metal=sgpu_metal,
        T_cuda=data.get("T_cuda"),
        T_metal=data.get("T_metal"),
        d_avail_cuda=data.get("d_avail_cuda"),
        d_avail_metal=data.get("d_avail_metal"),
        c_cpu=data.get("c_cpu", 0),
        c_gpu=data.get("c_gpu", 0),
        d_bytes_can_swap=data.get("d_bytes_can_swap", 0),
        d_swap_avail=data.get("d_swap_avail", 0),
    )


def load_model_profile_from_dict(data: Dict[str, Any]) -> ModelProfile:
    """Load ModelProfile from dictionary."""

    L = data.get("L", 36)

    # Get single values
    b_layer = (
        data["b"][1]
        if isinstance(data.get("b"), list) and len(data["b"]) > 1
        else data.get("b_layer", 74711040)
    )
    b_in = (
        data["b_i"][1]
        if isinstance(data.get("b_i"), list) and len(data["b_i"]) > 1
        else data.get("b_in", 28672000)
    )
    b_out = (
        data["b_o"][1]
        if isinstance(data.get("b_o"), list) and len(data["b_o"]) > 1
        else data.get("b_out", 28672000)
    )

    # Use f_by_quant if available
    if "f_by_quant" in data and data["f_by_quant"]:
        f_by_quant = data["f_by_quant"]
    else:
        f_base = (
            data["f_q"][1]
            if isinstance(data.get("f_q"), list) and len(data["f_q"]) > 1
            else 2972385280
        )
        f_by_quant = {
            "Q4_K": f_base * 0.125,
            "Q5_K": f_base * 0.156,
            "Q6_K": f_base * 0.187,
            "Q8_0": f_base * 0.25,
            "F16": f_base * 0.5,
            "F32": f_base,
        }

    f_out_by_quant = data.get("f_out_by_quant", f_by_quant)
    Q = data.get("Q", ["Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "F32"])

    return ModelProfile(
        L=L,
        b_layer=b_layer,
        b_in=b_in,
        b_out=b_out,
        hk=data.get("hk", 8),
        ek=data.get("ek", 128),
        hv=data.get("hv", 8),
        ev=data.get("ev", 128),
        n_kv=data.get("n_kv", 40960),
        e_embed=data.get("e_embed", 2560),
        V=data.get("V", 151936),
        f_by_quant=f_by_quant,
        f_out_by_quant=f_out_by_quant,
        Q=Q,
    )


def main():
    """Test the loader with gurobi solver."""
    import argparse

    parser = argparse.ArgumentParser(description="Load profiles and run gurobi solver")
    parser.add_argument("--devices", nargs="+", help="Device profile JSON files")
    parser.add_argument("--model", required=True, help="Model profile JSON file")
    parser.add_argument(
        "--combined", help="Combined JSON file (alternative to --devices and --model)"
    )
    parser.add_argument("--test", action="store_true", help="Run halda_solve test")

    args = parser.parse_args()

    try:
        if args.combined:
            # Load from combined file
            devices, model = load_from_combined_json(args.combined)
        else:
            # Load from separate files
            if not args.devices:
                print("Error: Either --combined or --devices must be specified")
                return 1
            devices, model = load_devices_and_model(args.devices, args.model)

        print(f"Loaded {len(devices)} devices")
        print(f"Model: L={model.L}, b_layer={model.b_layer/(1024**2):.1f}MB")
        print(
            f"Model params: hk={model.hk}, ek={model.ek}, n_kv={model.n_kv}, V={model.V}"
        )

        # Print device summary
        for dev in devices:
            ram_gb = dev.d_avail_ram / (1024**3)
            print(
                f"  - {dev.name}: {dev.os_type}, RAM={ram_gb:.1f}GB, "
                f"head={dev.is_head}, cuda={dev.has_cuda}, metal={dev.has_metal}"
            )
            if dev.has_metal and dev.d_avail_metal:
                metal_gb = dev.d_avail_metal / (1024**3)
                print(f"    Metal memory: {metal_gb:.1f}GB")
            if dev.has_cuda and dev.d_avail_cuda:
                cuda_gb = dev.d_avail_cuda / (1024**3)
                print(f"    CUDA memory: {cuda_gb:.1f}GB")

        if args.test:
            print("\nRunning HALDA solver...")
            result = halda_solve(
                devices,
                model,
                sdisk_threshold=None,
                time_limit_per_k=5.0,
                mip_gap=1e-4,
                max_outer_iters=10,
            )

            print("\n=== HALDA solution ===")
            print(f"k* = {result.k}")
            print("w* (layer windows per device):")
            for d, wi in zip(devices, result.w):
                print(f"  {d.name:40s}  w_m = {wi}")
            print("n* (GPU-assigned windows per device):")
            for d, ni in zip(devices, result.n):
                print(f"  {d.name:40s}  n_m = {ni}")
            print(f"Objective value: {result.obj_value:.4f}")
            print(f"Iterations: {result.iterations}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
