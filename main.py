#!/usr/bin/env python3
"""
Main script to run HALDA solver with device and model profiles.
This script loads profiles from JSON files and runs the gurobi solver.
"""

import argparse
import json
import sys
from typing import List

# Be resilient to being run as a script or module
try:
    # Package-style imports
    from src.dsolver.components.gurobi_loader import (
        load_devices_and_model,
        load_from_combined_json,
        load_from_profile_folder,
        load_device_profile,
        load_model_profile,
    )
    from src.dsolver.components.dataclasses import DeviceProfile, ModelProfile
    from src.dsolver.gurobi_solver import halda_solve
except Exception:
    # Script-style fallback
    from src.dsolver.components.gurobi_loader import (
        load_devices_and_model,
        load_from_combined_json,
        load_from_profile_folder,
        load_device_profile,
        load_model_profile,
    )
    from src.dsolver.components.dataclasses import DeviceProfile, ModelProfile
    from src.dsolver.gurobi_solver import halda_solve


def print_device_summary(devices: List[DeviceProfile]) -> None:
    """Print a summary of loaded devices."""
    print(f"\n{'='*60}")
    print(f"Loaded {len(devices)} device(s):")
    print(f"{'='*60}")

    for i, dev in enumerate(devices, 1):
        ram_gb = dev.d_avail_ram / (1024**3)
        print(f"\n{i}. {dev.name}")
        print(f"   OS Type: {dev.os_type}")
        print(f"   RAM: {ram_gb:.1f} GB")
        print(f"   Is Head: {dev.is_head}")
        print(f"   Unified Memory: {dev.is_unified_mem}")

        if dev.has_cuda and dev.d_avail_cuda:
            cuda_gb = dev.d_avail_cuda / (1024**3)
            print(f"   CUDA: {cuda_gb:.1f} GB")

        if dev.has_metal and dev.d_avail_metal:
            metal_gb = dev.d_avail_metal / (1024**3)
            print(f"   Metal: {metal_gb:.1f} GB")

        print(f"   Disk Speed: {dev.s_disk/(1024**2):.1f} MB/s")


def print_model_summary(model: ModelProfile) -> None:
    """Print a summary of the loaded model."""
    print(f"\n{'='*60}")
    print("Model Profile:")
    print(f"{'='*60}")
    print(f"  Layers (L): {model.L}")
    print(f"  Bytes per layer: {model.b_layer/(1024**2):.1f} MB")
    print(f"  Input bytes: {model.b_in/(1024**2):.1f} MB")
    print(f"  Output bytes: {model.b_out/(1024**2):.1f} MB")
    print(f"  Attention heads (k/v): {model.hk}/{model.hv}")
    print(f"  Head dimensions (k/v): {model.ek}/{model.ev}")
    print(f"  KV cache tokens: {model.n_kv}")
    print(f"  Embedding dimension: {model.e_embed}")
    print(f"  Vocabulary size: {model.V}")
    print(f"  Quantizations: {', '.join(model.Q)}")


def print_solution(result, devices: List[DeviceProfile]) -> None:
    """Print the HALDA solution in a formatted way."""
    print(f"\n{'='*60}")
    print("HALDA Solution")
    print(f"{'='*60}")

    print(f"\nOptimal k: {result.k}")
    print(f"Optimal batch size: {result.batch_size}")
    print(f"\nObjective value: {result.obj_value:.6f}")
    print(f"TPOT: {result.tpot}")
    # print(f"Iterations: {result.iterations}")

    print("\nLayer distribution (w):")
    total_layers = sum(result.w)
    for dev, wi in zip(devices, result.w):
        percentage = (wi / total_layers) * 100
        print(f"  {dev.name:40s}: {wi:3d} layers ({percentage:5.1f}%)")

    print("\nGPU assignments (n):")
    for dev, ni in zip(devices, result.n):
        if ni > 0:
            print(f"  {dev.name:40s}: {ni:3d} layers on GPU")
        else:
            print(f"  {dev.name:40s}: CPU only")

    print("\nDevice sets:")
    for set_name in ["M1", "M2", "M3"]:
        if result.sets[set_name]:
            device_names = [devices[i].name for i in result.sets[set_name]]
            print(f"  {set_name}: {', '.join(device_names)}")

    # if result.forced_M4:
    #    print("\nDevices forced to M4 during calibration:")
    #    for idx in result.forced_M4:
    #        print(f"  - {devices[idx].name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run HALDA solver for distributed LLM inference optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a profile folder
  python main.py --profile hermes_70b
  python main.py --profile profiles/qwen3_32b/8bit

  # Run with separate device and model files
  python main.py --devices profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m4.json \\
                 --model profiles/model_profile.json

  # Run with custom solver parameters
  python main.py --profile hermes_70b --time-limit 10 --max-iters 20 --mip-gap 0.001
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--devices", nargs="+", help="Device profile JSON files (requires --model)"
    )
    input_group.add_argument(
        "--profile",
        help="Profile folder path (e.g., 'hermes_70b' or 'profiles/hermes_70b')",
    )

    parser.add_argument(
        "--model", help="Model profile JSON file (required with --devices)"
    )

    # Solver parameters
    solver_group = parser.add_argument_group("solver parameters")
    solver_group.add_argument(
        "--time-limit",
        type=float,
        default=5.0,
        help="Time limit per k value in seconds (default: 5.0)",
    )
    solver_group.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="Maximum outer iterations (default: 50)",
    )
    solver_group.add_argument(
        "--mip-gap", type=float, default=1e-4, help="MIP gap tolerance (default: 1e-4)"
    )
    solver_group.add_argument(
        "--sdisk-threshold",
        type=float,
        help="Disk speed threshold for forcing devices to M4 (bytes/s)",
    )
    solver_group.add_argument(
        "--k-candidates",
        nargs="+",
        type=int,
        help="Specific k values to try (default: all factors of L)",
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument("--quiet", action="store_true", help="Minimal output")
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output including device and model summaries",
    )
    output_group.add_argument("--save-solution", help="Save solution to JSON file")
    output_group.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of k vs objective curve",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.devices and not args.model:
        parser.error("--devices requires --model")

    try:
        if args.profile:
            # Load from profile folder
            devices, model = load_from_profile_folder(args.profile)
            if not args.quiet:
                print(f"Loaded profile from: {args.profile}")
        else:
            # Load from separate files
            devices, model = load_devices_and_model(args.devices, args.model)
            if not args.quiet:
                print(f"Loaded {len(args.devices)} device file(s) and model")

        # Print summaries if verbose
        if args.verbose:
            print_device_summary(devices)
            print_model_summary(model)
        elif not args.quiet:
            print(f"\nLoaded {len(devices)} device(s) and model with {model.L} layers")

        # Run HALDA solver
        if not args.quiet:
            print(f"\n{'='*60}")
            print("Running HALDA solver...")
            print(f"{'='*60}")

        result = halda_solve(
            devices,
            model,
            sdisk_threshold=args.sdisk_threshold,
            k_candidates=args.k_candidates,

            mip_gap=args.mip_gap,

            plot=not args.no_plot,
        )

        # Print solution
        if args.quiet:
            # Minimal output
            print(f"k={result.k}, obj={result.obj_value:.6f}")
            for dev, wi in zip(devices, result.w):
                print(f"{dev.name}: {wi}")
        else:
            print_solution(result, devices)

        # Save solution if requested
        if args.save_solution:
            solution_data = {
                "k": result.k,
                "objective_value": result.obj_value,
                # "iterations": result.iterations,
                "layer_distribution": {
                    dev.name: {"w": wi, "n": ni}
                    for dev, wi, ni in zip(devices, result.w, result.n)
                },
                "sets": {
                    set_name: [devices[i].name for i in indices]
                    for set_name, indices in result.sets.items()
                },
                # "forced_M4": [devices[i].name for i in result.forced_M4],
            }

            with open(args.save_solution, "w") as f:
                json.dump(solution_data, f, indent=2)

            if not args.quiet:
                print(f"\nSolution saved to: {args.save_solution}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
