#!/usr/bin/env python3
"""
Main script to run HALDA solver with device and model profiles.
This script loads profiles from JSON files and runs the solver.
"""

import argparse
import json
import sys

from distilp.solver.components.loader import (
    load_devices_and_model,
    load_from_profile_folder,
)
from distilp.solver import halda_solve


def main() -> int:
    class ArgsNamespace(argparse.Namespace):
        devices: list[str] | None
        profile: str | None
        model: str | None
        time_limit: float
        max_iters: int
        mip_gap: float
        sdisk_threshold: float | None
        k_candidates: list[int] | None
        quiet: bool
        verbose: bool
        save_solution: str | None
        no_plot: bool

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

    args = parser.parse_args(namespace=ArgsNamespace())

    try:
        if args.profile:
            # Load from profile folder
            devices, model = load_from_profile_folder(args.profile)
            if not args.quiet:
                print(f"Loaded profile from: {args.profile}")
        elif args.devices and args.model:
            # Load from separate files
            devices, model = load_devices_and_model(args.devices, args.model)
            if not args.quiet:
                print(f"Loaded {len(args.devices)} device file(s) and model")
        else:
            if args.devices and not args.model:
                parser.error("--devices requires --model")
            else:
                parser.error(
                    "Either --profile or both --devices and --model must be provided."
                )
        # Print summaries if verbose
        if args.verbose:
            print(f"\n{'=' * 60}")
            print(f"Loaded {len(devices)} device(s):")
            print(f"{'=' * 60}")

            for i, dev in enumerate(devices, 1):
                print(f"\n{i}. {dev.name}")
                dev.print_summary()

            model.print_summary()
        elif not args.quiet:
            print(f"\nLoaded {len(devices)} device(s) and model with {model.L} layers")

        # Run HALDA solver
        if not args.quiet:
            print(f"\n{'=' * 60}")
            print("Running HALDA solver...")
            print(f"{'=' * 60}")

        result = halda_solve(
            devices, model, mip_gap=args.mip_gap, plot=not args.no_plot, kv_bits="4bit"
        )

        # Print solution
        if args.quiet:
            # Minimal output
            print(f"k={result.k}, obj={result.obj_value:.6f}")
            for dev, wi in zip(devices, result.w):
                print(f"{dev.name}: {wi}")
        else:
            result.print_solution(devices)

        # Save solution if requested
        if args.save_solution:
            solution_data = {
                "k": result.k,
                "objective_value": result.obj_value,
                "layer_distribution": {
                    dev.name: {"w": wi, "n": ni}
                    for dev, wi, ni in zip(devices, result.w, result.n)
                },
                "sets": {
                    set_name: [devices[i].name for i in indices]
                    for set_name, indices in result.sets.items()
                },
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
