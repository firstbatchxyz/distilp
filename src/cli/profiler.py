import argparse
from typing import Literal

from distilp.profiler import profile_model, profile_device


def main() -> int:
    class ArgsNamespace(argparse.Namespace):
        kind: Literal["device", "model"]
        model: str | None
        repo_id: str
        output_path: str | None
        debug_lvl: int
        seq_len: int
        max_batch_exp: int
        batches: str

    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="kind",
        type=str,
        choices=["device", "model"],
        help="Select kind of profiling: 'device' or 'model'",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        help=(
            "Optional mlx_lm model module override (e.g., 'qwen3'). "
            "If omitted, resolves from Hugging Face config.json 'model_type'."
        ),
    )
    parser.add_argument(
        "-r",
        "--repo",
        dest="repo_id",
        required=True,
        type=str,
        help="Hugging Face repository ID to load the config from (e.g., 'Qwen/Qwen3-4B-MLX-8bit')",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        help="Path to directory where we create the output file.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug_lvl",
        default=0,
        type=int,
        help="Debug logging level.",
    )
    parser.add_argument(
        "-s",
        "--sequence",
        dest="seq_len",
        type=int,
        default=512,
        help="Sequence length",
    )
    parser.add_argument(
        "--max-batch-exp",
        dest="max_batch_exp",
        default=2,
        type=int,
        help="Maximum batch exponent for device profiling (default: 6, which is 2^6=64)",
    )
    parser.add_argument(
        "--batches",
        dest="batches",
        type=str,
        default="1,2,4",
        help=(
            "Comma-separated batch sizes for per-batch profiling (e.g., '1,2,4,8'). "
            "The first value drives the top-level fields (b_i, b_o, f_q, f_out)."
        ),
    )

    # Phase flag removed: we always emit split prefill/decode outputs
    args = parser.parse_args(namespace=ArgsNamespace())

    # Load config from Hugging Face Hub using the API function
    if args.kind == "device":
        device_profile = profile_device(
            args.repo_id, args.model, args.max_batch_exp, debug=args.debug_lvl
        )
        output_str = device_profile.model_dump_json(indent=2)
        if args.output_path is None:
            print(output_str)
        else:
            with open(args.output_path, "w") as f:
                f.write(output_str)
        return 0

    elif args.kind == "model":
        # Parse batches argument into a list of ints (required)
        s = args.batches.strip()
        try:
            bs_list = [int(x.strip()) for x in s.split(",") if x.strip()]
        except ValueError:
            raise ValueError("--batches must be a comma-separated list of integers.")
        if len(bs_list) == 0:
            raise ValueError(
                "--batches must contain at least one batch size, e.g., --batches 1 or --batches 1,2,4."
            )

        model_profile = profile_model(
            args.repo_id,
            bs_list,
            args.seq_len,
        )

        output_str = model_profile.model_dump_json(indent=2)
        if args.output_path is None:
            print(output_str)
        else:
            with open(args.output_path, "w") as f:
                f.write(output_str)

    else:
        raise ValueError(
            "Unknown 'return' argument. We can only handle 'model' or 'device'."
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
