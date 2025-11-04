import sys
import json
import pprint
import importlib
from dataclasses import asdict
from argparse import ArgumentParser

from distilp.profiler import profile_model_split, profile_device
from distilp.profiler.api import load_config_from_repo


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--profile",
        dest="ret",
        type=str,
        help="Select return type: Device information vs Model information",
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
    parser.add_argument("-s", "--sequence", dest="L", type=int, help="Sequence length")
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
        required=True,
        help=(
            "Comma-separated batch sizes for per-batch profiling (e.g., '1,2,4,8'). "
            "The first value drives the top-level fields (b_i, b_o, f_q, f_out)."
        ),
    )
    # Phase flag removed: we always emit split prefill/decode outputs
    args = parser.parse_args()

    # Validate repository ID is provided
    if not args.repo_id:
        raise ValueError(
            "Repository ID (--repo) is required to load model config from Hugging Face Hub."
        )

    # Load config from Hugging Face Hub using the API function
    config_obj, config_dict, module_name = load_config_from_repo(
        repo_id=args.repo_id, model_name=args.model
    )

    if args.ret == "device":
        if args.output_path is None:
            ret = profile_device(config_obj, args.debug_lvl, args.max_batch_exp)
            pprint.pprint(ret)
            return 1  # FIXME: why?
        with open(args.output_path, "w") as f:
            ret = profile_device(config_obj, args.debug_lvl, args.max_batch_exp)
            if ret is not None:
                f.write(ret.json())
            else:
                raise RuntimeError("Unable to profile device.")
        return 1

    elif args.ret == "model":
        if args.output_path is None:
            raise ValueError("No output path given.")
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

        # Determine base batch size from --batches
        base_B = bs_list[0]

        # Instantiate model for profiling
        module = importlib.import_module(f"mlx_lm.models.{module_name}")
        Model = getattr(module, "Model")
        obj = Model(config_obj)

        with open(args.output_path, "w") as f:
            # Emit compact model profile with phase-split FLOPs only
            ret = profile_model_split(
                obj,
                config_obj,
                base_B,
                int(args.L),  # FIXME: ???
                config_dict,
                args.debug_lvl,
                bs_list=bs_list,
            )
            f.write(json.dumps(asdict(ret)))
        return 1

    else:
        raise ValueError(
            "Unknown 'return' argument. We can only handle 'model' or 'device'."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
