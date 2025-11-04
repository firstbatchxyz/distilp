import sys
import json
import pprint
import inspect
import importlib
from dataclasses import asdict
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download

from distilp.profiler import profile_model_split, profile_device
from distilp.profiler.api import _resolve_module_from_config


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

    # Load config from Hugging Face Hub
    if args.repo_id:
        try:
            config_path = hf_hub_download(repo_id=args.repo_id, filename="config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Unable to download config from Hugging Face Hub: {e}")
    else:
        raise ValueError(
            "Repository ID (--repo) is required to load model config from Hugging Face Hub."
        )

    # Resolve module name from CLI override or config.model_type
    module_name = args.model or _resolve_module_from_config(config_dict)

    # Import mlx_lm module and gather types
    try:
        module = importlib.import_module(f"mlx_lm.models.{module_name}")
    except ImportError as e:
        raise RuntimeError(
            f"Model '{module_name}' not found in mlx_lm registry. Error: {e}"
        )
    Model = getattr(module, "Model")
    ModelArgs = getattr(module, "ModelArgs")

    assert inspect.isclass(Model)
    assert inspect.isclass(ModelArgs)
    # FIXME: checking both is redundant?
    if Model is None or ModelArgs is None:
        raise RuntimeError(
            f"Could not import symbols 'Model' and 'ModelArgs' from mlx_lm.models.{module_name}"
        )

    try:
        modelargs_params = inspect.signature(ModelArgs.__init__).parameters
        # Skip 'self' parameter
        valid_params = [p for p in modelargs_params if p != "self"]
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}

        config_obj = ModelArgs(**filtered_config)
        obj = Model(config_obj)
    except Exception as e:
        raise RuntimeError(f"Unable to instantiate model object from config: {e}")

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
