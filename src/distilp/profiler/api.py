"""High-level API for profiling models using HF config.model_type."""

import json
import inspect
import importlib
from typing import Dict, List, Optional, Tuple, Any
from huggingface_hub import hf_hub_download

from .profiler import profile_model_split
from .profiler import profile_device as _profile_device
from ..common import ModelProfileSplit, DeviceProfileInfo

# Minimal aliasing from HF model_type to mlx_lm.models module names
_MODEL_TYPE_ALIASES: Dict[str, str] = {
    # Llama variants
    "llama": "llama",
    "llama2": "llama",
    "llama-2": "llama",
    "llama3": "llama",
    "llama-3": "llama",
    # Mistral
    "mistral": "mistral",
    # Qwen families
    "qwen2": "qwen2",
    "qwen-2": "qwen2",
    "qwen2_moe": "qwen2_moe",
    "qwen2-moe": "qwen2_moe",
    "qwen3": "qwen3",
    "qwen-3": "qwen3",
    "qwen3_moe": "qwen3_moe",
    "qwen3-moe": "qwen3_moe",
    # Gemma
    "gemma": "gemma2",
    "gemma2": "gemma2",
    # Phi
    "phi3": "phi3",
    # GPT-OSS example
    "gpt_oss": "gpt_oss",
}


def _resolve_module_from_config(config_dict: Dict[str, Any]) -> str:
    """Resolve mlx_lm.models.<module> from HF config.model_type.

    Raises a clear error if model_type is missing or does not map/import.
    """
    mt = (config_dict.get("model_type") or "").strip()
    if not mt:
        raise ValueError(
            "config.json is missing 'model_type'; repo may not be MLX-formatted. "
            "Pass model_name explicitly to override."
        )
    mt_norm = mt.replace(" ", "").lower()
    module_name = _MODEL_TYPE_ALIASES.get(mt_norm, mt_norm)

    # Validate importability early for better error messages
    try:
        importlib.import_module(f"mlx_lm.models.{module_name}")
    except Exception as e:
        raise ImportError(
            f"Unsupported or unknown model_type '{mt}' for MLX (resolved to '{module_name}'). "
            f"Ensure this repo targets mlx_lm or pass model_name explicitly. Error: {e}"
        )
    return module_name


def load_config_from_repo(
    repo_id: str, model_name: Optional[str] = None
) -> Tuple[Any, Dict, str]:
    """
    Load only configuration from a HuggingFace repository without creating the model.
    This is more memory-efficient than load_model_from_repo when only config is needed.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'Qwen/Qwen3-4B-MLX-8bit')
        model_name: Optional MLX model name. If not provided, will be inferred.

    Returns:
        Tuple of (config_obj, config_dict, module_name)
    """
    # Download and load config from HuggingFace Hub (to resolve model_type)
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Unable to download config from HuggingFace Hub for {repo_id}: {e}"
        )

    # Resolve module_name from config unless explicitly provided
    module_name = model_name or _resolve_module_from_config(config_dict)

    # Load the ModelArgs class from mlx_lm
    try:
        module = importlib.import_module(f"mlx_lm.models.{module_name}")
    except ImportError as e:
        raise ImportError(
            f"Model '{module_name}' not found in mlx_lm registry. "
            f"Ensure the HF repo is MLX-compatible or pass a supported model_name. Error: {e}"
        )

    ModelArgs = getattr(module, "ModelArgs", None)
    if ModelArgs is None:
        raise ImportError(
            f"Could not import 'ModelArgs' from mlx_lm.models.{module_name}"
        )

    # Filter config to only include valid ModelArgs parameters
    modelargs_params = inspect.signature(ModelArgs.__init__).parameters
    valid_params = [p for p in modelargs_params if p != "self"]
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}

    # Create model config only (no model instantiation)
    try:
        config_obj = ModelArgs(**filtered_config)
    except Exception as e:
        raise RuntimeError(f"Unable to instantiate config for {module_name}: {e}")

    return config_obj, config_dict, module_name


def profile_model(
    repo_id: str,
    batch_sizes: Optional[List[int]] = None,
    sequence_length: int = 512,
    model_name: Optional[str] = None,
    debug: int = 0,
) -> ModelProfileSplit:
    """
    Profile a model from a HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'Qwen/Qwen3-4B-MLX-8bit')
        batch_sizes: List of batch sizes to profile (default: [1, 2, 4, 8])
        sequence_length: Sequence length for profiling (default: 512)
        model_name: Optional MLX model name. If not provided, will be inferred.
        debug: Debug logging level (default: 0)

    Returns:
        ModelProfileSplit object with profiling results

    Example:
        >>> from src.dperf.api import profile_model
        >>> result = profile_model(
        ...     repo_id="Qwen/Qwen3-4B-MLX-8bit",
        ...     batch_sizes=[1, 2, 4],
        ...     sequence_length=512
        ... )
        >>> print(result)
    """
    # Set default batch sizes if not provided
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    # Load configuration and resolve module name
    config_obj, config_dict, module_name_final = load_config_from_repo(
        repo_id, model_name
    )

    # Create the model (only structure, no weights)
    module = importlib.import_module(f"mlx_lm.models.{module_name_final}")
    Model = getattr(module, "Model", None)
    model = Model(config_obj)  # type: ignore FIXME: !!!

    # Profile the model
    result = profile_model_split(
        model=model,
        config=config_obj,
        B=batch_sizes[0],  # Base batch size
        L=sequence_length,
        config_dict=config_dict,
        debug=debug,
        bs_list=batch_sizes,
    )

    return result


def profile_device(
    repo_id: str,
    model_name: Optional[str] = None,
    max_batch_exp: int = 6,
    debug: int = 0,
) -> DeviceProfileInfo:
    """
    Profile device capabilities using a model configuration from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'Qwen/Qwen3-4B-MLX-8bit')
        model_name: Optional MLX model name. If not provided, will be inferred.
        max_batch_exp: Maximum batch exponent for profiling (default: 6, which is 2^6=64)
        debug: Debug logging level (default: 0)

    Returns:
        DeviceProfileInfo object with device profiling results

    Example:
        >>> from src.dperf.api import profile_device
        >>> device_info = profile_device(
        ...     repo_id="Qwen/Qwen3-4B-MLX-8bit",
        ...     max_batch_exp=6
        ... )
        >>> print(device_info)
    """
    # Load ONLY the configuration (no model instantiation to save memory)
    config_obj, config_dict, _ = load_config_from_repo(repo_id, model_name)

    # Wire quantization fields onto the config object so lower layers can infer storage size
    try:
        if isinstance(config_dict.get("quantization"), dict):
            setattr(config_obj, "quantization", config_dict["quantization"])
        if isinstance(config_dict.get("quantization_config"), dict):
            setattr(
                config_obj, "quantization_config", config_dict["quantization_config"]
            )
        # Also expose dtype hints if not present on config object
        if not hasattr(config_obj, "torch_dtype") and config_dict.get("torch_dtype"):
            setattr(config_obj, "torch_dtype", config_dict.get("torch_dtype"))
        if not hasattr(config_obj, "dtype") and config_dict.get("dtype"):
            setattr(config_obj, "dtype", config_dict.get("dtype"))
    except Exception:
        pass

    # Profile the device
    device_info = _profile_device(
        config=config_obj, debug=debug, max_batch_exp=max_batch_exp
    )

    return device_info


__all__ = ["load_config_from_repo", "profile_model", "profile_device"]
