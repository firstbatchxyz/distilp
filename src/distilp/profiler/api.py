"""High-level API for profiling models using HF config.model_type."""

from typing import List, Optional


from .profiler.model import profile_model_split
from .profiler.device import profile_device as _profile_device
from ..common import ModelProfileSplit, DeviceProfileInfo
from .models import load_config_from_repo


def profile_model(
    repo_id: str,
    batch_sizes: Optional[List[int]] = None,
    sequence_length: int = 512,
    debug: int = 0,
) -> ModelProfileSplit:
    """
    Profile a model from a HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'Qwen/Qwen3-4B-MLX-8bit')
        batch_sizes: List of batch sizes to profile (default: [1, 2, 4, 8])
        sequence_length: Sequence length for profiling (default: 512)
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
    # config_obj, config_dict, module_name_final = load_config_from_repo(repo_id, model_name)
    config = load_config_from_repo(repo_id)

    # Profile the model
    result = profile_model_split(
        config,
        B=batch_sizes[0],  # Base batch size
        L=sequence_length,
        debug=debug,
        bs_list=batch_sizes,
    )

    return result


def profile_device(
    repo_id: str,
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
    cfg = load_config_from_repo(repo_id)
    device_info = _profile_device(config=cfg, debug=debug, max_batch_exp=max_batch_exp)

    return device_info


__all__ = ["load_config_from_repo", "profile_model", "profile_device"]
