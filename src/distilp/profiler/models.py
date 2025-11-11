from typing import Dict, Any, Literal
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
import importlib
import json

import mlx_lm.models.gpt_oss as GPTOSS
import mlx_lm.models.phi3 as PHI3
import mlx_lm.models.mixtral as MISTRAL
import mlx_lm.models.llama as LLAMA
import mlx_lm.models.qwen2 as QWEN2
import mlx_lm.models.qwen2_moe as QWEN2_MOE
import mlx_lm.models.qwen3 as QWEN3
import mlx_lm.models.qwen3_moe as QWEN3_MOE
import mlx_lm.models.gemma2 as GEMMA2


# import mlx_lm.models.deepseek_v2 as DEEPSEEK_V2


# Minimal aliasing from HF model_type to mlx_lm.models module names
type MODEL_ARCHS = Literal[
    "llama",
    "mistral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "gemma2",
    "phi3",
    "gpt_oss",
]


_MODEL_TYPE_ALIASES: Dict[str, MODEL_ARCHS] = {
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
    # GPT-OSS
    "gpt_oss": "gpt_oss",
}


class _MLX_ModelArgs_GPTOSS(BaseModel):
    module_name: Literal["gpt_oss"]
    args: GPTOSS.ModelArgs


class _MLX_ModelArgs_PHI3(BaseModel):
    module_name: Literal["phi3"]
    args: PHI3.ModelArgs


class _MLX_ModelArgs_LLAMA(BaseModel):
    module_name: Literal["llama"]
    args: LLAMA.ModelArgs


class _MLX_ModelArgs_MISTRAL(BaseModel):
    module_name: Literal["mistral"]
    args: MISTRAL.ModelArgs


class _MLX_ModelArgs_QWEN2(BaseModel):
    module_name: Literal["qwen2"]
    args: QWEN2.ModelArgs


class _MLX_ModelArgs_QWEN2_MOE(BaseModel):
    module_name: Literal["qwen2_moe"]
    args: QWEN2_MOE.ModelArgs


class _MLX_ModelArgs_QWEN3(BaseModel):
    module_name: Literal["qwen3"]
    args: QWEN3.ModelArgs


class _MLX_ModelArgs_QWEN3_MOE(BaseModel):
    module_name: Literal["qwen3_moe"]
    args: QWEN3_MOE.ModelArgs


class _MLX_ModelArgs_GEMMA2(BaseModel):
    module_name: Literal["gemma2"]
    args: GEMMA2.ModelArgs


class MLX_ModelArgs(BaseModel):
    """A discriminated union of all supported `mlx_lm` `ModelArgs` types."""

    module: (
        _MLX_ModelArgs_GPTOSS
        | _MLX_ModelArgs_PHI3
        | _MLX_ModelArgs_LLAMA
        | _MLX_ModelArgs_MISTRAL
        | _MLX_ModelArgs_QWEN2
        | _MLX_ModelArgs_QWEN2_MOE
        | _MLX_ModelArgs_QWEN3
        | _MLX_ModelArgs_QWEN3_MOE
        | _MLX_ModelArgs_GEMMA2
    ) = Field(discriminator="module_name")

    # raw dictionary of original config.json
    raw: Dict[str, Any] = Field(default_factory=dict)

    def model(self):
        if self.module.module_name == "gpt_oss":
            return GPTOSS.Model(self.module.args)
        elif self.module.module_name == "phi3":
            return PHI3.Model(self.module.args)
        elif self.module.module_name == "llama":
            return LLAMA.Model(self.module.args)
        elif self.module.module_name == "mistral":
            return MISTRAL.Model(self.module.args)
        elif self.module.module_name == "qwen2":
            return QWEN2.Model(self.module.args)
        elif self.module.module_name == "qwen2_moe":
            return QWEN2_MOE.Model(self.module.args)
        elif self.module.module_name == "qwen3":
            return QWEN3.Model(self.module.args)
        elif self.module.module_name == "qwen3_moe":
            return QWEN3_MOE.Model(self.module.args)
        elif self.module.module_name == "gemma2":
            return GEMMA2.Model(self.module.args)
        else:
            raise ValueError(f"Unsupported module_name: {self.module.module_name}")

    def hidden_size(self) -> int:
        return self.module.args.hidden_size

    def num_hidden_layers(self) -> int:
        return self.module.args.num_hidden_layers

    def intermediate_size(self) -> int:
        return self.module.args.intermediate_size

    def num_attention_heads(self) -> int:
        return self.module.args.num_attention_heads

    def max_position_embeddings(self, default: int) -> int:
        """Some models do not have max_position_embeddings defined; use default then."""
        if (
            self.module.module_name == "phi3"
            or self.module.module_name == "qwen2"
            or self.module.module_name == "qwen3"
            or self.module.module_name == "qwen3_moe"
        ):
            return self.module.args.max_position_embeddings
        else:
            return default

    def head_dim(self) -> int:
        if (
            self.module.module_name == "llama"
            or self.module.module_name == "phi3"
            or self.module.module_name == "mistral"
            or self.module.module_name == "qwen2"
            or self.module.module_name == "qwen2_moe"
        ):
            return self.module.args.hidden_size // self.module.args.num_attention_heads
        else:
            return self.module.args.head_dim

    def num_key_value_heads(self) -> int:
        ans = self.module.args.num_key_value_heads
        if ans is None:
            # fallback to `num_attention_heads`
            return self.num_attention_heads()
        return ans

    def model_type(self) -> str:
        return self.module.args.model_type

    def vocab_size(self) -> int:
        return self.module.args.vocab_size

    def moe_layer_freq(self) -> int:
        if self.module.module_name == "qwen3_moe":
            return self.module.args.decoder_sparse_step
        else:
            return 1  # indeed defaults to 1

    def n_routed_experts(self) -> int:
        """Get number of routed experts if applicable, else 0."""
        if self.module.module_name == "qwen2_moe" or self.module.module_name == "qwen3_moe":
            return self.module.args.num_experts
        elif self.module.module_name == "mistral" or self.module.module_name == "gpt_oss":
            return self.module.args.num_local_experts
        else:
            return 0

    def mlp_only_layers(self) -> list[int]:
        """Get list of MLP-only layer indices if applicable, else empty list."""
        if self.module.module_name == "qwen3_moe":
            return self.module.args.mlp_only_layers
        else:
            return []

    def moe_intermediate(self) -> int:
        """Get MoE intermediate size if applicable, else 0."""
        if self.module.module_name == "qwen2_moe" or self.module.module_name == "qwen3_moe":
            return self.module.args.moe_intermediate_size
        else:
            # uses standard intermediate size if no MoE
            return self.module.args.intermediate_size

    def shared_intermediate(self) -> int:
        """Get shared experts intermediate size if applicable, else 0."""
        if self.module.module_name == "qwen2_moe":
            return self.module.args.shared_expert_intermediate_size
        else:
            return self.moe_intermediate()

    def num_experts_tok(self) -> int:
        """Get number of experts per token if applicable, else 0."""
        if (
            self.module.module_name == "qwen2_moe"
            or self.module.module_name == "qwen3_moe"
            or self.module.module_name == "gpt_oss"
            or self.module.module_name == "mistral"
        ):
            return self.module.args.num_experts_per_tok
        else:
            raise ValueError(f"num_experts_tok is not applicable for {self.module.module_name}")

    def n_shared(self) -> int:
        """Get number of shared experts if applicable, else 0."""
        # FIXME: none of the models we have use of this?
        return 0


def load_config_from_repo(repo_id: str) -> MLX_ModelArgs:
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
        raise RuntimeError(f"Unable to download config from HuggingFace Hub for {repo_id}: {e}")

    # resolve module_name & load
    module_name = _resolve_module_from_config(config_dict)
    try:
        return MLX_ModelArgs.model_validate(
            {
                "module": {
                    "module_name": module_name,
                    "args": config_dict,
                },
                "raw": config_dict,
            }
        )
    except ImportError as e:
        raise ImportError(
            f"Model '{module_name}' not found in mlx_lm registry. "
            f"Ensure the HF repo is MLX-compatible or pass a supported model_name. Error: {e}"
        )


def _resolve_module_from_config(config_dict: Dict[str, Any]) -> str:
    """Resolve mlx_lm.models.<module> from HF config.model_type.

    Raises a clear error if model_type is missing or does not map/import.
    """
    # get model type from config
    mt: str | None = config_dict.get("model_type")
    if not mt:
        raise ValueError(
            "config.json is missing 'model_type'; repo may not be MLX-formatted. "
            "Pass model_name explicitly to override."
        )

    # get corresponding module name
    module_name: MODEL_ARCHS | None = _MODEL_TYPE_ALIASES.get(mt.strip().replace(" ", "").lower(), None)
    if not module_name:
        raise ValueError(
            f"Unsupported or unknown model_type '{mt}' for MLX. "
            f"Ensure this repo targets mlx_lm or pass model_name explicitly."
        )

    # Validate importability early for better error messages
    try:
        importlib.import_module(f"mlx_lm.models.{module_name}")
    except Exception as e:
        raise ImportError(
            f"Unsupported or unknown model_type '{mt}' for MLX (resolved to '{module_name}'). "
            f"Ensure this repo targets mlx_lm or pass model_name explicitly. Error: {e}"
        )
    return module_name
