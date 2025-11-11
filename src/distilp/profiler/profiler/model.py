from typing import List, Literal
from math import ceil
from fnmatch import fnmatch
from typing import Any

from pydantic import BaseModel


from ...common import (
    ModelProfileInfo,
    MoEModelProfileInfo,
    ModelProfileSplit,
    ModelProfilePhased,
)
from ..models import MLX_ModelArgs

type ModelPhase = Literal["merged", "prefill", "decode"]


class LayerMetadata(BaseModel):
    """Layer-level profiling metadata."""

    name: str = ""  # Name of the symbol
    submodules: Any = None  # Submodules decomposed into LayerMeta
    parent_layer: Any = None  # Parent Compount Layer
    layer: Any = None  # Original object
    flops: float = 0.0  # Estimated FLOPs to compute
    weight_bytes: int = 0  # Bytes of internal weight tensor
    input_bytes: int = 0  # Bytes of input tensor
    output_bytes: int = 0  # Bytes of output tensor
    kv_cache_t: int = 0  # Total tokens stored in KV Cache
    kv_cache_r: int = 0  # Bytes of KV Cache read
    kv_cache_w: int = 0  # Bytes of KV Cache written
    ram_vram_rw: int = 0  # Bytes of data transmitted between RAM <-> VRAM

    # Component breakdowns for MoE solver assignment
    attn_flops: float = 0.0  # Attention-specific FLOPs
    attn_bytes: int = 0  # Attention weight bytes
    moe_router_flops: float = 0.0  # MoE router/gate FLOPs
    moe_router_bytes: int = 0  # MoE router/gate bytes
    moe_expert_flops: float = 0.0  # Per-expert FLOPs (for routed experts)
    moe_expert_bytes: int = 0  # Per-expert bytes (for routed experts)
    moe_expert_flops_per_token: float = 0.0  # FLOPs for a single active expert on one token
    moe_shared_flops: float = 0.0  # Total shared experts FLOPs
    moe_shared_bytes: int = 0  # Total shared experts bytes
    is_moe_layer: bool = False  # Whether this layer uses MoE

    def __repr__(self):
        return (
            f"<LayerMeta {self.name}: "
            f"FLOPs={self.flops}, INPUT={self.input_bytes}, OUTPUT={self.output_bytes}, "
            f"WEIGHT={self.weight_bytes}, parent={self.parent_layer.__class__.__name__}"
        )


""" Estimate the FLOP count of all 'mlx_lm' models at a decoder level
    NOTE: Small OPs like RoPE and norms default to 0 FLOPs
    NOTE: FMA defaults to 2 FLOPs """

block_names = ["TransformerBlock", "DecoderLayer"]


# Add the quantization metadata to final byte count
def __quantized_bytes(n, d_bits: int, group_size: int, scale_bytes: int, zero_bytes: int) -> int:
    scaled_bits = n * d_bits
    code_bytes = ceil(scaled_bits / 8)
    if group_size and group_size > 0:
        groups = (n + group_size - 1) // group_size
        meta_bytes = groups * (scale_bytes + zero_bytes)
    else:
        meta_bytes = 0
    return code_bytes + meta_bytes


def in_profile_model(
    cfg: MLX_ModelArgs,
    B: int = 1,
    L: int = 4096,
    a_bits=16,
    w_bits=16,
    group_size=32,
    debug=0,
    phase: ModelPhase = "merged",  # 'prefill' | 'decode' | 'merged'
    exclude_patterns: list = [],
    fp_bits: int = 16,
):
    decoder_idx = 1
    layers: list[LayerMetadata] = []

    # Quantization hard-coded scale and zero bytes
    scale_bytes = 2
    zero_bytes = 0

    def is_excluded(path: str) -> bool:
        nonlocal exclude_patterns
        for pat in exclude_patterns:
            try:
                if fnmatch(path, pat):
                    return True
            except Exception:
                pass
        return False

    # Append a symbolic prefill layer to account for these FLOPs
    prefill = LayerMetadata()
    prefill.name = "prefill"
    layers.append(prefill)

    if debug >= 1:
        print("FMA: 2 FLOPs")
        # print(f"Quantization: {config.quantization.bits}")
        print(f"Parsing model {cfg.model_type()}:")
        print(f"Quantization: bits={w_bits}, group_size={group_size}")
        print(
            f"    hidden_size={cfg.hidden_size()},\n    vocab_size={cfg.vocab_size()},\n"
            f"    num_hidden_layers={cfg.num_hidden_layers()}"
        )

    for lyr in cfg.module.model.layers:
        lm = LayerMetadata()
        lm.layer = lyr
        lm.name = f"decoder_{decoder_idx}"
        if any(x in lyr.__class__.__name__ for x in ["TransformerBlock", "DecoderLayer"]):
            lm.input_bytes = (B * L * cfg.hidden_size() * a_bits) // 8
            lm.output_bytes = (B * L * cfg.hidden_size() * a_bits) // 8

            # Tokens processed per phase
            tokens_prefill = B * L  # Apply attention to whole sequence
            tokens_decode = B * 1  # Decode a single token
            if phase == "prefill":
                tokens = tokens_prefill
            elif phase == "decode":
                tokens = tokens_decode
            else:
                # merged: one full prefill pass + one decode step
                tokens = tokens_prefill + tokens_decode
            if debug >= 1:
                print(f"\nParsing [decoder.{decoder_idx}]:")
            for name, obj in lyr.named_modules():
                if name == "post_attention_layernorm" or name == "input_layernorm":
                    pass

                elif name in ("mlp", "ffn", "feed_forward", "feedforward", "ffn_layer"):
                    # MoE - check for various naming conventions (use cfg_get)
                    mlp_path = f"model.layers.{decoder_idx}.mlp"
                    n_experts = cfg.n_routed_experts()

                    layer_freq = cfg.moe_layer_freq()
                    mlp_only_layers = set(cfg.mlp_only_layers())

                    if n_experts != 0 and decoder_idx % layer_freq == 0 and decoder_idx not in mlp_only_layers:
                        lm.is_moe_layer = True
                        has_router_gate = False
                        found_switch_block = False
                        # Accumulators
                        gate_f = 0
                        smlp_f = 0
                        smlp_b = 0
                        se_f = 0
                        se_b = 0
                        # Initialize variables that may be used in debug statements
                        # moe_intermediate = 0
                        # shared_intermediate = None
                        # num_experts_tok = None
                        num_proj_smlp = 0
                        num_proj_se = 0
                        # Fallback detection flags for routed/shared experts
                        found_expert_up = False
                        found_expert_down = False
                        found_expert_gatep = False
                        found_shared_up = False
                        found_shared_down = False
                        found_shared_gatep = False
                        # Traverse the MLP object directly
                        for key, leaf in obj.named_modules():
                            key_l = key.lower()
                            if (
                                key == "gate"
                                or key_l.endswith(".gate")
                                or key_l.endswith(".router")
                                or key_l == "router"
                            ):
                                gate_f += 2 * tokens * cfg.hidden_size() * n_experts
                                lm.flops += gate_f
                                lm.moe_router_flops = gate_f
                                # Router weight bytes (apply per-module exclusion)
                                router_params = cfg.hidden_size() * n_experts
                                router_path = f"model.layers.{decoder_idx}.mlp.router"
                                local_w_bits = fp_bits if is_excluded(router_path) else w_bits
                                if local_w_bits < 16 and group_size is not None:
                                    lm.moe_router_bytes = __quantized_bytes(
                                        router_params,
                                        local_w_bits,
                                        group_size,
                                        scale_bytes,
                                        zero_bytes,
                                    )
                                else:
                                    lm.moe_router_bytes = ceil((router_params * local_w_bits) / 8)
                                has_router_gate = True

                            elif key == "switch_mlp":
                                moe_intermediate = cfg.moe_intermediate()
                                if moe_intermediate == 0:
                                    raise ValueError(
                                        "MoE layer detected but no valid intermediate size found in config"
                                    )
                                DS = cfg.hidden_size() * moe_intermediate
                                num_experts_tok = cfg.num_experts_tok()

                                num_proj_smlp = 2
                                for key2, proj in leaf.named_modules():
                                    if key2 == "gate_proj":
                                        smlp_f += 2 * tokens * num_experts_tok * DS
                                        num_proj_smlp = 3
                                    elif key2 in ["up_proj", "down_proj"]:
                                        smlp_f += 2 * tokens * num_experts_tok * DS
                                    elif key2 == "activations":
                                        # Activation FLOPs are small; include linear in L for completeness
                                        smlp_f += tokens * num_experts_tok * moe_intermediate

                                # Per-active-expert-per-token FLOPs (one token through one expert MLP)
                                # 2 * num_proj * H * D + activation_cost (approx D)
                                lm.moe_expert_flops_per_token = (
                                    2 * num_proj_smlp * cfg.hidden_size() * moe_intermediate + moe_intermediate
                                )
                                found_switch_block = True

                                # Add the quantization group overhead per projection matrix
                                local_w_bits = w_bits
                                # Optional: if entire mlp is excluded, use fp_bits
                                mlp_path = f"model.layers.{decoder_idx}.mlp"
                                if is_excluded(mlp_path):
                                    local_w_bits = fp_bits
                                if local_w_bits < 16 and group_size is not None:
                                    per_proj_params = cfg.hidden_size() * moe_intermediate
                                    per_proj_bytes = __quantized_bytes(
                                        per_proj_params,
                                        local_w_bits,
                                        group_size,
                                        scale_bytes,
                                        zero_bytes,
                                    )
                                    smlp_b = n_experts * num_proj_smlp * per_proj_bytes
                                else:
                                    smlp_b = ceil(
                                        (
                                            n_experts
                                            * num_proj_smlp
                                            * cfg.hidden_size()
                                            * moe_intermediate
                                            * local_w_bits
                                        )
                                        / 8
                                    )
                                lm.weight_bytes += smlp_b
                                lm.flops += smlp_f
                                # Per-expert metrics
                                lm.moe_expert_flops = smlp_f / n_experts if n_experts > 0 else 0
                                lm.moe_expert_bytes = smlp_b // n_experts if n_experts > 0 else 0

                            elif key == "shared_experts":
                                n_shared = cfg.n_shared()
                                shared_intermediate = cfg.shared_intermediate()
                                num_proj_se = 2
                                for key2, proj in leaf.named_modules():
                                    if key2 == "gate_proj":
                                        num_proj_se = 3
                                    if key2 in ["gate_proj", "up_proj", "down_proj"]:
                                        se_f += 2 * tokens * cfg.hidden_size() * n_shared * shared_intermediate

                                local_w_bits = w_bits
                                if is_excluded(mlp_path):
                                    local_w_bits = fp_bits
                                if local_w_bits < 16 and group_size is not None:
                                    per_proj_params = cfg.hidden_size() * shared_intermediate
                                    per_proj_bytes = __quantized_bytes(
                                        per_proj_params,
                                        local_w_bits,
                                        group_size,
                                        scale_bytes,
                                        zero_bytes,
                                    )
                                    se_b = n_shared * num_proj_se * per_proj_bytes
                                else:
                                    se_b = (
                                        n_shared * num_proj_se * cfg.hidden_size() * shared_intermediate * local_w_bits
                                    ) // 8
                                lm.weight_bytes += se_b
                                lm.flops += se_f
                                # Shared experts metrics
                                lm.moe_shared_flops = se_f
                                lm.moe_shared_bytes = se_b

                            # Fallback pattern detection for routed experts
                            if (
                                "experts" in key_l or "local_experts" in key_l or "routed_experts" in key_l
                            ) and not key_l.startswith("shared"):
                                if key_l.endswith("up_proj") or ".up_proj" in key_l:
                                    found_expert_up = True
                                if key_l.endswith("down_proj") or ".down_proj" in key_l:
                                    found_expert_down = True
                                if key_l.endswith("gate_proj") or ".gate_proj" in key_l:
                                    found_expert_gatep = True
                            # Fallback pattern detection for shared experts
                            if "shared" in key_l and ("experts" in key_l or "expert" in key_l):
                                if key_l.endswith("up_proj") or ".up_proj" in key_l:
                                    found_shared_up = True
                                if key_l.endswith("down_proj") or ".down_proj" in key_l:
                                    found_shared_down = True
                                if key_l.endswith("gate_proj") or ".gate_proj" in key_l:
                                    found_shared_gatep = True

                        # End of traversal: if no switch_mlp block but expert projections detected, compute generically
                        if not found_switch_block and (found_expert_up or found_expert_down or found_expert_gatep):
                            moe_intermediate = cfg.moe_intermediate()
                            DS = cfg.hidden_size() * moe_intermediate
                            num_experts_tok = cfg.num_experts_tok()

                            num_proj_smlp = int(found_expert_up) + int(found_expert_down) + int(found_expert_gatep)
                            # FLOPs for active experts
                            smlp_f = num_proj_smlp * (2 * tokens * num_experts_tok * DS)
                            # Activation cost (approximate)
                            smlp_f += tokens * num_experts_tok * moe_intermediate
                            # Bytes per projection matrix (quantized)
                            local_w_bits = w_bits
                            if is_excluded(mlp_path):
                                local_w_bits = fp_bits
                            if local_w_bits < 16 and group_size is not None:
                                per_proj_params = cfg.hidden_size() * moe_intermediate
                                per_proj_bytes = __quantized_bytes(
                                    per_proj_params,
                                    local_w_bits,
                                    group_size,
                                    scale_bytes,
                                    zero_bytes,
                                )
                                smlp_b = n_experts * num_proj_smlp * per_proj_bytes
                            else:
                                smlp_b = ceil(
                                    (n_experts * num_proj_smlp * cfg.hidden_size() * moe_intermediate * local_w_bits)
                                    / 8
                                )
                            lm.weight_bytes += smlp_b
                            lm.flops += smlp_f
                            lm.moe_expert_flops = smlp_f / n_experts if n_experts > 0 else 0
                            lm.moe_expert_bytes = smlp_b // n_experts if n_experts > 0 else 0
                            lm.moe_expert_flops_per_token = (
                                2 * num_proj_smlp * cfg.hidden_size() * moe_intermediate + moe_intermediate
                            )

                        # Shared experts fallback: if not computed but detected
                        if se_b == 0 and (found_shared_up or found_shared_down or found_shared_gatep):
                            n_shared: int = cfg.n_shared()
                            shared_intermediate: int = cfg.shared_intermediate()
                            if shared_intermediate:
                                num_proj_se = int(found_shared_up) + int(found_shared_down) + int(found_shared_gatep)
                                local_w_bits = w_bits
                                if is_excluded(mlp_path):
                                    local_w_bits = fp_bits
                                if local_w_bits < 16 and group_size is not None:
                                    per_proj_params = cfg.hidden_size() * shared_intermediate
                                    per_proj_bytes = __quantized_bytes(
                                        per_proj_params,
                                        local_w_bits,
                                        group_size,
                                        scale_bytes,
                                        zero_bytes,
                                    )
                                    se_b = n_shared * num_proj_se * per_proj_bytes
                                else:
                                    se_b = (
                                        n_shared * num_proj_se * cfg.hidden_size() * shared_intermediate * local_w_bits
                                    ) // 8
                                lm.weight_bytes += se_b
                                se_f = 2 * tokens * cfg.hidden_size() * n_shared * shared_intermediate * num_proj_se
                                lm.flops += se_f
                                lm.moe_shared_flops = se_f
                                lm.moe_shared_bytes = se_b

                        # Final generic fallback for routed experts if still zero
                        if smlp_b == 0 and (n_experts is not None and n_experts > 0):
                            moe_intermediate = cfg.moe_intermediate()
                            num_experts_tok = cfg.num_experts_tok()
                            if moe_intermediate and num_experts_tok:
                                num_proj_smlp = 3
                                DS = cfg.hidden_size() * moe_intermediate
                                smlp_f = num_proj_smlp * (2 * tokens * num_experts_tok * DS) + (
                                    tokens * num_experts_tok * moe_intermediate
                                )
                                local_w_bits = fp_bits if is_excluded(mlp_path) else w_bits
                                if local_w_bits < 16 and group_size is not None:
                                    per_proj_params = cfg.hidden_size() * moe_intermediate
                                    per_proj_bytes = __quantized_bytes(
                                        per_proj_params,
                                        local_w_bits,
                                        group_size,
                                        scale_bytes,
                                        zero_bytes,
                                    )
                                    smlp_b = n_experts * num_proj_smlp * per_proj_bytes
                                else:
                                    smlp_b = ceil(
                                        (
                                            n_experts
                                            * num_proj_smlp
                                            * cfg.hidden_size()
                                            * moe_intermediate
                                            * local_w_bits
                                        )
                                        / 8
                                    )
                                lm.weight_bytes += smlp_b
                                lm.flops += smlp_f
                                lm.moe_expert_flops = smlp_f / n_experts if n_experts > 0 else 0
                                lm.moe_expert_bytes = smlp_b // n_experts if n_experts > 0 else 0
                                lm.moe_expert_flops_per_token = (
                                    2 * num_proj_smlp * cfg.hidden_size() * moe_intermediate + moe_intermediate
                                )

                        # If router not found as submodule, estimate generically (common for MoE blocks)
                        if not has_router_gate and (n_experts is not None and n_experts > 0):
                            gate_f = 2 * tokens * cfg.hidden_size() * n_experts
                            lm.flops += gate_f
                            lm.moe_router_flops = gate_f
                            router_params = cfg.hidden_size() * n_experts
                            router_path = f"model.layers.{decoder_idx}.mlp.router"
                            local_w_bits = fp_bits if is_excluded(router_path) else w_bits
                            if local_w_bits < 16 and group_size is not None:
                                lm.moe_router_bytes = __quantized_bytes(
                                    router_params,
                                    local_w_bits,
                                    group_size,
                                    scale_bytes,
                                    zero_bytes,
                                )
                            else:
                                lm.moe_router_bytes = ceil((router_params * local_w_bits) / 8)

                        if debug >= 1:
                            print(
                                f"\tMoE Layer: FLOPs={smlp_f + se_f + gate_f} ({num_proj_smlp}x{num_experts_tok}x"
                                f"[{cfg.hidden_size()}, {moe_intermediate}] + {num_proj_se}x"
                                f"{n_shared}x[{cfg.hidden_size()}, {shared_intermediate}] + "
                                f"{B}x[{cfg.hidden_size()}, {n_experts}]), b={smlp_b + se_b} @ {w_bits}bits"
                                if has_router_gate
                                else f"), b={smlp_b + se_b} @ {w_bits}bits,",
                                end="",
                            )
                            print(
                                f" routed_experts={n_experts} with top-k={num_experts_tok}, ",
                                end="",
                            )
                            print(f" shared_experts={n_shared}")

                    # MLP
                    else:
                        num_proj = 2
                        proj_bytes = 0
                        # Traverse the MLP object directly
                        for key, leaf in obj.named_modules():
                            if key == "gate_proj":
                                num_proj = 3
                            if key in ["gate_proj", "up_proj", "down_proj"]:
                                lm.flops += 2 * tokens * cfg.hidden_size() * cfg.intermediate_size()
                                n = cfg.hidden_size() * cfg.intermediate_size()
                                if w_bits < 16 and group_size is not None:
                                    proj_bytes += __quantized_bytes(n, w_bits, group_size, scale_bytes, zero_bytes)
                                else:
                                    proj_bytes += ceil((n * w_bits) / 8)
                        lm.weight_bytes += proj_bytes

                        if debug >= 1:
                            print(
                                f"\tMLP Layer: FLOPs={num_proj * 2 * tokens * cfg.hidden_size() * cfg.intermediate_size()},"
                                f"  b={proj_bytes}"
                                f"( {num_proj} x [{cfg.hidden_size()}, {cfg.intermediate_size()}] @ {w_bits}),"
                                f"  b_i={B * L * cfg.hidden_size()}([{B}, {L}, {cfg.hidden_size()}])"
                            )

                # NOTE: We only compute projection bits then correct in the case of quantization
                elif name in ("self_attn", "attn", "self_attention"):
                    # Grouped Query Attention
                    is_gqa = False
                    if cfg.num_key_value_heads() != cfg.num_attention_heads():
                        is_gqa = True

                    # Low rank / Multi-head Latent Attention
                    is_mla = False
                    # if all(hasattr(config, k) for k in ["q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"]) and all(
                    #     getattr(config, k) is not None for k in ["q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"]
                    # ):
                    #     is_mla = True

                    if is_mla:
                        # Deepseek_v2,v3, Kimi_v1 and minicpm
                        # if any(hasattr(config, k) for k in ["kv_lora_rank", "v_head_dim"]):
                        #     # Q projections, flops and bytes
                        #     q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
                        #     q_a_proj = 2 * tokens * cfg.hidden_size() * config.q_lora_rank
                        #     q_b_proj = 2 * tokens * config.num_attention_heads * q_head_dim * config.q_lora_rank
                        #     q_a_proj_n = config.q_lora_rank * cfg.hidden_size()
                        #     q_b_proj_n = config.num_attention_heads * q_head_dim * config.q_lora_rank

                        #     # KV projections
                        #     if is_gqa:
                        #         pass  # TODO
                        #     else:
                        #         out_features = config.kv_lora_rank + config.qk_rope_head_dim
                        #         kv_a_proj_with_mqa = (
                        #             2 * tokens * (config.kv_lora_rank + config.qk_rope_head_dim) * cfg.hidden_size()
                        #         )
                        #         kv_b_proj = (
                        #             2
                        #             * tokens
                        #             * config.kv_lora_rank
                        #             * config.num_attention_heads
                        #             * (config.qk_nope_head_dim + config.v_head_dim)
                        #         )
                        #         kv_a_proj_n = out_features * cfg.hidden_size()
                        #         kv_b_proj_n = out_features * config.kv_lora_rank

                        #     # O projection
                        #     o_proj = 2 * tokens * config.num_attention_heads * config.v_head_dim * cfg.hidden_size()
                        #     o_proj_n = cfg.hidden_size() * config.num_attention_heads * config.v_head_dim

                        #     # KV Cache I/O
                        #     kv_elems = config.kv_lora_rank + config.qk_rope_head_dim
                        #     if phase == "prefill":
                        #         lm.kv_cache_w = (B * L * kv_elems * a_bits) / 8
                        #         lm.kv_cache_r = 0
                        #     elif phase == "decode":
                        #         lm.kv_cache_w = (B * 1 * kv_elems * a_bits) / 8
                        #         lm.kv_cache_r = (B * L * kv_elems * a_bits) / 8
                        #     else:
                        #         lm.kv_cache_w = (B * (L + 1) * kv_elems * a_bits) / 8
                        #         lm.kv_cache_r = (B * L * kv_elems * a_bits) / 8

                        #     # Totals
                        #     # Attention compute FLOPs per phase
                        #     attn_prefill = 4 * B * (config.num_attention_heads) * (L * L) * q_head_dim
                        #     attn_decode = 4 * B * (config.num_attention_heads) * L * q_head_dim
                        #     attn = (
                        #         attn_prefill
                        #         if phase == "prefill"
                        #         else attn_decode
                        #         if phase == "decode"
                        #         else attn_prefill + attn_decode
                        #     )
                        #     attn_layer_flops = q_a_proj + q_b_proj + kv_a_proj_with_mqa + kv_b_proj + o_proj + attn
                        #     lm.flops += attn_layer_flops
                        #     lm.attn_flops = attn_layer_flops

                        #     attn_path = f"model.layers.{decoder_idx}.{name}"
                        #     local_w_bits = fp_bits if is_excluded(attn_path) else w_bits
                        #     if local_w_bits < 16 and group_size is not None:
                        #         q_a_proj_bytes = __quantized_bytes(
                        #             q_a_proj_n,
                        #             local_w_bits,
                        #             group_size,
                        #             scale_bytes,
                        #             zero_bytes,
                        #         )
                        #         q_b_proj_bytes = __quantized_bytes(
                        #             q_b_proj_n,
                        #             local_w_bits,
                        #             group_size,
                        #             scale_bytes,
                        #             zero_bytes,
                        #         )
                        #         kv_a_proj_bytes = __quantized_bytes(
                        #             kv_a_proj_n,
                        #             local_w_bits,
                        #             group_size,
                        #             scale_bytes,
                        #             zero_bytes,
                        #         )
                        #         kv_b_proj_bytes = __quantized_bytes(
                        #             kv_b_proj_n,
                        #             local_w_bits,
                        #             group_size,
                        #             scale_bytes,
                        #             zero_bytes,
                        #         )
                        #         o_proj_bytes = __quantized_bytes(
                        #             o_proj_n,
                        #             local_w_bits,
                        #             group_size,
                        #             scale_bytes,
                        #             zero_bytes,
                        #         )
                        #         attn_bytes = (
                        #             q_a_proj_bytes + q_b_proj_bytes + kv_a_proj_bytes + kv_b_proj_bytes + o_proj_bytes
                        #         )
                        #     else:
                        #         q_a_proj_bytes = ceil((q_a_proj_n * local_w_bits) / 8)
                        #         q_b_proj_bytes = ceil((q_b_proj_n * local_w_bits) / 8)
                        #         kv_a_proj_bytes = ceil((kv_a_proj_n * local_w_bits) / 8)
                        #         kv_b_proj_bytes = ceil((kv_b_proj_n * local_w_bits) / 8)
                        #         o_proj_bytes = ceil((o_proj_n * local_w_bits) / 8)
                        #         attn_bytes = (
                        #             q_a_proj_bytes + q_b_proj_bytes + kv_a_proj_bytes + kv_b_proj_bytes + o_proj_bytes
                        #         )

                        #     # attn_layer_bytes = attn_bytes
                        #     lm.weight_bytes += attn_bytes
                        #     lm.attn_bytes = attn_bytes

                        #     if debug >= 1:
                        #         print(
                        #             f"\tMulti-head Latent Attention Layer {'with Group Query Attention' if is_gqa else ''}:"
                        #         )
                        #         print(
                        #             f"\t\tq_a_proj: [{cfg.hidden_size()}, {config.q_lora_rank} ], FLOPs={q_a_proj}, b={q_a_proj_bytes}"
                        #         )
                        #         print(
                        #             f"\t\tq_b_proj: [{config.num_attention_heads * q_head_dim}, {config.q_lora_rank}], "
                        #             f"FLOPs={q_b_proj}, b={q_b_proj_bytes}"
                        #         )
                        #         print(
                        #             f"\t\tkv_a_proj_with_mqa: [{cfg.hidden_size()}, {config.kv_lora_rank + config.qk_rope_head_dim}], "
                        #             f"FLOPs={kv_a_proj_with_mqa}, b={kv_a_proj_bytes}"
                        #         )
                        #         print(
                        #             f"\t\tkv_b_proj: [{config.kv_lora_rank}, "
                        #             f"{config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)}], "
                        #             f"FLOPs={kv_b_proj}, b={kv_b_proj_bytes}"
                        #         )
                        #         print(
                        #             f"\t\to_proj: [{config.num_attention_heads * config.v_head_dim}, {cfg.hidden_size()}], "
                        #             f"FLOPs={o_proj}, b={o_proj_bytes}"
                        #         )
                        #         print(
                        #             f"\t\tFLOPs={q_a_proj + q_b_proj + kv_a_proj_with_mqa + kv_b_proj + o_proj + attn} "
                        #             f"b={attn_bytes}"
                        #         )

                        #     continue

                        # Low-rank Replace (not implemented)
                        # else:
                        #     pass
                        pass

                    # Grouped Query Attention
                    elif is_gqa:
                        head_size = cfg.hidden_size() // cfg.num_attention_heads()
                        q_proj = 2 * tokens * cfg.hidden_size() * cfg.hidden_size()
                        q_proj_n = cfg.hidden_size() * cfg.hidden_size()

                        # K/V out dim = h_k * d = num_key_value_heads * head_size
                        k_out = cfg.num_key_value_heads() * head_size
                        v_out = cfg.num_key_value_heads() * head_size
                        k_proj = 2 * tokens * cfg.hidden_size() * k_out
                        k_proj_n = cfg.hidden_size() * k_out
                        v_proj = 2 * tokens * cfg.hidden_size() * v_out
                        v_proj_n = cfg.hidden_size() * v_out

                        o_proj = 2 * tokens * cfg.hidden_size() * cfg.hidden_size()
                        o_proj_n = cfg.hidden_size() * cfg.hidden_size()
                        # Attention FLOPs by phase
                        A = cfg.num_attention_heads()
                        attn_prefill = 4 * B * A * (L * L) * head_size
                        attn_decode = 4 * B * A * L * head_size
                        attn = (
                            attn_prefill
                            if phase == "prefill"
                            else attn_decode
                            if phase == "decode"
                            else attn_prefill + attn_decode
                        )

                        attn_layer_flops = q_proj + k_proj + v_proj + o_proj + attn
                        lm.flops += attn_layer_flops
                        lm.attn_flops = attn_layer_flops
                        # KV cache per phase
                        kv_elems = 2 * cfg.num_key_value_heads() * head_size
                        if phase == "prefill":
                            lm.kv_cache_w = (B * L * kv_elems * a_bits) // 8
                            lm.kv_cache_r = 0
                        elif phase == "decode":
                            lm.kv_cache_w = (B * 1 * kv_elems * a_bits) // 8
                            lm.kv_cache_r = (B * L * kv_elems * a_bits) // 8
                        else:
                            lm.kv_cache_w = (B * (L + 1) * kv_elems * a_bits) // 8
                            lm.kv_cache_r = (B * L * kv_elems * a_bits) // 8

                        attn_path = f"model.layers.{decoder_idx}.{name}"
                        local_w_bits = fp_bits if is_excluded(attn_path) else w_bits
                        if local_w_bits < 16 and group_size is not None:
                            q_proj_bytes = __quantized_bytes(
                                q_proj_n,
                                local_w_bits,
                                group_size,
                                scale_bytes,
                                zero_bytes,
                            )
                            k_proj_bytes = __quantized_bytes(
                                k_proj_n,
                                local_w_bits,
                                group_size,
                                scale_bytes,
                                zero_bytes,
                            )
                            v_proj_bytes = __quantized_bytes(
                                v_proj_n,
                                local_w_bits,
                                group_size,
                                scale_bytes,
                                zero_bytes,
                            )
                            o_proj_bytes = __quantized_bytes(
                                o_proj_n,
                                local_w_bits,
                                group_size,
                                scale_bytes,
                                zero_bytes,
                            )
                            attn_bytes = q_proj_bytes + k_proj_bytes + v_proj_bytes + o_proj_bytes
                        else:
                            q_proj_bytes = ceil((q_proj_n * local_w_bits) / 8)
                            k_proj_bytes = ceil((k_proj_n * local_w_bits) / 8)
                            v_proj_bytes = ceil((v_proj_n * local_w_bits) / 8)
                            o_proj_bytes = ceil((o_proj_n * local_w_bits) / 8)
                            attn_bytes = q_proj_bytes + k_proj_bytes + v_proj_bytes + o_proj_bytes

                        lm.weight_bytes += attn_bytes
                        lm.attn_bytes = attn_bytes

                        if debug >= 1:
                            print(
                                f"\tGrouped Query Attention Layer: "
                                f"FLOPs={q_proj + k_proj + v_proj + o_proj + attn}, "
                                f"b={attn_bytes} "
                                f"kv_cache_read={lm.kv_cache_r}, "
                                f"kv_cache_write={lm.kv_cache_w}"
                            )
                            print(f"\t\tq_proj: FLOPs={q_proj}, b={q_proj_bytes}")
                            print(f"\t\tk_proj: FLOPs={k_proj}, b={k_proj_bytes}")
                            print(f"\t\tv_proj: FLOPs={v_proj}, b={v_proj_bytes}")
                            print(f"\t\to_proj: FLOPs={o_proj}, b={o_proj_bytes}")

                    # MHA
                    else:
                        # Standard MHA (no GQA): k/v heads = A, head dim = H//A
                        A = cfg.num_attention_heads()
                        head_size = cfg.hidden_size() // A
                        q_proj = 2 * tokens * cfg.hidden_size() * cfg.hidden_size()
                        k_proj = 2 * tokens * cfg.hidden_size() * cfg.hidden_size()
                        v_proj = 2 * tokens * cfg.hidden_size() * cfg.hidden_size()
                        o_proj = 2 * tokens * cfg.hidden_size() * cfg.hidden_size()
                        attn_prefill = 4 * B * A * (L * L) * head_size
                        attn_decode = 4 * B * A * L * head_size
                        attn = (
                            attn_prefill
                            if phase == "prefill"
                            else attn_decode
                            if phase == "decode"
                            else attn_prefill + attn_decode
                        )
                        attn_layer_flops = q_proj + k_proj + v_proj + o_proj + attn
                        lm.flops += attn_layer_flops
                        lm.attn_flops = attn_layer_flops
                        # KV cache per phase
                        kv_elems = 2 * cfg.hidden_size()
                        if phase == "prefill":
                            lm.kv_cache_w = (B * L * kv_elems * a_bits) // 8
                            lm.kv_cache_r = 0
                        elif phase == "decode":
                            lm.kv_cache_w = (B * 1 * kv_elems * a_bits) // 8
                            lm.kv_cache_r = (B * L * kv_elems * a_bits) // 8
                        else:
                            lm.kv_cache_w = (B * (L + 1) * kv_elems * a_bits) // 8
                            lm.kv_cache_r = (B * L * kv_elems * a_bits) // 8
                        # Attention weights only (Q,K,V,O). MLP weights are accounted in MLP block.
                        n = 4 * cfg.hidden_size() * cfg.hidden_size()

                        attn_path = f"model.layers.{decoder_idx}.{name}"
                        local_w_bits = fp_bits if is_excluded(attn_path) else w_bits
                        if local_w_bits < 16 and group_size is not None:
                            attn_bytes = __quantized_bytes(n, local_w_bits, group_size, scale_bytes, zero_bytes)
                        else:
                            attn_bytes = ceil((n * local_w_bits) / 8)

                        lm.weight_bytes += attn_bytes
                        lm.attn_bytes = attn_bytes

                        if debug >= 1:
                            print(
                                f"\tAttention Layer: FLOPs={q_proj + k_proj + v_proj + o_proj + attn}, "
                                f"b={attn_bytes} "
                                f"kv_cache_read={lm.kv_cache_r}, "
                                f"kv_cache_write={lm.kv_cache_w}"
                            )

            decoder_idx += 1
        layers.append(lm)
    return layers


# Estimate FLOPs for Model
def profile_model(
    cfg: MLX_ModelArgs,
    B: int = 1,
    L: int = 4096,
    debug=0,
    bs_list: List[int] = [],
    phase: ModelPhase = "merged",
):
    dtype = None
    bits = 0
    group_size = 0

    # Prefer explicit quantization section for bit-width
    quant_method = None
    if isinstance(config_dict.get("quantization"), dict):
        q = config_dict["quantization"]
        bits = int(q.get("bits", 0) or 0)
        group_size = int(q.get("group_size", 0) or 0)
    elif isinstance(config_dict.get("quantization_config"), dict):
        q = config_dict["quantization_config"]
        bits = int(q.get("bits", 0) or 0)
        group_size = int(q.get("group_size", 0) or 0)
        quant_method = q.get("quant_method")
    # Fallback to dtype when no explicit quantization
    if bits == 0:
        # Try config_dict first, then the config object, then default to f16
        dtype = (
            config_dict.get("torch_dtype")
            or config_dict.get("dtype")
            or getattr(config, "torch_dtype", None)
            or getattr(config, "dtype", None)
        )
        # Map quant_method if present (e.g., mxfp4)
        if not dtype and isinstance(config_dict.get("quantization_config"), dict):
            quant_method = config_dict["quantization_config"].get("quant_method") or quant_method
        if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
            bits = 4
            # Default a group size if none provided
            if group_size == 0:
                group_size = 128
        if dtype:
            if dtype in ("bfloat16", "bf16"):
                bits = 16
            elif dtype in ("float16", "fp16"):
                bits = 16
            elif dtype in ("float32", "f32"):
                # Default to fp16 when quantization is not explicit
                bits = 16
        if bits == 0:
            # Default to fp16 if nothing explicit is set
            bits = 16

    # Determine fp_bits for exclusions (non-quantized modules)
    has_quant_cfg = isinstance(config_dict.get("quantization"), dict) or isinstance(
        config_dict.get("quantization_config"), dict
    )
    fp_bits = 16
    if has_quant_cfg:
        d_dtype = (
            config_dict.get("torch_dtype")
            or config_dict.get("dtype")
            or getattr(config, "torch_dtype", None)
            or getattr(config, "dtype", None)
        )
        if d_dtype in ("float32", "f32"):
            fp_bits = 32
        elif d_dtype in ("bfloat16", "bf16", "float16", "fp16") or d_dtype is None:
            fp_bits = 16

    # Quantization exclusions
    exclude_patterns = []
    if isinstance(config_dict.get("quantization_config"), dict):
        exclude_patterns = config_dict["quantization_config"].get("modules_to_not_convert", []) or []

    model_info = in_profile_model(
        cfg,
        B,
        L,
        16,
        bits,
        group_size,
        debug,
        phase,
        exclude_patterns,
        fp_bits,
    )
    ret = ModelProfileInfo()

    # Per-layer metrics (base batch)
    ret.b = [int(x.weight_bytes) for x in model_info]
    ret.b_i = [int(x.input_bytes) for x in model_info]
    ret.b_o = [int(x.output_bytes) for x in model_info]
    ret.f_q[f"b_{B}"] = [float(x.flops) for x in model_info]
    ret.f_out[f"b_{B}"] = ret.f_q[f"b_{B}"][-1] if ret.f_q[f"b_{B}"] else 0.0
    ret.seq_len = int(L)

    # Model-level metrics from config
    ret.L = cfg.num_hidden_layers()
    ret.e_embed = cfg.hidden_size()
    ret.V = cfg.vocab_size()

    # Attention head configuration
    ret.hk = cfg.num_key_value_heads()
    ret.hv = cfg.num_key_value_heads()

    # Calculate head dimension
    head_dim = cfg.head_dim()
    ret.ek = head_dim
    ret.ev = head_dim

    # KV cache tokens (using max position embeddings as proxy)
    ret.n_kv = cfg.max_position_embeddings(L)

    # Add quantization label
    # If no explicit quantization, default label to F16
    q_label = ""
    if isinstance(cfg.get("quantization"), dict) or isinstance(cfg.get("quantization_config"), dict):
        qbits = None
        if isinstance(cfg.get("quantization"), dict):
            qbits = cfg["quantization"].get("bits")
        else:
            qbits = cfg["quantization_config"].get("bits")
            quant_method = cfg["quantization_config"].get("quant_method")
        try:
            qbits = int(qbits) if qbits is not None else None
        except Exception:
            qbits = None
        if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
            q_label = "MXFP4"
        mapping = {4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q8_0", 16: "F16", 32: "F32"}
        if qbits in mapping:
            q_label = mapping[qbits]
        if not q_label:
            d = cfg.get("torch_dtype") or cfg.get("dtype")
            if d in ("bfloat16", "bf16"):
                q_label = "BF16"
            elif d in ("float16", "fp16"):
                q_label = "F16"
            elif d in ("float32", "f32"):
                q_label = "F32"
    else:
        q_label = "F16"
    ret.quantization = q_label

    # Multi-batch-size profiles: only if provided via --batches
    for Bx in bs_list:
        tag = f"b_{Bx}"
        layers_bx = in_profile_model(
            cfg,
            Bx,
            L,
            16,
            bits,
            group_size,
            0,
            phase,
            exclude_patterns,
            fp_bits,
        )
        ret.f_q[tag] = [float(x.flops) for x in layers_bx]
        ret.f_out[tag] = ret.f_q[tag][-1] if ret.f_q[tag] else 0.0

    return ret


def profile_moe_model(
    cfg: MLX_ModelArgs,
    B: int = 1,
    L: int = 4096,
    debug=0,
    bs_list: List[int] = [],
    phase: ModelPhase = "merged",
):
    """
    Profile an MoE model with component-level metrics for solver assignment.
    Returns MoEModelProfileInfo if MoE is detected, otherwise returns ModelProfileInfo.
    """
    dtype = None
    bits = 0
    group_size = 0

    # Try different field names for number of experts
    n_routed_experts = cfg.n_routed_experts()
    if n_routed_experts == 0:
        # Not an MoE model, use regular profiling
        return profile_model(cfg, B, L, debug, bs_list, phase)

    # Parse quantization info
    quant_method = None

    if isinstance(config_dict.get("quantization"), dict):
        q = config_dict["quantization"]
        bits = int(q.get("bits", 0) or 0)
        group_size = int(q.get("group_size", 0) or 0)
    elif isinstance(config_dict.get("quantization_config"), dict):
        q = config_dict["quantization_config"]
        bits = int(q.get("bits", 0) or 0)
        group_size = int(q.get("group_size", 0) or 0)
        quant_method = q.get("quant_method")
    if bits == 0:
        # Fallback: try config object then default to f16 or quant_method
        dtype = (
            config_dict.get("torch_dtype")
            or config_dict.get("dtype")
            or getattr(cfg.module.args, "torch_dtype", None)
            or getattr(cfg.module.args, "dtype", None)
        )
        if not dtype and isinstance(config_dict.get("quantization_config"), dict):
            quant_method = config_dict["quantization_config"].get("quant_method") or quant_method
        if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
            bits = 4
            if group_size == 0:
                group_size = 128
        if dtype:
            if dtype in ("bfloat16", "bf16"):
                bits = 16
            elif dtype in ("float16", "fp16"):
                bits = 16
            elif dtype in ("float32", "f32"):
                # Default to fp16 when quantization is not explicit
                bits = 16
        if bits == 0:
            # Default to fp16 if nothing explicit is set
            bits = 16

    # Prepare per-module quantization exclusions
    has_quant_cfg = isinstance(config_dict.get("quantization"), dict) or isinstance(
        config_dict.get("quantization_config"), dict
    )
    fp_bits = 16
    if has_quant_cfg:
        d_dtype = (
            config_dict.get("torch_dtype")
            or config_dict.get("dtype")
            or getattr(cfg.module.args, "torch_dtype", None)
            or getattr(cfg.module.args, "dtype", None)
        )
        if d_dtype in ("float32", "f32"):
            fp_bits = 32
        elif d_dtype in ("bfloat16", "bf16", "float16", "fp16") or d_dtype is None:
            # Default to fp16 when dtype is missing
            fp_bits = 16
    exclude_patterns = []
    if isinstance(config_dict.get("quantization_config"), dict):
        exclude_patterns = config_dict["quantization_config"].get("modules_to_not_convert", []) or []

    # Profile the model to get layer-level metrics
    model_info = in_profile_model(
        cfg,
        B,
        L,
        16,
        bits,
        group_size,
        debug,
        phase,
        exclude_patterns,
        fp_bits,
    )

    # Create MoE profile
    ret = MoEModelProfileInfo()

    # Populate base metrics
    ret.b = [int(x.weight_bytes) for x in model_info]
    ret.b_i = [int(x.input_bytes) for x in model_info]
    ret.b_o = [int(x.output_bytes) for x in model_info]
    ret.f_q[f"b_{B}"] = [float(x.flops) for x in model_info]
    ret.f_out[f"b_{B}"] = ret.f_q[f"b_{B}"][-1] if ret.f_q[f"b_{B}"] else 0.0
    ret.seq_len = int(L)

    # Model-level metrics
    ret.L = cfg.num_hidden_layers()
    ret.e_embed = cfg.hidden_size()
    ret.V = cfg.vocab_size()

    # Attention head configuration
    num_attention_heads = cfg.num_attention_heads()
    ret.hk = cfg.num_key_value_heads()
    ret.hv = cfg.num_key_value_heads()

    # Head dimension
    head_dim = cfg.head_dim()
    if head_dim == 0 and ret.e_embed > 0 and num_attention_heads > 0:
        head_dim = ret.e_embed // num_attention_heads
    ret.ek = head_dim
    ret.ev = head_dim
    ret.n_kv = cfg.max_position_embeddings(L)

    # MoE configuration - handle various naming conventions
    ret.n_routed_experts = n_routed_experts

    # Shared experts (Qwen3 uses shared_expert_intermediate_size to indicate shared experts)
    shared_expert_size = cfg.shared_intermediate()
    ret.n_shared_experts = 1 if shared_expert_size > 0 else 0  # Infer from shared_expert_intermediate_size

    # Experts per token (top-k selection)
    ret.experts_per_token = cfg.num_experts_tok()

    # MoE FFN hidden size - use intermediate_size for MoE models if no explicit MoE size
    ret.moe_intermediate_size = cfg.moe_intermediate()

    if ret.moe_intermediate_size == 0:
        raise ValueError(
            "MoE model detected but no valid intermediate/FFN size found. "
            "Config must have one of: moe_intermediate_size, expert_intermediate_size, "
            "intermediate_size, or ffn_dim"
        )

    # MoE layer frequency (which layers have MoE)
    ret.moe_layer_freq = cfg.moe_layer_freq()

    # First K dense layers (before MoE starts)
    # TODO: no `first_k_dense_replace` OR `num_dense_layers` in current models
    ret.first_k_dense_replace = 0

    # Determine MoE layer indices from parsed layers for accuracy
    moe_indices = [i for i, layer in enumerate(model_info[1:], 1) if getattr(layer, "is_moe_layer", False)]
    # Fallback to heuristic if parser didn't tag MoE layers
    if not moe_indices and ret.L:
        for layer_idx in range(1, ret.L + 1):
            if (
                layer_idx > ret.first_k_dense_replace
                and (ret.moe_layer_freq or 1) > 0
                and layer_idx % max(ret.moe_layer_freq, 1) == 0
            ):
                moe_indices.append(layer_idx)
    ret.moe_layer_indices = moe_indices
    ret.total_moe_layers = len(moe_indices)

    # Extract component metrics from layer
    ret.attn_bytes = []
    ret.attn_flops[f"b_{B}"] = []

    for idx, layer in enumerate(model_info[1:], 1):  # Skip prefill layer
        # Attention metrics (all layers have attention)
        ret.attn_bytes.append(layer.attn_bytes)
        ret.attn_flops[f"b_{B}"].append(layer.attn_flops)

        # MoE metrics (only for MoE layers)
        if (getattr(layer, "is_moe_layer", False)) or (idx in moe_indices):
            ret.bytes_per_expert[idx] = layer.moe_expert_bytes
            ret.bytes_shared_experts[idx] = layer.moe_shared_bytes
            ret.flops_per_expert[idx] = layer.moe_expert_flops
            ret.flops_shared_experts[idx] = layer.moe_shared_flops
            ret.router_flops[idx] = layer.moe_router_flops
            ret.router_bytes[idx] = layer.moe_router_bytes
            if hasattr(layer, "moe_expert_flops_per_token"):
                ret.flops_per_active_expert_per_token[idx] = layer.moe_expert_flops_per_token

    # Quantization label
    if isinstance(cfg.get("quantization"), dict) or isinstance(cfg.get("quantization_config"), dict):
        q_label = ""
        qbits = None
        if isinstance(cfg.get("quantization"), dict):
            qbits = cfg["quantization"].get("bits")
        else:
            qbits = cfg["quantization_config"].get("bits")
            quant_method = cfg["quantization_config"].get("quant_method")
        try:
            qbits = int(qbits) if qbits is not None else None
        except Exception:
            qbits = None
        if quant_method in ("mxfp4", "MXFP4", "mx_fp4"):
            q_label = "MXFP4"
        mapping = {4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q8_0", 16: "F16", 32: "F32"}
        if qbits in mapping:
            q_label = mapping[qbits]
        if not q_label:
            d = cfg.get("torch_dtype") or cfg.get("dtype")
            if d in ("bfloat16", "bf16"):
                q_label = "BF16"
            elif d in ("float16", "fp16"):
                q_label = "F16"
            elif d in ("float32", "f32"):
                q_label = "F32"
    else:
        q_label = "F16"
    ret.quantization = q_label

    # Multi-batch profiles
    for Bx in bs_list:
        tag = f"b_{Bx}"
        layers_bx = in_profile_model(
            cfg,
            Bx,
            L,
            16,
            bits,
            group_size,
            0,
            phase,
            exclude_patterns,
            fp_bits,
        )
        ret.f_q[tag] = [float(x.flops) for x in layers_bx]
        ret.f_out[tag] = ret.f_q[tag][-1] if ret.f_q[tag] else 0.0
        ret.attn_flops[tag] = [float(x.attn_flops) for x in layers_bx[1:]]  # Skip prefill

    return ret


def profile_model_phased(
    cfg: MLX_ModelArgs,
    B: int,
    L: int,
    debug=0,
    bs_list: List[int] = [],
):
    # use `profile_moe_model` which auto-detects MoE models
    prefill = profile_moe_model(
        cfg,
        B=B,
        L=L,
        debug=debug,
        bs_list=bs_list,
        phase="prefill",
    )
    decode = profile_moe_model(
        cfg,
        B=B,
        L=L,
        debug=debug,
        bs_list=bs_list,
        phase="decode",
    )
    return ModelProfilePhased(prefill=prefill, decode=decode)


def profile_model_split(
    cfg: MLX_ModelArgs,
    B: int,
    L: int,
    debug=0,
    bs_list: List[int] = [],
):
    phased = profile_model_phased(
        cfg,
        B=B,
        L=L,
        debug=debug,
        bs_list=bs_list,
    )
    pre, dec = phased.prefill, phased.decode

    # Create base split result
    result = ModelProfileSplit(
        b=pre.b,
        b_i=pre.b_i,
        b_o=pre.b_o,
        L=pre.L,
        hk=pre.hk,
        hv=pre.hv,
        ek=pre.ek,
        ev=pre.ev,
        n_kv=pre.n_kv,
        e_embed=pre.e_embed,
        V=pre.V,
        seq_len=pre.seq_len,
        f_q={
            "prefill": pre.f_q,
            "decode": dec.f_q,
        },
        f_out={
            "prefill": pre.f_out,
            "decode": dec.f_out,
        },
        quantization=pre.quantization,
    )

    # if this is an MoE model, populate MoE fields with the prefill
    if isinstance(pre, MoEModelProfileInfo):
        result.is_moe = True
        result.n_routed_experts = pre.n_routed_experts
        result.n_shared_experts = pre.n_shared_experts
        result.experts_per_token = pre.experts_per_token
        result.moe_intermediate_size = pre.moe_intermediate_size
        result.moe_layer_freq = pre.moe_layer_freq
        result.first_k_dense_replace = pre.first_k_dense_replace
        result.total_moe_layers = pre.total_moe_layers
        result.moe_layer_indices = pre.moe_layer_indices
        result.attn_bytes = pre.attn_bytes
        result.attn_flops = {
            "prefill": pre.attn_flops,
            "decode": dec.attn_flops if isinstance(dec, MoEModelProfileInfo) else {},
        }
        result.bytes_per_expert = pre.bytes_per_expert
        result.bytes_shared_experts = pre.bytes_shared_experts
        result.flops_per_expert = pre.flops_per_expert
        result.flops_shared_experts = pre.flops_shared_experts
        result.router_flops = pre.router_flops
        result.router_bytes = pre.router_bytes
        result.flops_per_active_expert_per_token = pre.flops_per_active_expert_per_token

    return result
