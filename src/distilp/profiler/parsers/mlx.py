import mlx.nn as nn

from math import ceil
from fnmatch import fnmatch
from typing import Any

from .meta import LayerMeta

""" Estimate the FLOP count of all 'mlx_lm' models at a decoder level
    NOTE: Small OPs like RoPE and norms default to 0 FLOPs
    NOTE: FMA defaults to 2 FLOPs """

block_names = ["TransformerBlock", "DecoderLayer"]


# Add the quantization metadata to final byte count
def __quantized_bytes(n, d_bits, group_size, scale_bytes, zero_bytes):
    scaled_bits = n * d_bits
    code_bytes = ceil(scaled_bits / 8)
    if group_size and group_size > 0:
        groups = (n + group_size - 1) // group_size
        meta_bytes = groups * (scale_bytes + zero_bytes)
    else:
        meta_bytes = 0
    return code_bytes + meta_bytes


def in_profile_model(
    m: nn.Module,
    config: Any,
    B: int = 1,
    L: int = 4096,
    a_bits=16,
    w_bits=16,
    group_size=32,
    debug=0,
    phase: str = "merged",  # 'prefill' | 'decode' | 'merged'
    cfg_dict: dict = None,
    exclude_patterns=None,
    fp_bits: int = 16,
):
    if not hasattr(m, "layers"):
        raise RuntimeError("Unable to profile a model without a '.layers' attribute.")

    decoder_idx = 1
    layers = []

    # Quantization hard-coded scale and zero bytes
    scale_bytes = 2
    zero_bytes = 0

    if exclude_patterns is None:
        exclude_patterns = []

    # Config getter that checks dict first then object
    cfg = cfg_dict or {}

    def cfg_get(key, default=None):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        return getattr(config, key, default)

    def is_excluded(path: str) -> bool:
        for pat in exclude_patterns:
            try:
                if fnmatch(path, pat):
                    return True
            except Exception:
                pass
        return False

    # Append a symbolic prefill layer to account for these FLOPs
    prefill = LayerMeta()
    prefill.name = "prefill"
    prefill.layer = None
    prefill.flops = 0
    prefill.kv_cache_r = 0
    prefill.kv_cache_w = 0
    layers.append(prefill)

    if debug >= 1:
        print("FMA: 2 FLOPs")
        # print(f"Quantization: {config.quantization.bits}")
        print(f"Parsing model {config.model_type}:")
        print(f"Quantization: bits={w_bits}, group_size={group_size}")
        print(
            f"    hidden_size={config.hidden_size},\n    vocab_size={config.vocab_size},\n"
            f"    num_hidden_layers={config.num_hidden_layers}"
        )

    for l in m.layers:
        lm = LayerMeta()
        lm.layer = l
        lm.name = f"decoder_{decoder_idx}"
        if any(x in l.__class__.__name__ for x in ["TransformerBlock", "DecoderLayer"]):
            lm.input_bytes = (B * L * config.hidden_size * a_bits) / 8
            lm.output_bytes = (B * L * config.hidden_size * a_bits) / 8
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
            for name, obj in l.named_modules():
                if name == "post_attention_layernorm" or name == "input_layernorm":
                    pass

                elif name in ("mlp", "ffn", "feed_forward", "feedforward", "ffn_layer"):
                    # MoE - check for various naming conventions (use cfg_get)
                    mlp_path = f"model.layers.{decoder_idx}.mlp"
                    n_experts = cfg_get(
                        "n_routed_experts",
                        cfg_get(
                            "num_experts",
                            cfg_get("num_local_experts", cfg_get("n_experts", None)),
                        ),
                    )

                    first_dense = cfg_get(
                        "first_k_dense_replace", cfg_get("num_dense_layers", 0)
                    )

                    layer_freq = cfg_get(
                        "moe_layer_freq",
                        cfg_get("decoder_sparse_step", cfg_get("expert_interval", 1)),
                    )

                    mlp_only_layers = set(cfg_get("mlp_only_layers", []))

                    if (
                        n_experts is not None
                        and n_experts > 0
                        and decoder_idx > first_dense
                        and decoder_idx % layer_freq == 0
                        and decoder_idx not in mlp_only_layers
                    ):
                        lm.is_moe_layer = True
                        has_router_gate = False
                        found_switch_block = False
                        # Accumulators
                        gate_f, smlp_f, smlp_b, se_f, se_b = 0, 0, 0, 0, 0
                        # Initialize variables that may be used in debug statements
                        moe_intermediate = None
                        shared_intermediate = None
                        num_experts_tok = None
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
                                gate_f += 2 * tokens * config.hidden_size * n_experts
                                lm.flops += gate_f
                                lm.moe_router_flops = gate_f
                                # Router weight bytes (apply per-module exclusion)
                                router_params = config.hidden_size * n_experts
                                router_path = f"model.layers.{decoder_idx}.mlp.router"
                                local_w_bits = (
                                    fp_bits if is_excluded(router_path) else w_bits
                                )
                                if local_w_bits < 16 and group_size is not None:
                                    lm.moe_router_bytes = __quantized_bytes(
                                        router_params,
                                        local_w_bits,
                                        group_size,
                                        scale_bytes,
                                        zero_bytes,
                                    )
                                else:
                                    lm.moe_router_bytes = ceil(
                                        (router_params * local_w_bits) / 8
                                    )
                                has_router_gate = True

                            elif key == "switch_mlp":
                                moe_intermediate = cfg_get(
                                    "moe_intermediate_size",
                                    cfg_get(
                                        "expert_intermediate_size",
                                        cfg_get("intermediate_size", None),
                                    ),
                                )
                                if moe_intermediate is None or moe_intermediate == 0:
                                    raise ValueError(
                                        "MoE layer detected but no valid intermediate size found in config"
                                    )
                                DS = config.hidden_size * moe_intermediate
                                num_experts_tok = cfg_get(
                                    "num_experts_per_tok",
                                    cfg_get(
                                        "num_experts_per_token",
                                        cfg_get("experts_per_token", None),
                                    ),
                                )
                                if num_experts_tok is None:
                                    raise ValueError(
                                        "MoE layer detected but num_experts_per_tok/experts_per_token not found in config"
                                    )
                                num_proj_smlp = 2
                                for key2, proj in leaf.named_modules():
                                    if key2 == "gate_proj":
                                        smlp_f += 2 * tokens * num_experts_tok * DS
                                        num_proj_smlp = 3
                                    elif key2 in ["up_proj", "down_proj"]:
                                        smlp_f += 2 * tokens * num_experts_tok * DS
                                    elif key2 == "activations":
                                        # Activation FLOPs are small; include linear in L for completeness
                                        smlp_f += (
                                            tokens * num_experts_tok * moe_intermediate
                                        )

                                # Per-active-expert-per-token FLOPs (one token through one expert MLP)
                                # 2 * num_proj * H * D + activation_cost (approx D)
                                lm.moe_expert_flops_per_token = (
                                    2
                                    * num_proj_smlp
                                    * config.hidden_size
                                    * moe_intermediate
                                    + moe_intermediate
                                )
                                found_switch_block = True

                                # Add the quantization group overhead per projection matrix
                                local_w_bits = w_bits
                                # Optional: if entire mlp is excluded, use fp_bits
                                mlp_path = f"model.layers.{decoder_idx}.mlp"
                                if is_excluded(mlp_path):
                                    local_w_bits = fp_bits
                                if local_w_bits < 16 and group_size is not None:
                                    per_proj_params = (
                                        config.hidden_size * moe_intermediate
                                    )
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
                                            * config.hidden_size
                                            * moe_intermediate
                                            * local_w_bits
                                        )
                                        / 8
                                    )
                                lm.weight_bytes += smlp_b
                                lm.flops += smlp_f
                                # Per-expert metrics
                                lm.moe_expert_flops = (
                                    smlp_f / n_experts if n_experts > 0 else 0
                                )
                                lm.moe_expert_bytes = (
                                    smlp_b // n_experts if n_experts > 0 else 0
                                )

                            elif key == "shared_experts":
                                n_shared = cfg_get(
                                    "n_shared_experts", cfg_get("num_shared_experts", 0)
                                )
                                shared_intermediate = cfg_get(
                                    "shared_expert_intermediate_size",
                                    cfg_get(
                                        "moe_intermediate_size",
                                        cfg_get("intermediate_size", None),
                                    ),
                                )
                                num_proj_se = 2
                                for key2, proj in leaf.named_modules():
                                    if key2 == "gate_proj":
                                        num_proj_se = 3
                                    if key2 in ["gate_proj", "up_proj", "down_proj"]:
                                        se_f += (
                                            2
                                            * tokens
                                            * config.hidden_size
                                            * n_shared
                                            * shared_intermediate
                                        )

                                local_w_bits = w_bits
                                if is_excluded(mlp_path):
                                    local_w_bits = fp_bits
                                if local_w_bits < 16 and group_size is not None:
                                    per_proj_params = (
                                        config.hidden_size * shared_intermediate
                                    )
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
                                        n_shared
                                        * num_proj_se
                                        * config.hidden_size
                                        * shared_intermediate
                                        * local_w_bits
                                    ) / 8
                                lm.weight_bytes += se_b
                                lm.flops += se_f
                                # Shared experts metrics
                                lm.moe_shared_flops = se_f
                                lm.moe_shared_bytes = se_b
                            # Fallback pattern detection for routed experts
                            if (
                                "experts" in key_l
                                or "local_experts" in key_l
                                or "routed_experts" in key_l
                            ) and not key_l.startswith("shared"):
                                if key_l.endswith("up_proj") or ".up_proj" in key_l:
                                    found_expert_up = True
                                if key_l.endswith("down_proj") or ".down_proj" in key_l:
                                    found_expert_down = True
                                if key_l.endswith("gate_proj") or ".gate_proj" in key_l:
                                    found_expert_gatep = True
                            # Fallback pattern detection for shared experts
                            if "shared" in key_l and (
                                "experts" in key_l or "expert" in key_l
                            ):
                                if key_l.endswith("up_proj") or ".up_proj" in key_l:
                                    found_shared_up = True
                                if key_l.endswith("down_proj") or ".down_proj" in key_l:
                                    found_shared_down = True
                                if key_l.endswith("gate_proj") or ".gate_proj" in key_l:
                                    found_shared_gatep = True

                        # End of traversal: if no switch_mlp block but expert projections detected, compute generically
                        if not found_switch_block and (
                            found_expert_up or found_expert_down or found_expert_gatep
                        ):
                            moe_intermediate = cfg_get(
                                "moe_intermediate_size",
                                cfg_get(
                                    "expert_intermediate_size",
                                    cfg_get("intermediate_size", None),
                                ),
                            )
                            if moe_intermediate is None or moe_intermediate == 0:
                                raise ValueError(
                                    "MoE layer detected but no valid intermediate size found in config"
                                )
                            DS = config.hidden_size * moe_intermediate
                            num_experts_tok = cfg_get(
                                "num_experts_per_tok",
                                cfg_get(
                                    "num_experts_per_token",
                                    cfg_get("experts_per_token", None),
                                ),
                            )
                            if num_experts_tok is None:
                                raise ValueError(
                                    "MoE layer detected but num_experts_per_tok/experts_per_token not found in config"
                                )
                            num_proj_smlp = (
                                int(found_expert_up)
                                + int(found_expert_down)
                                + int(found_expert_gatep)
                            )
                            # FLOPs for active experts
                            smlp_f = num_proj_smlp * (2 * tokens * num_experts_tok * DS)
                            # Activation cost (approximate)
                            smlp_f += tokens * num_experts_tok * moe_intermediate
                            # Bytes per projection matrix (quantized)
                            local_w_bits = w_bits
                            if is_excluded(mlp_path):
                                local_w_bits = fp_bits
                            if local_w_bits < 16 and group_size is not None:
                                per_proj_params = config.hidden_size * moe_intermediate
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
                                        * config.hidden_size
                                        * moe_intermediate
                                        * local_w_bits
                                    )
                                    / 8
                                )
                            lm.weight_bytes += smlp_b
                            lm.flops += smlp_f
                            lm.moe_expert_flops = (
                                smlp_f / n_experts if n_experts > 0 else 0
                            )
                            lm.moe_expert_bytes = (
                                smlp_b // n_experts if n_experts > 0 else 0
                            )
                            lm.moe_expert_flops_per_token = (
                                2
                                * num_proj_smlp
                                * config.hidden_size
                                * moe_intermediate
                                + moe_intermediate
                            )

                        # Shared experts fallback: if not computed but detected
                        if se_b == 0 and (
                            found_shared_up or found_shared_down or found_shared_gatep
                        ):
                            n_shared = cfg_get(
                                "n_shared_experts", cfg_get("num_shared_experts", 0)
                            )
                            shared_intermediate = cfg_get(
                                "shared_expert_intermediate_size",
                                cfg_get(
                                    "moe_intermediate_size",
                                    cfg_get("intermediate_size", None),
                                ),
                            )
                            if shared_intermediate:
                                num_proj_se = (
                                    int(found_shared_up)
                                    + int(found_shared_down)
                                    + int(found_shared_gatep)
                                )
                                local_w_bits = w_bits
                                if is_excluded(mlp_path):
                                    local_w_bits = fp_bits
                                if local_w_bits < 16 and group_size is not None:
                                    per_proj_params = (
                                        config.hidden_size * shared_intermediate
                                    )
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
                                        n_shared
                                        * num_proj_se
                                        * config.hidden_size
                                        * shared_intermediate
                                        * local_w_bits
                                    ) / 8
                                lm.weight_bytes += se_b
                                se_f = (
                                    2
                                    * tokens
                                    * config.hidden_size
                                    * n_shared
                                    * shared_intermediate
                                    * num_proj_se
                                )
                                lm.flops += se_f
                                lm.moe_shared_flops = se_f
                                lm.moe_shared_bytes = se_b

                        # Final generic fallback for routed experts if still zero
                        if smlp_b == 0 and (n_experts is not None and n_experts > 0):
                            moe_intermediate = cfg_get(
                                "moe_intermediate_size",
                                cfg_get(
                                    "expert_intermediate_size",
                                    cfg_get("intermediate_size", None),
                                ),
                            )
                            num_experts_tok = cfg_get(
                                "num_experts_per_tok",
                                cfg_get(
                                    "num_experts_per_token",
                                    cfg_get("experts_per_token", None),
                                ),
                            )
                            if moe_intermediate and num_experts_tok:
                                num_proj_smlp = 3
                                DS = config.hidden_size * moe_intermediate
                                smlp_f = num_proj_smlp * (
                                    2 * tokens * num_experts_tok * DS
                                ) + (tokens * num_experts_tok * moe_intermediate)
                                local_w_bits = (
                                    fp_bits if is_excluded(mlp_path) else w_bits
                                )
                                if local_w_bits < 16 and group_size is not None:
                                    per_proj_params = (
                                        config.hidden_size * moe_intermediate
                                    )
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
                                            * config.hidden_size
                                            * moe_intermediate
                                            * local_w_bits
                                        )
                                        / 8
                                    )
                                lm.weight_bytes += smlp_b
                                lm.flops += smlp_f
                                lm.moe_expert_flops = (
                                    smlp_f / n_experts if n_experts > 0 else 0
                                )
                                lm.moe_expert_bytes = (
                                    smlp_b // n_experts if n_experts > 0 else 0
                                )
                                lm.moe_expert_flops_per_token = (
                                    2
                                    * num_proj_smlp
                                    * config.hidden_size
                                    * moe_intermediate
                                    + moe_intermediate
                                )

                        # If router not found as submodule, estimate generically (common for MoE blocks)
                        if not has_router_gate and (
                            n_experts is not None and n_experts > 0
                        ):
                            gate_f = 2 * tokens * config.hidden_size * n_experts
                            lm.flops += gate_f
                            lm.moe_router_flops = gate_f
                            router_params = config.hidden_size * n_experts
                            router_path = f"model.layers.{decoder_idx}.mlp.router"
                            local_w_bits = (
                                fp_bits if is_excluded(router_path) else w_bits
                            )
                            if local_w_bits < 16 and group_size is not None:
                                lm.moe_router_bytes = __quantized_bytes(
                                    router_params,
                                    local_w_bits,
                                    group_size,
                                    scale_bytes,
                                    zero_bytes,
                                )
                            else:
                                lm.moe_router_bytes = ceil(
                                    (router_params * local_w_bits) / 8
                                )

                        if debug >= 1:
                            print(
                                f"\tMoE Layer: FLOPs={smlp_f + se_f + gate_f} ({num_proj_smlp}x{num_experts_tok}x"
                                f"[{config.hidden_size}, {moe_intermediate}] + {num_proj_se}x"
                                f"{n_shared}x[{config.hidden_size}, {shared_intermediate}] + "
                                f"{B}x[{config.hidden_size}, {n_experts}]), b={smlp_b + se_b} @ {w_bits}bits"
                                if has_router_gate
                                else f"), b={smlp_b + se_b} @ {w_bits}bits,",
                                end="",
                            )
                            print(
                                f" routed_experts={n_experts} "
                                f"with top-k={num_experts_tok}, ",
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
                                lm.flops += (
                                    2
                                    * tokens
                                    * config.hidden_size
                                    * config.intermediate_size
                                )
                                n = config.hidden_size * config.intermediate_size
                                if w_bits < 16 and group_size is not None:
                                    proj_bytes += __quantized_bytes(
                                        n, w_bits, group_size, scale_bytes, zero_bytes
                                    )
                                else:
                                    proj_bytes += ceil((n * w_bits) / 8)
                        lm.weight_bytes += proj_bytes

                        if debug >= 1:
                            print(
                                f"\tMLP Layer: FLOPs={num_proj * 2 * tokens * config.hidden_size * config.intermediate_size},"
                                f"  b={proj_bytes}"
                                f"( {num_proj} x [{config.hidden_size}, {config.intermediate_size}] @ {w_bits}),"
                                f"  b_i={B * L * config.hidden_size}([{B}, {L}, {config.hidden_size}])"
                            )

                # NOTE: We only compute projection bits then correct in the case of quantization
                elif name in ("self_attn", "attn", "self_attention"):
                    # attn_layer_flops = 0
                    # attn_layer_bytes = 0

                    is_gqa = False
                    is_mla = False

                    # Grouped Query Attention
                    if (
                        hasattr(config, "num_key_value_heads")
                        and config.num_key_value_heads != config.num_attention_heads
                    ):
                        is_gqa = True

                    # Low rank / Multi-head Latent Attention
                    if all(
                        hasattr(config, k)
                        for k in ["q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"]
                    ) and all(
                        getattr(config, k) is not None
                        for k in ["q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"]
                    ):
                        is_mla = True

                    if is_mla:
                        # Deepseek_v2,v3, Kimi_v1 and minicpm
                        if any(
                            hasattr(config, k) for k in ["kv_lora_rank", "v_head_dim"]
                        ):
                            # Q projections, flops and bytes
                            q_head_dim = (
                                config.qk_nope_head_dim + config.qk_rope_head_dim
                            )
                            q_a_proj = (
                                2 * tokens * config.hidden_size * config.q_lora_rank
                            )
                            q_b_proj = (
                                2
                                * tokens
                                * config.num_attention_heads
                                * q_head_dim
                                * config.q_lora_rank
                            )
                            q_a_proj_n = config.q_lora_rank * config.hidden_size
                            q_b_proj_n = (
                                config.num_attention_heads
                                * q_head_dim
                                * config.q_lora_rank
                            )

                            # KV projections
                            if is_gqa:
                                pass  # TODO
                            else:
                                out_features = (
                                    config.kv_lora_rank + config.qk_rope_head_dim
                                )
                                kv_a_proj_with_mqa = (
                                    2
                                    * tokens
                                    * (config.kv_lora_rank + config.qk_rope_head_dim)
                                    * config.hidden_size
                                )
                                kv_b_proj = (
                                    2
                                    * tokens
                                    * config.kv_lora_rank
                                    * config.num_attention_heads
                                    * (config.qk_nope_head_dim + config.v_head_dim)
                                )
                                kv_a_proj_n = out_features * config.hidden_size
                                kv_b_proj_n = out_features * config.kv_lora_rank

                            # O projection
                            o_proj = (
                                2
                                * tokens
                                * config.num_attention_heads
                                * config.v_head_dim
                                * config.hidden_size
                            )
                            o_proj_n = (
                                config.hidden_size
                                * config.num_attention_heads
                                * config.v_head_dim
                            )

                            # KV Cache I/O
                            kv_elems = config.kv_lora_rank + config.qk_rope_head_dim
                            if phase == "prefill":
                                lm.kv_cache_w = (B * L * kv_elems * a_bits) / 8
                                lm.kv_cache_r = 0
                            elif phase == "decode":
                                lm.kv_cache_w = (B * 1 * kv_elems * a_bits) / 8
                                lm.kv_cache_r = (B * L * kv_elems * a_bits) / 8
                            else:
                                lm.kv_cache_w = (B * (L + 1) * kv_elems * a_bits) / 8
                                lm.kv_cache_r = (B * L * kv_elems * a_bits) / 8

                            # Totals
                            # Attention compute FLOPs per phase
                            attn_prefill = (
                                4
                                * B
                                * (config.num_attention_heads)
                                * (L * L)
                                * q_head_dim
                            )
                            attn_decode = (
                                4 * B * (config.num_attention_heads) * L * q_head_dim
                            )
                            attn = (
                                attn_prefill
                                if phase == "prefill"
                                else attn_decode
                                if phase == "decode"
                                else attn_prefill + attn_decode
                            )
                            attn_layer_flops = (
                                q_a_proj
                                + q_b_proj
                                + kv_a_proj_with_mqa
                                + kv_b_proj
                                + o_proj
                                + attn
                            )
                            lm.flops += attn_layer_flops
                            lm.attn_flops = attn_layer_flops

                            attn_path = f"model.layers.{decoder_idx}.{name}"
                            local_w_bits = fp_bits if is_excluded(attn_path) else w_bits
                            if local_w_bits < 16 and group_size is not None:
                                q_a_proj_bytes = __quantized_bytes(
                                    q_a_proj_n,
                                    local_w_bits,
                                    group_size,
                                    scale_bytes,
                                    zero_bytes,
                                )
                                q_b_proj_bytes = __quantized_bytes(
                                    q_b_proj_n,
                                    local_w_bits,
                                    group_size,
                                    scale_bytes,
                                    zero_bytes,
                                )
                                kv_a_proj_bytes = __quantized_bytes(
                                    kv_a_proj_n,
                                    local_w_bits,
                                    group_size,
                                    scale_bytes,
                                    zero_bytes,
                                )
                                kv_b_proj_bytes = __quantized_bytes(
                                    kv_b_proj_n,
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
                                attn_bytes = (
                                    q_a_proj_bytes
                                    + q_b_proj_bytes
                                    + kv_a_proj_bytes
                                    + kv_b_proj_bytes
                                    + o_proj_bytes
                                )
                            else:
                                q_a_proj_bytes = ceil((q_a_proj_n * local_w_bits) / 8)
                                q_b_proj_bytes = ceil((q_b_proj_n * local_w_bits) / 8)
                                kv_a_proj_bytes = ceil((kv_a_proj_n * local_w_bits) / 8)
                                kv_b_proj_bytes = ceil((kv_b_proj_n * local_w_bits) / 8)
                                o_proj_bytes = ceil((o_proj_n * local_w_bits) / 8)
                                attn_bytes = (
                                    q_a_proj_bytes
                                    + q_b_proj_bytes
                                    + kv_a_proj_bytes
                                    + kv_b_proj_bytes
                                    + o_proj_bytes
                                )

                            # attn_layer_bytes = attn_bytes
                            lm.weight_bytes += attn_bytes
                            lm.attn_bytes = attn_bytes

                            if debug >= 1:
                                print(
                                    f"\tMulti-head Latent Attention Layer {'with Group Query Attention' if is_gqa else ''}:"
                                )
                                print(
                                    f"\t\tq_a_proj: [{config.hidden_size}, {config.q_lora_rank} ], FLOPs={q_a_proj}, b={q_a_proj_bytes}"
                                )
                                print(
                                    f"\t\tq_b_proj: [{config.num_attention_heads * q_head_dim}, {config.q_lora_rank}], "
                                    f"FLOPs={q_b_proj}, b={q_b_proj_bytes}"
                                )
                                print(
                                    f"\t\tkv_a_proj_with_mqa: [{config.hidden_size}, {config.kv_lora_rank + config.qk_rope_head_dim}], "
                                    f"FLOPs={kv_a_proj_with_mqa}, b={kv_a_proj_bytes}"
                                )
                                print(
                                    f"\t\tkv_b_proj: [{config.kv_lora_rank}, "
                                    f"{config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)}], "
                                    f"FLOPs={kv_b_proj}, b={kv_b_proj_bytes}"
                                )
                                print(
                                    f"\t\to_proj: [{config.num_attention_heads * config.v_head_dim}, {config.hidden_size}], "
                                    f"FLOPs={o_proj}, b={o_proj_bytes}"
                                )
                                print(
                                    f"\t\tFLOPs={q_a_proj + q_b_proj + kv_a_proj_with_mqa + kv_b_proj + o_proj + attn} "
                                    f"b={attn_bytes}"
                                )

                            continue

                        # Low-rank Replace (not implemented)
                        else:
                            pass

                    # Grouped Query Attention
                    elif is_gqa:
                        head_size = config.hidden_size // config.num_attention_heads
                        q_proj = 2 * tokens * config.hidden_size * config.hidden_size
                        q_proj_n = config.hidden_size * config.hidden_size

                        # K/V out dim = h_k * d = num_key_value_heads * head_size
                        k_out = config.num_key_value_heads * head_size
                        v_out = config.num_key_value_heads * head_size
                        k_proj = 2 * tokens * config.hidden_size * k_out
                        k_proj_n = config.hidden_size * k_out
                        v_proj = 2 * tokens * config.hidden_size * v_out
                        v_proj_n = config.hidden_size * v_out

                        o_proj = 2 * tokens * config.hidden_size * config.hidden_size
                        o_proj_n = config.hidden_size * config.hidden_size
                        # Attention FLOPs by phase
                        A = config.num_attention_heads
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
                        kv_elems = 2 * config.num_key_value_heads * head_size
                        if phase == "prefill":
                            lm.kv_cache_w = (B * L * kv_elems * a_bits) / 8
                            lm.kv_cache_r = 0
                        elif phase == "decode":
                            lm.kv_cache_w = (B * 1 * kv_elems * a_bits) / 8
                            lm.kv_cache_r = (B * L * kv_elems * a_bits) / 8
                        else:
                            lm.kv_cache_w = (B * (L + 1) * kv_elems * a_bits) / 8
                            lm.kv_cache_r = (B * L * kv_elems * a_bits) / 8

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
                            attn_bytes = (
                                q_proj_bytes
                                + k_proj_bytes
                                + v_proj_bytes
                                + o_proj_bytes
                            )
                        else:
                            q_proj_bytes = ceil((q_proj_n * local_w_bits) / 8)
                            k_proj_bytes = ceil((k_proj_n * local_w_bits) / 8)
                            v_proj_bytes = ceil((v_proj_n * local_w_bits) / 8)
                            o_proj_bytes = ceil((o_proj_n * local_w_bits) / 8)
                            attn_bytes = (
                                q_proj_bytes
                                + k_proj_bytes
                                + v_proj_bytes
                                + o_proj_bytes
                            )

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
                        A = config.num_attention_heads
                        head_size = config.hidden_size // A
                        q_proj = 2 * tokens * config.hidden_size * config.hidden_size
                        k_proj = 2 * tokens * config.hidden_size * config.hidden_size
                        v_proj = 2 * tokens * config.hidden_size * config.hidden_size
                        o_proj = 2 * tokens * config.hidden_size * config.hidden_size
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
                        kv_elems = 2 * config.hidden_size
                        if phase == "prefill":
                            lm.kv_cache_w = (B * L * kv_elems * a_bits) / 8
                            lm.kv_cache_r = 0
                        elif phase == "decode":
                            lm.kv_cache_w = (B * 1 * kv_elems * a_bits) / 8
                            lm.kv_cache_r = (B * L * kv_elems * a_bits) / 8
                        else:
                            lm.kv_cache_w = (B * (L + 1) * kv_elems * a_bits) / 8
                            lm.kv_cache_r = (B * L * kv_elems * a_bits) / 8
                        # Attention weights only (Q,K,V,O). MLP weights are accounted in MLP block.
                        n = 4 * config.hidden_size * config.hidden_size

                        attn_path = f"model.layers.{decoder_idx}.{name}"
                        local_w_bits = fp_bits if is_excluded(attn_path) else w_bits
                        if local_w_bits < 16 and group_size is not None:
                            attn_bytes = __quantized_bytes(
                                n, local_w_bits, group_size, scale_bytes, zero_bytes
                            )
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
