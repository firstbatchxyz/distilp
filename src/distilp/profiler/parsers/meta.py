from dataclasses import dataclass
from typing import Any


@dataclass
class LayerMeta:
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
    moe_expert_flops_per_token: float = (
        0.0  # FLOPs for a single active expert on one token
    )
    moe_shared_flops: float = 0.0  # Total shared experts FLOPs
    moe_shared_bytes: int = 0  # Total shared experts bytes
    is_moe_layer: bool = False  # Whether this layer uses MoE

    def __repr__(self):
        return (
            f"<LayerMeta {self.name}: "
            f"FLOPs={self.flops}, INPUT={self.input_bytes}, OUTPUT={self.output_bytes}, "
            f"WEIGHT={self.weight_bytes}, parent={self.parent_layer.__class__.__name__}"
        )
