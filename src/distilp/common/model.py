"""
Data classes for HALDA solver profiles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, List

from . import QuantPerf


# TODO: this is used by solver
# all else used by profiler
@dataclass
class ModelProfile:
    """
    Model-global constants (bytes, sizes, FLOPs) from profiler.
    """

    L: int  # total layers, L
    b_layer: int  # bytes per layer,   b  (paper notation: b)
    b_in: int  # input bytes,       b_i
    b_out: int  # output bytes,      b_o
    hk: int  # heads for keys,    h_k
    ek: int  # emb per head (k),  e_k
    hv: int  # heads for values,  h_v
    ev: int  # emb per head (v),  e_v
    n_kv: int  # tokens in KV cache, n_{kv}
    e_embed: int  # embedding size,    e
    V: int  # vocabulary size,   V

    # FLOPs per layer per batch_size, and for the output layer:
    f_q: Dict[str, QuantPerf]  # (batch_size, f_q) (per "typical" layer), per batch size
    f_out: Dict[str, QuantPerf]  # (batch_size,f_{q, out})   (for output layer)
    Q: Literal[
        "Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"
    ]  # Model quantization level (e.g., "Q4_K", "MXFP4", etc.)

    # Optional MoE-specific fields
    is_moe: bool = False  # Whether model uses Mixture of Experts
    n_routed_experts: int = 0  # Number of routed experts
    n_shared_experts: int = 0  # Number of shared experts
    experts_per_token: int = 0  # Number of experts activated per token
    moe_intermediate_size: int = 0  # MoE intermediate size
    moe_layer_freq: int = 1  # Frequency of MoE layers
    first_k_dense_replace: int = 0  # Number of dense layers before MoE
    total_moe_layers: int = 0  # Total number of MoE layers
    moe_layer_indices: Optional[list] = None  # Indices of MoE layers

    # Optional per-layer byte metrics (for MoE models)
    attn_bytes: Optional[list] = None  # Attention bytes per layer
    bytes_per_expert: Optional[Dict[str, int]] = None  # Bytes per expert per layer
    bytes_shared_experts: Optional[Dict[str, int]] = None  # Bytes for shared experts

    # Optional per-layer FLOP metrics (for MoE models)
    attn_flops: Optional[Dict[str, Dict[str, list]]] = (
        None  # Attention FLOPs (prefill/decode)
    )
    flops_per_expert: Optional[Dict[str, int]] = None  # FLOPs per expert per layer
    flops_shared_experts: Optional[Dict[str, int]] = None  # FLOPs for shared experts
    router_flops: Optional[Dict[str, int]] = None  # Router FLOPs per layer
    router_bytes: Optional[Dict[str, int]] = None  # Router bytes per layer
    flops_per_active_expert_per_token: Optional[Dict[str, int]] = (
        None  # Active expert FLOPs
    )

    def print_summary(self) -> None:
        """Print a summary of the loaded model."""
        print(f"\n{'=' * 60}")
        print("Model Profile:")
        print(f"{'=' * 60}")
        print(f"  Layers (L): {self.L}")
        print(f"  Bytes per layer: {self.b_layer / (1024**2):.1f} MB")
        print(f"  Input bytes: {self.b_in / (1024**2):.1f} MB")
        print(f"  Output bytes: {self.b_out / (1024**2):.1f} MB")
        print(f"  Attention heads (k/v): {self.hk}/{self.hv}")
        print(f"  Head dimensions (k/v): {self.ek}/{self.ev}")
        print(f"  KV cache tokens: {self.n_kv}")
        print(f"  Embedding dimension: {self.e_embed}")
        print(f"  Vocabulary size: {self.V}")
        print(f"  Quantizations: {', '.join(self.Q)}")


@dataclass
class ModelProfileInfo:
    """
    Model-global constants (bytes, sizes, FLOPs) from profiler.
    """

    # Per-layer metrics
    b: List[int] = []  # bytes per layer (weights)
    b_i: List[int] = []  # input bytes per layer (base batch)
    b_o: List[int] = []  # output bytes per layer (base batch)
    # FLOPs per layer for each batch size (e.g., {'b_1': [...], 'b_2': [...]})
    f_q: Dict[str, List[float]] = field(default_factory=dict)

    # Model-level metrics (new fields)
    L: int = 0  # total layers
    hk: int = 0  # heads for keys
    ek: int = 0  # emb per head (k)
    hv: int = 0  # heads for values
    ev: int = 0  # emb per head (v)
    n_kv: int = 0  # tokens in KV cache
    e_embed: int = 0  # embedding size
    V: int = 0  # vocabulary size

    # Quantization level label for this model/profile
    quantization: str = ""  # One of: Q4_K, Q5_K, Q6_K, Q8_0, BF16, F16, F32
    # Output-layer FLOPs per batch size (e.g., {'b_1': 123.0})
    f_out: Dict[str, float] = field(default_factory=dict)
    # Sequence length used for profiling
    seq_len: int = 0


@dataclass
class MoEModelProfileInfo(ModelProfileInfo):
    """
    MoE-specific model profile with component-level metrics for solver assignment.
    Inherits base metrics from ModelProfileInfo.
    """

    # MoE configuration
    n_routed_experts: int = 0  # Number of routed experts per MoE layer
    n_shared_experts: int = 0  # Number of always-active shared experts per MoE layer
    experts_per_token: int = 0  # Top-k experts selected per token
    moe_intermediate_size: int = 0  # FFN hidden dimension in each expert
    moe_layer_freq: int = 0  # Every N layers is MoE (1 = all MoE after first_k_dense)
    first_k_dense_replace: int = 0  # First K layers remain dense (no MoE)
    total_moe_layers: int = 0  # Total number of MoE layers in the model

    # Per-layer component metrics for solver assignment
    moe_layer_indices: List[int] = field(default_factory=list)  # Which layers are MoE

    # Attention component (same for MoE and dense, but tracked separately for assignment)
    attn_bytes: List[int] = field(
        default_factory=list
    )  # Attention weight bytes per layer
    attn_flops: Dict[str, List[float]] = field(
        default_factory=dict
    )  # Attention FLOPs per layer by batch size

    # MoE FFN component (per layer, indexed by layer number)
    bytes_per_expert: Dict[int, int] = field(
        default_factory=dict
    )  # Bytes per routed expert by layer
    bytes_shared_experts: Dict[int, int] = field(
        default_factory=dict
    )  # Total bytes for shared experts by layer
    flops_per_expert: Dict[int, float] = field(
        default_factory=dict
    )  # FLOPs per routed expert by layer
    flops_shared_experts: Dict[int, float] = field(
        default_factory=dict
    )  # Total shared experts FLOPs by layer
    router_flops: Dict[int, float] = field(
        default_factory=dict
    )  # Router/gate FLOPs by layer
    router_bytes: Dict[int, int] = field(
        default_factory=dict
    )  # Router/gate weight bytes by layer
    flops_per_active_expert_per_token: Dict[int, float] = field(
        default_factory=dict
    )  # Per-active-expert per-token FLOPs by layer


@dataclass
class ModelProfilePhased:
    prefill: ModelProfileInfo
    decode: ModelProfileInfo


@dataclass
class ModelProfileSplit:
    b: List[int]
    b_i: List[int]
    b_o: List[int]
    L: int
    hk: int
    hv: int
    ek: int
    ev: int
    n_kv: int
    e_embed: int
    V: int
    seq_len: int

    f_q: Dict[str, Dict[str, List[float]]]  # phase -> b_tag -> [FLOPs per layer]
    f_out: Dict[str, Dict[str, float]]  # phase -> b_tag -> output layer FLOPs
    quantization: str  # quantization label

    # MoE fields (optional, populated only for MoE models)
    is_moe: bool = False
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    moe_layer_freq: int = 0
    first_k_dense_replace: int = 0
    total_moe_layers: int = 0
    moe_layer_indices: List[int] = field(default_factory=list)

    # Component metrics for solver assignment
    attn_bytes: List[int] = field(default_factory=list)
    attn_flops: Dict[str, Dict[str, List[float]]] = field(
        default_factory=dict
    )  # phase -> b_tag -> [FLOPs]
    bytes_per_expert: Dict[int, int] = field(default_factory=dict)
    bytes_shared_experts: Dict[int, int] = field(default_factory=dict)
    flops_per_expert: Dict[int, float] = field(default_factory=dict)
    flops_shared_experts: Dict[int, float] = field(default_factory=dict)
    router_flops: Dict[int, float] = field(default_factory=dict)
    router_bytes: Dict[int, int] = field(default_factory=dict)
    flops_per_active_expert_per_token: Dict[int, float] = field(default_factory=dict)
