"""
Data classes for HALDA solver profiles.
"""

from __future__ import annotations
from typing import Dict, Literal, Optional, List
from pydantic import BaseModel, Field

type QuantizationLevel = Literal["Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"]


class ModelProfile(BaseModel):
    """
    Unified model profile for both profiler output and solver input.

    This class serves dual purposes:
    1. Profiler output: Can optionally include per-layer arrays (*_layers fields)
    2. Solver input: Uses scalar values representing a typical layer (b_layer, b_in, b_out, f_q, f_out)

    The solver requires the scalar fields to be populated. The profiler may optionally
    populate the per-layer array fields (*_layers) for more detailed analysis.

    When loading from profiler JSON with per-layer arrays, the loader extracts
    representative scalar values (typically from layer index 1, the first non-embedding layer).
    """

    # --- Model architecture (always required) ---
    L: int = 0  # total layers, L
    hk: int = 0  # heads for keys, h_k
    ek: int = 0  # embedding dimension per head (keys), e_k
    hv: int = 0  # heads for values, h_v
    ev: int = 0  # embedding dimension per head (values), e_v
    n_kv: int = 0  # tokens in KV cache, n_{kv}
    e_embed: int = 0  # embedding size, e
    V: int = 0  # vocabulary size, V

    # --- Solver format: scalar values for typical layer ---
    # These are the primary fields used by the solver for optimization
    b_layer: int = 0  # bytes per typical layer, b (paper notation)
    b_in: int = 0  # input layer bytes, b_i
    b_out: int = 0  # output layer bytes, b_o
    # FLOPs per batch size for typical layer: {"b_1": flops, "b_2": flops, ...}
    f_q: Dict[str, float] = Field(default_factory=dict)  # f_q (typical layer FLOPs)
    # FLOPs per batch size for output layer: {"b_1": flops, "b_2": flops, ...}
    f_out: Dict[str, float] = Field(default_factory=dict)  # f_{q,out} (output layer FLOPs)
    Q: QuantizationLevel = "F32"  # Model quantization level (e.g., "Q4_K", "F16", etc.)

    # --- Profiler format: optional per-layer arrays ---
    # These fields contain detailed per-layer data from the profiler
    # Format: arrays of length L+1 where index 0 is typically the embedding layer
    b_layers: Optional[List[int]] = None  # bytes per layer (array)
    b_i_layers: Optional[List[int]] = None  # input bytes per layer (array)
    b_o_layers: Optional[List[int]] = None  # output bytes per layer (array)
    # FLOPs per layer per batch: {"b_1": [flops_l0, flops_l1, ...], "b_2": [...], ...}
    f_q_layers: Optional[Dict[str, List[float]]] = None  # per-layer FLOPs (array)

    # --- Profiler metadata ---
    seq_len: int = 0  # sequence length used during profiling
    quantization: str = ""  # quantization label from profiler (may differ from Q)

    # --- MoE (Mixture of Experts) configuration ---
    is_moe: bool = False  # Whether model uses Mixture of Experts
    n_routed_experts: int = 0  # Number of routed experts per MoE layer
    n_shared_experts: int = 0  # Number of always-active shared experts per MoE layer
    experts_per_token: int = 0  # Top-k experts activated per token
    moe_intermediate_size: int = 0  # FFN hidden dimension in each expert
    moe_layer_freq: int = 1  # Every N layers is MoE (1 = all MoE after first_k_dense)
    first_k_dense_replace: int = 0  # First K layers remain dense (non-MoE)
    total_moe_layers: int = 0  # Total number of MoE layers in model
    moe_layer_indices: Optional[List[int]] = None  # Which layer indices are MoE layers

    # --- MoE per-layer component metrics (for detailed profiler output) ---
    # Attention component (same for MoE and dense layers)
    attn_bytes: Optional[List[int]] = None  # Attention weight bytes per layer
    # Attention FLOPs per batch size: {"b_1": [flops_layer0, flops_layer1, ...], "b_2": [...]}
    attn_flops: Optional[Dict[str, List[float]]] = None

    # MoE FFN components (indexed by layer number)
    bytes_per_expert: Optional[Dict[int, int]] = None  # Bytes per routed expert by layer
    bytes_shared_experts: Optional[Dict[int, int]] = None  # Total bytes for shared experts by layer
    flops_per_expert: Optional[Dict[int, float]] = None  # FLOPs per routed expert by layer
    flops_shared_experts: Optional[Dict[int, float]] = None  # Total shared expert FLOPs by layer
    router_flops: Optional[Dict[int, float]] = None  # Router/gate FLOPs by layer
    router_bytes: Optional[Dict[int, int]] = None  # Router/gate weight bytes by layer
    flops_per_active_expert_per_token: Optional[Dict[int, float]] = None  # Per-active-expert per-token FLOPs by layer

    def print_summary(self) -> None:
        """Print a summary of the loaded model."""
        print(f"\n{'=' * 60}")
        print("Model Profile:")
        print(f"{'=' * 60}")
        print(f"  Layers (L): {self.L}")
        if self.b_layer > 0:
            print(f"  Bytes per layer: {self.b_layer / (1024**2):.1f} MB")
        if self.b_in > 0:
            print(f"  Input bytes: {self.b_in / (1024**2):.1f} MB")
        if self.b_out > 0:
            print(f"  Output bytes: {self.b_out / (1024**2):.1f} MB")
        print(f"  Attention heads (k/v): {self.hk}/{self.hv}")
        print(f"  Head dimensions (k/v): {self.ek}/{self.ev}")
        print(f"  KV cache tokens: {self.n_kv}")
        print(f"  Embedding dimension: {self.e_embed}")
        print(f"  Vocabulary size: {self.V}")
        print(f"  Quantization: {self.Q}")


class ModelProfilePhased(BaseModel):
    """
    Container for separate prefill and decode phase profiles.

    Used internally by the profiler to separate prefill and decode metrics
    before combining them into ModelProfileSplit.
    """

    prefill: ModelProfile
    decode: ModelProfile


class ModelProfileSplit(BaseModel):
    """
    Profiler output format with per-layer arrays split by phase (prefill/decode).

    This is the raw output format from the profiler API (profile_model_split).
    It contains per-layer arrays and phase-specific metrics. The loader
    (loader.py) converts this format to ModelProfile by extracting scalar
    values from the arrays.

    Used only as:
    - Profiler output: profile_model_split() returns this
    - Loader input: load_model_profile_from_dict() accepts this format
    """

    # Per-layer arrays (length L+1, index 0 is embedding layer)
    b: List[int]  # bytes per layer
    b_i: List[int]  # input bytes per layer
    b_o: List[int]  # output bytes per layer

    # Model architecture
    L: int  # total layers
    hk: int  # heads for keys
    hv: int  # heads for values
    ek: int  # embedding dimension per head (keys)
    ev: int  # embedding dimension per head (values)
    n_kv: int  # tokens in KV cache
    e_embed: int  # embedding size
    V: int  # vocabulary size
    seq_len: int  # sequence length used during profiling

    # Phase-split FLOPs: {"prefill": {"b_1": [flops per layer], ...}, "decode": {...}}
    f_q: Dict[str, Dict[str, List[float]]]  # phase -> batch_size -> [FLOPs per layer]
    f_out: Dict[str, Dict[str, float]]  # phase -> batch_size -> output layer FLOPs
    quantization: str  # quantization label (e.g., "Q4_K", "F16")

    # MoE fields (optional, populated only for MoE models)
    is_moe: bool = False
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    moe_layer_freq: int = 0
    first_k_dense_replace: int = 0
    total_moe_layers: int = 0
    moe_layer_indices: List[int] = Field(default_factory=list)

    # Component metrics for solver assignment
    attn_bytes: List[int] = Field(default_factory=list)
    attn_flops: Dict[str, Dict[str, List[float]]] = Field(default_factory=dict)  # phase -> b_tag -> [FLOPs]
    bytes_per_expert: Dict[int, int] = Field(default_factory=dict)
    bytes_shared_experts: Dict[int, int] = Field(default_factory=dict)
    flops_per_expert: Dict[int, float] = Field(default_factory=dict)
    flops_shared_experts: Dict[int, float] = Field(default_factory=dict)
    router_flops: Dict[int, float] = Field(default_factory=dict)
    router_bytes: Dict[int, int] = Field(default_factory=dict)
    flops_per_active_expert_per_token: Dict[int, float] = Field(default_factory=dict)
