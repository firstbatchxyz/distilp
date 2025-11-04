from typing import Literal, Dict

type QuantizationType = Literal["Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"]

type QuantPerf = Dict[str, Dict[str, float]]
"""Type-alias for quantization performance dictionary."""

# TODO: why is this also called QuantPerf?
type QuantPerfOther = Dict[str, float]
