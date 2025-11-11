from typing import Literal

type ModelPhase = Literal["merged", "prefill", "decode"]
type QuantizationLevel = Literal["Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"]
