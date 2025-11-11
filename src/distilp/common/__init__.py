from typing import Literal
from .device import DeviceProfile
from .model import (
    ModelProfile,
    ModelProfilePhased,
    ModelProfileSplit,
)

type QuantizationLevel = Literal["Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16", "F32"]

__all__ = [
    # devices
    "DeviceProfile",
    # models
    "ModelProfile",
    "ModelProfilePhased",
    "ModelProfileSplit",
    # types
    "QuantizationLevel",
]
