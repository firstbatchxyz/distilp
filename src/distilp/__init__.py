"""
DistilP: A Python package for solving layer assignment problems in distributed inference
"""

from .halda_p_solver import halda_solve
from .components.dataclasses import DeviceProfile, ModelProfile
from .components.loader import (
    load_device,
    load_model,
    load_model_profile_from_dict,
    load_model_profile_split_from_json_string,
    load_device_profile_from_dict,
)
from .components.dense_common import HALDAResult

__all__ = [
    "halda_solve",
    "DeviceProfile",
    "ModelProfile",
    "HALDAResult",
    "load_device",
    "load_model",
    "load_model_profile_from_dict",
    "load_model_profile_split_from_json_string",
    "load_device_profile_from_dict",
]

__version__ = "0.1.2"
