"""
DistilP: A Python package for solving layer assignment problems in distributed inference
"""

from .halda_p_solver import halda_solve
from .components.dense_common import HALDAResult

__all__ = [
    "halda_solve",
    "HALDAResult",
]

__version__ = "0.1.2"
