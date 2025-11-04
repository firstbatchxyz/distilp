"""
DistilP: A Python library for MILP-based layer/expert assignment
for distributed inference across heterogeneous devices.

Submodules are available based on installed optional dependencies:
- distilp.solver: Install with `pip install distilp[solver]`
- distilp.profiler: Install with `pip install distilp[profiler]`

Usage:
    # For solver functionality
    from distilp.solver import halda_solve, DeviceProfile, ModelProfile

    # For profiler functionality
    from distilp.profiler import profile_model, profile_device, DeviceInfo
"""

__version__ = "0.1.2"