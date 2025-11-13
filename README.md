# DistilP

**DistilP** is a Python library for MILP-based layer/expert assignment for distributed inference across heterogeneous devices. It profiles device and model characteristics, then solves an optimization problem to determine the optimal layer distribution for distributed Large Language Model (LLM) inference.

## Features

- **Device Profiling**: Measure CPU/GPU throughput, memory capacity, disk I/O, and communication characteristics
- **Model Profiling**: Analyze layer-wise memory requirements and computational costs for LLMs
- **HALDA Solver**: Mixed Integer Linear Programming solver for optimal layer assignment across heterogeneous devices
- **Pydantic-based**: Type-safe profile schemas with automatic validation and serialization

## Installation

DistilP requires:

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Install from Package

```bash
# Install solver only
uv add distilp[solver]

# Install profiler only (macOS with MLX support)
uv add distilp[profiler]

# Install both solver and profiler
uv add distilp[solver,profiler]

# Install with plotting support
uv add distilp[solver,solver-plotting]
```

### Local Development

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/firstbatchxyz/distilp.git
cd distilp

# Install all dependencies for development
uv sync --extra dev
```

This installs all optional dependencies including pytest, matplotlib, and profiling tools.

## Usage

### CLI

The typical workflow involves three steps:

1. **Profile your devices** - Generate device performance profiles
2. **Profile your model** - Generate model layer characteristics
3. **Run the solver** - Compute optimal layer distribution

#### 1. Profile a Device

Profile the current device to measure its computational capabilities:

```bash
# Profile device for a specific model
uv run profiler device -r "Qwen/Qwen3-4B-MLX-8bit" -o device_profile.json

# Profile with higher batch sizes (default max_batch_exp is 2, meaning 2^2=4)
uv run profiler device -r "Qwen/Qwen3-4B-MLX-8bit" -o device_profile.json --max-batch-exp 6
```

This will output a JSON file, at the path `-o`. If no `-o` is given, it will print to console.

#### 2. Profile a Model

Profile a model to measure layer-wise memory and computational requirements:

```bash
# Profile model with default batch sizes (1,2,4)
uv run profiler model -r "Qwen/Qwen3-4B-MLX-8bit" -o model_profile.json

# Profile with custom batch sizes and sequence length
uv run profiler model -r "Qwen/Qwen3-4B-MLX-8bit" \
  -o model_profile.json \
  --batches "1,2,4,8" \
  --sequence 1024
```

This will also output a JSON.

#### 3. Run the Solver

Use the profiles to compute the optimal layer distribution:

```bash
# Run solver with a profile folder
uv run solver --profile test/profiles/hermes_70b

# Run solver with separate device and model files
uv run solver \
  --devices device_profile_1.json device_profile_2.json \
  --model model_profile.json

# Run with custom solver parameters
uv run solver --profile hermes_70b \
  --time-limit 10 \
  --max-iters 20 \
  --mip-gap 0.001

# Save solution to JSON
uv run solver --profile hermes_70b --save-solution solution.json

# Disable plotting
uv run solver --profile hermes_70b --no-plot
```

**Output**: Optimal layer distribution showing:

- Number of pipeline stages (k)
- Layer assignment per device (w)
- Objective value (latency estimate)
- Device grouping for optimization

### Library

DistilP can be used programmatically in Python scripts. This is useful for integrating profiling and solving into automated workflows or custom applications.

#### Basic Profiling

```python
from distilp.profiler import profile_device, profile_model

# Profile the current device
device_profile = profile_device(
    repo_id="Qwen/Qwen3-4B-MLX-8bit",
    max_batch_exp=6,  # Profile up to batch size 2^6=64
    debug=0
)

# Profile a model
model_profile = profile_model(
    repo_id="Qwen/Qwen3-4B-MLX-8bit",
    batch_sizes=[1, 2, 4, 8],
    sequence_length=512,
    debug=0
)

print(f"Device: {device_profile.name}")
print(f"Model: {model_profile.L} layers, {model_profile.V} vocab size")
```

#### Saving and Loading Profiles

Profiles are Pydantic models with built-in serialization:

```python
from distilp.common import DeviceProfile, ModelProfileSplit
import json

# Save profiles to JSON
with open("device_profile.json", "w") as f:
    f.write(device_profile.model_dump_json(indent=2))

with open("model_profile.json", "w") as f:
    f.write(model_profile.model_dump_json(indent=2))

# Load profiles from JSON
with open("device_profile.json", "r") as f:
    device_profile = DeviceProfile.model_validate_json(f.read())

with open("model_profile.json", "r") as f:
    model_profile = ModelProfileSplit.model_validate_json(f.read())
```

#### Running the Solver

```python
from distilp.solver import halda_solve
from distilp.common import DeviceProfile, ModelProfile

# Load or create device and model profiles
devices = [device_profile_1, device_profile_2]  # List of DeviceProfile objects
model = model_profile  # ModelProfile or ModelProfileSplit object

# Run the HALDA solver
result = halda_solve(
    devs=devices,
    model=model,
    k_candidates=None,  # None = try all factors of L
    mip_gap=1e-4,       # MIP gap tolerance
    plot=True,          # Show k vs objective plot
    kv_bits="4bit",     # KV cache quantization
)

# Access results
print(f"Optimal k: {result.k}")
print(f"Objective value: {result.obj_value:.6f}")
print(f"Layer distribution: {result.w}")
print(f"Device grouping: {result.sets}")

# Access per-device assignments
for i, (device, w_i, n_i) in enumerate(zip(devices, result.w, result.n)):
    print(f"Device {i} ({device.name}): {w_i} layer groups, {n_i} layers total")
```

## Testing

We use pytest:

```bash
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest test/test_integration.py -v

# Run specific test function (can give file name too)
uv run pytest -k test_method_name
```

## License

You can find the license [here](./LICENSE).
