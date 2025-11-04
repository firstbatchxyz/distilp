# distilp

## Installation

distilp requires:

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

```sh
# Install solver only
uv add distilp[solver]

# Install profiler only
uv add distilp[profiler]

# Install both
uv add distilp[solver,profiler]

# Install profiler with CUDA 12 support
uv add distilp[profiler,cuda12]

# Install everything for development
uv add distilp[solver,profiler,torch]
```

### Local Development

Add all optional dependencies with:

```sh
uv sync --extra dev
```

TODO: !!!
