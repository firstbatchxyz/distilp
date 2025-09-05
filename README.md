# DNET Solver

Run to install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then
```bash
git clone https://github.com/firstbatchxyz/dsolver.git
cd dsolver
uv sync
uv run python main.py --profile qwen3_32b/6bit --verbose
uv run python main.py --profile hermes_70b --verbose
```


