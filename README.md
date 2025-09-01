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
uv run python main.py --devices profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m4.json --model profiles/model_profile.json
```

