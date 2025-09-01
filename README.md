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
uv run python main.py --devices profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m4.json profiles/device_profile_mac_m4.json profiles/device_profile_mac_m4.json --model profiles/model_profile_qwen3_4b_8bit.json


uv run python main.py --devices profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m4.json --model profiles/model_profile_qwen3_32b_8bit.json



uv run python main.py --devices profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m1_max.json profiles/device_profile_mac_m4.json profiles/device_profile_mac_m4.json profiles/device_profile_mac_m4.json --model profiles/model_profile_qwen3_32_fp16.json
```


