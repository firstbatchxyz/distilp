import json
import inspect
import importlib
from huggingface_hub import hf_hub_download
from distilp.profiler import profile_model

B = 1
L = 128


def load_config(repo):
    try:
        config_path = hf_hub_download(repo_id=repo, filename="config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return config_dict
    except Exception as e:
        raise RuntimeError(f"Unable to download config from Hugging Face: {e}")


def filter_config(obj, config_dict):
    try:
        modelargs_params = inspect.signature(obj.__init__).parameters
        valid_config = [p for p in modelargs_params if p != "self"]
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_config}
        return filtered_config
    except Exception as e:
        raise RuntimeError(f"Unable to get signature of config object: {e}")


def get_model_objs(model):
    try:
        module = importlib.import_module(f"mlx_lm.models.{model}")
    except ImportError as e:
        raise RuntimeError(f"Model {model} not found in mlx_lm registry.") from e
    Model = getattr(module, "Model")
    ModelArgs = getattr(module, "ModelArgs")
    assert inspect.isclass(Model)
    assert inspect.isclass(ModelArgs)

    if Model is None or ModelArgs is None:
        raise RuntimeError(
            "Could not import symbols 'Model' and 'ModelArgs' from mlx.models.{model}"
        )

    return Model, ModelArgs


def run_profiler(model, repo):
    config = load_config(repo)
    Model, ModelArgs = get_model_objs(model)
    filtered_config = filter_config(ModelArgs, config)
    config_obj = ModelArgs(**filtered_config)
    obj = Model(config_obj)
    return profile_model(obj, config_obj, B, L, config)  # type: ignore # FIXME: ???


def profile_qwen3_6b():
    model = "qwen3"
    repo = "Qwen/Qwen3-32B-MLX-6bit"
    data = run_profiler(model, repo)
    assert data.L == 64
    assert data.V == 151936
    assert data.e_embed == 5120
    assert data.ek == 128
    assert data.ev == 128
    assert data.b[3] == 346214400.0
    assert data.b_i[3] == 1310720.0
    assert data.f_q[3] == 907018240.0  # type: ignore # FIXME: ???
    print("Qwen/Qwen3-32B-MLX-6bit -- PASS")


def profile_llama_70b_4b():
    model = "llama"
    repo = "mlx-community/Meta-Llama-3-70B-4bit"
    data = run_profiler(model, repo)
    assert data.L == 80
    assert data.V == 128256
    assert data.e_embed == 8192
    assert data.ek is None  # FIXME: ???
    assert data.ev is None  # FIXME: ???
    assert data.b[3] == 454557696.0
    assert data.b_i[3] == 2097152.0
    assert data.f_q[3] == 1715470336.0
    print("mlx-community/Meta-Llama-3-70B-4bit -- PASS")


def profile_deepseek_v2_4b():
    model = "deepseek_v2"
    repo = "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"
    data = run_profiler(model, repo)
    assert data.L == 27
    assert data.V == 102400
    assert data.e_embed == 2048
    assert data.ek == 128
    assert data.ev == 128
    assert data.b[3] == 44634112
    assert data.b_i[3] == 524288.0
    assert data.f_q[3] == 173277184.0  # type: ignore # FIXME: ???
    print("mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx -- PASS")


def profile_qwen3_f16():
    model = "qwen3"
    repo = "Qwen/Qwen3-32B-MLX-bf16"
    data = run_profiler(model, repo)
    assert data.L == 64
    assert data.V == 151936
    assert data.e_embed == 5120
    assert data.ek == 128
    assert data.ev == 128
    assert data.b[3] == 904396800
    assert data.b_i[3] == 1310720.0
    assert data.f_q[3] == 907018240.0  # type: ignore # FIXME: ???
    print("Qwen/Qwen3-32B-MLX-bf16 -- PASS")


def profile_qwen3_8b():
    model = "qwen3"
    repo = "Qwen/Qwen3-14B-MLX-8bit"
    data = run_profiler(model, repo)
    assert data.L == 40
    assert data.V == 151936
    assert data.e_embed == 5120
    assert data.ek == 128
    assert data.ev == 128
    assert data.b[3] == 335462400.0
    assert data.b_i[3] == 1310720.0
    assert data.f_q[3] == 663224320.0  # type: ignore # FIXME: ???
    print("Qwen/Qwen3-32B-MLX-bf16 -- PASS")


if __name__ == "__main__":
    profile_qwen3_6b()
    profile_qwen3_8b()
    profile_qwen3_f16()
    profile_llama_70b_4b()
    profile_deepseek_v2_4b()
