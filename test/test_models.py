from distilp.profiler import profile_model

batch_sizes = [1, 2, 4]
seq_len = 128


def test_profile_qwen3_6b():
    repo = "Qwen/Qwen3-32B-MLX-6bit"
    data = profile_model(repo, batch_sizes=batch_sizes, sequence_length=seq_len)
    assert data.L == 64
    assert data.V == 151936
    assert data.e_embed == 5120
    assert data.ek == 128
    assert data.ev == 128
    assert data.b[3] == 346214400.0
    assert data.b_i[3] == 1310720.0
    assert data.f_q["decode"]["b_1"][3] == 907018240.0
    assert data.quantization == "Q6_K"


def test_profile_llama_70b_4b():
    repo = "mlx-community/Meta-Llama-3-70B-4bit"
    data = profile_model(repo, batch_sizes=batch_sizes, sequence_length=seq_len)
    assert data.L == 80
    assert data.V == 128256
    assert data.e_embed == 8192
    assert data.ek == 128
    assert data.ev == 128
    assert data.b[3] == 454557696.0
    assert data.b_i[3] == 2097152.0
    assert data.f_q["decode"]["b_1"][3] == 1715470336.0
    assert data.quantization == "Q4_K"


# def test_profile_deepseek_v2_4b():
#     repo = "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"
#     data = profile_model(repo, batch_sizes=batch_sizes, sequence_length=seq_len)
#     assert data.L == 27
#     assert data.V == 102400
#     assert data.e_embed == 2048
#     assert data.ek == 128
#     assert data.ev == 128
#     assert data.b[3] == 44634112
#     assert data.b_i[3] == 524288.0
#     assert data.f_q["decode"]["b_1"][3] == 173277184.0
#     assert data.quantization == "F16"


def test_profile_qwen3_bf16():
    repo = "Qwen/Qwen3-32B-MLX-bf16"
    data = profile_model(repo, batch_sizes=batch_sizes, sequence_length=seq_len)
    assert data.L == 64
    assert data.V == 151936
    assert data.e_embed == 5120
    assert data.ek == 128
    assert data.ev == 128
    assert data.b[3] == 904396800
    assert data.b_i[3] == 1310720
    assert data.f_q["decode"]["b_1"][3] == 907018240.0
    assert data.quantization == "BF16"


def test_profile_qwen3_8b():
    repo = "Qwen/Qwen3-14B-MLX-8bit"
    data = profile_model(repo, batch_sizes=batch_sizes, sequence_length=seq_len)
    assert data.L == 40
    assert data.V == 151936
    assert data.e_embed == 5120
    assert data.ek == 128
    assert data.ev == 128
    assert data.b[3] == 335462400.0
    assert data.b_i[3] == 1310720.0
    assert data.f_q["decode"]["b_1"][3] == 663224320.0
    assert data.quantization == "Q8_0"
