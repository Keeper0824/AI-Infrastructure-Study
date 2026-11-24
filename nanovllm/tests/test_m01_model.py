import os

import pytest
import torch


def _tiny_config(num_hidden_layers=2):
    from nanovllm.models.qwen3 import Qwen3Config

    return Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        vocab_size=100,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )


@pytest.fixture(scope="module")
def hf_model():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model_dir = "models/Qwen3-0.6B"
    if not os.path.isdir(model_dir):
        pytest.skip("model not downloaded")

    return (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )


def test_config():
    from nanovllm.models.qwen3 import Qwen3Config

    c = Qwen3Config.from_pretrained("models/Qwen3-0.6B")
    assert (
        c.num_hidden_layers,
        c.hidden_size,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.head_dim,
    ) == (28, 1024, 16, 8, 128)
    assert c.intermediate_size == 3072
    assert c.vocab_size == 151936
    assert c.rope_theta == 1_000_000
    assert c.max_position_embeddings == 40960
    assert c.tie_word_embeddings is True
    assert c.rope_scaling is None


def test_rms_norm():
    from nanovllm.models.qwen3 import RMSNorm

    torch.manual_seed(0)
    x, w = torch.randn(4, 8), torch.randn(8)
    norm = RMSNorm(8, 1e-6)
    norm.weight.data.copy_(w)

    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-6)


def test_rms_norm_bf16():
    from nanovllm.models.qwen3 import RMSNorm

    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(8, dtype=torch.bfloat16)
    norm = RMSNorm(8, 1e-6).to(torch.bfloat16)
    norm.weight.data.copy_(w)

    xf = x.float()
    expected = (
        (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(w.dtype) * w
    ).to(x.dtype)
    torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("T", [1, 17])
def test_rms_norm_T(T):
    from nanovllm.models.qwen3 import RMSNorm

    torch.manual_seed(0)
    x = torch.randn(T, 128)
    w = torch.randn(128)
    norm = RMSNorm(128, 1e-6)
    norm.weight.data.copy_(w)

    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-6)


def test_rope_rotate_half():
    from nanovllm.models.qwen3 import rotate_half

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_rope_relative_position():
    from nanovllm.models.qwen3 import RotaryEmbedding

    torch.manual_seed(0)
    rope = RotaryEmbedding(8, 32, 1_000_000)
    v = torch.randn(1, 2, 8)

    q3, _ = rope(torch.tensor([3]), v, v)
    _, k5 = rope(torch.tensor([5]), v, v)
    q4, _ = rope(torch.tensor([4]), v, v)
    _, k6 = rope(torch.tensor([6]), v, v)

    dot35 = (q3 * k5).sum(-1)
    dot46 = (q4 * k6).sum(-1)
    dot36 = (q3 * k6).sum(-1)

    # 相对距离都是 2：点积必须相同
    torch.testing.assert_close(dot35, dot46, atol=1e-5, rtol=1e-5)

    # 相对距离不同：点积必须变化；如果 RoPE 是空操作，这条会失败
    assert not torch.allclose(dot35, dot36, atol=1e-4)


def test_rope_shape():
    from nanovllm.models.qwen3 import RotaryEmbedding

    rope = RotaryEmbedding(8, 32, 1_000_000)
    q = torch.randn(5, 4, 8)
    k = torch.randn(5, 2, 8)

    q_rot, k_rot = rope(torch.arange(5), q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


@pytest.mark.parametrize("T", [1, 17])
def test_rope_shape_T(T):
    from nanovllm.models.qwen3 import RotaryEmbedding

    rope = RotaryEmbedding(128, 64, 1_000_000)
    q = torch.randn(T, 4, 128)
    k = torch.randn(T, 2, 128)

    q_rot, k_rot = rope(torch.arange(T), q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_mlp_formula():
    from nanovllm.models.qwen3 import Qwen3MLP

    torch.manual_seed(0)
    mlp = Qwen3MLP(8, 16)
    x = torch.randn(3, 8)

    expected = mlp.down_proj(
        torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
    )
    torch.testing.assert_close(mlp(x), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("T", [1, 17])
def test_mlp_shape_T(T):
    from nanovllm.models.qwen3 import Qwen3MLP

    mlp = Qwen3MLP(128, 256)
    x = torch.randn(T, 128)

    assert mlp(x).shape == x.shape


@pytest.mark.gpu
def test_mlp_hf(hf_model):
    from nanovllm.models.qwen3 import Qwen3MLP

    nano = Qwen3MLP(1024, 3072).to(device="cuda", dtype=torch.bfloat16)

    hf_mlp = hf_model.model.layers[0].mlp
    nano.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight)
    nano.up_proj.weight.data.copy_(hf_mlp.up_proj.weight)
    nano.down_proj.weight.data.copy_(hf_mlp.down_proj.weight)

    torch.manual_seed(0)
    x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        got = nano(x)
        want = hf_mlp(x)

    torch.testing.assert_close(got, want, atol=2e-2, rtol=1e-2)


def test_attention_naive_matches_sdpa():
    torch.manual_seed(0)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)

    scores = q @ k.transpose(-1, -2) / (8**0.5)
    causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    want = torch.softmax(scores, dim=-1) @ v

    got = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("T", [1, 5, 17])
def test_attention_shape_cpu(T):
    from nanovllm.models.qwen3 import Qwen3Attention, Qwen3Config

    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        vocab_size=100,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    attn = Qwen3Attention(cfg)
    x = torch.randn(T, 16)

    out = attn(x, torch.arange(T))

    assert out.shape == x.shape


@pytest.mark.gpu
def test_attention_hf(hf_model):
    from nanovllm.models.qwen3 import Qwen3Attention, Qwen3Config

    model_dir = "models/Qwen3-0.6B"
    cfg = Qwen3Config.from_pretrained(model_dir)
    nano = Qwen3Attention(cfg).to(device="cuda", dtype=torch.bfloat16)

    hf_attn = hf_model.model.layers[0].self_attn
    nano.q_proj.weight.data.copy_(hf_attn.q_proj.weight)
    nano.k_proj.weight.data.copy_(hf_attn.k_proj.weight)
    nano.v_proj.weight.data.copy_(hf_attn.v_proj.weight)
    nano.o_proj.weight.data.copy_(hf_attn.o_proj.weight)
    nano.q_norm.weight.data.copy_(hf_attn.q_norm.weight)
    nano.k_norm.weight.data.copy_(hf_attn.k_norm.weight)

    T = 7
    torch.manual_seed(0)
    hidden = torch.randn(1, T, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.arange(T, device="cuda").unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(hidden, position_ids)

    mask = torch.full((T, T), -1e9, device="cuda", dtype=torch.bfloat16)
    mask = torch.triu(mask, diagonal=1)[None, None]

    with torch.no_grad():
        want, _ = hf_attn(hidden, (cos, sin), attention_mask=mask)
        got = nano(hidden[0], torch.arange(T, device="cuda"))

    torch.testing.assert_close(got, want[0], atol=2e-2, rtol=1e-2)


def test_model_decoder_layer_shape():
    from nanovllm.models.qwen3 import Qwen3DecoderLayer

    cfg = _tiny_config()
    layer = Qwen3DecoderLayer(cfg)
    x = torch.randn(5, cfg.hidden_size)

    out = layer(x, torch.arange(5))

    assert out.shape == x.shape


def test_model_shape():
    from nanovllm.models.qwen3 import Qwen3Model

    cfg = _tiny_config()
    model = Qwen3Model(cfg)
    input_ids = torch.tensor([1, 2, 3, 4])

    hidden = model(input_ids, torch.arange(4))

    assert hidden.shape == (4, cfg.hidden_size)


def test_lm_head_logits():
    from nanovllm.models.qwen3 import Qwen3ForCausalLM

    cfg = _tiny_config()
    model = Qwen3ForCausalLM(cfg)
    input_ids = torch.tensor([1, 2, 3, 4])
    positions = torch.arange(4)

    logits = model(input_ids, positions)
    assert logits.shape == (4, cfg.vocab_size)

    expected = model.compute_logits(model.model(input_ids, positions))
    torch.testing.assert_close(logits, expected)


def test_load_weights():
    pytest.importorskip("safetensors")
    from nanovllm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    model_dir = "models/Qwen3-0.6B"
    if not os.path.isdir(model_dir):
        pytest.skip("model not downloaded")

    cfg = Qwen3Config.from_pretrained(model_dir)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)

    loaded = model.load_weights(model_dir)

    assert len(model.state_dict()) == 311
    assert len(loaded) in (310, 311)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
