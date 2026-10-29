import types
import torch
import pytest
from bluepixel.text_precision import source_rms_norm, source_attention, reference_sdpa, reference_frequencies


def test_qwen_rms_cast_before_scale():
    torch.manual_seed(17)
    values = torch.randn(3, 7, 128).bfloat16()
    weight = torch.randn(128).bfloat16()
    module = types.SimpleNamespace(eps=1e-6, scale=lambda: weight)
    expected = values.float()
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
    expected = weight * expected.to(torch.bfloat16)
    assert torch.equal(source_rms_norm(module, values), expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression"))])
def test_source_rope_rounding_at_token_459(device):
    positions = torch.tensor([[459]], device=device)
    cos, half_sin, negative_half_sin = reference_frequencies(128, 1000000.0, positions, device)
    # Golden BF16 source value; GPU-initialized inverse frequencies round this
    # to -0.0009002685546875 on the verified cu128 environment.
    assert half_sin.bfloat16()[0, 0, 0, 18].item() == -0.000904083251953125
    assert torch.equal(half_sin, -negative_half_sin)
    assert cos.shape == (1, 1, 1, 128)


@pytest.mark.parametrize("previous", [False, True])
def test_attention_precision_preference_restored_on_dispatch_failure(monkeypatch, previous):
    state = {"allowed": previous}
    monkeypatch.setattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", lambda: state["allowed"])
    monkeypatch.setattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", lambda value: state.update(allowed=value))
    def fail(*args, **kwargs):
        assert not state["allowed"]
        raise RuntimeError("dispatch failure")
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fail)
    with pytest.raises(RuntimeError, match="dispatch failure"):
        reference_sdpa(types.SimpleNamespace(is_cuda=True), None, None)
    assert state["allowed"] == previous


def test_source_attention_matches_unfused_reference():
    torch.manual_seed(18)
    dtype = torch.float32
    batch, length, heads, kv_heads, dim, width = 1, 5, 2, 1, 8, 16
    linear = lambda out: torch.nn.Linear(width, out, bias=False, dtype=dtype)
    module = types.SimpleNamespace(num_heads=heads, num_kv_heads=kv_heads, head_dim=dim,
        inner_size=heads * dim, q_proj=linear(heads * dim), k_proj=linear(kv_heads * dim),
        v_proj=linear(kv_heads * dim), o_proj=torch.nn.Linear(heads * dim, width, bias=False),
        q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity())
    values = torch.randn(batch, length, width)
    positions = torch.arange(length).float().unsqueeze(-1)
    frequencies = positions / (1000000 ** (torch.arange(0, dim, 2).float() / dim))
    full = torch.cat((frequencies, frequencies), dim=-1).view(1, 1, length, dim)
    cos, sin = full.cos(), full.sin()
    mask = torch.ones(length, length).triu(1).masked_fill(torch.ones(length, length).triu(1).bool(), -1e30)
    actual, cache = source_attention(module, values, mask, (cos, sin[..., :dim // 2], -sin[..., dim // 2:]))
    q = module.q_proj(values).view(batch, length, heads, dim).transpose(1, 2)
    k = module.k_proj(values).view(batch, length, kv_heads, dim).transpose(1, 2)
    v = module.v_proj(values).view(batch, length, kv_heads, dim).transpose(1, 2)
    rotate = lambda x: torch.cat((-x[..., dim // 2:], x[..., :dim // 2]), dim=-1)
    q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
    k, v = k.repeat_interleave(heads // kv_heads, dim=1), v.repeat_interleave(heads // kv_heads, dim=1)
    scores = (q @ k.transpose(-1, -2)) / dim ** 0.5
    scores = scores.masked_fill(mask.bool(), float("-inf"))
    expected = torch.softmax(scores, -1) @ v
    expected = module.o_proj(expected.transpose(1, 2).contiguous().reshape(batch, length, -1))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert cache is None
