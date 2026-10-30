from types import SimpleNamespace

import pytest
import torch

from bluepixel.text_precision import reference_sdpa, source_attention


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("query_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_dtype", [torch.float16, torch.bfloat16])
def test_low_precision_mask_supports_fp32_and_bf16_queries(device, query_dtype, mask_dtype):
    generator = torch.Generator(device=device).manual_seed(82)
    query, key, value = [torch.randn((1, 2, 5, 8), generator=generator, device=device, dtype=query_dtype)
                         for _ in range(3)]
    mask = torch.zeros(5, 5, device=device, dtype=mask_dtype)
    mask[:, -1] = float("-inf")
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask.to(query_dtype))
    actual = reference_sdpa(query, key, value, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == query_dtype and torch.isfinite(actual).all()
    assert mask.dtype == mask_dtype


def test_boolean_mask_keeps_true_means_attend_semantics():
    query = torch.zeros(1, 1, 1, 4)
    key = torch.zeros(1, 1, 2, 4)
    value = torch.tensor([[[[2., 2., 2., 2.], [100., 100., 100., 100.]]]])
    mask = torch.tensor([[True, False]])
    actual = reference_sdpa(query, key, value, attn_mask=mask)
    assert torch.equal(actual, torch.full_like(actual, 2.))
    assert torch.equal(mask, torch.tensor([[True, False]]))


@pytest.mark.parametrize("device", DEVICES)
def test_native_bf16_mask_path_retains_its_values(device):
    generator = torch.Generator(device=device).manual_seed(83)
    query, key, value = [torch.randn((1, 2, 7, 8), generator=generator, device=device).bfloat16()
                         for _ in range(3)]
    mask = torch.zeros(7, 7, device=device, dtype=torch.bfloat16)
    mask[:, -2:] = torch.finfo(torch.bfloat16).min
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    actual = reference_sdpa(query, key, value, attn_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_text_attention_survives_backend_upcasting_qkv(monkeypatch):
    torch.manual_seed(84)
    linear = lambda width: torch.nn.Linear(16, width, bias=False, dtype=torch.bfloat16)
    module = SimpleNamespace(num_heads=2, num_kv_heads=1, head_dim=8, inner_size=16,
        q_proj=linear(16), k_proj=linear(8), v_proj=linear(8), o_proj=linear(16),
        q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity())
    hidden = torch.randn(1, 5, 16).bfloat16()
    mask = torch.zeros(5, 5, dtype=torch.bfloat16)
    mask[:, -1] = torch.finfo(torch.bfloat16).min
    freqs = (torch.ones(1, 1, 5, 8), torch.zeros(1, 1, 5, 4), torch.zeros(1, 1, 5, 4))
    original = torch.nn.functional.scaled_dot_product_attention
    seen = []

    def upcast(query, key, value, **kwargs):
        seen.append(kwargs["attn_mask"].dtype)
        return original(query.float(), key.float(), value.float(), **kwargs).to(query.dtype)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", upcast)
    actual, cache = source_attention(module, hidden, mask, freqs)
    assert actual.shape == hidden.shape and actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all() and cache is None
    assert seen == [torch.bfloat16, torch.float32]


@pytest.mark.parametrize("mask", [None, torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, 2, dtype=torch.bfloat16)])
def test_unrelated_backend_failure_is_not_retried(monkeypatch, mask):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("backend dispatch failed")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fail)
    query = torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="backend dispatch failed"):
        reference_sdpa(query, query, query, attn_mask=mask)
    assert len(calls) == 1
