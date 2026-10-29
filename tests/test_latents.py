import pytest
import torch
from bluepixel.latents import pack_latent, unpack_latent


def test_packed_permutation_scatter_and_scale():
    torch.manual_seed(22)
    raw = torch.randn(2, 128, 3, 5)
    packed = pack_latent(raw)
    order = torch.randperm(15)
    packed["samples"] = packed["samples"][:, order]
    packed["ids"] = packed["ids"][:, order]
    torch.testing.assert_close(unpack_latent(packed)["samples"], raw)


def test_packed_rejects_duplicate_and_reference_coordinates():
    packed = pack_latent(torch.zeros(1, 128, 2, 2))
    packed["ids"] = packed["ids"].clone()
    packed["ids"][:, 1] = packed["ids"][:, 0]
    with pytest.raises(ValueError, match="exactly once"):
        unpack_latent(packed)
    packed["ids"][:, :, 0] = 10
    with pytest.raises(ValueError, match="T=0"):
        unpack_latent(packed)


def test_source_bf16_division_is_preserved():
    scaled = torch.randn(1, 128, 2, 3).bfloat16()
    result = unpack_latent(pack_latent(scaled, "scaled_dcae_bchw"))
    assert torch.equal(result["samples"], scaled / 0.4538)
