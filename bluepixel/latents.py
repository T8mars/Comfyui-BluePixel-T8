"""Explicit source packed tokens and native spatial latent boundaries."""

import torch
from .config import LATENT_CHANNELS, LATENT_SCALE


def pack_latent(samples, convention="raw_dcae_bchw"):
    if samples.ndim != 4 or samples.shape[1] != LATENT_CHANNELS:
        raise ValueError("Expected native [B,128,H,W] samples")
    if convention not in {"raw_dcae_bchw", "scaled_dcae_bchw"}:
        raise ValueError("Unknown latent scale convention")
    batch, channels, height, width = samples.shape
    scaled = samples * LATENT_SCALE if convention == "raw_dcae_bchw" else samples
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
    return {"samples": scaled.flatten(2).transpose(1, 2).contiguous(),
            "ids": ids.unsqueeze(0).expand(batch, -1, -1).to(samples.device),
            "width": width * 64, "height": height * 64, "convention": "scaled_dcae_blc"}


def unpack_latent(packed):
    samples, ids = packed["samples"], packed["ids"]
    width, height = int(packed["width"]) // 64, int(packed["height"]) // 64
    if packed.get("convention") != "scaled_dcae_blc":
        raise ValueError("Source packed samples must declare scaled_dcae_blc")
    if min(width, height) < 1 or packed["width"] % 64 or packed["height"] % 64:
        raise ValueError("Packed output dimensions must be positive multiples of64")
    if samples.ndim != 3 or samples.shape[1:] != (width * height, LATENT_CHANNELS):
        raise ValueError("Packed samples and output dimensions disagree")
    if ids.shape != (samples.shape[0], width * height, 4) or not torch.equal(ids, ids.round()):
        raise ValueError("Expected integral source coordinates [B,L,4]")
    if not torch.all((ids[..., 0] == 0) & (ids[..., 3] == 0)):
        raise ValueError("Only generated-image T=0,L=0 coordinates are accepted")
    if not torch.all((ids[..., 1] >= 0) & (ids[..., 1] < height) & (ids[..., 2] >= 0) & (ids[..., 2] < width)):
        raise ValueError("Packed coordinates lie outside the output image")
    indices = (ids[..., 1].long() * width + ids[..., 2].long()).to(samples.device)
    expected = torch.arange(width * height, device=samples.device).expand(samples.shape[0], -1)
    if not torch.equal(indices.sort(dim=1).values, expected):
        raise ValueError("Packed coordinates must cover every spatial position exactly once")
    spatial = torch.empty(samples.shape[0], width * height, LATENT_CHANNELS, device=samples.device, dtype=samples.dtype)
    spatial.scatter_(1, indices.unsqueeze(-1).expand_as(samples), samples)
    # Preserve the source dtype at its scale division boundary.
    spatial = (spatial / LATENT_SCALE).transpose(1, 2).reshape(samples.shape[0], LATENT_CHANNELS, height, width)
    return {"samples": spatial, "downscale_ratio_spacial": 64, "bluepixel_latent_convention": "raw_dcae_bchw"}
