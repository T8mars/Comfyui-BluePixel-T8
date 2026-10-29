"""Source-compatible resolution selection and RGB preprocessing."""

import math
import numpy as np
from PIL import Image
import torch


def select_size(width, height, target_pixels=0):
    if width < 1 or height < 1 or target_pixels < 0:
        raise ValueError("Image dimensions must be positive and target_pixels non-negative")
    original_width, original_height = width, height
    if target_pixels:
        scale = math.sqrt(target_pixels / (width * height))
        width, height = int(width * scale), int(height * scale)
    width_lo, width_hi = max(64, (width - 64) // 64 * 64), (width + 64) // 64 * 64
    height_lo, height_hi = max(64, (height - 64) // 64 * 64), (height + 64) // 64 * 64
    ratio = original_width / original_height
    best, best_error = (width, height), float("inf")
    for candidate_width in range(width_lo, width_hi + 1, 64):
        for candidate_height in range(height_lo, height_hi + 1, 64):
            error = abs(candidate_width / candidate_height - ratio)
            if error < best_error:
                best, best_error = (candidate_width, candidate_height), error
    return best


def prepare_image(images, target_pixels=0, output_width=0, output_height=0):
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("Expected ComfyUI IMAGE [B,H,W,3+] in the 0..1 range")
    source_height, source_width = images.shape[1:3]
    selected_width, selected_height = select_size(source_width, source_height, target_pixels)
    # The upstream Flux2ImageProcessor uses 2*64, not 64, for its second resize.
    processed_width = selected_width - selected_width % 128
    processed_height = selected_height - selected_height % 128
    if min(processed_width, processed_height) < 128:
        raise ValueError("The upstream 128-pixel processor would create an empty dimension; increase target_pixels")
    width, height = output_width or selected_width, output_height or selected_height
    if width < 64 or height < 64 or width % 64 or height % 64:
        raise ValueError("Output width and height must be positive multiples of 64")
    prepared = []
    for image in images:
        pixels = (image[..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        pil = Image.fromarray(pixels, "RGB").resize((selected_width, selected_height), Image.Resampling.LANCZOS)
        if pil.size != (processed_width, processed_height):
            pil = pil.resize((processed_width, processed_height), Image.Resampling.LANCZOS)
        prepared.append(torch.from_numpy(np.asarray(pil).copy()).float() / 255.0)
    metadata = {"source_size": [source_width, source_height], "selected_size": [selected_width, selected_height],
                "processed_reference_size": [processed_width, processed_height], "output_size": [width, height],
                "target_pixels": target_pixels, "image_seq_len": (width // 64) * (height // 64)}
    return torch.stack(prepared), width, height, metadata
