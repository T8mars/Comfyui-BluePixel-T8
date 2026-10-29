"""Compare pinned source BF16 and native edit fixtures without a quality claim."""

import argparse
import json
from pathlib import Path
import sys


def metrics(actual, expected):
    import torch
    delta = actual.double() - expected.double()
    mse = delta.square().mean()
    return {"max_abs": delta.abs().max().item(), "mean_abs": delta.abs().mean().item(),
            "relative_l2": (delta.norm() / expected.double().norm().clamp_min(1e-20)).item(),
            "rmse": mse.sqrt().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source oracle PNG")
    parser.add_argument("--native", type=Path, required=True, help="Native PNG")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from bluepixel.artifacts import atomic_json
    torch.set_num_threads(4)
    with Image.open(args.source) as value:
        source_pixels = torch.from_numpy(np.asarray(value.convert("RGB")).copy()).float() / 255
    with Image.open(args.native) as value:
        native_pixels = torch.from_numpy(np.asarray(value.convert("RGB")).copy()).float() / 255
    if source_pixels.shape != native_pixels.shape:
        raise ValueError("Output image sizes differ")
    height, width = source_pixels.shape[:2]
    source = load_file(str(args.source.with_suffix(".latents.safetensors")))
    native = load_file(str(args.native.with_suffix(".latents.safetensors")))
    steps = sorted(int(key.split("_")[1]) for key in source if key.endswith("output_packed"))
    results = {}
    for index in steps:
        packed = source[f"step_{index}_output_packed"]
        expected = packed.transpose(1, 2).reshape(packed.shape[0], 128, height // 64, width // 64)
        if index == steps[-1]:
            # Source divides its stored BF16 final latent before FP32 VAE decode.
            expected = (expected / 0.4538).float()
            actual = native["final_latent"]
        else:
            actual = native[f"step_{index + 1}_input"]
        results[f"step_{index + 1}"] = metrics(actual, expected)
    results["image"] = metrics(native_pixels, source_pixels)
    results["image"]["psnr_db"] = float(-20 * np.log10(max(results["image"]["rmse"], 1e-20)))
    results.update(source=str(args.source), native=str(args.native), size=[width, height],
                   interpretation="Same prompt/seed; errors include native backend differences and optional quantization. No perceptual equivalence assertion.")
    atomic_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
