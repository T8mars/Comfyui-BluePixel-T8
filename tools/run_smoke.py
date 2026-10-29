"""Full native editing with preverified text fixtures and saved sampler evidence."""

import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, action="append", default=[])
    parser.add_argument("--instruction", default="Change the gray sweater to burgundy red.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-pixels", type=int, default=1048576)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file, save_file
    import comfy.cli_args
    comfy.cli_args.args.use_pytorch_cross_attention = True
    comfy.cli_args.args.disable_xformers = True
    from bluepixel.native import load_diffusion, BluePixelVAE
    from bluepixel.execution import run_edit, save_png_atomic
    from bluepixel.artifacts import atomic_json, sha256_file
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data = load_file(str(args.embeddings))
    positive = [[data["instruction.embedding"], {}]]
    negative = [[data["empty.embedding"], {}]]
    model = load_diffusion(args.diffusion, dtype="bf16", disable_dynamic=True)
    vae = BluePixelVAE(args.vae, tiling=True)
    with Image.open(args.image) as source:
        image = torch.from_numpy(np.asarray(source.convert("RGB")).copy()).float().unsqueeze(0) / 255
    if args.reference_image:
        images = [image]
        for path in args.reference_image:
            with Image.open(path) as source:
                images.append(torch.from_numpy(np.asarray(source.convert("RGB")).copy()).float().unsqueeze(0) / 255)
        image = images
    evidence = {}

    def callback(index, denoised, sample, total):
        evidence[f"step_{index}_input"] = sample.detach().cpu().clone()
        print(json.dumps({"stage": "sample", "step": index + 1, "total": total,
                          "dtype": str(sample.dtype), "shape": list(sample.shape)}), flush=True)

    started = time.perf_counter()
    with torch.inference_mode():
        output, latent, metadata = run_edit(model, None, vae,
            args.instruction, image, args.steps,
            args.guidance, args.seed, args.target_pixels, args.width, args.height,
            prompt_embeds=positive, empty_embeds=negative, callback=callback)
    metadata.update(total_seconds=time.perf_counter() - started,
                    diffusion_file=args.diffusion.name, vae_file=args.vae.name,
                    precision=model.bluepixel_metadata["bluepixel.precision"],
                    embeddings_sha256=sha256_file(args.embeddings),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    output_shape=list(output.shape), finite=torch.isfinite(output).all().item())
    if not metadata["finite"]:
        raise AssertionError("Non-finite generated image")
    save_png_atomic(output[0], args.output, metadata)
    evidence["final_latent"] = latent["samples"].cpu()
    save_file(evidence, str(args.output.with_suffix(".latents.safetensors")))
    atomic_json(args.output.with_suffix(".json"), metadata)
    print(json.dumps({"stage": "complete", "output": str(args.output), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
