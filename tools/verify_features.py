"""Full-weight checks for non-reference editing, CFG edges, batch refs and hooks."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import faulthandler
    faulthandler.dump_traceback_later(90, repeat=True)
    sys.path.insert(0, str(args.comfyui.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    import comfy.cli_args
    comfy.cli_args.args.use_pytorch_cross_attention = True
    comfy.cli_args.args.disable_xformers = True
    from PIL import Image
    from safetensors.torch import load_file
    from bluepixel.native import load_diffusion, BluePixelVAE
    from bluepixel.execution import run_edit, save_png_atomic
    from bluepixel.sampling import seed_list_noise
    from bluepixel.artifacts import atomic_json
    torch.set_num_threads(4)
    data = load_file(str(args.embeddings))
    positive, negative = [[data["instruction.embedding"], {}]], [[data["empty.embedding"], {}]]
    print(json.dumps({"stage": "load_diffusion"}), flush=True)
    model = load_diffusion(args.diffusion, "bf16")
    print(json.dumps({"stage": "load_vae"}), flush=True)
    vae = BluePixelVAE(args.vae, True)
    settings = dict(model=model, clip=None, vae=vae, instruction="Change the gray sweater to burgundy red.",
        steps=2, seed=42, width=512, height=512, target_pixels=262144,
        prompt_embeds=positive, empty_embeds=negative)
    records = {}
    with torch.inference_mode():
        print(json.dumps({"stage": "no_reference"}), flush=True)
        zero_image, zero_latent, zero_metadata = run_edit(**settings, guidance=0)
        one_image, one_latent, _ = run_edit(**settings, guidance=1)
        if not torch.equal(zero_latent["samples"], one_latent["samples"]):
            raise AssertionError("CFG<=1 source conditional-only paths differ")
        save_png_atomic(zero_image[0], args.output.parent / "text_only.png", zero_metadata)
        records["no_reference_cfg_0_1"] = {"latent_equal": True, "shape": list(zero_image.shape),
                                           "finite": torch.isfinite(zero_image).all().item()}
        images = []
        for path in args.images:
            with Image.open(path) as image:
                images.append(torch.from_numpy(np.asarray(image.convert("RGB")).copy()).float().unsqueeze(0) / 255)
        callbacks = []
        print(json.dumps({"stage": "references_batch_hooks"}), flush=True)
        def step_end(controller, index, time, values):
            callbacks.append({"step": index, "packed_shape": list(values["latents"].shape),
                              "time": time.item()})
            update = {"prompt_embeds": negative[0][0]}
            if index == 1:
                update["latents"] = torch.zeros_like(values["latents"])
            return update
        noise = seed_list_noise((2, 128, 8, 8), [42, 43], "cuda")
        image, latent, metadata = run_edit(**settings, image=images, batch_size=2, guidance=3,
            mu_mode="training", initial_noise=noise, callback_on_step_end=step_end,
            callback_on_step_end_tensor_inputs=("latents", "prompt_embeds"))
        if torch.count_nonzero(latent["samples"]):
            raise AssertionError("Final step callback latent replacement was lost")
        if len(callbacks) != 2 or image.shape != (2, 512, 512, 3) or not torch.isfinite(image).all():
            raise AssertionError("Reference/batch/callback execution failed")
        records["two_references_batch_two_training_callback"] = {"callbacks": callbacks,
            "final_latent_zero": True, "output_shape": list(image.shape), "metadata": metadata}
        try:
            bad = {**settings, "prompt_embeds": [[torch.zeros(1, 512, 4096), {}]]}
            run_edit(**bad)
        except ValueError as error:
            records["invalid_embedding_rejected"] = str(error)
        else:
            raise AssertionError("Malformed custom conditioning was accepted")
    atomic_json(args.output, records)
    print(json.dumps(records, indent=2), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
