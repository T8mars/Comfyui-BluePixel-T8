"""Run the pinned official inference code with preverified text embeddings.

Only the missing attention_kwargs property alias and staged text loading are
adapted. The official image, model, schedule, CFG and denoising code runs intact.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-code", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, action="append", default=[])
    parser.add_argument("--instruction", default="Change the gray sweater to burgundy red.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-pixels", type=int, default=1048576)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-vae", action="store_true", help="Offload AE after encoding and DiT before decoding")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from diffusers import AutoencoderDC, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
    from bluepixel.artifacts import atomic_json, sha256_file
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    original_code = args.reference_code.read_text(encoding="utf-8")
    callback_bug = "callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}"
    if original_code.count(callback_bug) != 1:
        raise ValueError("Pinned callback source changed; inspect the oracle patch")
    import re
    patched_code = re.sub(r"(?m)^([ \t]*)" + re.escape(callback_bug),
        lambda match: match.group(1) + "callback_state = locals()\n" + match.group(1) +
            "callback_kwargs = {k: callback_state[k] for k in callback_on_step_end_tensor_inputs}", original_code)
    import difflib
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Persist the adapted reference for inspection, then load it as a Python module.
    patched_path = args.output.with_suffix(".oracle.py")
    patched_path.write_text(patched_code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("bluepixel_official_reference", patched_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference module: {patched_path}")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    args.output.with_suffix(".oracle.patch").write_text("".join(difflib.unified_diff(
        original_code.splitlines(keepends=True), patched_code.splitlines(keepends=True),
        fromfile="vins_infer.py (pinned)", tofile="vins_infer.py (oracle runtime)")), encoding="utf-8")
    print(json.dumps({"stage": "load_vae"}), flush=True)
    vae = AutoencoderDC.from_pretrained(args.source, subfolder="vae", torch_dtype=torch.float32,
                                       local_files_only=True).eval().requires_grad_(False).to("cuda")
    vae.enable_tiling()
    print(json.dumps({"stage": "load_transformer"}), flush=True)
    transformer = Flux2Transformer2DModel.from_pretrained(args.source, subfolder="transformer",
        torch_dtype=torch.bfloat16, local_files_only=True).eval().requires_grad_(False).to("cuda")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.source, subfolder="scheduler", local_files_only=True)
    scheduler = reference.configure_inference_scheduler(scheduler, 0, "empirical")
    pipe = reference.FluxKleinPipeline(transformer, vae, scheduler, None, None, vae_scale_factor=64)
    if not hasattr(type(pipe), "attention_kwargs"):
        type(pipe).attention_kwargs = property(lambda instance: instance._attention_kwargs)
    data = load_file(str(args.embeddings))

    def cached_encode(self, prompt=None, num_images_per_prompt=1, **kwargs):
        key = "empty.embedding" if prompt == "" else "instruction.embedding"
        embeddings = data[key].to("cuda", torch.bfloat16).repeat_interleave(num_images_per_prompt, dim=0)
        return embeddings, reference._prepare_text_ids(embeddings).to("cuda")

    pipe.encode_prompt = types.MethodType(cached_encode, pipe)
    if args.stage_vae:
        source_encode, source_decode = vae.encode, vae.decode

        def staged_encode(*values, **kwargs):
            vae.to("cuda")
            result = source_encode(*values, **kwargs)
            vae.to("cpu")
            torch.cuda.empty_cache()
            return result

        def staged_decode(*values, **kwargs):
            transformer.to("cpu")
            torch.cuda.empty_cache()
            vae.to("cuda")
            return source_decode(*values, **kwargs)

        vae.encode, vae.decode = staged_encode, staged_decode
    evidence = {}

    def callback(self, index, timestep, values):
        evidence[f"step_{index}_output_packed"] = values["latents"].detach().cpu().contiguous()
        print(json.dumps({"stage": "sample", "step": index + 1}), flush=True)
        return {}

    image = Image.open(args.image).convert("RGB")
    if args.reference_image:
        image = [image] + [Image.open(path).convert("RGB") for path in args.reference_image]
    started = time.perf_counter()
    output = pipe(image=image, instruction=args.instruction,
        num_inference_steps=4, guidance_scale=3.0, target_pixels=args.target_pixels,
        width=args.width or None, height=args.height or None,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        callback_on_step_end=callback).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    save_file(evidence, str(args.output.with_suffix(".latents.safetensors")))
    metadata = {"reference_code_sha256": sha256_file(args.reference_code), "source_revision": "e9c7a641a4bcd9851eef50c1ce9b07fa44f6044c",
        "seed": args.seed, "steps": 4, "guidance": 3, "target_pixels": args.target_pixels,
        "output_size": list(output.size), "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "stage_vae": args.stage_vae,
        "instruction": args.instruction, "reference_images": 1 + len(args.reference_image),
        "patches": ["missing attention_kwargs property alias", "fix callback comprehension locals scope",
                    "preverified Qwen embeddings to avoid simultaneous full text+DiT GPU residency"]}
    atomic_json(args.output.with_suffix(".json"), metadata)
    print(json.dumps({"stage": "complete", **metadata}), flush=True)


if __name__ == "__main__":
    main()
