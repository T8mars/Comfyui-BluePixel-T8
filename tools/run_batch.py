"""Run BluePixel JSONL tasks using ComfyUI's native components and sampler."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path)
    parser.add_argument("--diffusion", type=Path)
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--diffusion-dtype", choices=["default", "bf16", "fp32"], default="default")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--instruction-field", default="edit_prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--target-pixels", type=int, default=16777216)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--mu-mode", choices=["empirical", "training"], default="empirical")
    parser.add_argument("--shift", type=float, default=0)
    parser.add_argument("--noise-mode", choices=["upstream_cuda_bf16", "comfy_cpu"], default="upstream_cuda_bf16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata-file", default="metadata.jsonl")
    parser.add_argument("--lock-timeout", type=float, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.check and args.jsonl is None:
        from importlib.metadata import version, PackageNotFoundError
        packages = {}
        for name in ("torch", "diffusers", "transformers", "comfy-kitchen", "safetensors", "filelock"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        print(json.dumps({"stage": "environment", "python": sys.version,
            "packages": packages, "comfyui": str(args.comfyui) if args.comfyui else None,
            "comfyui_found": bool(args.comfyui and (args.comfyui / "main.py").is_file()),
            "models_loaded": False}), flush=True)
        return
    missing = ["--" + name.replace("_", "-") for name in
               ("comfyui", "diffusion", "text_encoder", "vae", "jsonl", "out_dir") if getattr(args, name) is None]
    if missing:
        parser.error("Required for task execution/status: " + ", ".join(missing))
    from bluepixel.artifacts import sha256_file
    from bluepixel.tasks import plan_tasks, TaskStore, inference_config
    settings = {"steps": args.steps, "guidance": args.guidance, "target_pixels": args.target_pixels,
                "width": args.width, "height": args.height, "mu_mode": args.mu_mode,
                "shift": args.shift, "noise_mode": args.noise_mode}
    config = inference_config(settings, {key: sha256_file(path) for key, path in
                 [("diffusion", args.diffusion), ("text_encoder", args.text_encoder), ("vae", args.vae)]}, args.vae_tiling,
                 args.diffusion_dtype)
    seeds = [int(item.strip()) for item in args.seeds.split(",")] if args.seeds is not None else [args.seed]
    tasks = plan_tasks(args.jsonl, seeds, config, args.instruction_field, args.limit)
    store = TaskStore(args.out_dir, args.metadata_file)
    store.validate_task_paths(tasks)
    print(json.dumps({"stage": "plan", **store.status(tasks), "settings": settings}), flush=True)
    if args.dry_run or args.check:
        return
    store.register_config(config)
    status = store.status(tasks)
    if not tasks or (not args.no_resume and status["remaining_tasks"] == 0):
        print(json.dumps({"stage": "complete", "success": 0, "failed": 0,
            "skipped": len(tasks), **status, "models_loaded": False}), flush=True)
        return
    sys.path.insert(0, str(args.comfyui.resolve(strict=True)))
    # Device selection occurs once in this dedicated process, before ComfyUI imports.
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    import comfy.options
    comfy.options.args_parsing = False
    from bluepixel.native import load_diffusion, load_clip, BluePixelVAE
    from bluepixel.execution import run_tasks
    model = load_diffusion(args.diffusion, args.diffusion_dtype)
    clip = load_clip(args.text_encoder)
    vae = BluePixelVAE(args.vae, args.vae_tiling)
    result = run_tasks(tasks, args.out_dir, model, clip, vae, settings, not args.no_resume,
                       args.metadata_file, args.lock_timeout,
                       progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True), config=config)
    print(json.dumps({"stage": "complete", **result}), flush=True)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
