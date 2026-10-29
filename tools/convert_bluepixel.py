"""Convert a local BluePixel snapshot to three independent ComfyUI model files."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--component", choices=["all", "transformer", "text_encoder", "vae"], default="all")
    parser.add_argument("--precision", choices=["int8_convrot", "original", "bf16"], default="int8_convrot")
    parser.add_argument("--text-precision", choices=["original", "int8_convrot", "bf16"], default="original")
    parser.add_argument("--group-size", type=int, choices=[64, 256], default=256)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sft-ckpt", type=Path)
    parser.add_argument("--sft-lora-ckpt", type=Path)
    parser.add_argument("--lora-scale", type=float)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.threads < 1 or args.row_chunk < 1:
        parser.error("threads and row-chunk must be positive")
    import torch
    torch.set_num_threads(args.threads)
    from bluepixel.conversion import convert_component

    def progress(event):
        if event.get("index") == 1 or event.get("index") == event.get("total") or event.get("index", 0) % max(1, args.progress_every) == 0:
            print(json.dumps(event), flush=True)

    components = ["transformer", "text_encoder", "vae"] if args.component == "all" else [args.component]
    for component in components:
        precision = args.precision if component == "transformer" else args.text_precision if component == "text_encoder" else "original"
        folder = {"transformer": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}[component]
        filename = {"transformer": f"bluepixel_turbo_{precision}.safetensors",
                    "text_encoder": f"bluepixel_qwen3_8b_{precision}.safetensors",
                    "vae": "bluepixel_dcae_f64.safetensors"}[component]
        result = convert_component(args.source, args.output / folder / filename, component,
                                   precision, args.group_size, args.row_chunk, args.overwrite,
                                   progress=progress,
                                   sft_checkpoint=args.sft_ckpt if component == "transformer" else None,
                                   peft_directory=args.sft_lora_ckpt if component == "transformer" else None,
                                   lora_scale=args.lora_scale)
        print(json.dumps({"stage": "complete", "component": component, **result["output"]}), flush=True)


if __name__ == "__main__":
    main()
