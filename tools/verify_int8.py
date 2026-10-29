"""Load the complete converted DiT and prove native INT8 ConvRot execution."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import comfy.cli_args
    comfy.cli_args.args.use_pytorch_cross_attention = True
    comfy.cli_args.args.disable_xformers = True
    import comfy.model_management as mm
    from comfy_kitchen.tensor import QuantizedTensor
    from safetensors.torch import load_file
    from bluepixel.native import load_diffusion
    from bluepixel.artifacts import atomic_json
    torch.set_num_threads(4)
    model = load_diffusion(args.artifact, dtype="bf16", disable_dynamic=True)
    quantized = [(name, module) for name, module in model.model.diffusion_model.named_modules()
                 if isinstance(getattr(module, "weight", None), QuantizedTensor)]
    if not quantized or any(not module.weight._params.convrot for _, module in quantized):
        raise AssertionError("Model did not retain native ConvRot quantized parameters")
    mm.load_models_gpu([model], force_full_load=True)
    data = load_file(str(args.embeddings))
    context = data.get("instruction.embedding", data.get("embedding")).to("cuda", torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(119)
    x = torch.randn(1, 128, 16, 16, device="cuda", dtype=torch.bfloat16, generator=generator)
    reference = torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator)
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        output = model.model.diffusion_model(x, torch.tensor([0.9], device="cuda"), context,
                                            ref_latents=[reference])
        torch.cuda.synchronize()
    operations = {event.key: event.count for event in profile.key_averages()
                  if "int_mm" in event.key or "int8_linear" in event.key}
    result = {"file": args.artifact.name, "quantized_linear_layers": len(quantized),
        "convrot_groups": sorted({module.weight._params.convrot_groupsize for _, module in quantized}),
        "latent_ratio": model.get_model_object("latent_format").spacial_downscale_ratio,
        "output_shape": list(output.shape), "output_finite": torch.isfinite(output).all().item(),
        "integer_operations": operations, "torch": torch.__version__,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["output_finite"] or not any("int_mm" in key for key in operations):
        raise AssertionError("Native integer GEMM execution was not proven")


if __name__ == "__main__":
    main()
