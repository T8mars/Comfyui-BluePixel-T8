"""Compare an independent tiny Flux2 forward with ComfyUI's native backbone.

This checks conversion math without needing the large checkpoint. It does not
replace full BluePixel image/weight/INT8 qualification.
"""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve(strict=True)))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import comfy.cli_args
    comfy.cli_args.args.use_pytorch_cross_attention = True
    comfy.cli_args.args.disable_xformers = True
    import comfy.ops
    from comfy.ldm.flux.model import Flux
    from diffusers import Flux2Transformer2DModel
    from bluepixel.conversion import transformer_mapping
    torch.set_num_threads(4)
    torch.manual_seed(119)
    reference = Flux2Transformer2DModel(in_channels=128, num_layers=1,
        num_single_layers=1, num_attention_heads=1, attention_head_dim=128,
        joint_attention_dim=384, axes_dims_rope=(32, 32, 32, 32),
        guidance_embeds=False).eval().to(args.device)
    source = reference.state_dict()
    mapped = {}
    for target, parts in transformer_mapping(source).items():
        values = []
        for _, key, transform in parts:
            tensor = source[key]
            if transform == "swap_scale_shift":
                scale, shift = tensor.chunk(2)
                tensor = torch.cat((shift, scale))
            values.append(tensor)
        mapped[target] = values[0] if len(values) == 1 else torch.cat(values)
    native = Flux(in_channels=128, out_channels=128, vec_in_dim=None,
        context_in_dim=384, hidden_size=128, mlp_ratio=3.0, num_heads=1,
        depth=1, depth_single_blocks=1, axes_dim=[32, 32, 32, 32], theta=2000,
        patch_size=1, qkv_bias=False, guidance_embed=False, txt_ids_dims=[3],
        global_modulation=True, mlp_silu_act=True, ops_bias=False,
        default_ref_method="index", ref_index_scale=10.0,
        dtype=torch.float32, device=args.device, operations=comfy.ops.disable_weight_init).eval()
    native.load_state_dict(mapped, strict=True)
    x = torch.randn(1, 128, 4, 4, device=args.device)
    ref = torch.randn(1, 128, 3, 2, device=args.device)
    context = torch.randn(1, 7, 384, device=args.device)
    timestep = torch.tensor([0.9], device=args.device)

    def image_tokens(value, t):
        height, width = value.shape[-2:]
        coordinates = torch.cartesian_prod(torch.tensor([t], device=args.device),
            torch.arange(height, device=args.device), torch.arange(width, device=args.device),
            torch.tensor([0], device=args.device)).float()
        return value.flatten(2).transpose(1, 2), coordinates

    target_tokens, target_ids = image_tokens(x, 0)
    ref_tokens, ref_ids = image_tokens(ref, 10)
    text_ids = torch.zeros(context.shape[1], 4, device=args.device)
    text_ids[:, 3] = torch.arange(context.shape[1], device=args.device)
    with torch.inference_mode():
        expected = reference(hidden_states=torch.cat((target_tokens, ref_tokens), dim=1),
            encoder_hidden_states=context, timestep=timestep, guidance=torch.ones_like(timestep),
            img_ids=torch.cat((target_ids, ref_ids)), txt_ids=text_ids, return_dict=False)[0]
        expected = expected[:, :16].transpose(1, 2).reshape_as(x)
        actual = native(x, timestep, context, ref_latents=[ref])
    difference = (actual - expected).abs()
    result = {"test": "tiny_flux2_diffusers_to_comfyui_forward", "device": args.device,
              "source_keys": len(source), "target_keys": len(mapped),
              "max_abs_error": difference.max().item(), "mean_abs_error": difference.mean().item(),
              "relative_l2": (torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)).item()}
    print(json.dumps(result, indent=2), flush=True)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    main()
