"""Verify full-weight DCAE-F64 encode/decode through the native VAE interface."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--tiling", action="store_true")
    parser.add_argument("--owned-tiling", action="store_true", help="Exercise official VAE Encode/Decode Tiled calls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import faulthandler
    faulthandler.dump_traceback_later(90, repeat=True)
    sys.path.insert(0, str(args.comfyui.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import comfy.cli_args
    comfy.cli_args.args.use_pytorch_cross_attention = True
    comfy.cli_args.args.disable_xformers = True
    from bluepixel.native import BluePixelVAE
    from bluepixel.artifacts import atomic_json
    from diffusers import AutoencoderDC
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(119)
    print(json.dumps({"stage": "load_native"}), flush=True)
    native = BluePixelVAE(args.artifact, args.tiling)
    print(json.dumps({"stage": "load_reference"}), flush=True)
    reference = AutoencoderDC.from_pretrained(args.source, subfolder="vae", local_files_only=True,
                                            torch_dtype=torch.float32).eval().requires_grad_(False).to("cuda")
    if args.tiling or args.owned_tiling:
        reference.enable_tiling()
    pixels = torch.rand(1, args.size, args.size, 3)
    print(json.dumps({"stage": "forward"}), flush=True)
    with torch.inference_mode():
        source_input = (pixels * 2 - 1).movedim(-1, 1).cuda()
        native_input = native.process_input(native.vae_encode_crop_pixels(pixels).movedim(-1, 1)).to("cuda")
        if not torch.equal(source_input, native_input):
            raise AssertionError("Native and source normalized input differ")
        expected_latent = reference.encode(source_input, return_dict=False)[0].cpu()
        actual_latent = native.encode_tiled(pixels, tile_x=512, tile_y=512, overlap=64) if args.owned_tiling else native.encode(pixels)
        direct_latent = native.first_stage_model.encode(source_input).cpu()
        repeat_latent = reference.encode(source_input, return_dict=False)[0].cpu()
        expected_pixels = reference.decode(expected_latent.cuda(), return_dict=False)[0]
        expected_pixels = ((expected_pixels + 1) / 2).clamp(0, 1).movedim(1, -1).cpu()
        actual_pixels = native.decode_tiled(expected_latent, tile_x=8, tile_y=8, overlap=1) if args.owned_tiling else native.decode(expected_latent)
        roundtrip_pixels = native.decode_tiled(actual_latent, tile_x=8, tile_y=8, overlap=1) if args.owned_tiling else native.decode(actual_latent)
        layout_pixels = reference.decode(expected_latent.contiguous().cuda(), return_dict=False)[0]
        layout_pixels = ((layout_pixels + 1) / 2).clamp(0, 1).movedim(1, -1).cpu()
    result = {"size": args.size, "tiling": args.tiling, "owned_tiling": args.owned_tiling, "latent_shape": list(actual_latent.shape),
              "encode_max_abs": (actual_latent - expected_latent).abs().max().item(),
              "direct_encode_max_abs": (direct_latent - expected_latent).abs().max().item(),
              "repeat_encode_max_abs": (repeat_latent - expected_latent).abs().max().item(),
              "input_strides": [list(source_input.stride()), list(native_input.stride())],
              "decode_max_abs": (actual_pixels - expected_pixels).abs().max().item(),
              "roundtrip_max_abs": (roundtrip_pixels - expected_pixels).abs().max().item(),
              "roundtrip_mean_abs": (roundtrip_pixels - expected_pixels).abs().mean().item(),
              "source_decode_layout_max_abs": (layout_pixels - expected_pixels).abs().max().item(),
              "roundtrip_vs_same_layout_max_abs": (roundtrip_pixels - layout_pixels).abs().max().item(),
              "latent_strides": [list(expected_latent.stride()), list(actual_latent.stride())],
              "finite": torch.isfinite(actual_pixels).all().item(), "vae_dtype": str(native.vae_dtype)}
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    torch.testing.assert_close(actual_latent, expected_latent, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_pixels, expected_pixels, atol=1e-5, rtol=1e-5)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
