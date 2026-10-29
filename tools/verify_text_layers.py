"""Locate Qwen3 precision differences with source/native layers sharing weights."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import comfy.cli_args
    comfy.cli_args.args.use_pytorch_cross_attention = True
    comfy.cli_args.args.disable_xformers = True
    import comfy.model_management as mm
    from transformers import Qwen3Config, Qwen3Model
    from safetensors.torch import load_file, save_file
    from bluepixel.native import load_clip
    from bluepixel.artifacts import atomic_json
    torch.set_num_threads(4)
    clip = load_clip(args.artifact)
    clip.load_model()
    native = clip.cond_stage_model.qwen3_8b.transformer
    config = Qwen3Config.from_pretrained(args.source / "text_encoder", local_files_only=True)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        reference = Qwen3Model(config)
    state = native.model.state_dict()
    reference.load_state_dict({key: state[key] for key in reference.state_dict()}, strict=True, assign=True)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    reference.rotary_emb = Qwen3RotaryEmbedding(config, device="cuda")
    reference.eval()
    data = load_file(str(args.fixtures))
    result = {}
    details, detail_metadata = {}, {}
    for label, prompt in (("chinese", "把灰色毛衣换成酒红色，保持人物和背景不变。"),
                          ("overlength", "Change the sweater to burgundy red. " * 200)):
        expected, actual, handles = {}, {}, []
        def capture(target, index):
            def hook(module, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                target[index] = value.detach().cpu().contiguous()
            return hook
        for index, layer in enumerate(reference.layers):
            handles.append(layer.register_forward_hook(capture(expected, index)))
        for index, layer in enumerate(native.model.layers):
            handles.append(layer.register_forward_hook(capture(actual, index)))
        previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        try:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
            with torch.inference_mode():
                reference(input_ids=data[label + ".input_ids"].to("cuda"),
                    attention_mask=data[label + ".attention_mask"].to("cuda"), use_cache=False)
        finally:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
        with torch.inference_mode():
            clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        for handle in handles:
            handle.remove()
        differences = []
        for index in expected:
            x, y = expected[index].float(), actual[index].float()
            delta = y - x
            differences.append({"layer": index, "max_abs": delta.abs().max().item(),
                "relative_l2": (delta.norm() / x.norm()).item(), "different": (delta != 0).sum().item()})
        result[label] = differences
        first = next((value for value in differences if value["different"]), None)
        print(json.dumps({"case": label, "first_difference": first}), flush=True)
        if first is not None:
            layer_index = first["layer"]
            trace_state = {"backend": "source", "index": 0}
            original_sdpa = torch.nn.functional.scaled_dot_product_attention
            def traced_sdpa(query, key, value, *values, **kwargs):
                output = original_sdpa(query, key, value, *values, **kwargs)
                if trace_state["index"] == layer_index:
                    prefix = label + "." + trace_state["backend"] + ".sdpa."
                    for name, tensor in (("query", query), ("key", key), ("value", value), ("output", output)):
                        details[prefix + name] = tensor.detach().cpu().contiguous()
                    mask = kwargs.get("attn_mask")
                    if mask is not None:
                        details[prefix + "mask"] = mask.detach().cpu().contiguous()
                    detail_metadata[prefix] = {"strides": [list(tensor.stride()) for tensor in (query, key, value)],
                        "kwargs": {key: str(value) for key, value in kwargs.items() if key != "attn_mask"}}
                trace_state["index"] += 1
                return output
            torch.nn.functional.scaled_dot_product_attention = traced_sdpa
            hooks = []
            for backend, layer in (("source", reference.layers[layer_index]), ("native", native.model.layers[layer_index])):
                for name in ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_norm", "self_attn.k_norm",
                             "self_attn.o_proj", "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                    def capture_detail(module, inputs, output, key=label + "." + backend + "." + name):
                        details[key] = output.detach().cpu().contiguous()
                    hooks.append(layer.get_submodule(name).register_forward_hook(capture_detail))
            try:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
                with torch.inference_mode():
                    reference(input_ids=data[label + ".input_ids"].to("cuda"),
                        attention_mask=data[label + ".attention_mask"].to("cuda"), use_cache=False)
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
                trace_state.update(backend="native", index=0)
                with torch.inference_mode():
                    clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            finally:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
                torch.nn.functional.scaled_dot_product_attention = original_sdpa
                for handle in hooks:
                    handle.remove()
    atomic_json(args.output, result)
    save_file(details, str(args.output.with_suffix(".details.safetensors")))
    atomic_json(args.output.with_suffix(".details.json"), detail_metadata)


if __name__ == "__main__":
    main()
