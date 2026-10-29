"""Export native or Transformers Qwen3 fixtures for full-weight comparison."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "transformers"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--comfyui", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Change the gray sweater to burgundy red.")
    parser.add_argument("--suite", action="store_true", help="Also check empty, Chinese and truncated long instructions")
    parser.add_argument("--trace", action="store_true", help="Save first-block intermediates for precision diagnostics")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--hidden-layers", default="9,18,27")
    parser.add_argument("--check-defaults-after-custom", action="store_true", help="Also test that CLIP clones do not leak custom taps/length")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    torch.set_num_threads(4)
    from safetensors.torch import save_file
    from bluepixel.config import CHAT_TEMPLATE
    cases = {"instruction": args.prompt}
    if args.suite:
        cases.update(empty="", chinese="把灰色毛衣换成酒红色，保持人物和背景不变。",
                     overlength="Change the sweater to burgundy red. " * 200)
    tensors = {}
    layers = [int(value.strip()) for value in args.hidden_layers.split(",")]
    trace_case = {"label": ""}

    def trace_model(model):
        if not args.trace:
            return
        layer = model.model.layers[0]
        for name in ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            module = layer.get_submodule(name)
            def capture(module, inputs, output, name=name):
                key = trace_case["label"] + ".trace." + name
                if key not in tensors:
                    tensors[key] = output.detach().cpu().contiguous()
            module.register_forward_hook(capture)
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        def trace_sdpa(query, key, value, *values, **kwargs):
            result = original_sdpa(query, key, value, *values, **kwargs)
            prefix = trace_case["label"] + ".trace.sdpa."
            if prefix + "query" not in tensors:
                tensors.update({prefix + name: tensor.detach().cpu().contiguous()
                                for name, tensor in (("query", query), ("key", key), ("value", value), ("output", result))})
            return result
        torch.nn.functional.scaled_dot_product_attention = trace_sdpa
    if args.backend == "native":
        if args.comfyui is None:
            parser.error("--comfyui is required for native verification")
        sys.path.insert(0, str(args.comfyui.resolve()))
        import comfy.cli_args
        comfy.cli_args.args.use_pytorch_cross_attention = True
        comfy.cli_args.args.disable_xformers = True
        from bluepixel.native import load_clip
        clip = load_clip(args.source)
        import importlib.util
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("bluepixel_verification_plugin", root / "__init__.py",
                                                    submodule_search_locations=[str(root)])
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = plugin
        spec.loader.exec_module(plugin)
        encoder_node = plugin.NODE_CLASS_MAPPINGS["BluePixelTextEncode"]()
        trace_model(clip.cond_stage_model.qwen3_8b.transformer)
        for label, prompt in cases.items():
            trace_case["label"] = label
            tokens = clip.tokenize(prompt, max_sequence_length=args.max_sequence_length)
            embeddings = encoder_node.encode(clip, prompt, args.max_sequence_length, args.hidden_layers)[0][0][0]
            ids = torch.tensor([[pair[0] for pair in tokens["qwen3_8b"][0]]], dtype=torch.long)
            prefix = label + "." if args.suite else ""
            tensors.update({prefix + "embedding": embeddings.cpu(), prefix + "input_ids": ids})
        if args.check_defaults_after_custom:
            for label, prompt in cases.items():
                trace_case["label"] = "default." + label
                default_embeddings = encoder_node.encode(clip, prompt, 512, "9,18,27")[0][0][0]
                tensors["default." + label + ".embedding"] = default_embeddings.cpu()
    else:
        from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
        tokenizer = Qwen2TokenizerFast.from_pretrained(args.source / "tokenizer", local_files_only=True)
        model = Qwen3ForCausalLM.from_pretrained(args.source / "text_encoder", dtype=torch.bfloat16,
            local_files_only=True, attn_implementation="sdpa").eval().to("cuda")
        trace_model(model)
        for label, prompt in cases.items():
            trace_case["label"] = label
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if text != CHAT_TEMPLATE.format(prompt):
                raise ValueError("Embedded chat template differs from source")
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)
            with torch.inference_mode():
                output = model(**{k: v.to("cuda") for k, v in inputs.items()},
                               output_hidden_states=True, use_cache=False)
            embeddings = torch.cat([output.hidden_states[index] for index in layers], dim=-1)
            prefix = label + "." if args.suite else ""
            tensors.update({prefix + "embedding": embeddings.cpu(), prefix + "input_ids": inputs["input_ids"],
                            prefix + "attention_mask": inputs["attention_mask"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    print(json.dumps({"backend": args.backend, "output": str(args.output),
        "cases": list(cases), "shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
        "max_sequence_length": args.max_sequence_length, "hidden_layers": layers,
        "defaults_after_custom": args.check_defaults_after_custom,
        "finite": all(torch.isfinite(value).all().item() for value in tensors.values())}), flush=True)


if __name__ == "__main__":
    main()
