"""Auditable BluePixel single-file conversion, including native INT8 ConvRot.

The output uses ComfyUI's int8_tensorwise/convrot metadata and row scales.
No full-model dequantization is needed at load time.
"""

from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import torch
from safetensors import safe_open

from .artifacts import StreamingSafeWriter, SourceFileSnapshot, atomic_json, sha256_file
from .config import FORMAT_VERSION, SOURCE_REPO, SOURCE_REVISION, validate_transformer_config


class TensorSource:
    """Lazy, indexed, checked access to a local snapshot component."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        indexes = list(self.directory.glob("*.safetensors.index.json"))
        if len(indexes) > 1:
            raise ValueError(f"Ambiguous shard indexes: {indexes}")
        self.map = {}
        self.stack = ExitStack()
        self.readers = {}
        self.inputs = {}
        self.aux_inputs = {}
        if indexes:
            self.aux_inputs[indexes[0].name] = SourceFileSnapshot(indexes[0])
            self.map = json.loads(indexes[0].read_text(encoding="utf-8"))["weight_map"]
            self.aux_inputs[indexes[0].name].verify_unchanged()
        else:
            files = list(self.directory.glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError(f"Expected one safetensors file or an index in {directory}")
            if not files[0].resolve().is_relative_to(self.directory):
                raise ValueError(f"Shard path escapes snapshot: {files[0].name}")
            self.inputs[files[0].name] = SourceFileSnapshot(files[0])
            with safe_open(str(files[0]), framework="pt", device="cpu") as source:
                self.map = {key: files[0].name for key in source.keys()}
            self.inputs[files[0].name].verify_unchanged()
        if not self.map:
            raise ValueError("Empty checkpoint index")
        for name in set(self.map.values()):
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError(f"Shard path escapes snapshot: {name}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing shard: {path}")
            if name not in self.inputs:
                self.inputs[name] = SourceFileSnapshot(self.directory / name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def get(self, key):
        name = self.map[key]
        self.inputs[name].verify_unchanged()
        if name not in self.readers:
            reader = self.stack.enter_context(safe_open(str(self.directory / name), framework="pt", device="cpu"))
            indexed = {k for k, filename in self.map.items() if filename == name}
            if indexed != set(reader.keys()):
                raise ValueError(f"Index and shard keys differ: {name}")
            self.inputs[name].verify_unchanged()
            self.readers[name] = reader
        return self.readers[name].get_tensor(key)

    def verify_unchanged(self):
        for snapshot in (*self.inputs.values(), *self.aux_inputs.values()):
            snapshot.verify_unchanged()


def transformer_mapping(keys):
    """Map Diffusers Flux2 names to ComfyUI names, with explicit QKV fusion."""
    targets = {}
    top = {
        "x_embedder": "img_in", "context_embedder": "txt_in",
        "time_guidance_embed.timestep_embedder.linear_1": "time_in.in_layer",
        "time_guidance_embed.timestep_embedder.linear_2": "time_in.out_layer",
        "double_stream_modulation_img.linear": "double_stream_modulation_img.lin",
        "double_stream_modulation_txt.linear": "double_stream_modulation_txt.lin",
        "single_stream_modulation.linear": "single_stream_modulation.lin",
        "norm_out.linear": "final_layer.adaLN_modulation.1",
        "proj_out": "final_layer.linear",
    }
    dual = {
        "attn.norm_q": "img_attn.norm.query_norm", "attn.norm_k": "img_attn.norm.key_norm",
        "attn.norm_added_q": "txt_attn.norm.query_norm", "attn.norm_added_k": "txt_attn.norm.key_norm",
        "attn.to_out.0": "img_attn.proj", "attn.to_add_out": "txt_attn.proj",
        "ff.linear_in": "img_mlp.0", "ff.linear_out": "img_mlp.2",
        "ff_context.linear_in": "txt_mlp.0", "ff_context.linear_out": "txt_mlp.2",
    }
    single = {"attn.norm_q": "norm.query_norm", "attn.norm_k": "norm.key_norm",
              "attn.to_qkv_mlp_proj": "linear1", "attn.to_out": "linear2"}
    for key in sorted(keys):
        base, suffix = key.rsplit(".", 1)
        rank = 0
        transform = None
        if base in top:
            target = top[base] + "." + suffix
            if base == "norm_out.linear":
                transform = "swap_scale_shift"
        else:
            match = re.fullmatch(r"(single_transformer_blocks|transformer_blocks)\.(\d+)\.(.+)", base)
            if not match:
                raise ValueError(f"Unmapped BluePixel transformer key: {key}")
            kind, index, tail = match.groups()
            if kind == "single_transformer_blocks":
                if tail not in single:
                    raise ValueError(f"Unmapped single block key: {key}")
                target = f"single_blocks.{index}.{single[tail]}.{suffix}"
            else:
                qkv = {"attn.to_q": ("img", 0), "attn.to_k": ("img", 1), "attn.to_v": ("img", 2),
                       "attn.add_q_proj": ("txt", 0), "attn.add_k_proj": ("txt", 1),
                       "attn.add_v_proj": ("txt", 2)}
                if tail in qkv:
                    stream, rank = qkv[tail]
                    target = f"double_blocks.{index}.{stream}_attn.qkv.{suffix}"
                elif tail in dual:
                    target = f"double_blocks.{index}.{dual[tail]}.{suffix}"
                else:
                    raise ValueError(f"Unmapped dual block key: {key}")
        targets.setdefault(target, []).append((rank, key, transform))
    for target, parts in targets.items():
        if ".qkv." in target:
            if sorted(rank for rank, _, _ in parts) != [0, 1, 2]:
                raise ValueError(f"Incomplete QKV group: {target}")
        elif len(parts) != 1:
            raise ValueError(f"Duplicate target: {target}")
        parts.sort()
    return targets


def expected_transformer_shapes(config):
    hidden = config["attention_head_dim"] * config["num_attention_heads"]
    mlp = int(hidden * config.get("mlp_ratio", 3.0))
    channels = config["in_channels"]
    expected = {
        "img_in.weight": (hidden, channels),
        "txt_in.weight": (hidden, config["joint_attention_dim"]),
        "time_in.in_layer.weight": (hidden, 256), "time_in.out_layer.weight": (hidden, hidden),
        "double_stream_modulation_img.lin.weight": (6 * hidden, hidden),
        "double_stream_modulation_txt.lin.weight": (6 * hidden, hidden),
        "single_stream_modulation.lin.weight": (3 * hidden, hidden),
        "final_layer.adaLN_modulation.1.weight": (2 * hidden, hidden),
        "final_layer.linear.weight": (config.get("out_channels") or channels, hidden),
    }
    for index in range(config["num_layers"]):
        for stream in ("img", "txt"):
            prefix = f"double_blocks.{index}.{stream}"
            expected[prefix + "_attn.qkv.weight"] = (3 * hidden, hidden)
            expected[prefix + "_attn.proj.weight"] = (hidden, hidden)
            expected[prefix + "_attn.norm.query_norm.weight"] = (config["attention_head_dim"],)
            expected[prefix + "_attn.norm.key_norm.weight"] = (config["attention_head_dim"],)
            expected[prefix + "_mlp.0.weight"] = (2 * mlp, hidden)
            expected[prefix + "_mlp.2.weight"] = (hidden, mlp)
    for index in range(config["num_single_layers"]):
        prefix = f"single_blocks.{index}"
        expected[prefix + ".linear1.weight"] = (3 * hidden + 2 * mlp, hidden)
        expected[prefix + ".linear2.weight"] = (hidden, hidden + mlp)
        expected[prefix + ".norm.query_norm.weight"] = (config["attention_head_dim"],)
        expected[prefix + ".norm.key_norm.weight"] = (config["attention_head_dim"],)
    return expected


def quantize_convrot(tensor, group_size=256, row_chunk=256):
    """Use the very same layout implementation that native ComfyUI loads."""
    from comfy_kitchen.tensor import TensorWiseINT8Layout
    if tensor.ndim != 2 or not tensor.is_floating_point():
        raise ValueError("INT8 ConvRot expects a floating-point matrix")
    if group_size < 4 or group_size & (group_size - 1) or (group_size.bit_length() - 1) % 2 or tensor.shape[1] % group_size:
        raise ValueError(f"Invalid ConvRot group {group_size} for {tuple(tensor.shape)}")
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    codes, scales = [], []
    for start in range(0, tensor.shape[0], row_chunk):
        block = tensor[start:start + row_chunk].float().contiguous()
        if not torch.isfinite(block).all():
            raise ValueError("Non-finite source weight")
        quantized, params = TensorWiseINT8Layout.quantize(
            block, is_weight=True, per_channel=True, convrot=True,
            convrot_groupsize=group_size, stochastic_rounding=0)
        codes.append(quantized.cpu())
        scales.append(params.scale.cpu().float())
    return torch.cat(codes, dim=0), torch.cat(scales, dim=0)


def convert_component(snapshot, output, component="transformer", precision="int8_convrot",
                      group_size=256, row_chunk=256, overwrite=False, progress=None,
                      sft_checkpoint=None, peft_directory=None, lora_scale=None):
    if component not in {"transformer", "text_encoder", "vae"}:
        raise ValueError(f"Unknown component: {component}")
    if precision not in {"original", "bf16", "int8_convrot"}:
        raise ValueError(f"Unknown precision: {precision}")
    if precision == "int8_convrot" and (isinstance(group_size, bool) or not isinstance(group_size, int)
            or group_size < 64 or group_size & (group_size - 1) or (group_size.bit_length() - 1) % 2):
        raise ValueError("INT8 conversion requires a ConvRot group of at least 64 and a power of four")
    if component == "vae" and precision != "original":
        raise ValueError("Keep the DCAE-F64 in its original FP32 precision")
    snapshot, output = Path(snapshot).absolute(), Path(output).absolute()
    directory = snapshot / component
    config_file = SourceFileSnapshot(directory / "config.json")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config_file.verify_unchanged()
    if component == "transformer":
        validate_transformer_config(config)
    metadata = {
        "format": "pt", "bluepixel.format_version": FORMAT_VERSION,
        "bluepixel.component": component, "bluepixel.precision": precision,
        "bluepixel.source_repo": SOURCE_REPO, "bluepixel.source_revision": SOURCE_REVISION,
        "bluepixel.config": json.dumps(config, separators=(",", ":")),
        "bluepixel.latent_convention": "raw_dcae_bchw",
    }
    progress = progress or (lambda event: None)
    report = {"format_version": FORMAT_VERSION, "component": component, "precision": precision,
              "source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION,
              "convrot_groupsize": group_size, "lossless": precision == "original", "tensors": []}
    with TensorSource(directory) as source:
        from .adapters import SFTOverlay, PeftOverlay
        overlays = []
        auxiliary = {f"{component}/config.json": config_file,
                     **{f"{component}/{name}": value for name, value in source.aux_inputs.items()}}
        if sft_checkpoint or peft_directory:
            if component != "transformer":
                raise ValueError("SFT/LoRA overrides apply only to the diffusion transformer")
            if sft_checkpoint:
                overlays.append(SFTOverlay(sft_checkpoint, source.map))
            if peft_directory:
                overlays.append(PeftOverlay(peft_directory, source.map, lora_scale))
            report["adapters"] = [overlay.report for overlay in overlays]
            metadata["bluepixel.adapters"] = json.dumps(report["adapters"], separators=(",", ":"))
        mapping = transformer_mapping(source.map) if component == "transformer" else {
            key: [(0, key, None)] for key in source.map}
        expected = expected_transformer_shapes(config) if component == "transformer" else None
        if expected is not None and set(mapping) != set(expected):
            raise ValueError(f"Incomplete model: missing={sorted(set(expected)-set(mapping))}, unexpected={sorted(set(mapping)-set(expected))}")
        with StreamingSafeWriter(output, metadata, overwrite=overwrite) as writer:
            if component == "text_encoder":
                tokenizer_file = Path(snapshot) / "tokenizer" / "tokenizer.json"
                if not tokenizer_file.is_file():
                    raise FileNotFoundError("The snapshot must include tokenizer/tokenizer.json")
                auxiliary["tokenizer/tokenizer.json"] = SourceFileSnapshot(tokenizer_file)
                writer.add("bluepixel_tokenizer_json", torch.frombuffer(bytearray(tokenizer_file.read_bytes()), dtype=torch.uint8))
            for index, (target, parts) in enumerate(sorted(mapping.items())):
                values = []
                for _, key, transform in parts:
                    value = source.get(key)
                    for overlay in overlays:
                        value = overlay.apply(key, value)
                    if transform == "swap_scale_shift":
                        scale, shift = value.chunk(2, dim=0)
                        value = torch.cat((shift, scale), dim=0)
                    values.append(value)
                tensor = values[0] if len(values) == 1 else torch.cat(values, dim=0)
                if expected is not None and tuple(tensor.shape) != expected[target]:
                    raise ValueError(f"Incorrect shape for {target}: {tuple(tensor.shape)} != {expected[target]}")
                record = {"target": target, "sources": [p[1] for p in parts],
                          "shape": list(tensor.shape), "source_dtype": str(tensor.dtype)}
                is_matrix = tensor.ndim == 2 and target.endswith(".weight") and tensor.is_floating_point()
                layer_group = group_size if is_matrix else 0
                while layer_group >= 4 and tensor.shape[1] % layer_group:
                    layer_group //= 4
                if precision == "int8_convrot" and is_matrix and layer_group >= 64:
                    quantized, scale = quantize_convrot(tensor, layer_group, row_chunk)
                    layer = target[:-len(".weight")]
                    conf = {"format": "int8_tensorwise", "convrot": True,
                            "convrot_groupsize": layer_group}
                    writer.add(target, quantized)
                    writer.add(layer + ".weight_scale", scale)
                    writer.add(layer + ".comfy_quant", torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8))
                    record["output_dtype"] = "torch.int8"
                    record["quantization"] = conf
                else:
                    if precision == "bf16" and tensor.is_floating_point():
                        tensor = tensor.to(torch.bfloat16)
                    writer.add(target, tensor)
                    record["output_dtype"] = str(tensor.dtype)
                    if precision == "int8_convrot" and is_matrix:
                        record["kept_original_reason"] = "input width not divisible by ConvRot group"
                report["tensors"].append(record)
                progress({"stage": "tensor", "component": component, "index": index + 1,
                          "total": len(mapping), "key": target, "dtype": record["output_dtype"]})
                del tensor, values
            # Freeze provenance before the writer publishes the new artifact.
            # All payloads have already been copied into its private spool.
            source.verify_unchanged()
            for overlay in overlays:
                overlay.verify_unchanged()
            report["source_files"] = [{"path": name, "sha256": source.inputs[name].sha256()}
                                      for name in sorted(source.inputs)]
            report["source_aux_files"] = [{"path": name, "sha256": value.sha256()}
                                          for name, value in sorted(auxiliary.items())]
            source.verify_unchanged()
            for overlay in overlays:
                overlay.verify_unchanged()
            for value in auxiliary.values():
                value.verify_unchanged()
    report["output"] = {"file": Path(output).name, "bytes": Path(output).stat().st_size,
                        "sha256": sha256_file(output)}
    report_path = Path(str(output) + ".manifest.json")
    atomic_json(report_path, report)
    return report
