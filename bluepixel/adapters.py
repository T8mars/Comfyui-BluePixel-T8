"""Full-SFT and PEFT linear-LoRA inputs applied before single-file quantization."""

import json
import math
from pathlib import Path
import re
import torch
from safetensors import safe_open
from .artifacts import sha256_file


def normalize_key(key):
    if key.startswith("base_model.model."):
        key = key[len("base_model.model."):]
    return key.replace(".base_layer.", ".")


class SFTOverlay:
    def __init__(self, path, available_keys):
        self.path = Path(path)
        self.reader = None
        if self.path.suffix == ".safetensors":
            self.reader = safe_open(str(self.path), framework="pt", device="cpu")
            keys = self.reader.keys()
        else:
            self.state = torch.load(str(self.path), map_location="cpu", weights_only=True, mmap=True)
            if "state_dict" in self.state:
                self.state = self.state["state_dict"]
            keys = self.state.keys()
        self.map = {}
        self.skipped = []
        for key in keys:
            normalized = normalize_key(key)
            if "lora_" in normalized or normalized not in available_keys:
                self.skipped.append(key)
                continue
            if normalized in self.map:
                raise ValueError(f"Duplicate SFT key after prefix normalization: {normalized}")
            self.map[normalized] = key
        if not self.map:
            raise ValueError("SFT checkpoint matches no transformer weights")
        self.report = {"file": self.path.name, "sha256": sha256_file(self.path),
                       "matched": len(self.map), "base_fallback": len(available_keys) - len(self.map),
                       "skipped": self.skipped}

    def apply(self, key, base):
        if key not in self.map:
            return base
        source_key = self.map[key]
        value = self.reader.get_tensor(source_key) if self.reader else self.state[source_key]
        if not isinstance(value, torch.Tensor) or value.shape != base.shape:
            raise ValueError(f"SFT tensor shape differs from the model: {key}")
        return value.to(base.dtype)


class PeftOverlay:
    def __init__(self, directory, available_keys, scale=None):
        directory = Path(directory)
        self.config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
        if self.config.get("peft_type", "LORA") != "LORA" or self.config.get("use_dora"):
            raise ValueError("This converter accepts linear PEFT LoRA; DoRA/other adapter types require their own merge")
        self.reader = safe_open(str(directory / "adapter_model.safetensors"), framework="pt", device="cpu")
        self.pairs = {}
        unknown = []
        for key in self.reader.keys():
            normalized = normalize_key(key)
            match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.[^.]+)?\.weight", normalized)
            if not match:
                unknown.append(key)
                continue
            module, direction = match.groups()
            target = module + ".weight"
            if target not in available_keys:
                raise ValueError(f"LoRA target does not exist in this model: {target}")
            pair = self.pairs.setdefault(target, {})
            if direction in pair:
                raise ValueError(f"Duplicate LoRA factor: {target} {direction}")
            pair[direction] = key
        if unknown or not self.pairs or any(set(pair) != {"A", "B"} for pair in self.pairs.values()):
            raise ValueError(f"Incomplete/unsupported PEFT LoRA state, unhandled keys: {unknown}")
        if scale is not None and not math.isfinite(scale):
            raise ValueError("lora_scale must be finite")
        # Preserve the source's legacy rule: negative scale leaves PEFT's default scale unchanged.
        self.multiplier = scale if scale is not None and scale >= 0 else 1.0
        self.report = {"file": "adapter_model.safetensors", "sha256": sha256_file(directory / "adapter_model.safetensors"),
                       "config_sha256": sha256_file(directory / "adapter_config.json"),
                       "scale": scale, "effective_multiplier": self.multiplier,
                       "targets": sorted(self.pairs)}

    def _pattern_value(self, name, module, default):
        patterns = self.config.get(name, {})
        # Match PEFT's get_pattern_key: an optional module prefix, then the
        # configured regex suffix, taking the first match in config order.
        for pattern, value in patterns.items():
            if re.match(rf"(.*\.)?({pattern})$", module):
                return value
        return default

    def apply(self, key, base):
        if key not in self.pairs or self.multiplier == 0:
            return base
        pair = self.pairs[key]
        a, b = self.reader.get_tensor(pair["A"]).float(), self.reader.get_tensor(pair["B"]).float()
        if base.ndim != 2 or a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0] or (b.shape[0], a.shape[1]) != tuple(base.shape):
            raise ValueError(f"LoRA dimensions do not match {key}")
        rank = a.shape[0]
        configured_rank = self._pattern_value("rank_pattern", key[:-7], self.config.get("r", rank))
        if configured_rank != rank:
            raise ValueError(f"LoRA rank differs from adapter_config: {key}")
        alpha = self._pattern_value("alpha_pattern", key[:-7], self.config.get("lora_alpha", rank))
        scale = alpha / (math.sqrt(rank) if self.config.get("use_rslora") else rank) * self.multiplier
        result = base.float().clone()
        for start in range(0, base.shape[0], 256):
            result[start:start + 256].addmm_(b[start:start + 256], a, beta=1.0, alpha=scale)
        return result.to(base.dtype)
