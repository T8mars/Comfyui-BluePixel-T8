import json
import torch
import pytest
from safetensors.torch import save_file
from bluepixel.adapters import SFTOverlay, PeftOverlay


def test_sft_prefixes_partial_base_and_shape_check(tmp_path):
    path = tmp_path / "sft.safetensors"
    save_file({"base_model.model.block.base_layer.weight": torch.ones(3, 4)}, str(path))
    overlay = SFTOverlay(path, {"block.weight", "other.weight"})
    assert overlay.report["base_fallback"] == 1
    assert torch.equal(overlay.apply("block.weight", torch.zeros(3, 4)), torch.ones(3, 4))
    with pytest.raises(ValueError, match="shape"):
        overlay.apply("block.weight", torch.zeros(4, 4))


@pytest.mark.parametrize("zip_serialization", [False, True])
def test_sft_pytorch_checkpoint_formats_are_supported(tmp_path, zip_serialization):
    path = tmp_path / "sft.pt"
    expected = torch.randn(2, 3)
    torch.save({"state_dict": {"block.weight": expected}}, path,
               _use_new_zipfile_serialization=zip_serialization)
    overlay = SFTOverlay(path, {"block.weight"})
    assert torch.equal(overlay.apply("block.weight", torch.zeros_like(expected)), expected)


@pytest.mark.parametrize("scale,effective", [(None, 1), (0, 0), (0.5, 0.5), (-1, 1)])
def test_peft_merge_and_legacy_negative_scale(tmp_path, scale, effective):
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    save_file({"base_model.model.block.lora_A.weight": a, "base_model.model.block.lora_B.weight": b},
              str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "peft_type": "LORA"}))
    overlay = PeftOverlay(tmp_path, {"block.weight"}, scale)
    base = torch.randn(3, 4)
    torch.testing.assert_close(overlay.apply("block.weight", base), base + effective * 2 * (b @ a))


@pytest.mark.parametrize("config", [
    {"r": 4, "lora_alpha": 8, "rank_pattern": {r"block\.\d+\.proj": 2},
     "alpha_pattern": {r"block\.\d+\.proj": 4}},
    {"r": 2, "lora_alpha": 2,
     "alpha_pattern": {"proj": 4, "encoder.block.0.proj": 8}},
])
def test_peft_patterns_follow_regex_suffix_and_first_match(tmp_path, config):
    module = "encoder.block.0.proj"
    a, b = torch.ones(2, 4), torch.ones(3, 2)
    save_file({f"base_model.model.{module}.lora_A.weight": a,
               f"base_model.model.{module}.lora_B.weight": b},
              str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    overlay = PeftOverlay(tmp_path, {module + ".weight"})
    # PEFT resolves rank=2 and alpha=4 in both cases, including when a later,
    # longer pattern also matches. The LoRA delta is (alpha/rank) * B @ A.
    torch.testing.assert_close(overlay.apply(module + ".weight", torch.zeros(3, 4)),
                               2 * (b @ a))
