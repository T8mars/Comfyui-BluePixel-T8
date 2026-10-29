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


@pytest.mark.parametrize("scale,effective", [(None, 1), (0, 0), (0.5, 0.5), (-1, 1)])
def test_peft_merge_and_legacy_negative_scale(tmp_path, scale, effective):
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    save_file({"base_model.model.block.lora_A.weight": a, "base_model.model.block.lora_B.weight": b},
              str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "peft_type": "LORA"}))
    overlay = PeftOverlay(tmp_path, {"block.weight"}, scale)
    base = torch.randn(3, 4)
    torch.testing.assert_close(overlay.apply("block.weight", base), base + effective * 2 * (b @ a))
