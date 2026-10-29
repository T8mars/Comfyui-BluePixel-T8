import pytest
import torch
from safetensors.torch import save_file

from bluepixel.artifacts import read_header
from bluepixel.conversion import convert_component


@pytest.fixture
def tiny_snapshot(tmp_path, monkeypatch):
    directory = tmp_path / "snapshot" / "transformer"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    save_file({"x_embedder.weight": torch.ones(4, 64)}, str(directory / "model.safetensors"))
    # Exercise the real conversion/writer with a tiny matrix, not a 9B allocation.
    monkeypatch.setattr("bluepixel.conversion.validate_transformer_config", lambda config: None)
    monkeypatch.setattr("bluepixel.conversion.expected_transformer_shapes",
                        lambda config: {"img_in.weight": (4, 64)})
    return directory.parent


@pytest.mark.parametrize("group", [0, -64, 16, 63, 128, True, 256.0])
def test_invalid_int8_group_is_rejected_before_touching_output(tmp_path, group):
    output = tmp_path / "existing.safetensors"
    output.write_bytes(b"existing checkpoint")
    with pytest.raises(ValueError, match="ConvRot group"):
        convert_component(tmp_path / "not-read", output, group_size=group, overwrite=True)
    assert output.read_bytes() == b"existing checkpoint"
    assert sorted(path.name for path in tmp_path.iterdir()) == [output.name]


@pytest.mark.parametrize("group", [64, 256, 1024])
def test_valid_int8_groups_preserve_width_based_fallback(tiny_snapshot, tmp_path, group):
    output = tmp_path / "converted.safetensors"
    report = convert_component(tiny_snapshot, output, group_size=group)
    header = read_header(output)
    assert header["img_in.weight"]["dtype"] == "I8"
    assert "img_in.comfy_quant" in header
    assert report["tensors"][0]["quantization"]["convrot_groupsize"] == 64
    assert report["precision"] == "int8_convrot"


def test_nonquantized_conversion_ignores_unused_group_option(tiny_snapshot, tmp_path):
    output = tmp_path / "original.safetensors"
    report = convert_component(tiny_snapshot, output, precision="original", group_size=0)
    assert read_header(output)["img_in.weight"]["dtype"] == "F32"
    assert report["lossless"] is True
