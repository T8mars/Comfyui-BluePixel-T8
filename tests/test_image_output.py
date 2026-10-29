import numpy as np
from PIL import Image
import torch
import pytest
from bluepixel.execution import save_png_atomic
from bluepixel.image import prepare_image


def test_atomic_png_roundtrip_with_metadata(tmp_path):
    values = torch.rand(32, 48, 3)
    path = tmp_path / "nested" / "output.png"
    save_png_atomic(values, path, {"seed": 42, "instruction": "换成红色"})
    with Image.open(path) as result:
        result.load()
        assert result.size == (48, 32)
        assert '"seed": 42' in result.info["bluepixel"]
        actual = torch.from_numpy(np.asarray(result).copy()).float() / 255
    assert (actual - values).abs().max() <= 1 / 255
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_output_cannot_replace_a_valid_result(tmp_path, value):
    path = tmp_path / "output.png"
    save_png_atomic(torch.zeros(3, 4, 3), path)
    before = path.read_bytes()
    pixels = torch.zeros(3, 4, 3)
    pixels[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        save_png_atomic(pixels, path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_bfloat16_image_input_and_output_match_their_float32_values(tmp_path):
    image = torch.rand(1, 256, 256, 3).bfloat16()
    actual, width, height, metadata = prepare_image(image, 65536)
    expected, expected_width, expected_height, expected_metadata = prepare_image(image.float(), 65536)
    assert torch.equal(actual, expected)
    assert (width, height, metadata) == (expected_width, expected_height, expected_metadata)
    bf16, fp32 = tmp_path / "bf16.png", tmp_path / "fp32.png"
    save_png_atomic(image[0], bf16)
    save_png_atomic(image[0].float(), fp32)
    assert bf16.read_bytes() == fp32.read_bytes()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reference_is_rejected_instead_of_becoming_black_pixels(value):
    image = torch.zeros(1, 128, 128, 3)
    image[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        prepare_image(image, 65536)
