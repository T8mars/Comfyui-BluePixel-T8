import hashlib
from io import BytesIO
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from bluepixel.execution import _load_task_image


def encoded(color, mode="RGB"):
    buffer = BytesIO()
    Image.new(mode, (2, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def task_for(path, data):
    path.write_bytes(data)
    return SimpleNamespace(image_path=str(path), image_sha256=hashlib.sha256(data).hexdigest())


def test_replacement_between_verification_and_decode_cannot_change_used_pixels(tmp_path, monkeypatch):
    path = tmp_path / "input.png"
    red, blue = encoded("red"), encoded("blue")
    task = task_for(path, red)
    real_open = Image.open

    def replace_before_decode(value, *args, **kwargs):
        path.write_bytes(blue)
        return real_open(value, *args, **kwargs)

    monkeypatch.setattr(Image, "open", replace_before_decode)
    result = _load_task_image(task)
    assert path.read_bytes() == blue
    assert result.shape == (1, 3, 2, 3) and result.dtype == torch.float32
    assert torch.equal(result[0, 0, 0], torch.tensor([1., 0., 0.]))


def test_image_changed_before_snapshot_is_rejected(tmp_path):
    path = tmp_path / "input.png"
    task = task_for(path, encoded("red"))
    path.write_bytes(encoded("blue"))
    with pytest.raises(ValueError, match="changed after task planning"):
        _load_task_image(task)


def test_snapshot_retains_rgb_conversion_for_grayscale_inputs(tmp_path):
    task = task_for(tmp_path / "input.png", encoded(255, "L"))
    result = _load_task_image(task)
    assert result.shape == (1, 3, 2, 3)
    assert torch.all(result == 1)
