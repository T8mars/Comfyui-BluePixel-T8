import numpy as np
from PIL import Image
import torch
from bluepixel.execution import save_png_atomic


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
