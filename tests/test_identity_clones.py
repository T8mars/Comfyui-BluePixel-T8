import hashlib
import os
from types import SimpleNamespace

import pytest

from bluepixel.identity import file_stamp, mark_loaded, verify_loaded


class Patcher:
    """Small double of ComfyUI's attachment API; native clones are probed separately."""

    def __init__(self):
        self.attachments = {}

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def get_attachment(self, key):
        return self.attachments.get(key)

    def clone(self):
        clone = Patcher()
        clone.attachments = self.attachments.copy()
        return clone


@pytest.fixture(params=["model", "clip"])
def cloned_component(tmp_path, request):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"original weights")
    patcher = Patcher()
    component = patcher if request.param == "model" else SimpleNamespace(patcher=patcher)
    component.bluepixel_load_dtype = "bf16"
    mark_loaded(component, path, file_stamp(path))
    clone = patcher.clone()
    if request.param == "clip":
        clone = SimpleNamespace(patcher=clone)
    return path, clone, hashlib.sha256(path.read_bytes()).hexdigest()


def test_unchanged_native_clone_retains_verifiable_artifact_identity(cloned_component):
    _, clone, digest = cloned_component
    verify_loaded(clone, digest, "component", load_dtype="bf16")


def test_cloned_component_still_rejects_wrong_task_hash(cloned_component):
    _, clone, _ = cloned_component
    with pytest.raises(ValueError, match="differs from the task plan"):
        verify_loaded(clone, "0" * 64, "component")


def test_cloned_component_still_rejects_wrong_precision(cloned_component):
    _, clone, digest = cloned_component
    with pytest.raises(ValueError, match="dtype differs"):
        verify_loaded(clone, digest, "component", load_dtype="fp32")


def test_cloned_component_still_rejects_replaced_checkpoint(cloned_component):
    path, clone, _ = cloned_component
    replacement = path.with_name("replacement")
    replacement.write_bytes(b"replacement weights")
    os.replace(replacement, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="changed after loading"):
        verify_loaded(clone, digest, "component")


def test_unmarked_native_patcher_is_not_accepted():
    with pytest.raises(ValueError, match="Reload the BluePixel"):
        verify_loaded(Patcher().clone(), "0" * 64, "diffusion")
