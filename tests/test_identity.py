import hashlib
import os
from types import SimpleNamespace

import pytest
from bluepixel.identity import file_stamp, mark_loaded, verify_loaded


def test_loaded_component_matches_its_recorded_file(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"old model")
    component = SimpleNamespace()
    mark_loaded(component, path, file_stamp(path))
    verify_loaded(component, hashlib.sha256(b"old model").hexdigest(), "diffusion")


def test_replanning_cannot_relabel_cached_weights_after_atomic_replacement(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"old model")
    component = SimpleNamespace()
    original_stat = path.stat()
    mark_loaded(component, path, file_stamp(path))
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"new model")
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="changed after loading"):
        verify_loaded(component, hashlib.sha256(b"new model").hexdigest(), "diffusion")


def test_model_changed_during_loading_is_not_bound(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"old")
    before = file_stamp(path)
    path.write_bytes(b"different size")
    with pytest.raises(ValueError, match="changed while loading"):
        mark_loaded(SimpleNamespace(), path, before)


def test_loaded_file_must_match_task_plan_hash(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights")
    component = SimpleNamespace()
    mark_loaded(component, path, file_stamp(path))
    with pytest.raises(ValueError, match="differs from the task plan"):
        verify_loaded(component, hashlib.sha256(b"other weights").hexdigest(), "text_encoder")
