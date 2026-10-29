import hashlib
from types import SimpleNamespace

import pytest
from bluepixel.identity import file_stamp, mark_loaded, verify_loaded
from bluepixel.tasks import INFERENCE_DEFAULTS, fingerprint, inference_config


def test_default_precision_keeps_existing_task_fingerprints():
    legacy = {"pipeline_version": 1, "settings": dict(INFERENCE_DEFAULTS),
              "model_hashes": {"diffusion": "sha"}, "vae_tiling": True}
    assert fingerprint(inference_config({}, {"diffusion": "sha"})) == fingerprint(legacy)


def test_precision_modes_cannot_share_a_resume_identity():
    configs = [inference_config({}, {"diffusion": "sha"}, diffusion_dtype=dtype)
               for dtype in ("default", "bf16", "fp32")]
    assert len({fingerprint(config) for config in configs}) == 3


def test_integral_step_values_share_one_identity():
    integer = inference_config({"steps": 4}, {})
    floating = inference_config({"steps": 4.0}, {})
    assert type(floating["settings"]["steps"]) is int
    assert fingerprint(integer) == fingerprint(floating)


def test_invalid_precision_is_rejected():
    with pytest.raises(ValueError, match="diffusion_dtype"):
        inference_config({}, {}, diffusion_dtype="fp16")


def test_same_file_with_another_loaded_precision_cannot_satisfy_plan(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"same weights")
    component = SimpleNamespace(bluepixel_load_dtype="fp32")
    mark_loaded(component, path, file_stamp(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_loaded(component, digest, "diffusion", load_dtype="fp32")
    with pytest.raises(ValueError, match="dtype differs"):
        verify_loaded(component, digest, "diffusion", load_dtype="default")
