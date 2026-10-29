import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
import torch

from bluepixel import execution
from bluepixel.identity import file_stamp, mark_loaded
from bluepixel.tasks import inference_config, plan_tasks


@pytest.fixture
def batch(tmp_path, monkeypatch):
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"image_name": "input.png", "edit_prompt": "edit"}) + "\n")
    paths = {label: tmp_path / label for label in ("diffusion", "text_encoder", "vae")}
    model = SimpleNamespace(bluepixel_load_dtype="default")
    clip = SimpleNamespace()
    vae = SimpleNamespace(first_stage_model=SimpleNamespace(autoencoder=SimpleNamespace(use_tiling=True)))
    components = dict(zip(paths, (model, clip, vae)))
    for label, path in paths.items():
        path.write_bytes(label.encode())
        mark_loaded(components[label], path, file_stamp(path))
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in paths.items()}
    config = inference_config({}, hashes)
    tasks = plan_tasks(source, [42], config)
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    mm = ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.InterruptProcessingException = type("InterruptProcessingException", (Exception,), {})
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    calls = []

    def sample(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 2, 2, 3), {}, {"sampling_stubbed": True}

    monkeypatch.setattr(execution, "run_edit", sample)
    return SimpleNamespace(root=tmp_path, output=tmp_path / "output", paths=paths,
        components=components, config=config, tasks=tasks, calls=calls)


def run(batch, settings=None):
    return execution.run_tasks(batch.tasks, batch.output, *batch.components.values(),
        batch.config["settings"] if settings is None else settings, config=batch.config)


def assert_not_started(batch):
    assert not batch.calls
    assert not list((batch.output / "img").glob("*.png"))
    assert not list((batch.output / "_configs").glob("*.json"))


def test_matching_plan_and_loaded_components_execute(batch):
    result = run(batch)
    assert result["success"] == result["done_tasks"] == 1
    assert len(batch.calls) == 1


@pytest.mark.parametrize("label", ["diffusion", "text_encoder", "vae"])
def test_replaced_checkpoint_loaded_after_planning_is_rejected(batch, label):
    path = batch.paths[label]
    path.write_bytes(b"replaced before loading")
    mark_loaded(batch.components[label], path, file_stamp(path))
    with pytest.raises(ValueError, match="differs from the task plan"):
        run(batch)
    assert_not_started(batch)


def test_changed_runtime_settings_cannot_use_the_original_task_identity(batch):
    with pytest.raises(ValueError, match="Execution settings differ"):
        run(batch, {**batch.config["settings"], "steps": 8})
    assert_not_started(batch)


def test_equivalent_settings_are_normalized_before_execution(batch):
    result = run(batch, {"steps": 4.0, "guidance": 3, "shift": 0})
    assert result["success"] == 1
    assert type(batch.calls[0]["steps"]) is int


def test_wrong_runtime_precision_is_rejected_at_the_execution_entry(batch):
    batch.components["diffusion"].bluepixel_load_dtype = "fp32"
    with pytest.raises(ValueError, match="dtype differs"):
        run(batch)
    assert_not_started(batch)


def test_wrong_vae_tiling_is_rejected_at_the_execution_entry(batch):
    batch.components["vae"].first_stage_model.autoencoder.use_tiling = False
    with pytest.raises(ValueError, match="tiling differs"):
        run(batch)
    assert_not_started(batch)
