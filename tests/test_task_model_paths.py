import json
import os
from pathlib import Path

import pytest

from bluepixel.tasks import TaskStore, fingerprint, plan_tasks


@pytest.fixture
def task(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"input image")
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"image_name": str(image), "edit_prompt": "edit"}))
    return plan_tasks(source, [42], {})[0]


def test_metadata_cannot_overwrite_protected_model(tmp_path, task):
    directory = tmp_path / "models"
    directory.mkdir()
    model = directory / "vae.safetensors"
    model.write_bytes(b"model weights")
    store = TaskStore(directory, model.name, protected_paths=[model])
    with pytest.raises(ValueError, match="metadata must not overwrite.*model file"):
        store.record(task, "failed", error="probe")
    assert model.read_bytes() == b"model weights"


def test_planned_output_cannot_overwrite_protected_model(tmp_path, task):
    directory = tmp_path / "output"
    model = directory / "img" / f"{task.task_id}_seed{task.seed}.png"
    store = TaskStore(directory, protected_paths=[model])
    model.write_bytes(b"model weights")
    with pytest.raises(ValueError, match="output must not overwrite.*model file"):
        store.validate_task_paths([task])
    assert model.read_bytes() == b"model weights"


def test_explicit_record_output_cannot_be_protected_model(tmp_path, task):
    model = tmp_path / "output" / "checkpoint.safetensors"
    store = TaskStore(model.parent, protected_paths=[model])
    model.write_bytes(b"model weights")
    with pytest.raises(ValueError, match="output must not overwrite.*model file"):
        store.record(task, "success", model)
    assert model.read_bytes() == b"model weights"
    assert not store.metadata.exists()


@pytest.mark.parametrize("kind", ["done", "task_lock", "owner", "metadata_lock", "config"])
def test_claim_protects_models_before_internal_writes(tmp_path, task, kind):
    directory = tmp_path / "output"
    model = {
        "done": directory / "_done" / (task.task_id + ".json"),
        "task_lock": directory / "_locks" / (task.task_id + ".lock"),
        "owner": directory / "_locks" / (task.task_id + ".owner.json"),
        "metadata_lock": directory / "_locks" / "metadata.lock",
        "config": directory / "_configs" / (task.config_fingerprint + ".json"),
    }[kind]
    store = TaskStore(directory, protected_paths=[model])
    model.write_bytes(b"model weights")
    with pytest.raises(ValueError, match="internal state must not overwrite.*model file"):
        with store.claim(task):
            pytest.fail("A model collision must fail before claiming")
    assert model.read_bytes() == b"model weights"


@pytest.mark.parametrize("target", ["metadata", "output"])
def test_hardlink_aliases_of_models_remain_protected(tmp_path, task, target):
    model = tmp_path / "checkpoint.safetensors"
    model.write_bytes(b"model weights")
    store = TaskStore(tmp_path / "output", protected_paths=[model])
    alias = store.metadata if target == "metadata" else store.output_path(task)
    os.link(model, alias)
    with pytest.raises(ValueError, match="must not overwrite.*model file"):
        store.validate_task_paths([task])
    assert alias.read_bytes() == model.read_bytes() == b"model weights"


def test_protected_file_identity_is_refreshed_before_each_validation(tmp_path, task):
    model = tmp_path / "checkpoint.safetensors"
    model.write_bytes(b"model weights")
    store = TaskStore(tmp_path / "output", protected_paths=[model])
    store.metadata.write_bytes(b"log that must stay intact")
    store.validate_task_paths([task])
    model.unlink()
    os.link(store.metadata, model)
    with pytest.raises(ValueError, match="metadata must not overwrite.*model file"):
        store.validate_task_paths([task])
    assert store.metadata.read_bytes() == b"log that must stay intact"


def test_relative_protected_paths_survive_working_directory_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "checkpoint.safetensors"
    model.write_bytes(b"model weights")
    store = TaskStore(tmp_path, model.name, protected_paths=[Path(model.name)])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError, match="metadata must not overwrite.*model file"):
        store.validate_task_paths([])
    assert model.read_bytes() == b"model weights"


def test_register_config_protects_model_target_even_with_empty_plan(tmp_path):
    config = {"fixture": "config"}
    model = tmp_path / "output" / "_configs" / (fingerprint(config) + ".json")
    store = TaskStore(model.parent.parent, protected_paths=[model])
    model.write_bytes(b"model weights")
    store.validate_task_paths([])
    with pytest.raises(ValueError, match="output must not overwrite.*model file"):
        store.register_config(config)
    assert model.read_bytes() == b"model weights"


def test_unrelated_protected_model_does_not_block_config_storage(tmp_path):
    model = tmp_path / "checkpoint.safetensors"
    model.write_bytes(b"model weights")
    store = TaskStore(tmp_path / "output", protected_paths=[model])
    config = {"fixture": "config"}
    identity = store.register_config(config)
    assert identity == fingerprint(config)
    assert json.loads((store.directory / "_configs" / (identity + ".json")).read_text()) == {
        "config_fingerprint": identity, "config": config}
    assert model.read_bytes() == b"model weights"
