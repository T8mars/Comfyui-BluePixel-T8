import json
import os
from pathlib import Path

import pytest
from bluepixel.tasks import TaskStore, plan_tasks


def plan_and_store(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"image contents")
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"image_name": str(image), "edit_prompt": "edit"}) + "\n")
    store = TaskStore(tmp_path / "output")
    return source, plan_tasks(source, [42], {})[0], store


def internal_target(store, task, kind):
    return {
        "done": store.done_path(task),
        "task_lock": store.directory / "_locks" / (task.task_id + ".lock"),
        "owner": store.directory / "_locks" / (task.task_id + ".owner.json"),
        "metadata_lock": store.directory / "_locks" / "metadata.lock",
        "config": store.directory / "_configs" / (task.config_fingerprint + ".json"),
    }[kind]


@pytest.mark.parametrize("kind", ["done", "task_lock", "owner", "metadata_lock", "config"])
def test_source_jsonl_cannot_be_an_internal_write_target(tmp_path, kind):
    source, task, store = plan_and_store(tmp_path)
    target = internal_target(store, task, kind)
    original = source.read_bytes()
    target.write_bytes(original)
    replanned = plan_tasks(target, [42], {})[0]
    assert replanned.task_id == task.task_id
    with pytest.raises(ValueError, match="internal state must not overwrite"):
        store.validate_task_paths([replanned])
    assert target.read_bytes() == original


@pytest.mark.parametrize("kind", ["task_lock", "owner", "metadata_lock"])
def test_claim_rejects_internal_input_alias_before_acquiring_lock(tmp_path, kind):
    _, task, store = plan_and_store(tmp_path)
    target = internal_target(store, task, kind)
    original = Path(task.image_path).read_bytes()
    os.link(task.image_path, target)
    with pytest.raises(ValueError, match="internal state must not overwrite"):
        with store.claim(task):
            pytest.fail("Collision must fail before claiming the task")
    assert target.read_bytes() == Path(task.image_path).read_bytes() == original


def test_record_protects_source_from_done_path_overwrite(tmp_path):
    source, task, store = plan_and_store(tmp_path)
    target = store.done_path(task)
    original = source.read_bytes()
    target.write_bytes(original)
    replanned = plan_tasks(target, [42], {})[0]
    with pytest.raises(ValueError, match="internal state must not overwrite"):
        store.record(replanned, "failed", error="probe")
    assert target.read_bytes() == original
    assert not store.metadata.exists()


def test_metadata_alias_cannot_overwrite_done_record(tmp_path):
    _, task, store = plan_and_store(tmp_path)
    target = store.done_path(task)
    target.write_bytes(b"existing done record")
    os.link(target, store.metadata)
    with pytest.raises(ValueError, match="metadata must not overwrite internal state"):
        store.record(task, "failed")
    assert target.read_bytes() == b"existing done record"


def test_explicit_output_cannot_be_a_done_record(tmp_path):
    _, task, store = plan_and_store(tmp_path)
    target = store.done_path(task)
    target.write_bytes(b"image output that must not be replaced by JSON")
    with pytest.raises(ValueError, match="output must not overwrite internal state"):
        store.record(task, "success", target)
    assert target.read_bytes() == b"image output that must not be replaced by JSON"
