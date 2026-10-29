from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
from bluepixel.tasks import Task, TaskStore, fingerprint, plan_tasks


def planned_task(directory):
    image = directory / "input.png"
    image.write_bytes(b"image bytes that must be preserved")
    source = directory / "tasks.jsonl"
    source.write_text(json.dumps({"image_name": "input.png", "edit_prompt": "edit"}) + "\n")
    return plan_tasks(source, [42], {})[0]


@pytest.mark.parametrize("metadata_name", ["input.png", "tasks.jsonl"])
def test_metadata_cannot_modify_task_image_or_source_jsonl(tmp_path, metadata_name):
    task = planned_task(tmp_path)
    original = (tmp_path / metadata_name).read_bytes()
    store = TaskStore(tmp_path, metadata_name)
    with pytest.raises(ValueError, match="metadata must not overwrite"):
        store.record(task, "failed", error="injected")
    assert (tmp_path / metadata_name).read_bytes() == original
    assert not store.done_path(task).exists()


def test_future_task_input_is_checked_before_any_batch_output(tmp_path):
    task = planned_task(tmp_path)
    store = TaskStore(tmp_path / "output")
    output = store.output_path(task)
    output.write_bytes(b"future input")
    future = replace(task, task_id="future", image_path=str(output))
    with pytest.raises(ValueError, match="output must not overwrite"):
        store.validate_task_paths([task, future])
    assert output.read_bytes() == b"future input"


@pytest.mark.parametrize("target", ["image", "jsonl"])
def test_metadata_hardlink_to_input_is_rejected(tmp_path, target):
    task = planned_task(tmp_path)
    source = Path(task.image_path if target == "image" else task.source_jsonl)
    metadata = tmp_path / "log.jsonl"
    os.link(source, metadata)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="metadata must not overwrite"):
        TaskStore(tmp_path, metadata.name).record(task, "failed")
    assert source.read_bytes() == original


def test_explicit_record_output_cannot_be_a_source_input(tmp_path):
    task = planned_task(tmp_path)
    store = TaskStore(tmp_path)
    original = Path(task.image_path).read_bytes()
    with pytest.raises(ValueError, match="output must not overwrite"):
        store.record(task, "success", Path(task.image_path))
    assert Path(task.image_path).read_bytes() == original


def test_explicit_output_must_not_alias_metadata(tmp_path):
    task = planned_task(tmp_path)
    store = TaskStore(tmp_path)
    store.metadata.write_bytes(b"preserve this log")
    output = tmp_path / "other-output.png"
    os.link(store.metadata, output)
    with pytest.raises(ValueError, match="metadata and output paths"):
        store.record(task, "success", output)
    assert store.metadata.read_bytes() == b"preserve this log"


def test_source_tracking_does_not_change_existing_task_identity(tmp_path):
    task = planned_task(tmp_path)
    assert task.source_jsonl == str((tmp_path / "tasks.jsonl").resolve())
    identity = {"image_sha256": task.image_sha256, "instruction": task.instruction,
                "seed": task.seed, "config_fingerprint": task.config_fingerprint}
    assert task.task_id == fingerprint(identity)
    assert Task("id", "image", "hash", "edit", 42, "config").source_jsonl is None


def test_deduplicated_tasks_keep_all_source_images_protected(tmp_path):
    image_a, image_b = tmp_path / "a.png", tmp_path / "b.png"
    original = b"identical image contents"
    image_a.write_bytes(original)
    image_b.write_bytes(original)
    source = tmp_path / "tasks.jsonl"
    source.write_text("\n".join(json.dumps({"image_name": name, "edit_prompt": "same edit"})
                                for name in ("a.png", "b.png")) + "\n")
    tasks = plan_tasks(source, [42], {})
    assert len(tasks) == 1 and tasks[0].image_path == str(image_b.resolve())
    assert set(tasks[0].input_aliases) == {str(image_a.resolve()), str(image_b.resolve())}
    store = TaskStore(tmp_path, image_a.name)
    with pytest.raises(ValueError, match="metadata must not overwrite"):
        store.record(tasks[0], "failed", error="probe")
    assert image_a.read_bytes() == image_b.read_bytes() == original
