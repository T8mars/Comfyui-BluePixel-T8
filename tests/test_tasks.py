import json
from pathlib import Path
import pytest
from bluepixel.tasks import plan_tasks, TaskStore, inference_config, fingerprint


def make_plan(tmp_path, config=None):
    image = tmp_path / "test.png"
    image.write_bytes(b"fixture image bytes")
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"image_name": "test.png", "edit_prompt": "变成红色"}, ensure_ascii=False), encoding="utf-8")
    return plan_tasks(source, [42, 43], config or {"model_hash": "model-v1", "steps": 4})


def test_content_identity_tracks_model_image_instruction_and_seed(tmp_path):
    tasks = make_plan(tmp_path)
    assert len(tasks) == 2 and tasks[0].task_id != tasks[1].task_id
    changed = make_plan(tmp_path, {"model_hash": "model-v2", "steps": 4})
    assert tasks[0].task_id != changed[0].task_id
    (tmp_path / "test.png").write_bytes(b"changed image")
    replanned = plan_tasks(tmp_path / "tasks.jsonl", [42], {"model_hash": "model-v1", "steps": 4})
    assert replanned[0].task_id != tasks[0].task_id


def test_task_lock_resume_failed_retry_and_corruption(tmp_path):
    tasks = make_plan(tmp_path)
    store = TaskStore(tmp_path / "output")
    second = TaskStore(tmp_path / "output")
    task = tasks[0]
    with store.claim(task) as acquired:
        assert acquired
        with second.claim(task) as duplicate:
            assert not duplicate
        store.record(task, "failed", error="injected failure")
    assert not store.is_done(task)
    with store.claim(task) as acquired:
        assert acquired
        output = store.output_path(task)
        output.write_bytes(b"real result fixture")
        store.record(task, "success", output)
    assert store.is_done(task)
    with store.claim(task) as acquired:
        assert not acquired
    with store.claim(task, resume=False) as acquired:
        assert acquired
    output.write_bytes(b"corrupt result")
    assert not store.is_done(task)
    assert store.status(tasks)["remaining_tasks"] == 2
    records = [json.loads(line) for line in store.metadata.read_text(encoding="utf-8").splitlines()]
    assert [record["status"] for record in records] == ["failed", "success"]


def test_metadata_stays_under_output_directory(tmp_path):
    with pytest.raises(ValueError):
        TaskStore(tmp_path / "output", "../escaped.jsonl")


def test_graph_cli_defaults_have_identical_task_identity():
    graph = inference_config({"guidance": 3, "shift": 0}, {"diffusion": "sha"})
    cli = inference_config({"guidance": 3.0, "shift": 0.0}, {"diffusion": "sha"}, True)
    assert fingerprint(graph) == fingerprint(cli)
    assert fingerprint(cli) != fingerprint(inference_config({}, {"diffusion": "sha"}, False))


def test_configuration_provenance_is_resolvable_and_checked(tmp_path):
    store = TaskStore(tmp_path / "output")
    config = inference_config({}, {"diffusion": "actual-model-sha"})
    identity = store.register_config(config)
    path = store.directory / "_configs" / (identity + ".json")
    assert json.loads(path.read_text())["config"]["model_hashes"]["diffusion"] == "actual-model-sha"
    assert store.register_config(config) == identity
    path.write_text('{}')
    with pytest.raises(ValueError, match="provenance"):
        store.register_config(config)


@pytest.mark.parametrize("damaged", [None, [], 42, "invalid", {"output": None},
                                    {"output": []}, {"output": "missing.png"}])
def test_damaged_completion_record_can_be_reclaimed(tmp_path, damaged):
    task = make_plan(tmp_path)[0]
    store = TaskStore(tmp_path / "output")
    store.done_path(task).write_text(json.dumps(damaged), encoding="utf-8")
    assert not store.is_done(task)
    assert store.status([task])["remaining_tasks"] == 1
    with store.claim(task) as acquired:
        assert acquired
        output = store.output_path(task)
        output.write_bytes(b"regenerated result")
        store.record(task, "success", output)
    assert store.is_done(task)


def test_completion_record_cannot_redirect_outside_output_directory(tmp_path):
    task = make_plan(tmp_path)[0]
    store = TaskStore(tmp_path / "output")
    output = store.output_path(task)
    output.write_bytes(b"result")
    record = store.record(task, "success", output)
    external = tmp_path / "external.png"
    external.write_bytes(b"result")
    record["output"] = str(external)
    store.done_path(task).write_text(json.dumps(record), encoding="utf-8")
    assert not store.is_done(task)
