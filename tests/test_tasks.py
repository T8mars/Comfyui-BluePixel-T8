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


@pytest.mark.parametrize("seed", [42.9, -0.5, True, False, float("nan"), float("inf"), None])
def test_invalid_seeds_are_not_silently_coerced(tmp_path, seed):
    make_plan(tmp_path)
    with pytest.raises(ValueError, match="unsigned 64-bit integers"):
        plan_tasks(tmp_path / "tasks.jsonl", [seed], {})


def test_integral_seed_forms_keep_the_same_task_identity(tmp_path):
    make_plan(tmp_path)
    tasks = [plan_tasks(tmp_path / "tasks.jsonl", [seed], {})[0]
             for seed in (42, 42.0, "42")]
    assert {task.seed for task in tasks} == {42}
    assert len({task.task_id for task in tasks}) == 1
    assert plan_tasks(tmp_path / "tasks.jsonl", [0, 2**64 - 1], {})[-1].seed == 2**64 - 1


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


@pytest.mark.parametrize("filename", ["", "img", "_locks"])
def test_invalid_metadata_filename_is_rejected_before_execution(tmp_path, filename):
    with pytest.raises(ValueError, match="metadata_file"):
        TaskStore(tmp_path / "output", filename)


def test_metadata_filename_cannot_name_existing_directory(tmp_path):
    directory = tmp_path / "output"
    (directory / "metadata.jsonl").mkdir(parents=True)
    with pytest.raises(ValueError, match="metadata_file"):
        TaskStore(directory)


def test_task_lock_releases_even_if_owner_lease_cleanup_fails(tmp_path, monkeypatch):
    from filelock import FileLock
    task = make_plan(tmp_path)[0]
    store = TaskStore(tmp_path / "output")
    original_unlink = Path.unlink

    def fail_owner_unlink(path, *args, **kwargs):
        if path.name.endswith(".owner.json"):
            raise PermissionError("injected owner lease cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_owner_unlink)
    with pytest.raises(PermissionError, match="owner lease") as retained_error:
        with store.claim(task) as acquired:
            assert acquired
    # Retain the exception and its traceback: release must not rely on object GC.
    assert retained_error.value is not None
    with FileLock(str(store.directory / "_locks" / (task.task_id + ".lock")), timeout=0):
        pass


@pytest.mark.parametrize("prefix,previous", [
    (b'{"status":"suc', []),
    (b'{"instruction":"\xe4\xb8', []),
    (b'{"previous":"kept"}', [{"previous": "kept"}]),
    (b'{"previous":"kept"}\n{"partial"', [{"previous": "kept"}]),
    (b'{"previous":"kept"}\n' + json.dumps({"long": "中" * 5000}, ensure_ascii=False).encode("utf-8"),
     [{"previous": "kept"}, {"long": "中" * 5000}]),
], ids=["partial-json", "partial-utf8", "json-no-newline", "preserve-history", "long-unicode-tail"])
def test_metadata_append_recovers_unfinished_tail(tmp_path, prefix, previous):
    task = make_plan(tmp_path)[0]
    store = TaskStore(tmp_path / "output")
    store.metadata.write_bytes(prefix)
    output = store.output_path(task)
    output.write_bytes(b"result")
    record = store.record(task, "success", output)
    records = [json.loads(line) for line in store.metadata.read_text(encoding="utf-8").splitlines()]
    assert records == previous + [record]
    assert store.is_done(task)


def test_metadata_tail_recovery_does_not_read_entire_log():
    from io import BytesIO
    from bluepixel.tasks import _recover_metadata_tail

    class ReadCountingStream(BytesIO):
        bytes_read = 0

        def read(self, size=-1):
            assert size >= 0
            result = super().read(size)
            self.bytes_read += len(result)
            return result

    prior = b'{"previous":"kept"}\n' * 200000
    stream = ReadCountingStream(prior + b'{"partial"')
    _recover_metadata_tail(stream)
    assert stream.getvalue() == prior
    assert stream.bytes_read <= 8193


@pytest.mark.parametrize("filename", ["_LOCKS", "_DONE", "_WORKERS", "_CONFIGS", "IMG"])
def test_metadata_internal_names_are_reserved_case_insensitively(tmp_path, filename):
    # On Windows these names alias the directories created by this constructor.
    with pytest.raises(ValueError, match="plain filename"):
        TaskStore(tmp_path / "fresh-output", filename)


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
