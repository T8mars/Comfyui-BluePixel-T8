"""Cross-platform, content-addressed batch tasks with process-owned locks."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from filelock import FileLock, Timeout
from .artifacts import atomic_json, sha256_file

INFERENCE_DEFAULTS = {"steps": 4, "guidance": 3.0, "target_pixels": 16777216,
    "width": 0, "height": 0, "mu_mode": "empirical", "shift": 0.0,
    "noise_mode": "upstream_cuda_bf16"}


def inference_config(settings, model_hashes, vae_tiling=True):
    if not isinstance(settings, dict) or set(settings) - set(INFERENCE_DEFAULTS):
        raise ValueError(f"Unknown inference settings; allowed: {sorted(INFERENCE_DEFAULTS)}")
    resolved = {**INFERENCE_DEFAULTS, **settings}
    from .scheduling import schedule_values
    import math
    for name in ("guidance", "shift"):
        if isinstance(resolved[name], bool) or not isinstance(resolved[name], (int, float)):
            raise ValueError(f"{name} must be numeric")
        resolved[name] = float(resolved[name])
    schedule_values(resolved["steps"], 1, resolved["mu_mode"], resolved["shift"])
    if not math.isfinite(resolved["guidance"]) or resolved["guidance"] < 0:
        raise ValueError("guidance must be finite and non-negative")
    for name in ("width", "height", "target_pixels"):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        if name != "target_pixels" and value % 64:
            raise ValueError(f"{name} must be zero or a positive multiple of64")
    if resolved["noise_mode"] not in {"upstream_cuda_bf16", "comfy_cpu"}:
        raise ValueError("Unknown noise mode")
    return {"pipeline_version": 1, "settings": resolved, "model_hashes": dict(model_hashes),
            "vae_tiling": bool(vae_tiling)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Task:
    task_id: str
    image_path: str
    image_sha256: str
    instruction: str
    seed: int
    config_fingerprint: str


def plan_tasks(jsonl_path, seeds, config, instruction_field="edit_prompt", limit=None):
    path = Path(jsonl_path).resolve(strict=True)
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("At least one seed is required")
    if any(seed < 0 or seed > 0xffffffffffffffff for seed in seeds):
        raise ValueError("Seeds must be unsigned 64-bit integers")
    if limit is not None and (isinstance(limit, bool) or int(limit) != limit or limit < 1):
        raise ValueError("limit must be a positive integer, or omitted")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds would create duplicate tasks")
    config_id = fingerprint(config)
    tasks, source_rows = [], 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if limit is not None and source_rows >= limit:
                break
            try:
                row = json.loads(line)
                image = Path(row["image_name"])
                if not image.is_absolute():
                    image = path.parent / image
                image = image.resolve(strict=True)
                instruction = row[instruction_field]
                if not isinstance(instruction, str):
                    raise ValueError("instruction is not a string")
            except (KeyError, ValueError, TypeError, OSError) as error:
                raise ValueError(f"Invalid task at {path}:{line_number}: {error}") from error
            image_hash = sha256_file(image)
            for seed in seeds:
                identity = {"image_sha256": image_hash, "instruction": instruction,
                            "seed": seed, "config_fingerprint": config_id}
                tasks.append(Task(fingerprint(identity), str(image), image_hash,
                                  instruction, seed, config_id))
            source_rows += 1
    # Identical rows are a single content-addressed task, not concurrent duplicate writes.
    return list({task.task_id: task for task in tasks}.values())


class TaskStore:
    def __init__(self, directory, metadata_file="metadata.jsonl"):
        self.directory = Path(directory).resolve()
        filename = Path(metadata_file)
        if filename.name != metadata_file or filename.is_absolute():
            raise ValueError("metadata_file must be a plain filename")
        self.metadata = self.directory / filename
        for folder in ("_locks", "_done", "_workers", "_configs", "img"):
            (self.directory / folder).mkdir(parents=True, exist_ok=True)
        self.owner = f"{socket.gethostname()}:{os.getpid()}"

    def register_config(self, config):
        identity = fingerprint(config)
        path = self.directory / "_configs" / (identity + ".json")
        record = {"config_fingerprint": identity, "config": config}
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise ValueError("Stored configuration provenance is damaged")
        else:
            atomic_json(path, record)
        return identity

    def output_path(self, task):
        return self.directory / "img" / f"{task.task_id}_seed{task.seed}.png"

    def done_path(self, task):
        return self.directory / "_done" / (task.task_id + ".json")

    def is_done(self, task, verify_hash=True):
        try:
            record = json.loads(self.done_path(task).read_text(encoding="utf-8"))
            output = Path(record["output"])
            return (record["status"] == "success" and record["task_id"] == task.task_id and
                    record["config_fingerprint"] == task.config_fingerprint and output.is_file() and
                    (not verify_hash or sha256_file(output) == record["output_sha256"]))
        except (OSError, ValueError, KeyError):
            return False

    @contextmanager
    def claim(self, task, resume=True, timeout=0):
        lock_path = self.directory / "_locks" / (task.task_id + ".lock")
        lock = FileLock(str(lock_path), timeout=timeout)
        acquired = False
        try:
            try:
                lock.acquire()
                acquired = True
            except Timeout:
                yield False
                return
            if resume and self.is_done(task):
                yield False
                return
            # The OS lock is held throughout inference. A stale timestamp cannot steal a live job.
            lease = lock_path.with_suffix(".owner.json")
            atomic_json(lease, {"owner": self.owner, "task_id": task.task_id, "started": time.time()})
            yield True
        finally:
            if acquired:
                lock_path.with_suffix(".owner.json").unlink(missing_ok=True)
                lock.release()

    def record(self, task, status, output=None, error=None, **metadata):
        if status not in {"success", "failed"}:
            raise ValueError("Task status must be success or failed")
        record = {**metadata, **asdict(task), "status": status, "owner": self.owner, "timestamp": time.time()}
        if status == "success":
            output = Path(output).resolve(strict=True)
            if not output.is_relative_to(self.directory):
                raise ValueError("Batch output must stay within the configured output directory")
            record.update(output=str(output), output_sha256=sha256_file(output))
        elif error is not None:
            record["error"] = str(error)
        with FileLock(str(self.directory / "_locks" / "metadata.lock"), timeout=60):
            with self.metadata.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(self.done_path(task), record)
        return record

    def status(self, tasks):
        completed = sum(self.is_done(task) for task in tasks)
        return {"total_tasks": len(tasks), "done_tasks": completed, "remaining_tasks": len(tasks) - completed}
