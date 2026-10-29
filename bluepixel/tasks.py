"""Cross-platform, content-addressed batch tasks with process-owned locks."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
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


def inference_config(settings, model_hashes, vae_tiling=True, diffusion_dtype="default"):
    if not isinstance(settings, dict) or set(settings) - set(INFERENCE_DEFAULTS):
        raise ValueError(f"Unknown inference settings; allowed: {sorted(INFERENCE_DEFAULTS)}")
    resolved = {**INFERENCE_DEFAULTS, **settings}
    from .scheduling import schedule_values, positive_int
    import math
    for name in ("guidance", "shift"):
        if isinstance(resolved[name], bool) or not isinstance(resolved[name], (int, float)):
            raise ValueError(f"{name} must be numeric")
        resolved[name] = float(resolved[name])
    resolved["steps"] = positive_int(resolved["steps"], "steps")
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
    if diffusion_dtype not in {"default", "bf16", "fp32"}:
        raise ValueError("diffusion_dtype must be default, bf16 or fp32")
    config = {"pipeline_version": 1, "settings": resolved, "model_hashes": dict(model_hashes),
              "vae_tiling": bool(vae_tiling)}
    # Keep existing default-mode task fingerprints compatible.
    if diffusion_dtype != "default":
        config["diffusion_dtype"] = diffusion_dtype
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _recover_metadata_tail(stream):
    """Repair only an unfinished final JSONL entry while holding metadata.lock."""
    stream.seek(0, os.SEEK_END)
    end = stream.tell()
    if not end:
        return
    stream.seek(end - 1)
    if stream.read(1) == b"\n":
        return
    cursor, chunks, tail_start = end, [], 0
    while cursor:
        start = max(0, cursor - 8192)
        stream.seek(start)
        chunk = stream.read(cursor - start)
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            tail_start = start + newline + 1
            chunks.append(chunk[newline + 1:])
            break
        chunks.append(chunk)
        cursor = start
    tail = b"".join(reversed(chunks))
    try:
        json.loads(tail.decode("utf-8"))
    except (ValueError, UnicodeError):
        stream.truncate(tail_start)
    else:
        stream.seek(0, os.SEEK_END)
        stream.write(b"\n")
    stream.seek(0, os.SEEK_END)


@dataclass(frozen=True)
class Task:
    task_id: str
    image_path: str
    image_sha256: str
    instruction: str
    seed: int
    config_fingerprint: str
    source_jsonl: str | None = None
    input_aliases: tuple[str, ...] = ()


def plan_tasks(jsonl_path, seeds, config, instruction_field="edit_prompt", limit=None):
    path = Path(jsonl_path).resolve(strict=True)
    normalized_seeds = []
    for seed in seeds:
        try:
            value = int(seed)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Seeds must be unsigned 64-bit integers") from error
        if isinstance(seed, bool) or (not isinstance(seed, str) and value != seed):
            raise ValueError("Seeds must be unsigned 64-bit integers")
        normalized_seeds.append(value)
    seeds = normalized_seeds
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
                                  instruction, seed, config_id, str(path)))
            source_rows += 1
    # Identical rows are a single content-addressed task, not concurrent duplicate writes.
    unique, aliases = {}, {}
    for task in tasks:
        previous = unique.get(task.task_id)
        if previous is not None:
            paths = aliases.setdefault(task.task_id, {previous.image_path: None})
            paths[task.image_path] = None
        unique[task.task_id] = task
    return [replace(task, input_aliases=tuple(aliases[identity])) if identity in aliases else task
            for identity, task in unique.items()]


class TaskStore:
    def __init__(self, directory, metadata_file="metadata.jsonl", *, protected_paths=()):
        self.directory = Path(directory).resolve()
        # Keep logical absolute paths so a later symlink retarget or hardlink
        # replacement is checked against the files present at each write.
        self.protected_paths = tuple(Path(path).absolute() for path in protected_paths)
        filename = Path(metadata_file)
        folders = ("_locks", "_done", "_workers", "_configs", "img")
        if (not metadata_file or filename.name != metadata_file or filename.is_absolute()
                or filename.name.casefold() in folders):
            raise ValueError("metadata_file must be a plain filename")
        self.metadata = self.directory / filename
        if self.metadata.is_dir():
            raise ValueError("metadata_file must name a file, not an existing directory")
        for folder in folders:
            (self.directory / folder).mkdir(parents=True, exist_ok=True)
        self.owner = f"{socket.gethostname()}:{os.getpid()}"

    def validate_task_paths(self, tasks, outputs=()):
        """Keep writable batch artifacts separate from every planned input."""
        tasks = list(tasks)
        cache = {}

        def identity(value):
            path = Path(value)
            if path not in cache:
                resolved = path.resolve()
                try:
                    stat = resolved.stat()
                except FileNotFoundError:
                    inode = None
                else:
                    inode = (stat.st_dev, stat.st_ino) if stat.st_ino else None
                cache[path] = resolved, inode
            return cache[path]

        input_paths, input_inodes = set(), set()
        for source in self.protected_paths:
            path, inode = identity(source)
            input_paths.add(path)
            if inode is not None:
                input_inodes.add(inode)
        for task in tasks:
            for source in (task.image_path, getattr(task, "source_jsonl", None),
                           *getattr(task, "input_aliases", ())):
                if source is None:
                    continue
                path, inode = identity(source)
                input_paths.add(path)
                if inode is not None:
                    input_inodes.add(inode)

        def overlaps_input(value):
            path, inode = value
            return path in input_paths or inode is not None and inode in input_inodes

        metadata_identity = identity(self.metadata)
        if overlaps_input(metadata_identity):
            raise ValueError("Batch metadata must not overwrite an input image, source JSONL or model file")
        internal = {self.directory / "_locks" / "metadata.lock"}
        for task in tasks:
            lock = self.directory / "_locks" / (task.task_id + ".lock")
            internal.update((self.done_path(task), lock, lock.with_suffix(".owner.json"),
                             self.directory / "_configs" / (task.config_fingerprint + ".json")))
        internal_identities = [identity(path) for path in internal]
        if any(overlaps_input(value) for value in internal_identities):
            raise ValueError("Batch internal state must not overwrite an input image, source JSONL or model file")
        internal_paths = {path for path, _ in internal_identities}
        internal_inodes = {inode for _, inode in internal_identities if inode is not None}

        def overlaps_internal(value):
            path, inode = value
            return path in internal_paths or inode is not None and inode in internal_inodes

        if overlaps_internal(metadata_identity):
            raise ValueError("Batch metadata must not overwrite internal state")
        for output in [*(self.output_path(task) for task in tasks), *outputs]:
            output_identity = identity(output)
            if overlaps_input(output_identity):
                raise ValueError("Batch output must not overwrite an input image, source JSONL or model file")
            if (output_identity[0] == metadata_identity[0] or
                    output_identity[1] is not None and output_identity[1] == metadata_identity[1]):
                raise ValueError("Batch metadata and output paths must be different")
            if overlaps_internal(output_identity):
                raise ValueError("Batch output must not overwrite internal state")

    def register_config(self, config):
        identity = fingerprint(config)
        path = self.directory / "_configs" / (identity + ".json")
        self.validate_task_paths((), outputs=[path])
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
            if not isinstance(record, dict) or not isinstance(record.get("output"), str):
                return False
            output = Path(record["output"])
            return (record["status"] == "success" and record["task_id"] == task.task_id and
                    record["config_fingerprint"] == task.config_fingerprint and
                    output.resolve().is_relative_to(self.directory) and output.is_file() and
                    (not verify_hash or sha256_file(output) == record["output_sha256"]))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    @contextmanager
    def claim(self, task, resume=True, timeout=0):
        self.validate_task_paths([task])
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
                try:
                    lock_path.with_suffix(".owner.json").unlink(missing_ok=True)
                finally:
                    lock.release()

    def record(self, task, status, output=None, error=None, **metadata):
        if status not in {"success", "failed"}:
            raise ValueError("Task status must be success or failed")
        self.validate_task_paths([task], outputs=[output] if status == "success" and output is not None else [])
        record = {**metadata, **asdict(task), "input_aliases": list(task.input_aliases),
                  "status": status, "owner": self.owner, "timestamp": time.time()}
        if status == "success":
            output = Path(output).resolve(strict=True)
            if not output.is_relative_to(self.directory):
                raise ValueError("Batch output must stay within the configured output directory")
            record.update(output=str(output), output_sha256=sha256_file(output))
        elif error is not None:
            record["error"] = str(error)
        payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with FileLock(str(self.directory / "_locks" / "metadata.lock"), timeout=60):
            with self.metadata.open("a+b") as stream:
                _recover_metadata_tail(stream)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(self.done_path(task), record)
        return record

    def status(self, tasks):
        completed = sum(self.is_done(task) for task in tasks)
        return {"total_tasks": len(tasks), "done_tasks": completed, "remaining_tasks": len(tasks) - completed}
