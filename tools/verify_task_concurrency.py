"""Exercise live/stale-owner exclusion and crash recovery in separate processes."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def hold(task, directory, ready):
    from bluepixel.tasks import TaskStore
    with TaskStore(directory).claim(task) as acquired:
        if not acquired:
            raise RuntimeError("Holder could not acquire fixture lock")
        ready.set()
        import time
        while True:
            time.sleep(1)


def probe(task, directory, result):
    from bluepixel.tasks import TaskStore
    with TaskStore(directory).claim(task) as acquired:
        result.put(acquired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from bluepixel.tasks import plan_tasks
    from bluepixel.artifacts import atomic_json
    parent = args.output.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    ready, result = context.Event(), context.Queue()
    workers = []
    with tempfile.TemporaryDirectory(prefix="bluepixel-lock-", dir=parent) as temporary:
        directory = Path(temporary).resolve()
        if not directory.is_relative_to(parent):
            raise ValueError("Unexpected temporary workspace")
        (directory / "input.dat").write_bytes(b"Synthetic lock fixture, not a model/image output")
        (directory / "tasks.jsonl").write_text(json.dumps({"image_name": "input.dat", "edit_prompt": "lock fixture"}))
        task = plan_tasks(directory / "tasks.jsonl", [42], {"fixture": True})[0]
        store_directory = directory / "results"
        holder = context.Process(target=hold, args=(task, store_directory, ready))
        workers.append(holder)
        try:
            holder.start()
            if not ready.wait(20):
                raise TimeoutError("Fixture holder did not start")
            lease = store_directory / "_locks" / (task.task_id + ".owner.json")
            owner = json.loads(lease.read_text())
            owner["started"] = 0
            atomic_json(lease, owner)
            rival = context.Process(target=probe, args=(task, store_directory, result))
            workers.append(rival)
            rival.start()
            rival_acquired = result.get(timeout=20)
            rival.join(20)
            if rival.exitcode != 0 or rival_acquired:
                raise AssertionError("A stale timestamp stole a live OS lock")
            holder.terminate()  # This tool's own helper, simulating a worker crash.
            holder.join(20)
            stale_owner = lease.exists()
            recovery = context.Process(target=probe, args=(task, store_directory, result))
            workers.append(recovery)
            recovery.start()
            recovered = result.get(timeout=20)
            recovery.join(20)
            if recovery.exitcode != 0 or not recovered or lease.exists():
                raise AssertionError("A crashed worker left an unrecoverable task lock")
            report = {"separate_processes": True, "stale_timestamp_rival_acquired": rival_acquired,
                      "owner_record_remained_after_crash": stale_owner, "recovered": recovered,
                      "lease_cleaned": not lease.exists(), "neural_inference": False}
            atomic_json(args.output, report)
            print(json.dumps(report), flush=True)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                if worker.pid is not None:
                    worker.join(20)
            result.close()
            result.join_thread()


if __name__ == "__main__":
    main()
