"""Submit an exported native graph to a running ComfyUI and retain its history."""

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request


def request(url, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=payload,
            headers={"Content-Type": "application/json"}), timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8196")
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bluepixel.artifacts import atomic_json, sha256_file
    graph = json.loads(args.workflow.read_text(encoding="utf-8"))
    server = args.server.rstrip("/")
    started = time.perf_counter()
    queued = request(server + "/prompt", {"prompt": graph, "client_id": "bluepixel-validation"})
    prompt_id = queued["prompt_id"]
    print(json.dumps({"stage": "queued", "prompt_id": prompt_id}), flush=True)
    while time.perf_counter() - started < args.timeout:
        history = request(server + "/history/" + prompt_id)
        if prompt_id in history:
            record = history[prompt_id]
            result = {"workflow": str(args.workflow), "workflow_sha256": sha256_file(args.workflow),
                      "prompt_id": prompt_id, "seconds": time.perf_counter() - started, "history": record}
            atomic_json(args.output, result)
            if record.get("status", {}).get("status_str") != "success":
                raise RuntimeError(json.dumps(record.get("status"), ensure_ascii=False))
            print(json.dumps({"stage": "complete", "prompt_id": prompt_id,
                              "seconds": result["seconds"], "outputs": record.get("outputs")}), flush=True)
            return
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} still pending/running; inspect its queue before submitting again")


if __name__ == "__main__":
    main()
