"""Download an immutable BluePixel snapshot; never prints account tokens."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bluepixel.config import SOURCE_REPO, SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component", choices=["all", "transformer", "text_encoder", "vae"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(SOURCE_REPO, revision=SOURCE_REVISION, files_metadata=True)
    if info.sha != SOURCE_REVISION:
        raise ValueError("Hub did not resolve the pinned revision")
    files = [f for f in info.siblings if not f.rfilename.startswith(".") and
             (args.component == "all" or not f.rfilename.endswith(".safetensors") or
              f.rfilename.startswith(args.component + "/"))]
    print(json.dumps({"repo_id": SOURCE_REPO, "revision": SOURCE_REVISION,
                      "files": len(files), "bytes": sum(f.size or 0 for f in files),
                      "output": str(args.output.resolve()), "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for item in files:
        print(json.dumps({"stage": "download", "file": item.rfilename,
                          "bytes": item.size}), flush=True)
        hf_hub_download(SOURCE_REPO, item.rfilename, revision=SOURCE_REVISION,
                        local_dir=str(args.output))
    print(json.dumps({"stage": "download_complete", "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
