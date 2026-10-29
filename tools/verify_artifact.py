"""Verify an artifact against its manifest and the local immutable source."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from bluepixel.artifacts import sha256_file
    from bluepixel.conversion import TensorSource
    torch.set_num_threads(4)
    report = json.loads(Path(str(args.artifact) + ".manifest.json").read_text(encoding="utf-8"))
    if sha256_file(args.artifact) != report["output"]["sha256"]:
        raise ValueError("Artifact SHA256 does not match the manifest")
    checked = 0
    with safe_open(str(args.artifact), framework="pt", device="cpu") as artifact:
        expected_keys = set()
        for entry in report["tensors"]:
            expected_keys.add(entry["target"])
            if "quantization" in entry:
                layer = entry["target"][:-7]
                expected_keys.update((layer + ".weight_scale", layer + ".comfy_quant"))
        if report["component"] == "text_encoder":
            expected_keys.add("bluepixel_tokenizer_json")
        if set(artifact.keys()) != expected_keys:
            raise ValueError("Artifact keys differ from the conversion report")
        if args.source:
            directory = args.source / report["component"]
            for source_file in report["source_files"]:
                if sha256_file(directory / source_file["path"]) != source_file["sha256"]:
                    raise ValueError(f"Source changed: {source_file['path']}")
            if report["precision"] == "original" and not report.get("adapters"):
                with TensorSource(directory) as source:
                    for entry in report["tensors"]:
                        tensors = [source.get(key) for key in entry["sources"]]
                        expected = tensors[0] if len(tensors) == 1 else torch.cat(tensors)
                        if entry["target"] == "final_layer.adaLN_modulation.1.weight":
                            scale, shift = expected.chunk(2)
                            expected = torch.cat((shift, scale))
                        actual = artifact.get_tensor(entry["target"])
                        if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                            raise ValueError(f"Tensor is not lossless: {entry['target']}")
                        checked += 1
    print(json.dumps({"artifact": args.artifact.name, "sha256_verified": True,
                      "precision": report["precision"], "lossless_tensors_verified": checked}), flush=True)


if __name__ == "__main__":
    main()
