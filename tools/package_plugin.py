"""Build a clean custom_nodes ZIP, explicitly excluding private work and weights."""

import argparse
import hashlib
import re
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"',
                        (root / "pyproject.toml").read_text(encoding="utf-8")).group(1)
    output = args.output or root / "dist" / f"ComfyUI-BluePixel-{version}.zip"
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    files = [root / name for name in ("__init__.py", "bluepixel_nodes.py", "README.md", "README_EN.md", "LICENSE",
        "NOTICE", "requirements.txt", "pyproject.toml", "MANIFEST.in", ".gitignore", ".comfyignore")]
    for name in ("bluepixel", "tools", "docs", "workflows", "tests"):
        files.extend(path for path in (root / name).rglob("*") if path.is_file()
                     and path.suffix in {".py", ".json", ".md", ".txt"}
                     and "__pycache__" not in path.parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files):
                relative = path.relative_to(root)
                if relative.name == "roadmap.md" or any(part in {".research", "models", ".git"} for part in relative.parts):
                    raise ValueError(f"Private entry in source package: {relative}")
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"Source symlink escapes workspace: {relative}")
                archive.write(path, "ComfyUI-BluePixel/" + relative.as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    import json
    print(json.dumps({"output": str(output), "files": len(files), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
