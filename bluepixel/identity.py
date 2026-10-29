"""Bind cached components to the artifact version they actually loaded."""

from pathlib import Path
from .artifacts import sha256_file


def file_stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def mark_loaded(component, path, before):
    path = Path(path).resolve(strict=True)
    if file_stamp(path) != before:
        raise ValueError("Model file changed while loading; reload the component")
    component.bluepixel_path = str(path)
    component.bluepixel_file_stamp = before


def verify_loaded(component, expected_hash, label):
    path = getattr(component, "bluepixel_path", None)
    before = getattr(component, "bluepixel_file_stamp", None)
    if path is None or before is None:
        raise ValueError(f"Reload the BluePixel {label} component before running this task plan")
    if file_stamp(path) != before:
        raise ValueError(f"The {label} file changed after loading; reload it and replan the batch")
    matches = sha256_file(path) == expected_hash
    if file_stamp(path) != before:
        raise ValueError(f"The {label} file changed during verification; reload it and replan the batch")
    if not matches:
        raise ValueError(f"Connected {label} differs from the task plan; replan before running")
