"""Atomic JSON and bounded-memory safetensors output for multi-shard models."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceFileSnapshot:
    """Detect changed conversion inputs without copying their payloads."""

    def __init__(self, path):
        # Keep the logical filename: resolving a leaf symlink here would hide a
        # later retargeting from the reader, which opens that logical filename.
        self.path = Path(path).absolute()
        self.stamp = self._stat()

    def _stat(self):
        value = self.path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def verify_unchanged(self):
        try:
            unchanged = self._stat() == self.stamp
        except OSError as error:
            raise ValueError(f"Conversion input changed or became unavailable: {self.path}") from error
        if not unchanged:
            raise ValueError(f"Conversion input changed while being read: {self.path}")

    def sha256(self):
        self.verify_unchanged()
        value = sha256_file(self.path)
        self.verify_unchanged()
        return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_header(path):
    with open(path, "rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length > 100_000_000:
            raise ValueError("Unexpectedly large safetensors header")
        header = source.read(length)
        if len(header) != length:
            raise ValueError("Truncated safetensors header")
        return json.loads(header)


class StreamingSafeWriter:
    """Spool tensor payloads, then atomically publish a complete safetensors file.

    Memory use is bounded by the caller's current tensor rather than the model.
    Existing output files are never overwritten unless explicitly requested.
    """

    def __init__(self, path, metadata=None, overwrite=False):
        self.path = Path(path)
        self.metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
        self.overwrite = overwrite
        self.header = {}
        self.offset = 0
        self.payload_path = None
        self.final_path = None

    def __enter__(self):
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".payload", dir=self.path.parent)
        self.payload_path = Path(name)
        self.payload = os.fdopen(fd, "wb")
        return self

    def add(self, key, tensor):
        import torch
        if key in self.header or key == "__metadata__":
            raise ValueError(f"Duplicate or reserved tensor key: {key}")
        dtypes = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
                  torch.float64: "F64", torch.int64: "I64", torch.int32: "I32",
                  torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}
        if tensor.dtype not in dtypes:
            raise ValueError(f"Unsupported output dtype: {tensor.dtype}")
        tensor = tensor.detach().cpu().contiguous()
        raw = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        size = raw.nbytes
        self.header[key] = {"dtype": dtypes[tensor.dtype], "shape": list(tensor.shape),
                            "data_offsets": [self.offset, self.offset + size]}
        self.payload.write(raw)
        self.offset += size

    def __exit__(self, kind, error, traceback):
        try:
            self.payload.flush()
            os.fsync(self.payload.fileno())
            self.payload.close()
            if kind is not None:
                return False
            if self.path.exists() and not self.overwrite:
                raise FileExistsError(self.path)
            if not self.header:
                raise ValueError("Cannot publish an empty model")
            header = {"__metadata__": self.metadata, **self.header}
            encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
            encoded += b" " * (-len(encoded) % 8)
            fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
            self.final_path = Path(name)
            with os.fdopen(fd, "wb") as result, open(self.payload_path, "rb") as payload:
                result.write(struct.pack("<Q", len(encoded)))
                result.write(encoded)
                shutil.copyfileobj(payload, result, length=8 * 1024 * 1024)
                result.flush()
                os.fsync(result.fileno())
            from safetensors import safe_open
            with safe_open(str(self.final_path), framework="pt", device="cpu") as check:
                if set(check.keys()) != set(self.header):
                    raise ValueError("Serialized tensor keys did not round trip")
            if self.overwrite:
                os.replace(self.final_path, self.path)
            elif os.name == "nt":
                # Windows rename atomically refuses an existing destination.
                os.rename(self.final_path, self.path)
            else:
                # POSIX rename replaces existing files. Linking the complete
                # temporary file creates the destination atomically instead.
                os.link(self.final_path, self.path)
        finally:
            if not self.payload.closed:
                self.payload.close()
            for temporary in (self.payload_path, self.final_path):
                if temporary and temporary.exists():
                    temporary.unlink()
        return False
