"""Bind cached components to the artifact version they actually loaded."""

from pathlib import Path
from functools import partial
from .artifacts import sha256_file

_IDENTITY_ATTACHMENT = "bluepixel.artifact_identity"


def register_patcher_loader(patcher, loader, path, **options):
    # ComfyUI reloads through this factory when a clone needs independent weights.
    # Keyword binding allows clone(disable_dynamic=True) to override the default.
    patcher.cached_patcher_init = (partial(loader, **options), (str(Path(path).resolve()),))


def file_stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def mark_loaded(component, path, before):
    path = Path(path).resolve(strict=True)
    if file_stamp(path) != before:
        raise ValueError("Model file changed while loading; reload the component")
    component.bluepixel_path = str(path)
    component.bluepixel_file_stamp = before
    patcher = getattr(component, "patcher", component)
    if hasattr(patcher, "set_attachments"):
        # Native MODEL/CLIP clones retain attachments, unlike custom attributes
        # on their patcher or CLIP wrapper. Keep the shared record immutable.
        patcher.set_attachments(_IDENTITY_ATTACHMENT,
            (str(path), before, getattr(component, "bluepixel_load_dtype", None)))


def verify_loaded(component, expected_hash, label, load_dtype=None):
    path = getattr(component, "bluepixel_path", None)
    before = getattr(component, "bluepixel_file_stamp", None)
    dtype = getattr(component, "bluepixel_load_dtype", None)
    if path is None or before is None:
        patcher = getattr(component, "patcher", component)
        saved = patcher.get_attachment(_IDENTITY_ATTACHMENT) if hasattr(patcher, "get_attachment") else None
        if saved is not None:
            path, before, dtype = saved
    if path is None or before is None:
        raise ValueError(f"Reload the BluePixel {label} component before running this task plan")
    if load_dtype is not None and dtype != load_dtype:
        raise ValueError(f"Connected {label} dtype differs from the task plan; select the planned dtype and reload")
    if file_stamp(path) != before:
        raise ValueError(f"The {label} file changed after loading; reload it and replan the batch")
    matches = sha256_file(path) == expected_hash
    if file_stamp(path) != before:
        raise ValueError(f"The {label} file changed during verification; reload it and replan the batch")
    if not matches:
        raise ValueError(f"Connected {label} differs from the task plan; replan before running")


def verify_components(model, clip, vae, config):
    hashes = config.get("model_hashes", {})
    if not isinstance(hashes, dict) or {"diffusion", "text_encoder", "vae"} - hashes.keys():
        raise ValueError("Task configuration must identify all three BluePixel model files")
    autoencoder = getattr(getattr(vae, "first_stage_model", None), "autoencoder", None)
    if autoencoder is None or not hasattr(autoencoder, "use_tiling"):
        raise ValueError("Connect the BluePixel DCAE-F64 component for this task plan")
    if bool(autoencoder.use_tiling) != config.get("vae_tiling", True):
        raise ValueError("Connected VAE tiling differs from the task plan; replan before running")
    for label, component in (("diffusion", model), ("text_encoder", clip), ("vae", vae)):
        dtype = config.get("diffusion_dtype", "default") if label == "diffusion" else None
        verify_loaded(component, hashes[label], label, load_dtype=dtype)
