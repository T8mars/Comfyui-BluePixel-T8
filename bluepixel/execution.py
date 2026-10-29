"""Editing and batch execution through ComfyUI's standard sampler."""

import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from PIL import Image
import torch

from .artifacts import sha256_file
from .config import DEFAULT_PIXELS
from .image import prepare_image, image_to_uint8
from .scheduling import schedule_values
from .tasks import TaskStore, plan_tasks
from .conditioning import append_references


def run_edit(model, clip, vae, instruction, image=None, steps=4, guidance=3.0,
             seed=42, target_pixels=DEFAULT_PIXELS, width=0, height=0,
             mu_mode="empirical", shift=0.0, batch_size=1, noise_mode="upstream_cuda_bf16",
             prompt_embeds=None, empty_embeds=None, initial_noise=None, callback=None,
             callback_on_step_end=None, callback_on_step_end_tensor_inputs=("latents",)):
    import comfy.model_management as mm
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview
    if not np.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance must be finite and non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if model.get_model_object("latent_format").spacial_downscale_ratio != 64:
        raise ValueError("Select the BluePixel f64 diffusion loader")
    if vae.downscale_ratio != 64 or vae.latent_channels != 128:
        raise ValueError("Select the BluePixel DCAE-F64")
    metadata = {"seed": int(seed), "guidance": guidance, "noise_mode": noise_mode}
    references = []
    if image is not None:
        image_list = image if isinstance(image, list) else [image]
        for index, reference in enumerate(image_list):
            if reference.shape[0] != 1:
                raise ValueError("A reference IMAGE must have batch=1; use task batches for independent images")
            prepared, resolved_width, resolved_height, sizes = prepare_image(reference, target_pixels, width, height)
            references.append(vae.encode(prepared))
            if index == 0:
                width, height = resolved_width, resolved_height
                metadata["resolution"] = sizes
    width, height = width or 1024, height or 1024
    if width % 64 or height % 64 or min(width, height) < 64:
        raise ValueError("Output dimensions must be positive multiples of 64")
    if prompt_embeds is None:
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(instruction))
    else:
        positive = prompt_embeds
    if guidance > 1:
        negative = empty_embeds if empty_embeds is not None else clip.encode_from_tokens_scheduled(clip.tokenize(""))
    else:
        negative = positive
    for label, conditioning in (("positive", positive), ("negative", negative)):
        if not conditioning:
            raise ValueError(f"{label} conditioning is empty")
        for embedding, _ in conditioning:
            if embedding.ndim != 3 or embedding.shape[-1] != 12288 or embedding.shape[0] not in (1, batch_size):
                raise ValueError(f"{label} embeddings must be [1 or batch,sequence,12288]")
            if not embedding.is_floating_point() or not torch.isfinite(embedding).all():
                raise ValueError(f"{label} embeddings must be finite floating point")
    controller = None
    if callback_on_step_end is not None:
        if len(positive) != 1:
            raise ValueError("Source-style callbacks require a single positive conditioning block")
        from .hooks import StepController
        controller = StepController(callback_on_step_end, positive[0][0], callback_on_step_end_tensor_inputs)
        model = model.clone()
        model.set_model_unet_function_wrapper(controller.diffusion_wrapper(model.model_options.get("model_function_wrapper")))
    if references:
        positive = append_references(positive, references)
        negative = append_references(negative, references)
    latent = torch.zeros((batch_size, 128, height // 64, width // 64), device=mm.intermediate_device())
    if initial_noise is not None:
        if tuple(initial_noise.shape) != tuple(latent.shape):
            raise ValueError("Explicit noise shape does not match output latent")
        noise = initial_noise
    elif noise_mode == "upstream_cuda_bf16":
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise ValueError("upstream_cuda_bf16 noise requires CUDA")
        noise = torch.randn(latent.shape, device=device, dtype=torch.bfloat16,
                            generator=torch.Generator(device=device).manual_seed(seed)).to(mm.intermediate_device())
    elif noise_mode == "comfy_cpu":
        noise = comfy.sample.prepare_noise(latent, seed)
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")
    values, schedule_metadata = schedule_values(steps, latent.shape[-2] * latent.shape[-1], mu_mode, shift)
    metadata.update(schedule_metadata)
    sigmas = torch.tensor(values, dtype=torch.float32)
    if callback is None:
        callback = latent_preview.prepare_callback(model, steps)
    started = time.perf_counter()
    from .sampling import make_sampler
    samples = comfy.sample.sample_custom(model, noise, guidance if guidance > 1 else 1.0,
        make_sampler(controller), sigmas, positive, negative, latent,
        callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    metadata["sampling_seconds"] = time.perf_counter() - started
    output = vae.decode(samples)
    return output, {"samples": samples, "bluepixel_latent_convention": "raw_dcae_bchw"}, metadata


def save_png_atomic(image, path, metadata=None):
    from PIL.PngImagePlugin import PngInfo
    pixels = image_to_uint8(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    if metadata:
        info.add_text("bluepixel", json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Image.fromarray(pixels).save(temporary, format="PNG", pnginfo=info)
        with open(temporary, "r+b") as source:
            os.fsync(source.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_tasks(tasks, output_dir, model, clip, vae, settings, resume=True,
              metadata_file="metadata.jsonl", lock_timeout=0, fail_fast=False, progress=None, config=None):
    store = TaskStore(output_dir, metadata_file)
    store.validate_task_paths(tasks)
    if config is not None:
        identity = store.register_config(config)
        if any(task.config_fingerprint != identity for task in tasks):
            raise ValueError("Task plan differs from the recorded model/inference configuration")
    counts = {"success": 0, "failed": 0, "skipped": 0}
    progress = progress or (lambda event: None)
    for task in tasks:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
        with store.claim(task, resume=resume, timeout=lock_timeout) as acquired:
            if not acquired:
                counts["skipped"] += 1
                continue
            try:
                if sha256_file(task.image_path) != task.image_sha256:
                    raise ValueError("Input image changed after task planning; replan the batch")
                with Image.open(task.image_path) as source:
                    image = torch.from_numpy(np.asarray(source.convert("RGB")).copy()).float().unsqueeze(0) / 255
                output, latent, metadata = run_edit(model, clip, vae, task.instruction,
                    image, seed=task.seed, **settings)
                if output.shape[0] != 1:
                    raise ValueError("A JSONL task must generate one image; use multiple seeds")
                output_path = store.output_path(task)
                save_png_atomic(output[0], output_path, metadata)
                record = store.record(task, "success", output_path, inference=metadata)
                counts["success"] += 1
            except mm.InterruptProcessingException:
                raise
            except Exception as error:
                record = store.record(task, "failed", error=error)
                counts["failed"] += 1
                if fail_fast:
                    raise
            progress(record)
    return {**counts, **store.status(tasks)}
