"""Composable ComfyUI nodes. All neural interfaces use native ComfyUI objects."""

import json
import math
from pathlib import Path
import torch
import folder_paths
import comfy.model_management as mm
import comfy.samplers
import comfy.utils

from .bluepixel.config import DEFAULT_PIXELS
from .bluepixel.image import prepare_image
from .bluepixel.scheduling import schedule_values
from .bluepixel.native import BluePixelVAE, load_clip, load_diffusion
from .bluepixel.conditioning import append_references


class BluePixelDiffusionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (folder_paths.get_filename_list("diffusion_models"),),
                              "dtype": (["default", "bf16", "fp32"],)},
                "optional": {"disable_dynamic": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "BluePixel/loaders"

    def load(self, model_name, dtype="default", disable_dynamic=False):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return (load_diffusion(path, dtype, disable_dynamic),)


class BluePixelCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (folder_paths.get_filename_list("text_encoders"),),
                              "device": (["default", "cpu"],)}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "BluePixel/loaders"

    def load(self, model_name, device="default"):
        return (load_clip(folder_paths.get_full_path_or_raise("text_encoders", model_name), device),)


class BluePixelVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (folder_paths.get_filename_list("vae"),),
                              "tiling": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "BluePixel/loaders"

    def load(self, model_name, tiling=True):
        return (BluePixelVAE(folder_paths.get_full_path_or_raise("vae", model_name), tiling),)


class BluePixelTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",),
                              "instruction": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                              "max_sequence_length": ("INT", {"default": 512, "min": 1, "max": 32768})},
                "optional": {"hidden_layers": ("STRING", {"default": "9,18,27"})}}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "empty_instruction")
    FUNCTION = "encode"
    CATEGORY = "BluePixel/conditioning"

    def encode(self, clip, instruction, max_sequence_length=512, hidden_layers="9,18,27"):
        layers = [int(value.strip()) for value in hidden_layers.split(",")]
        if len(layers) != 3 or len(set(layers)) != 3 or any(layer < 0 or layer > 36 for layer in layers):
            raise ValueError("Choose three distinct hidden layers in 0..36; trained defaults are9,18,27")
        if layers != [9, 18, 27]:
            clip = clip.clone()
            clip.clip_layer(sorted(layers))
        tokens = clip.tokenize(instruction, max_sequence_length=max_sequence_length)
        empty = clip.tokenize("", max_sequence_length=max_sequence_length)
        positive = clip.encode_from_tokens_scheduled(tokens)
        negative = clip.encode_from_tokens_scheduled(empty)
        if layers != sorted(layers):
            order = [sorted(layers).index(layer) for layer in layers]
            def reorder(conditioning):
                return [[torch.cat([value[..., index * 4096:(index + 1) * 4096] for index in order], dim=-1), metadata]
                        for value, metadata in conditioning]
            positive, negative = reorder(positive), reorder(negative)
        return positive, negative


class BluePixelPrepareImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "target_pixels": ("INT", {"default": DEFAULT_PIXELS, "min": 0, "max": 268435456})},
                "optional": {"output_width": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 64}),
                              "output_height": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 64})}}
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("reference_image", "output_width", "output_height", "metadata")
    FUNCTION = "prepare"
    CATEGORY = "BluePixel/image"

    def prepare(self, image, target_pixels=DEFAULT_PIXELS, output_width=0, output_height=0):
        prepared, width, height, metadata = prepare_image(image, target_pixels, output_width, output_height)
        return prepared, width, height, json.dumps(metadata, ensure_ascii=False)


class BluePixelEmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT", {"default": 4096, "min": 64, "max": 32768, "step": 64}),
                              "height": ("INT", {"default": 4096, "min": 64, "max": 32768, "step": 64}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "BluePixel/latent"

    def generate(self, width=4096, height=4096, batch_size=1):
        if width < 64 or height < 64 or width % 64 or height % 64 or batch_size < 1:
            raise ValueError("Width/height must be positive multiples of 64; batch_size must be positive")
        samples = torch.zeros((batch_size, 128, height // 64, width // 64), device=mm.intermediate_device())
        return ({"samples": samples, "downscale_ratio_spacial": 64, "bluepixel_latent_convention": "raw_dcae_bchw"},)


class BluePixelScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "steps": ("INT", {"default": 4, "min": 1, "max": 4096}),
                              "mu_mode": (["empirical", "training"],),
                              "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01})}}
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_metadata")
    FUNCTION = "schedule"
    CATEGORY = "BluePixel/sampling"

    def schedule(self, latent, steps=4, mu_mode="empirical", shift=0.0):
        samples = latent["samples"]
        if samples.ndim != 4 or samples.shape[1] != 128:
            raise ValueError("BluePixelScheduler requires [B,128,H/64,W/64] latents")
        values, metadata = schedule_values(steps, samples.shape[-2] * samples.shape[-1], mu_mode, shift)
        return torch.tensor(values, dtype=torch.float32), json.dumps(metadata)


class SourceNoise:
    def __init__(self, seed, mode="upstream_cuda_bf16"):
        self.seed = int(seed)
        self.mode = mode

    def generate_noise(self, latent):
        if self.mode == "comfy_cpu":
            import comfy.sample
            return comfy.sample.prepare_noise(latent["samples"], self.seed, latent.get("batch_index"))
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise ValueError("upstream_cuda_bf16 noise requires CUDA; choose comfy_cpu on this device")
        generator = torch.Generator(device=device).manual_seed(self.seed)
        return torch.randn(latent["samples"].shape, generator=generator, device=device,
                           dtype=torch.bfloat16).to(mm.intermediate_device())


class BluePixelNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                              "mode": (["upstream_cuda_bf16", "comfy_cpu"],)}}
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "create"
    CATEGORY = "BluePixel/sampling"

    def create(self, seed=42, mode="upstream_cuda_bf16"):
        return (SourceNoise(seed, mode),)


class BluePixelEditConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                              "reference": ("LATENT",)}}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    FUNCTION = "apply"
    CATEGORY = "BluePixel/conditioning"

    def apply(self, positive, negative, reference):
        samples = reference["samples"]
        if samples.ndim != 4 or samples.shape[1] != 128:
            raise ValueError("Use the BluePixel DCAE-F64 to encode the reference")
        return append_references(positive, [samples]), append_references(negative, [samples])


class BluePixelCFGGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                              "guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1})}}
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create"
    CATEGORY = "BluePixel/sampling"

    def create(self, model, positive, negative, guidance=3.0):
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and non-negative")
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        # The source explicitly skips unconditional inference for every cfg<=1.
        guider.set_cfg(guidance if guidance > 1 else 1.0)
        return (guider,)


class BluePixelLatentInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "convention": (["raw_dcae_bchw", "scaled_dcae_bchw"],)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "convert"
    CATEGORY = "BluePixel/latent"

    def convert(self, latent, convention="raw_dcae_bchw"):
        samples = latent["samples"]
        if samples.ndim != 4 or samples.shape[1] != 128:
            raise ValueError("Expected a spatial [B,128,H,W] latent; packed BLC tokens are not a Comfy LATENT")
        result = latent.copy()
        result["samples"] = samples / 0.4538 if convention == "scaled_dcae_bchw" else samples
        result["downscale_ratio_spacial"] = 64
        result["bluepixel_latent_convention"] = "raw_dcae_bchw"
        return (result,)


class BluePixelDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"model": ("MODEL",), "vae": ("VAE",), "latent": ("LATENT",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"
    CATEGORY = "BluePixel/tools"
    OUTPUT_NODE = True

    def inspect(self, model=None, vae=None, latent=None):
        info = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(mm.get_torch_device())}
        if model is not None:
            info["model"] = getattr(model, "bluepixel_metadata", {})
            info["latent_ratio"] = model.get_model_object("latent_format").spacial_downscale_ratio
            from comfy_kitchen.tensor import QuantizedTensor
            quantized = [name for name, value in model.model.named_parameters() if isinstance(value, QuantizedTensor)]
            info["quantized_parameters"] = quantized
        if vae is not None:
            info["vae"] = {"dtype": str(vae.vae_dtype), "ratio": vae.downscale_ratio, "channels": vae.latent_channels}
        if latent is not None:
            info["latent"] = {"shape": list(latent["samples"].shape), "dtype": str(latent["samples"].dtype)}
        text = json.dumps(info, ensure_ascii=False, indent=2)
        return {"ui": {"text": [text]}, "result": (text,)}


class BluePixelEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
            "instruction": ("STRING", {"multiline": True}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 4096}),
            "guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0}),
            "target_pixels": ("INT", {"default": DEFAULT_PIXELS, "min": 0, "max": 268435456}),
            "mu_mode": (["empirical", "training"],),
            "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0})},
            "optional": {"image": ("IMAGE",), "reference_2": ("IMAGE",),
            "width": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 64}),
            "height": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 64}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "positive": ("CONDITIONING",), "empty_instruction": ("CONDITIONING",),
            "noise": ("LATENT",)}}
    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    FUNCTION = "execute"
    CATEGORY = "BluePixel"

    def execute(self, model, clip, vae, instruction, seed=42, steps=4, guidance=3.0,
                target_pixels=DEFAULT_PIXELS, mu_mode="empirical", shift=0.0,
                image=None, reference_2=None, width=0, height=0, batch_size=1,
                positive=None, empty_instruction=None, noise=None):
        from .bluepixel.execution import run_edit
        references = [reference for reference in (image, reference_2) if reference is not None]
        output, latent, metadata = run_edit(model, clip, vae, instruction,
            references or None, steps, guidance, seed, target_pixels, width, height,
            mu_mode, shift, batch_size, prompt_embeds=positive, empty_embeds=empty_instruction,
            initial_noise=noise["samples"] if noise is not None else None)
        return output, latent, json.dumps(metadata, ensure_ascii=False)


class BluePixelTaskPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "jsonl_path": ("STRING", {"default": "tasks.jsonl"}),
            "diffusion_name": (folder_paths.get_filename_list("diffusion_models"),),
            "text_encoder_name": (folder_paths.get_filename_list("text_encoders"),),
            "vae_name": (folder_paths.get_filename_list("vae"),),
            "seeds": ("STRING", {"default": "42"}),
            "instruction_field": ("STRING", {"default": "edit_prompt"}),
            "limit": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            "vae_tiling": ("BOOLEAN", {"default": True}),
            "settings_json": ("STRING", {"multiline": True,
                "default": '{"steps":4,"guidance":3.0,"target_pixels":16777216,"mu_mode":"empirical","shift":0.0}'})}}
    RETURN_TYPES = ("BLUEPIXEL_TASK_PLAN", "STRING")
    FUNCTION = "plan"
    CATEGORY = "BluePixel/batch"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def plan(self, jsonl_path, diffusion_name, text_encoder_name, vae_name, seeds="42",
             instruction_field="edit_prompt", limit=0, settings_json="{}", vae_tiling=True):
        from .bluepixel.artifacts import sha256_file
        from .bluepixel.tasks import plan_tasks, inference_config
        settings = json.loads(settings_json)
        allowed = {"steps", "guidance", "target_pixels", "width", "height", "mu_mode", "shift", "noise_mode"}
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError(f"Only these inference settings are allowed: {sorted(allowed)}")
        paths = {"diffusion": folder_paths.get_full_path_or_raise("diffusion_models", diffusion_name),
                 "text_encoder": folder_paths.get_full_path_or_raise("text_encoders", text_encoder_name),
                 "vae": folder_paths.get_full_path_or_raise("vae", vae_name)}
        identities = {key: sha256_file(path) for key, path in paths.items()}
        config = inference_config(settings, identities, vae_tiling)
        settings = config["settings"]
        tasks = plan_tasks(jsonl_path, [int(value.strip()) for value in seeds.split(",")],
                           config, instruction_field, limit or None)
        plan = {"tasks": tasks, "config": config, "settings": settings, "model_hashes": identities}
        return plan, json.dumps({"total_tasks": len(tasks), "config": config}, ensure_ascii=False)


class BluePixelBatchRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("BLUEPIXEL_TASK_PLAN",), "model": ("MODEL",), "clip": ("CLIP",),
            "vae": ("VAE",), "output_dir": ("STRING", {"default": "bluepixel_batch"}),
            "resume": ("BOOLEAN", {"default": True}), "dry_run": ("BOOLEAN", {"default": False})},
            "optional": {"metadata_file": ("STRING", {"default": "metadata.jsonl"}),
                         "lock_timeout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "BluePixel/batch"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Task state, files and locks can change outside the ComfyUI graph.
        return float("nan")

    def run(self, plan, model, clip, vae, output_dir, resume=True, dry_run=False,
            metadata_file="metadata.jsonl", lock_timeout=0.0):
        from .bluepixel.artifacts import sha256_file
        from .bluepixel.tasks import TaskStore
        from .bluepixel.execution import run_tasks
        directory = Path(output_dir)
        if not directory.is_absolute():
            directory = Path(folder_paths.get_output_directory()) / directory
        if dry_run:
            result = {"dry_run": True, **TaskStore(directory, metadata_file).status(plan["tasks"])}
        else:
            if bool(vae.first_stage_model.autoencoder.use_tiling) != plan["config"]["vae_tiling"]:
                raise ValueError("Connected VAE tiling differs from the task plan; replan before running")
            for key, component in (("diffusion", model), ("text_encoder", clip), ("vae", vae)):
                path = getattr(component, "bluepixel_path", None)
                if path is None or sha256_file(path) != plan["model_hashes"][key]:
                    raise ValueError(f"Connected {key} differs from the task plan; replan before running")
            result = run_tasks(plan["tasks"], directory, model, clip, vae, plan["settings"],
                               resume, metadata_file, lock_timeout, config=plan["config"])
        text = json.dumps(result, ensure_ascii=False)
        return {"ui": {"text": [text]}, "result": (text,)}


class BluePixelTensorNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise": ("LATENT",), "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff})}}
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "create"
    CATEGORY = "BluePixel/sampling"

    def create(self, noise, seed=42):
        class ExplicitNoise:
            def __init__(self, samples, seed):
                self.samples, self.seed = samples, seed

            def generate_noise(self, latent):
                if self.samples.shape != latent["samples"].shape:
                    raise ValueError("Explicit noise shape differs from the destination latent")
                return self.samples.to(mm.intermediate_device())
        return (ExplicitNoise(noise["samples"], seed),)


class BluePixelEulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "create"
    CATEGORY = "BluePixel/sampling"

    def create(self):
        from .bluepixel.sampling import make_sampler
        return (make_sampler(),)


class BluePixelPackLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",),
            "convention": (["raw_dcae_bchw", "scaled_dcae_bchw"],)}}
    RETURN_TYPES = ("BLUEPIXEL_PACKED_LATENT",)
    FUNCTION = "pack"
    CATEGORY = "BluePixel/latent"

    def pack(self, latent, convention="raw_dcae_bchw"):
        from .bluepixel.latents import pack_latent
        return (pack_latent(latent["samples"], convention),)


class BluePixelUnpackLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"packed": ("BLUEPIXEL_PACKED_LATENT",)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "unpack"
    CATEGORY = "BluePixel/latent"

    def unpack(self, packed):
        from .bluepixel.latents import unpack_latent
        return (unpack_latent(packed),)


class BluePixelSeedListNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seeds": ("STRING", {"default": "42,43"}),
            "mode": (["upstream_cuda_bf16", "comfy_cpu"],)}}
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "create"
    CATEGORY = "BluePixel/sampling"

    def create(self, seeds="42,43", mode="upstream_cuda_bf16"):
        values = [int(value.strip()) for value in seeds.split(",")]
        if not values or any(value < 0 or value > 0xffffffffffffffff for value in values):
            raise ValueError("Provide one unsigned 64-bit seed per batch image")
        class SeedListNoise:
            seed = values[0]

            def generate_noise(self, latent):
                from .bluepixel.sampling import seed_list_noise
                device = mm.get_torch_device() if mode == "upstream_cuda_bf16" else torch.device("cpu")
                return seed_list_noise(latent["samples"].shape, values, device, mode).to(mm.intermediate_device())
        return (SeedListNoise(),)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    BluePixelDiffusionLoader, BluePixelCLIPLoader, BluePixelVAELoader,
    BluePixelTextEncode, BluePixelPrepareImage, BluePixelEmptyLatent,
    BluePixelScheduler, BluePixelNoise, BluePixelEditConditioning,
    BluePixelCFGGuider, BluePixelLatentInput, BluePixelDiagnostics,
    BluePixelEdit, BluePixelTaskPlan, BluePixelBatchRun, BluePixelTensorNoise,
    BluePixelEulerSampler, BluePixelPackLatent, BluePixelUnpackLatent, BluePixelSeedListNoise)}
NODE_DISPLAY_NAME_MAPPINGS = {name: name.replace("BluePixel", "BluePixel ", 1) for name in NODE_CLASS_MAPPINGS}
