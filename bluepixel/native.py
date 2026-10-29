"""Native ComfyUI model objects. Import only inside a ComfyUI process."""

import json
import logging
from pathlib import Path
import torch
import comfy.latent_formats
import comfy.model_management as mm
import comfy.model_patcher
import comfy.sd
import comfy.utils
from .artifacts import read_header
from .identity import file_stamp, mark_loaded, register_patcher_loader
from .config import CHAT_TEMPLATE, CONTEXT_DIM, FORMAT_VERSION, LATENT_CHANNELS, LATENT_SCALE, SPATIAL_RATIO


def artifact_metadata(path, component):
    metadata = read_header(path).get("__metadata__", {})
    if metadata.get("bluepixel.component") != component:
        raise ValueError(f"Select a converted BluePixel {component} file, not another model: {Path(path).name}")
    if metadata.get("bluepixel.format_version") != str(FORMAT_VERSION):
        raise ValueError("Unsupported BluePixel artifact version; run the matching converter")
    return metadata


class BluePixelLatentFormat(comfy.latent_formats.LatentFormat):
    latent_channels = LATENT_CHANNELS
    spacial_downscale_ratio = SPATIAL_RATIO
    scale_factor = LATENT_SCALE

    def process_out(self, latent):
        # Core hands FP32 samples to process_out; the source divides its stored
        # BF16 sample by the AE scale before converting the result to AE FP32.
        return (latent.to(torch.bfloat16) / self.scale_factor).float()


def load_diffusion(path, dtype="default", disable_dynamic=False):
    before = file_stamp(path)
    metadata = artifact_metadata(path, "transformer")
    header = read_header(path)
    from .config import validate_transformer_config
    from .conversion import expected_transformer_shapes
    config = json.loads(metadata["bluepixel.config"])
    validate_transformer_config(config)
    expected = expected_transformer_shapes(config)
    actual_weights = {k for k in header if k.endswith(".weight")}
    if actual_weights != set(expected):
        raise ValueError("Diffusion artifact has missing or unexpected model weights")
    for key, shape in expected.items():
        if tuple(header[key]["shape"]) != shape:
            raise ValueError(f"Incorrect diffusion weight shape: {key}")
    quantized = [key for key in header if key.endswith(".comfy_quant")]
    if metadata["bluepixel.precision"] == "int8_convrot" and not quantized:
        raise ValueError("The INT8 ConvRot file has no quantization metadata")
    options = {}
    if dtype != "default":
        options["dtype"] = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model = comfy.sd.load_diffusion_model(str(path), model_options=options, disable_dynamic=disable_dynamic)
    if model is None:
        raise ValueError("ComfyUI could not load the BluePixel diffusion model")
    actual = model.model.model_config.unet_config
    if actual.get("in_channels") != 128 or actual.get("patch_size") != 1 or actual.get("context_in_dim") != CONTEXT_DIM:
        raise ValueError("Loaded architecture does not match BluePixel")
    # Only this instance changes; ordinary FLUX.2 models keep their f16 format.
    model.model.latent_format = BluePixelLatentFormat()
    model.model.model_config.latent_format = model.model.latent_format
    # Unlike the generic Flux2 wrapper, the source accepts explicitly short text sequences.
    import types
    from comfy import model_base
    model.model.extra_conds = types.MethodType(model_base.Flux.extra_conds, model.model)
    from .sampling import source_timestep, source_post_cfg
    model.model.model_sampling.timestep = types.MethodType(source_timestep, model.model.model_sampling)
    model.set_model_sampler_post_cfg_function(source_post_cfg)
    model.bluepixel_metadata = metadata
    model.bluepixel_load_dtype = dtype
    mark_loaded(model, path, before)
    register_patcher_loader(model, load_diffusion, model.bluepixel_path,
                            dtype=dtype, disable_dynamic=disable_dynamic)
    logging.info("BluePixel: %s; f64 latents; %d native quantized layers", metadata["bluepixel.precision"], len(quantized))
    return model


class BluePixelTokenizer:
    """Exact upstream fixed-length no-thinking Qwen tokenization."""

    def __init__(self, tokenizer_json):
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
        self.serialized = tokenizer_json
        self.backend = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(tokenizer_json),
                                              pad_token="<|endoftext|>", eos_token="<|im_end|>",
                                              padding_side="right", truncation_side="right")

    def tokenize_with_weights(self, text, return_word_ids=False, max_sequence_length=512, **kwargs):
        max_sequence_length = int(max_sequence_length)
        if not 1 <= max_sequence_length <= 32768:
            raise ValueError("max_sequence_length must be in 1..32768")
        encoded = self.backend(CHAT_TEMPLATE.format(text), padding="max_length", truncation=True,
                               max_length=max_sequence_length, add_special_tokens=False)
        tokens = [(int(token), 1.0, 0) if return_word_ids else (int(token), 1.0)
                  for token in encoded["input_ids"]]
        return {"qwen3_8b": [tokens]}

    def state_dict(self):
        return {"bluepixel_tokenizer_json": torch.frombuffer(bytearray(self.serialized.encode()), dtype=torch.uint8)}

    def untokenize(self, token_weight_pair):
        return self.backend.decode([token[0] for token in token_weight_pair])


def load_clip(path, device="default", disable_dynamic=False):
    before = file_stamp(path)
    artifact_metadata(path, "text_encoder")
    state, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    serialized = state.pop("bluepixel_tokenizer_json", None)
    if serialized is None:
        raise ValueError("Text artifact is missing its tokenizer; reconvert the snapshot")
    state, _ = comfy.utils.convert_old_quants(state, metadata=metadata)
    options = {"dtype": torch.bfloat16}
    if device == "cpu":
        options.update(load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    clip = comfy.sd.load_text_encoder_state_dicts([state], clip_type=comfy.sd.CLIPType.FLUX2,
                                                model_options=options, disable_dynamic=disable_dynamic)
    # Generic CLIP uses FP32 token embeddings/activations. The published Qwen
    # reference uses BF16; padding hidden states can differ substantially if we
    # silently keep the generic activation convention.
    transformer = clip.cond_stage_model.qwen3_8b.transformer
    from .text_precision import install_reference_precision, install_reference_layer_options
    install_reference_precision(transformer)
    install_reference_layer_options(clip.cond_stage_model.qwen3_8b)
    clip.patcher.set_model_compute_dtype(torch.bfloat16)
    clip.tokenizer = BluePixelTokenizer(serialized.numpy().tobytes().decode("utf-8"))
    mark_loaded(clip, path, before)
    register_patcher_loader(clip.patcher, _load_clip_patcher, clip.bluepixel_path,
                            device=device, disable_dynamic=disable_dynamic)
    return clip


def _load_clip_patcher(path, device="default", disable_dynamic=False):
    return load_clip(path, device, disable_dynamic).patcher


class DCAEComponent(torch.nn.Module):
    """Tensor-only bridge around the autoencoder component, not a pipeline."""

    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def encode(self, image):
        return self.autoencoder.encode(image, return_dict=False)[0]

    def decode(self, latent, **kwargs):
        return self.autoencoder.decode(latent, return_dict=False)[0]

    def encode_tiled(self, image):
        # BluePixelVAE configures the source tile geometry before entering the
        # native VAE owned-tiling path. AutoencoderDC.encode honors use_tiling.
        return self.encode(image)

    def decode_tiled(self, latent):
        return self.decode(latent)


class BluePixelVAE(comfy.sd.VAE):
    def __init__(self, path, tiling=False):
        before = file_stamp(path)
        metadata = artifact_metadata(path, "vae")
        config = json.loads(metadata["bluepixel.config"])
        if config.get("latent_channels") != 128 or config.get("scaling_factor") != LATENT_SCALE:
            raise ValueError("Incorrect BluePixel DCAE configuration")
        from diffusers import AutoencoderDC
        state = comfy.utils.load_torch_file(str(path), safe_load=True)
        with torch.device("meta"):
            autoencoder = AutoencoderDC.from_config(config)
        autoencoder.load_state_dict(state, strict=True, assign=True)
        autoencoder = autoencoder.eval().requires_grad_(False).to(dtype=torch.float32)
        if tiling:
            autoencoder.enable_tiling()
        self.first_stage_model = DCAEComponent(autoencoder).eval()
        self.device = mm.vae_device()
        self.output_device = mm.intermediate_device()
        self.vae_dtype = torch.float32
        self.working_dtypes = [torch.float32]
        self.downscale_ratio = self.upscale_ratio = 64
        self.latent_channels = 128
        self.latent_dim = 2
        self.output_channels = 3
        self.pad_channel_value = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
        self.disable_offload = False
        self.not_video = False
        self.size = None
        self.downscale_index_formula = self.upscale_index_formula = None
        self.extra_1d_channel = None
        self.crop_input = True
        self.handles_tiling = False
        self.format_encoded = None
        # Conservative per-tile activations; ComfyUI owns allocation/offload and OOM fallback.
        self.memory_used_encode = lambda shape, dtype: 2048 * min(shape[-2], 512 if tiling else shape[-2]) * min(shape[-1], 512 if tiling else shape[-1]) * 4
        self.memory_used_decode = lambda shape, dtype: 2048 * min(shape[-2] * 64, 512 if tiling else shape[-2] * 64) * min(shape[-1] * 64, 512 if tiling else shape[-1] * 64) * 4
        self.patcher = comfy.model_patcher.ModelPatcher(self.first_stage_model,
            load_device=self.device, offload_device=mm.vae_offload_device())
        mm.archive_model_dtypes(self.first_stage_model)
        self.model_size()
        mark_loaded(self, path, before)
        register_patcher_loader(self.patcher, _load_vae_patcher, self.bluepixel_path, tiling=tiling)

    def _source_tiling(self, samples, encode, tile_x, tile_y, overlap):
        autoencoder = self.first_stage_model.autoencoder
        # ComfyUI decode tile dimensions are in latent units; encode dimensions are pixels.
        multiplier = 1 if encode else 64
        width = max(128, int(tile_x) * multiplier)
        height = max(128, int(tile_y) * multiplier)
        overlap = max(64, int(overlap) * multiplier)
        if width % 64 or height % 64 or overlap % 64 or overlap >= min(width, height):
            raise ValueError("DCAE tiles and overlap must align to 64 pixels, with overlap smaller than each tile")
        names = ("use_tiling", "tile_sample_min_height", "tile_sample_min_width",
                 "tile_sample_stride_height", "tile_sample_stride_width",
                 "tile_latent_min_height", "tile_latent_min_width")
        old = {name: getattr(autoencoder, name) for name in names}
        try:
            autoencoder.enable_tiling(tile_sample_min_height=height, tile_sample_min_width=width,
                tile_sample_stride_height=height - overlap, tile_sample_stride_width=width - overlap)
            if encode:
                return self._encode_tiled_owned(samples)
            return self._decode_tiled_owned(samples)
        finally:
            for name, value in old.items():
                setattr(autoencoder, name, value)

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        return self._source_tiling(pixel_samples, True, tile_x, tile_y, overlap)

    def decode_tiled_(self, samples, tile_x=8, tile_y=8, overlap=1):
        return self._source_tiling(samples, False, tile_x, tile_y, overlap)


def _load_vae_patcher(path, tiling=False, disable_dynamic=False):
    # This bridge already uses the non-dynamic ModelPatcher for every instance.
    return BluePixelVAE(path, tiling).patcher
