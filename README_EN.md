# ComfyUI BluePixel T8

[简体中文](README.md) | **English**

A native ComfyUI integration for [BluePixel-Turbo](https://github.com/vivo/BluePixel). Load the diffusion model, Qwen3 text encoder and DCAE-F64 separately, then connect them to ComfyUI's standard sampling and VAE nodes for four-step native 4K image editing.

[Models](https://huggingface.co/t8star/BluePixel-Comfy) · [Registry](https://registry.comfy.org/nodes/bluepixel-t8) · [Workflows](workflows) · [Validation](docs/validation.md)

## Features

- 20 nodes, with native INT8 ConvRot for the diffusion model and original precision for the text encoder and VAE.
- Standard `MODEL`, `CLIP`, `VAE` and `CONDITIONING` interfaces, including `SamplerCustomAdvanced` support.
- Single-image and multiple-reference editing, aspect-ratio preservation, scheduling options, seed batches and resumable JSONL jobs.
- Conversion with SFT / PEFT LoRA merging; external text conditioning, noise and latent inputs.
- Eight example workflows, each supplied as UI JSON and API JSON.

## Installation

Search for **BluePixel T8** in ComfyUI Manager, or install manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-BluePixel-T8.git
cd Comfyui-BluePixel-T8
python -m pip install -r requirements.txt
```

Use the Python environment that runs ComfyUI, then restart ComfyUI. Requires Python 3.10+, ComfyUI 0.43.0+ and Diffusers 0.40+. CUDA INT8 execution depends on PyTorch, comfy-kitchen and GPU support. ComfyUI supplies PyTorch.

## Models

Download these three files from [t8star/BluePixel-Comfy](https://huggingface.co/t8star/BluePixel-Comfy/tree/main) into the matching folders. Allow **28.19 GB** of disk space. Weights are distributed separately from the node package.

| Component | File | ComfyUI folder | Size |
| --- | --- | --- | --- |
| Diffusion · INT8 ConvRot | [bluepixel_turbo_int8_convrot.safetensors](https://huggingface.co/t8star/BluePixel-Comfy/resolve/main/diffusion_models/bluepixel_turbo_int8_convrot.safetensors) | `models/diffusion_models/` | 9.09 GB |
| Qwen3-8B · BF16 | [bluepixel_qwen3_8b_original.safetensors](https://huggingface.co/t8star/BluePixel-Comfy/resolve/main/text_encoders/bluepixel_qwen3_8b_original.safetensors) | `models/text_encoders/` | 16.39 GB |
| DCAE-F64 · FP32 | [bluepixel_dcae_f64.safetensors](https://huggingface.co/t8star/BluePixel-Comfy/resolve/main/vae/bluepixel_dcae_f64.safetensors) | `models/vae/` | 2.71 GB |

Use this package's three **BluePixel Loader** nodes, then connect their outputs to standard ComfyUI nodes. DCAE-F64 has 64× spatial compression; ordinary Flux2 VAE and empty-latent nodes are incompatible.

## Usage

1. Import the [1K preview](workflows/02_edit_1k_preview.json) or [4K editing](workflows/01_edit_4k_int8_convrot.json) workflow.
2. Select all three models and an input image, then enter an editing instruction.
3. Start with **4 steps and CFG 3**. The 4K example explicitly outputs 4096×4096.

Other examples cover aspect-ratio preservation, training μ, fixed shift, multiple references, experimental text-only generation and seed batches. See the [usage guide, in Chinese](docs/usage.zh-CN.md) for conversion and batch commands, and [advanced inputs](docs/advanced.md) for additional interfaces.

## Validation and licensing

Validated with 149 unit tests, 8/8 executed API workflows, an actual 4096×4096 edit, and text-encoder / VAE comparisons. INT8 quantization is lossy. Speed, memory use and output quality depend on hardware and task; see [validation](docs/validation.md) and the [public inference feature coverage](docs/function-coverage.md).

Node code is [GPL-3.0-or-later](LICENSE). Model weights retain their upstream terms, including the FLUX Non-Commercial License for FLUX.2 Klein 9B derivatives. This project does not relicense the weights. See [NOTICE](NOTICE) and the [model license](https://huggingface.co/t8star/BluePixel-Comfy/blob/main/LICENSE.md).

Credits: [vivo/BluePixel](https://github.com/vivo/BluePixel), [openvivo/BluePixel-Turbo](https://huggingface.co/openvivo/BluePixel-Turbo), [ComfyUI](https://github.com/Comfy-Org/ComfyUI), Qwen and NVIDIA DCAE.

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
