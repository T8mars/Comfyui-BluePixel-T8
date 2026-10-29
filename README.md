# ComfyUI BluePixel T8

**简体中文** | [English](README_EN.md)

[BluePixel-Turbo](https://github.com/vivo/BluePixel) 的 ComfyUI 原生适配：独立加载扩散模型、Qwen3 文本编码器和 DCAE-F64，连接官方采样与 VAE 节点，支持四步原生 4K 图像编辑。

[模型下载](https://huggingface.co/t8star/BluePixel-Comfy) · [节点注册表](https://registry.comfy.org/nodes/bluepixel-t8) · [示例工作流](workflows) · [验证记录](docs/validation.md)

## 功能

- 20 个节点；扩散主模型使用原生 INT8 ConvRot，文本编码器与 VAE 保留原始精度。
- 返回标准 `MODEL`、`CLIP`、`VAE`、`CONDITIONING` 等接口，支持 `SamplerCustomAdvanced`。
- 单图编辑、多参考图、比例保持、调度设置、种子批量与 JSONL 断点续跑。
- 提供 SFT / PEFT LoRA 合并转换、外部文本条件、噪声和 latent 输入接口。
- 8 组工作流，每组同时提供 UI JSON 和 API JSON。

## 安装

在 ComfyUI Manager 中搜索 **BluePixel T8**，或手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-BluePixel-T8.git
cd Comfyui-BluePixel-T8
python -m pip install -r requirements.txt
```

请使用运行 ComfyUI 的 Python，然后重启 ComfyUI。要求 Python 3.10+、ComfyUI 0.43.0+、Diffusers 0.40+；CUDA INT8 运算取决于 PyTorch、comfy-kitchen 与显卡支持。PyTorch 由宿主 ComfyUI 提供。

## 模型

从 [t8star/BluePixel-Comfy](https://huggingface.co/t8star/BluePixel-Comfy/tree/main) 下载以下三个文件，放入对应目录。约需 **28.19 GB** 磁盘空间，插件本身不包含权重。

| 组件 | 文件 | ComfyUI 目录 | 大小 |
| --- | --- | --- | --- |
| 扩散主模型 · INT8 ConvRot | [bluepixel_turbo_int8_convrot.safetensors](https://huggingface.co/t8star/BluePixel-Comfy/resolve/main/diffusion_models/bluepixel_turbo_int8_convrot.safetensors) | `models/diffusion_models/` | 9.09 GB |
| Qwen3-8B · BF16 | [bluepixel_qwen3_8b_original.safetensors](https://huggingface.co/t8star/BluePixel-Comfy/resolve/main/text_encoders/bluepixel_qwen3_8b_original.safetensors) | `models/text_encoders/` | 16.39 GB |
| DCAE-F64 · FP32 | [bluepixel_dcae_f64.safetensors](https://huggingface.co/t8star/BluePixel-Comfy/resolve/main/vae/bluepixel_dcae_f64.safetensors) | `models/vae/` | 2.71 GB |

三个组件使用本项目的 **BluePixel Loader** 加载，再连接官方节点。DCAE-F64 的空间压缩倍率为 64，不能与普通 Flux2 的 VAE 或空 latent 混用。

## 使用

1. 导入 [1K 预览工作流](workflows/02_edit_1k_preview.json) 或 [4K 编辑工作流](workflows/01_edit_4k_int8_convrot.json)。
2. 选择三个模型和输入图片，填写编辑指令。
3. 以 **4 步、CFG 3** 开始，运行工作流。4K 示例显式输出 4096×4096。

其余示例包括保留比例、训练 μ、固定 shift、多参考图、实验性无参考图生成和种子批量。完整操作、转换和批处理见[使用说明](docs/usage.zh-CN.md)，高级接口见[高级输入](docs/advanced.md)。

## 验证与许可

已完成 225 项单测、8/8 API 工作流执行、实际 4096×4096 编辑，以及文本编码 / VAE 对照。INT8 是有损量化；速度、显存与画质随硬件和任务变化。测试条件与误差见[验证记录](docs/validation.md)，公开推理功能范围见[覆盖清单](docs/function-coverage.md)。

节点代码使用 [GPL-3.0-or-later](LICENSE)。模型保留各上游许可，FLUX.2 Klein 9B 衍生权重受 FLUX 非商业许可约束；本项目不重新授权模型。详见 [NOTICE](NOTICE) 和[模型许可](https://huggingface.co/t8star/BluePixel-Comfy/blob/main/LICENSE.md)。

感谢 [vivo/BluePixel](https://github.com/vivo/BluePixel)、[openvivo/BluePixel-Turbo](https://huggingface.co/openvivo/BluePixel-Turbo)、[ComfyUI](https://github.com/Comfy-Org/ComfyUI)、Qwen 和 NVIDIA DCAE。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
