# ComfyUI-BluePixel

BluePixel 原生图像编辑节点与分体模型转换工具。扩散主模型支持 ComfyUI 原生 **INT8 ConvRot**；Qwen3 文本编码器和 DCAE-F64 自编码器独立加载，默认保留原始精度。

已完成公开推理功能的原生适配，实测完整 INT8 模型、1K/4096×4096 编辑、8组官方API工作流和高级输入。功能范围见[覆盖清单](function-coverage.md)，性能与量化误差见[验证记录](validation.md)，不以单张样例代表全部任务画质。

## 模型与数据口

| 组件 | 文件 | 原生接口 |
| --- | --- | --- |
| FLUX.2 Klein 衍生 9B DiT | `models/diffusion_models/bluepixel_turbo_int8_convrot.safetensors` | MODEL |
| Qwen3-8B | `models/text_encoders/bluepixel_qwen3_8b_original.safetensors` | CLIP |
| DCAE-F64，128 通道，scale 0.4538 | `models/vae/bluepixel_dcae_f64.safetensors` | VAE |

DiT 使用原生 `int8_tensorwise`、`convrot=true`、逐行 scale；矩阵输入维度决定 64/256 旋转分组，遵守 ConvRot 的 4 的整数幂约束。不是把浮点权重简单转成 INT8。非矩阵 norm 权重保留原精度，DCAE 默认 FP32。每个文件可以独立加载、卸载；无需在节点内部加载完整 Diffusers Pipeline。

主链使用 ComfyUI 原生 `ModelPatcher`、CLIP、VAE、ReferenceLatent、CFG guider 与采样器接口。`BluePixelEulerSampler` 保留原作者每步 BF16 舍入，直接接官方 SamplerCustomAdvanced。**BluePixel 与普通 Flux2 虽然都有 128 latent 通道，但分别是 64 与 16 倍空间压缩，不能互换空 latent 或 VAE。**

分体文件使用本项目的三个 BluePixel Loader 识别，随后连接官方采样与 VAE 节点。普通 UNET/CLIP/VAE Loader 直接识别全部组件不在此版本支持范围。

## 安装

把目录放入 `ComfyUI/custom_nodes/ComfyUI-BluePixel`，使用该 ComfyUI 的 Python 安装依赖：

```powershell
python -m pip install -r requirements.txt
```

本地开发目录可运行 `python tools/package_plugin.py` 生成当前版本的 `dist/ComfyUI-BluePixel-<version>.zip`，解压到 `custom_nodes`。模型另行放入 ComfyUI 的 `models/diffusion_models`、`models/text_encoders`、`models/vae`，不会包含在插件ZIP里。

Torch 由宿主 ComfyUI 提供，本插件不在加载时安装/替换 Torch。研究基线是 ComfyUI `0df1646a9cd71d4c990a87014fb37797a2956ead`；旧版如果缺少 Flux2/Klein、Qwen3-8B、INT8 ConvRot 支持，需要更新。当前工具需要 Python 3.10+、Diffusers 0.40+；原作者 requirements 的 0.36 限制不能直接作为正确骨干对照环境。

CUDA 模型推理与高效 INT8 kernel 的实际可用性还依赖 Torch、comfy-kitchen 和显卡。诊断输出会保留原生/模拟算子情况，不把 INT8 存储等同于提速保证。

## 下载与转换

先下载固定版本的官方快照，或使用已有本地快照：

```powershell
python tools/download_models.py --output D:/models/BluePixel-source --dry-run
python tools/download_models.py --output D:/models/BluePixel-source
python tools/convert_bluepixel.py --source D:/models/BluePixel-source --output D:/ComfyUI/models
```

默认导出 DiT INT8 ConvRot、原精度 Qwen3 和 FP32 DCAE 三个单体。源权重约 55.4 GB；还需输出和临时文件空间。转换逐张量执行，合并 QKV、交换最终 scale/shift 顺序，写入版本/config/tokenizer 元数据和 SHA256 manifest。已有文件默认拒绝覆盖，明确使用 `--overwrite` 才覆盖。

可只处理一个组件，或保留未量化主模型进行对照：

```powershell
python tools/convert_bluepixel.py --source D:/models/BluePixel-source --output D:/ComfyUI/models --component transformer --precision original
python tools/convert_bluepixel.py --source D:/models/BluePixel-source --output D:/ComfyUI/models --component transformer --precision int8_convrot --group-size 256
```

SFT 和 PEFT LoRA 在量化前合并；保持 base → full SFT → LoRA 的顺序：

```powershell
python tools/convert_bluepixel.py --source D:/models/BluePixel-source --output D:/ComfyUI/models --component transformer --sft-ckpt D:/models/sft.safetensors --sft-lora-ckpt D:/models/peft-adapter --lora-scale 0.8
```

PEFT 目录需 `adapter_config.json` 和 `adapter_model.safetensors`。支持普通线性 LoRA、alpha/rank patterns 和 rsLoRA；DoRA/其它 PEFT 类型会明确报错。负 `lora_scale` 保持原作者的“采用默认倍率”语义，0 表示不应用增量。标准 ComfyUI 格式 LoRA 也可在 MODEL 后接官方 LoRA 节点，但特定模型/适配器组合仍须验证。

## 工作流

`workflows/` 同时提供可编辑 UI JSON 和 API JSON：

- `01_edit_4k_int8_convrot`：明确输出 4096×4096，4 步、CFG 3、seed 42。
- `02_edit_1k_preview`：1024×1024 预览。
- `03_preserve_source_aspect`：按上游像素预算和比例选择输出。
- `04_training_mu`：训练 μ 调度。
- `05_fixed_shift`：固定 shift=3 调度。
- `06_multiple_references`：依次加入两张参考图，正负条件都携带相同参考；示例把第一张图的人物移入森林。
- `07_text_only_experimental`：源码允许的无参考图路径；画质不等同于专用文生图模型。
- `08_seed_list_batch`：同一参考图一次生成两张，分别使用 seed42/43。

加载后选择输入图片和三个模型。默认工作流使用 `SamplerCustomAdvanced` 官方节点；其输入为独立的 noise、guider、sampler、sigmas 和空 latent。`BluePixelEulerSampler` 通过原生 SAMPLER 接口补齐上游 BF16 step/CFG/time 舍入约定；普通 Euler 也能连接，但不采用相同的逐步精度约定。`BluePixelEdit` 是复用相同组件的便捷入口，支持无参考图、两张参考图、自定义 CONDITIONING 和初始 noise。

注意源代码的尺寸规则：按比例在 64 像素邻域搜索，比例相同时取第一个候选。因此 4096² 像素预算可能选中 4032²；图像 processor 又按 128 取整。比例工作流保留该行为，4K 工作流显式设置输出宽高为 4096，节点 metadata 给出实际参考/目标尺寸。

CFG >1 执行条件/空指令两路前向，两个分支使用相同参考图；CFG≤1 按上游只使用条件预测。负向条件默认是 Qwen3 空指令的真实编码，不是全零张量。使用 `BluePixelCFGGuider` 可保证这个边界。

默认 `BluePixelNoise` 在 CUDA BF16 生成上游风格噪声；`comfy_cpu` 提供 Comfy 通常的随机方式。同 seed 在这两种模式下不保证同 noise。精确对照用 `BluePixelTensorNoise` 注入同一张量。

## JSONL 批任务与续跑

每行一条任务，图片相对路径以 JSONL 所在目录为基准：

```json
{"image_name":"input.jpg","edit_prompt":"Change the gray sweater to burgundy red."}
```

```powershell
python tools/run_batch.py --comfyui D:/ComfyUI --diffusion D:/ComfyUI/models/diffusion_models/bluepixel_turbo_int8_convrot.safetensors --text-encoder D:/ComfyUI/models/text_encoders/bluepixel_qwen3_8b_original.safetensors --vae D:/ComfyUI/models/vae/bluepixel_dcae_f64.safetensors --jsonl tasks.jsonl --out-dir output --seeds 42,43 --dry-run
```

去掉 `--dry-run` 执行。可用 `--instruction-field`、`--limit`、`--metadata-file`、`--no-resume`、`--vae-tiling/--no-vae-tiling`、`--mu-mode`、`--shift`、`--width/--height`。单独 `--check --comfyui D:/ComfyUI` 检查环境，不加载模型；携带完整任务参数的 `--check` 显示任务状态。`BluePixelDiagnostics` 显示运行环境与已加载组件。CLI中全部任务已完成时，续跑直接退出并保留结果，不加载神经网络。

任务身份包含图片内容、指令、seed、模型文件内容哈希和推理设置。成功输出经过校验才跳过，删除/损坏结果会重新执行；失败任务可以重跑。跨进程锁由操作系统持有，不能仅因时间戳较旧就抢走仍存活任务。共享网络盘还需在该盘验证文件锁语义。多 GPU 使用各自的 runner 进程和 `--device-id`，不在一个图节点内启动隐式 worker。

`BluePixelTaskPlan` 不加载神经网络即可生成任务计划；图内 `BluePixelBatchRun` 连接真实组件执行。图内 dry-run 不会采样，但已经连接的 Loader 仍由 ComfyUI 按图依赖执行；完全不加载模型请使用 CLI dry-run 或只执行 TaskPlan。

高级接口见 [高级输入与回调](advanced.md)。`BluePixelPackLatent` / `BluePixelUnpackLatent` 在上游 packed tokens 和 ComfyUI 空间 latent 之间显式转换；`BluePixelSeedListNoise` 接官方采样节点，要求 seed 数等于 latent batch。

## 测试

```powershell
python -m pytest -q
python tools/verify_native.py --comfyui D:/ComfyUI --device cpu
```

单测覆盖模型文件读写、QKV 映射、ConvRot 数学、scheduler 与 Diffusers 对照、SFT/LoRA、任务身份与锁恢复。小骨干对照检查真实 Diffusers/ComfyUI 前向，完整权重文本、4K 输出和 INT8 画质检查单独记录。

本次交付覆盖已公开的 Turbo 模型与推理功能。官网尚未公开的 Agent、BluePixReward、专家工具和训练工程暂不开发。已有权重内的通用编辑技能可通过编辑指令使用。

模型来源：[vivo/BluePixel](https://github.com/vivo/BluePixel)、[openvivo/BluePixel-Turbo](https://huggingface.co/openvivo/BluePixel-Turbo)。模型与代码的授权材料不同，见 [NOTICE](../NOTICE)。模型文件和本地研究文件不包含在 Git 仓库中。
