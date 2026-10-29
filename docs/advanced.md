# 高级输入与回调

图接口使用真实 MODEL、CLIP、VAE、CONDITIONING、LATENT、NOISE、SAMPLER。标准 LATENT 一律是 DCAE 原始空间 `[B,128,H/64,W/64]`；缩放由模型边界完成。

`BluePixelPackLatent` 输出专用 `BLUEPIXEL_PACKED_LATENT`，包含按行排列的 `[B,L,128]`、四维坐标、输出宽高和 `scaled_dcae_blc` 声明。`BluePixelUnpackLatent` 支持坐标重排，检查缺失、重复、范围和参考图 T 坐标后返回原生 LATENT。原始 BF16 缩放/除法会舍入，格式转换不保证恢复已经舍入丢失的位；要保留原始 packed 输出，可在 step-end 回调读取。

每图 generator 列表通过 `BluePixelSeedListNoise` 表达，seed 数必须等于 batch；已有噪声张量通过 `BluePixelTensorNoise` 注入。参考图列表属于同一次编辑的条件；JSONL 多任务则由 TaskPlan/BatchRun 管理。参考图 batch 应为1，由原生 conditioning 按输出 batch 复制。

`BluePixelTextEncode` 支持固定最大长度、三个不同的 layer taps（默认9/18/27，允许0..36并保持用户顺序），输出宽度始终12288。自定义 CONDITIONING 需要浮点、有限值、batch=1或输出 batch。上游 `caption_upsample_temperature` 未执行 caption upsampler，本项目保留这一事实。

Python 执行接口 `bluepixel.execution.run_edit` 的 `callback` 是标准 ComfyUI 采样进度回调；另提供 `callback_on_step_end`，在完成每一步 BF16 Euler 更新后执行，兼容上游回调的四参数形式：

```python
def after_step(context, index, timestep, values):
    # values["latents"]: scaled BF16 [B,L,128]
    # values["prompt_embeds"]: [B,sequence,12288]，仅在 tensor_inputs 指定时提供
    return {"latents": values["latents"]}

image, latent, metadata = run_edit(
    model, clip, vae, "Change the sweater to burgundy red.", image=input_image,
    callback_on_step_end=after_step,
    callback_on_step_end_tensor_inputs=("latents", "prompt_embeds"),
)
```

`context` 是本次采样的控制对象，不是 Diffusers Pipeline。回调可返回同形状的 `latents` 和/或 `prompt_embeds`；只更新正向文本，空指令分支与参考图保持原值。更新包含最终一步。禁止改变 batch/长度/宽度、返回 NaN/Inf 或未知字段。使用模型 clone 和实例 wrapper 传递更新；先前已有的 wrapper 会继续调用。取消通过 ComfyUI 的标准中断接口传播，批任务锁在退出时释放。

上游 `attention_kwargs` 的 LoRA scale 对应转换器的 `--lora-scale` 或官方 LoRA 节点的 strength；不将任意 Diffusers processor 参数转发给不同实现。SFT/PEFT 在量化前合并。DoRA 和其它未实现适配器会报错。

按本次交付范围，官网展示但尚未公开的 Agent、BluePixReward、专家工具与训练工程暂不开发。通用编辑能力以公开 Turbo 权重实际输出为准。
