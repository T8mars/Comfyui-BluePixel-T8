# 验证记录

日期2026-10-29。环境：Python3.10.10、Torch2.7.0+cu128、RTX5090 Laptop 24GB、ComfyUI `0df1646a9cd71d4c990a87014fb37797a2956ead`、隔离 Diffusers0.40.0。共享 ComfyUI 的 Torch/依赖未被替换。

## 实际模型文件

| 文件 | bytes | SHA256 |
| --- | ---: | --- |
| bluepixel_turbo_int8_convrot.safetensors | 9,085,830,591 | `00cbd3100708c072bcae85f7322851f8f96282915898769345e49c6b1bb76b8e` |
| bluepixel_qwen3_8b_original.safetensors | 16,392,941,666 | `baa5482d84d89b3cc11154669d468d5d32815563790cc9d29e2ac1edf48e5d26` |
| bluepixel_dcae_f64.safetensors | 2,707,684,888 | `e118ff218bb28cbc150777aaef0a50796938c8ba150cf4a90ba2e6e7f538d778` |

文本399个源张量、DCAE418个源张量（含I64 buffers）逐值与dtype复核。源DiT233个张量映射为原生201个参数；121个矩阵采用真正INT8 ConvRot，非矩阵norm保持原精度。旋转组为64/256，必须是4的整数幂。

## 已执行的正确性检查

- 完整单测107项通过。覆盖流式原子文件、key/shape、QKV融合、最终调制顺序、ConvRot数学、SFT/PEFT合并、调度、尺寸、内容身份、锁/恢复、packed坐标、逐图seed、step-end回调、backend偏好恢复、RoPE舍入回归、完整配置追溯与链式参考路由。
- 3种scheduler × 6种token length × 4种steps逐项对照Diffusers，包含4300/4301分段与终点0。
- 小型Flux2带参考图与四维IDs，Diffusers/ComfyUI CPU FP32前向：max abs `2.980232238769531e-7`，relative L2 `1.0175625675401534e-7`。
- 全量INT8骨干前向finite；Profiler记录121次 `comfy_kitchen::int8_linear` 和121次 `aten::_int_mm`，峰值9,821,536,768 bytes。并非仅INT8存储或浮点模拟。
- 完整Qwen3：英文、空指令、中文、超长截断四组token ids及完整 `[1,512,12288]` embedding全部逐元素一致，max/mean/relativeL2均为0。
- 高级文本：长度127、层顺序36/0/18四组亦全部逐元素一致；调用后恢复默认512/层9/18/27再对照，差异仍为0。修复了通用Qwen默认层列表会忽略自定义列表的问题，适配限定在BluePixel实例。
- 文本修复保持BF16激活、RMSNorm先舍入后乘权重、BF16 RoPE和GQA；Windows cu128无FlashAttention，Comfy低精度math归约曾使长文本误差达3.2%。最终剩余误差来自RoPE inverse frequencies的CPU/GPU pow差异，在token459的两个通道跨过BF16舍入边界；已按源CPU初始化复现。调用内保持源归约精度并恢复宿主偏好，未持久更改其他模型配置。
- DCAE512全图、1024官方VAE Tiled入口对照：编码与同输入解码max abs均为0，1024 tiled完整roundtrip为0。独立验证发现同值但不同stride可引入约3.48e-5的源CUDA解码误差，因此另报告输入布局，未放宽原门槛掩盖错误。
- 完整高级路径：无参考图CFG0/1 latent逐元素相同；双参考、batch2、逐图seed42/43、trainingμ、真实回调替换文本与最后latent均通过；畸形自定义embedding被拒绝。
- 官方ComfyUI API工作流8/8均已完整执行，包含干净ZIP解压安装的4096×4096完整链（129.88秒，含模型/文本加载）；保比例3200×5312、trainingμ、fixed shift、双参考、无参考、batch2都有真实PNG和history。首次1024²全链约51.4秒，缓存后约6.1秒。
- JSONL真实批任务两个seed均成功；最终规范化配置续跑skipped2、models_loaded=false，文件哈希复核后保留原图。任务指纹可在`_configs`反查模型SHA和完整推理设置。

## 完整编辑与BF16对照

下表使用预先核验的文本embedding，计时从组件CPU加载之后开始，包含参考编码、GPU采样与解码；不包含Qwen编码或组件CPU文件加载。三组件完整API链的含加载耗时另列为129.88秒，不将下表峰值当作整套软件最低显存要求。

| INT8案例 | steps / CFG | 采样秒 | 总秒 | 峰值allocated bytes |
| --- | --- | ---: | ---: | ---: |
| 保比例768×1280 | 4 / 3 | 10.36 | 19.47 | 13,131,191,808 |
| 显式4096×4096 | 4 / 3 | 38.42 | 75.95 | 17,827,159,552 |

均为finite实际图片，已查看。原作者推理主体亦已执行同prompt/seed/尺寸，使用预先核验的Qwen embedding；仅修复其缺失属性别名/回调locals错误，并在4K对照分阶段offload。源脚本、补丁和逐步latent均保留在本地证据中。

INT8相对源BF16的图像PSNR：1K预算29.52dB，4K26.85dB；mean abs分别0.00991/0.01405（0..1）。这些指标包含量化和原生kernel差异。最终latent relative L2分别0.156/0.234，不能宣称无损或画质等价。人工检查：主要换色有效，人物/衣服细节存在变化；源BF16在4K也有部分细纹。

额外原生BF16对照：同1K预算、prompt/seed/调度，图像PSNR35.35dB、最终latent relativeL2=0.05495；说明原生backend也存在BF16数值差异，不能把全部误差算为量化损失。双参考初测的拼接/风格漂移，定位为链式追加把路由字符串拼成`indexindex`；已改为仅追加latent列表、固定路由标量。干净安装包复测完成森林背景替换（58.89秒），默认单参考亦再验证成功；旧失败证据保留，不将其归因于量化。源BF16与原生BF16的双参考对照均完成相同意图，人物细节仍有差异。无参考图为源API允许的实验功能。

以上均为单机少量样例，无预热/多轮统计。源4K曾受主存压力影响，不能据此建立速度倍数；cu128未启用部分cu130优化kernel。未复现网页所述P800 XPU耗时，不承诺特定提速或最低显存。

## 交付与限制

- 20个节点，8组UI/API工作流，128个API节点连接已通过真实ComfyUI类型检查；完整图执行单独记录。
- 最终ZIP清单不含`roadmap.md`、`.research/`、模型权重；实际从ZIP干净解压测试，生产源码哈希与工作区相同。
- UI视觉/拖拽验收尚未完成：Computer Use无法可靠判定Windows Chrome URL，工具中止了那次UI操作。没有用其它UI自动化绕过该限制。
- 全类别编辑质量、共享网络盘锁、多GPU/多机器硬件、新环境安装仍需相应环境测试。
- 按用户确认，未公开的Agent/BluePixReward/专家工具/训练工程不在本次交付范围。

本地证据不进入Git。源码公开材料只保留验证结论与可重跑工具，不包含机器任务台账或私有规划。
