# 公开功能覆盖

本次范围为固定版本的官方Turbo模型、`vins_infer.py`和批脚本。网页尚未公开的Agent、自动评分/选图、专家工具、训练工程，按用户要求不纳入。

| 上游功能 | ComfyUI交付 | 实际验证/边界 |
| --- | --- | --- |
| 指令编辑，steps/CFG/seed | TextEncode、CFGGuider、EulerSampler及官方采样器 | 4步CFG3；CFG0/1条件单路latent一致 |
| 4K与像素预算 | PrepareImage、F64 EmptyLatent | 4096²和3200×5312完整API执行；64候选/128预处理分开报告 |
| empirical/training/fixed调度 | BluePixelScheduler | 公式单测及3种实际API工作流 |
| DCAE全图/tiling | 真实comfy.sd.VAE与官方VAE节点 | 512全图/1024 tiled与源逐元素一致；4K解码完成 |
| 独立加载/卸载 | MODEL / CLIP / VAE三个Loader | 独立文件，原生ModelPatcher管理，不加载完整Diffusers Pipeline |
| INT8 ConvRot | 流式转换器、原生int8_tensorwise | 121矩阵；GPU实际整数GEMM；norm/CLIP/VAE精度单独保留 |
| Full SFT覆盖 | 转换前SFTOverlay | prefix/形状/未知key报告，缺失项使用base并记录 |
| PEFT LoRA与倍率 | 转换前合并、官方LoRA接口 | 普通线性LoRA、rank/alpha patterns、rsLoRA；DoRA/其它类型明确拒绝 |
| JSONL与自定义指令字段 | TaskPlan、BatchRun、CLI | 中文、空行、坏行、相对路径、limit与多seed |
| 输出目录/元数据 | 原子PNG、JSONL、_done/_configs | 内容身份含模型SHA、配置、图片、指令、seed；不沿用不完整的上游命名身份 |
| resume/no-resume | TaskStore与runner | 真实两任务生成及不加载模型续跑；坏结果重算、显式重跑 |
| 并发/崩溃恢复 | 跨进程FileLock | Windows独立进程验证活锁不可被旧时间戳抢走；强制退出自有worker后可恢复 |
| dry-run/check/device-id | CLI与Diagnostics | 纯计划/状态不加载模型；独立进程选GPU；节点不修改全局设备 |
| 自定义embeddings/层/长度 | CONDITIONING、TextEncode | 默认512和自定义127/层36,0,18四组均逐元素一致；自定义后默认恢复一致 |
| 输入noise与generator列表 | TensorNoise、SeedListNoise | CUDA BF16源噪声、Comfy CPU噪声；seed数等于batch |
| num_images_per_prompt | 原生latent batch | 官方batch2图实际输出两张，不等同JSONL任务列表 |
| latent/图像输出 | 官方Sampler LATENT、VAEDecode IMAGE | DCAE raw BCHW；显式packed导入导出检查坐标与缩放 |
| step-end callback | run_edit高级接口、原生SAMPLER/wrapper | 实测更新正向文本与最终latent；标准Comfy中断/进度 |
| attention kwargs中的LoRA scale | 合并倍率或原生LoRA strength | 不将任意Diffusers processor参数错误转发到另一实现 |
| 无参考图 | 实验T2I图与Edit可选image | 实际API出图；不承诺专用T2I画质 |
| 多参考图 | 反复追加ReferenceLatent或Edit输入 | 两张图/batch2实际执行；链式路由修复后，INT8示例完成森林背景替换 |
| caption_upsample_temperature | 保留上游事实说明 | 源码接受但未执行，不伪造caption upsampler |

工作流的API执行、模型数值、图像质量分别验收。接口可运行不意味着每条指令都成功；量化和源模型的能力限制见[验证记录](validation.md)。
