# 已自行调研的资源与执行选择

核查日期：2026-11-01。所有资源由本项目自行检查硬件、访问官方来源并核验可得性；下文区分已获取、可公开获取与尚未运行，不把下载链接或缓存目录视为已经完成实验。

## 当前决定

| 资源 | 已落实内容 | 下一阶段使用方式 |
| --- | --- | --- |
| 本机算力 | RTX 5070 Laptop，8151 MiB 显存；31.50 GiB RAM；当前 torch 为 CPU 版 | 先完成真实数据与协议检查；CUDA 环境在独立目录验证后再运行本地 GPU |
| 模型权重 | 已有完整 Qwen3-VL-2B 缓存；Qwen2.5-VL-3B/7B 官方权重和固定 revision 已核实 | 2B 可作为后续本地适配候选；原定 3B/7B 主实验使用显存更充足的配置，不能混称同模型结果 |
| 即时视频与答案 | 已下载官方 Perception Test sample：8 视频、36 MCQA、71 动作标注 | 用来验证真实视频读取、时间坐标、答案格式及任务设计 |
| 长程第一视角数据 | EPIC-KITCHENS 官方单视频和动作 CSV 已核实；已下载短片 P03_15 及其 3 条真实动作标注 | 主线数据候选，按主体划分；根据动作/时间标注定义可核实的派生问题 |
| 长视频/记忆外部评估 | 已核查 LongVideoBench、Video-MME、EgoSchema、Memento 等访问方式 | 对开放标签与视频授权分别检查，作为独立外部验证，不把它们自动视为同一种长期记忆任务 |
| 实验预算 | 优先在本机检查小样本；为 3B/7B 查明 24/48 GB 云 GPU 候选及公开价 | 尚未创建实例、预订或付费；完整预算以实测吞吐和 paired readout 次数估算 |

详细依据分别见 [算力与模型](compute_resources.md)、[数据集资源](dataset_resources.md)、[实验任务与约束](resource_experiment_plan.md)。

## 已下载的真实资源包

位置：`data/perception_test_sample/`。官方来源及下载字节数、SHA-256、时间、ETag 保存于 `provenance.json`；原始 README 和数据归属说明同时保存。

| 文件 | 内容 |
| --- | --- |
| `media/videos/*.mp4` | 8 段原始视频，未转码 |
| `annotations/sample.json` | 官方原始多任务标注，未修改 |
| `videos.jsonl` | 视频路径、原始 metadata、原 train/valid split、文件哈希 |
| `questions.jsonl` | 36 道官方 MCQA，保留选项及 0-based answer index |
| `memory_questions.jsonl` | 官方 `area=memory` 的 1 道题 |
| `actions.jsonl` | 71 个动作片段，官方微秒时间换算为秒 |
| `media_checks.json` | 每个视频首、中、末帧解码结果与帧数核对 |
| `UPSTREAM_README.md` / `ATTRIBUTION.md` | 官方说明快照、CC-BY-4.0 归属与转换说明 |

下载视频 archive 225,370,639 bytes，标注 archive 2,305,618 bytes。8 个视频的文件哈希已核验、24 个抽样帧均可解码，OpenCV 读取的帧数全部与官方标注相符。这里只验证抽样解码与完整文件哈希，没有宣称逐帧 PTS 都均匀。

复现获取命令（输出目录须为空）：

```powershell
python -m spectral_memory.prepare_resources --output-dir data/perception_test_sample_new
```

下载器限制单个视频 archive 小于 256 MiB，检查解压路径、禁止符号链接和目录穿越，保留来源/哈希。`--annotations-only` 可仅获取标注。当前已获取的目录可直接使用，不必重复下载。

## 这些样例能证明什么

样例包含一个明确的开头/结尾顺序比较：`video_1580` 的问题要求判断打乱后的字母顺序是否与开头一致，官方答案为 `No`。这适合检查“任务需要比较时间状态”的数据和回答链路。

但 36 道问答不等于 36 道长期记忆题：官方只有 1 道标为 memory，其余包括语义、物理和抽象任务。这 8 段短视频不够构成论文主评估，也不支撑跨主体、跨会话或长时记忆结论。资源 JSONL 中未知主体/会话保存为 null，没有把视频 ID 伪装成主体 ID。

官方正确答案只是任务 ground truth，**不是** `R(with_memory)-R(without_memory)`。当前资源包没有任何伪造的 utility；所有问答标记 `utility_status=not_evaluated`。它不是可以直接传给 `spectral_memory run` 的已标注 probe manifest。

## 长程主线的选材决定

优先 EPIC-KITCHENS 的连续第一视角视频，因为其官方动作标注提供 participant ID、video ID、动作起止时间与 verb/noun。候选 `P01_01` 约 27.54 分钟，官方文件约 6.218 GB；本轮只核查远端可得性，没有下载这个大视频。可先用小体积单视频验证解码与标注，再按主体分层抽取连续片段。

EPIC 的动作标签不能直接当成“最后物体位置”的答案。第一批可从明确标注生成动作先后/发生时间问题，记录派生规则，并人工或程序核查边界；最终位置、动作完成状态、实体身份歧义需要额外证据确认。不能凭动作名称自动补出未标注事实。

第一视角短样例已实际获取到 `data/epic_kitchens_p03_15/`：`P03_15.MP4` 为 39,548,685 bytes，时长 10.477133 秒；原始 train CSV 明确给出主体 `P03`，筛选得到 3 条动作标注，保存于 `official_actions.jsonl`。已核验完整视频 SHA-256 并解码首/中/末 3 帧，结果写入 `media_checks.json`；原 CSV、视频信息及来源哈希同时保留。它没有原生未来问答或记忆效用标签，亦不代表长片实验已完成。

Memento 的公开对话标注可作为与长期流式记忆更贴近的后续资源，但其视频依赖 Ego4D 授权；不把公开代码许可当成底层视频许可。LongVideoBench/Video-MME 的大视频包和公开答案可用性逐项见数据资源表，当前不整库下载。

## 剩余工程边界

1. 已在独立 `.venv-cuda` 安装并验证 CUDA PyTorch，完成真实 GPU 推理；没有修改现有全局 CPU 安装。
2. 已缓存 Qwen3-VL-2B 的专用适配器已完成；已有 Qwen2.5-VL 提取器仍不支持直接替换模型。
3. 公开样例无主体身份。后续 loader 要允许真实 unknown 身份，并显式使用 video-level 划分；当前不生成假 subject ID 来满足旧 schema。
4. 问题涉及整段视频开头/结尾，采样必须覆盖全片；不能只截最前 16 帧就假定保留答案证据。非均匀 PTS 必须按明确规则重采样，记录实际时间格。
5. 已接入真实 readout，完成 36 QA × 4 条件，产出 108 条相对空记忆的实测效用。仍未训练真实效用探针，也未实现固定存储预算的记忆算法。问答载荷与完整视频 hidden NPZ 不同，不能直接按视频 ID 强行合并标签。

本轮资源调研、可得性核验、小样例获取、来源归档、真实模型读出和分层提取已完成；未发生云 GPU 采购。实际结果见 [小模型试点](small_model_pilot.md)，后续继续沿公共资源路线扩展。
