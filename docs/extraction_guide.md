# Qwen2.5-VL / Qwen3-VL 隐状态提取协议

本项目将预采样本地 RGB 帧变成 `hidden_states[T,D]` 和真实时间坐标。提取过程不生成未来问题或效用标签。Qwen2.5-VL 与 Qwen3-VL 使用独立适配器，不能互换模型类或套用同一视觉 Token 布局。下述原协议描述 Qwen2.5-VL；Qwen3 的差异与实测见末节。

## 依赖与一手依据

实现锁定 `transformers==4.57.1`，使用官方慢速 `Qwen2VLImageProcessor` 视频路径、`AutoTokenizer` 和 `Qwen2_5_VLForConditionalGeneration.model`。最后一个调用跳过词表 logits，返回 decoder 隐状态。这个版本的慢速视频路径仍可用，但已标记弃用，不能直接升级 Transformers 5 后照搬。

可选环境需要 `torch`、`transformers==4.57.1`、`Pillow` 和基础 `numpy`；无需为预采样帧引入 torchvision、视频解码器或 qwen-vl-utils。本次实际验证环境为 Python 3.13.7、torch 2.10.0、Pillow 11.3.0、numpy 2.4.2。GPU torch 版本应与运行机器匹配。

一手来源（核查日期：2026-11-01）：

- [4.57.1 模型文档](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)。
- [processor 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py)：视频 token 数为 `prod(video_grid_thw)/merge_size²`，时间间隔为 `temporal_patch_size/fps`。
- [慢速预处理源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)：连续帧组成 temporal patch；无法整除时会补帧。本适配器拒绝这种输入。
- [模型源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)：视觉窗口重排在进入语言模型前恢复；最后一项 hidden state 包含最终 RMSNorm。
- [官方模型配置](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json)。官方 processor、预处理源码和模型配置的原始 URL 均返回 HTTP 200；正式实验需固定 revision，不能把可变 main 当固定快照。

## 输入 JSONL

一行对应一个事件，`sample_id` 必须唯一。帧路径相对于输入 JSONL 所在目录，或使用绝对路径。时间戳使用原视频中的秒数，并落在事件区间内。

```json
{"sample_id":"subject01_session01_event001","video_id":"video01","subject_id":"subject01","session_id":"session01","event_start_s":10.0,"event_end_s":18.0,"frames":["frames/000.png","frames/001.png","frames/002.png","frames/003.png","frames/004.png","frames/005.png","frames/006.png","frames/007.png","frames/008.png","frames/009.png","frames/010.png","frames/011.png","frames/012.png","frames/013.png","frames/014.png","frames/015.png"],"timestamps_s":[10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,15.5,16.0,16.5,17.0,17.5],"split":"train"}
```

可选字段如 `query_id`、`memory_state_id`、`evaluation_group`、`cost`、`cost_unit`、`utility`、`context` 原样保留，**不会拼入模型提示词**。`run` 阶段需要任务/收益字段；提取阶段可以暂缺，随后用 `labels` 命令按成对回答日志补齐。未来答案或人工真值不能进入部署时可见的 `context`。

上游采样应记录原视频、解码器、目标帧率和实际选中帧时间戳。本脚本不解码视频、不猜帧率、不伪造时间坐标。可变帧率输入须经过明确的上游重采样并保留记录；不能仅重写时间戳以通过检查。

## 运行

```powershell
python -m spectral_memory.extract_qwen --help
python -m spectral_memory.extract_qwen --input data/frames.jsonl --output-dir runs/qwen3b_hidden --model Qwen/Qwen2.5-VL-3B-Instruct --device cuda --layers auto
python -m spectral_memory run --manifest runs/qwen3b_hidden/manifest.layer_0.jsonl --output-dir runs/qwen3b_probe
```

最后一条命令需先将真实成对回答日志产生的 utility 及任务字段合入 manifest，不能把未标注的提取结果直接用于训练。`--layers auto` 选七个均匀分布的 hidden-state tuple 索引，也可显式传入模型范围内的 `--layers 0,7,14,21,28`。

索引 0 是 decoder 输入嵌入，视觉位置已替换为视觉投影输出；1 到 N 是后续 decoder 状态，N 包含最终 RMSNorm。它们不是 ViT 层。不同层各有独立 manifest，不应混成同一训练集。

默认只读本地缓存；缺少权重时明确报错。显式 `--allow-download` 才允许下载。正式实验用 `--revision <commit>` 固定版本；本地模型还应记录权重校验和。默认 CPU 为 float32；支持 bf16 的 CUDA 默认 bfloat16，可用 `--dtype` 覆盖。

输出目录必须不存在或为空，避免覆盖旧实验。异常后 `extraction.json` 保持 `status: incomplete`；全部文件写完才标记 complete。修正错误后使用新的输出目录，不把残留输出当完整数据。

## 时间和 Token 对齐

1. 帧时间戳严格递增且均匀，容差 `rtol=1e-5, atol=1e-7`；至少产生八个时间格。
2. 帧数必须能被 `temporal_patch_size` 整除，拒绝自动补帧。
3. 从模型配置获取 `video_token_id` 并与 tokenizer 核对，仅选择该 token，不依赖 `mm_token_type_ids`。
4. 校验单视频、连续 video-token 区间、空间网格可整除、token 总数与时间格数量。
5. reshape 为 `[grid_t, grid_h*grid_w/merge_size², d]`，沿空间维度平均池化。
6. 时间格坐标取对应帧时间戳均值；显式设置 `second_per_grid_ts=temporal_patch_size/input_fps`。

16 帧按 2 FPS 采样，temporal patch 大小为 2，输出 8 个时间点，隐状态采样率为 **1 Hz**，Nyquist 为 **0.5 Hz**。输入仅 1 FPS 时，隐状态采样率是 0.5 Hz，Nyquist 只有 0.25 Hz，必须下调频带上界。不能把原始帧率当作隐状态采样率。

脚本不做时间重采样。默认每帧最多 `128*28*28` 像素、最多 8192 个序列 token；超限报错，要求上游明确调整片段或采样。所有对照须保持相同像素和 token 限制。

## 输出和溯源

输出包含 `extraction.json`、每层 `manifest.layer_N.jsonl`、`hidden/*.layer_N.npz`。每个 NPZ 含 `hidden_states: float32[T,D]` 和 `timestamps_s: float64[T]`。`hidden_path` 相对于输出 manifest。

manifest 中原有 `timestamps_s` 仍指输入帧；新增 `hidden_timestamps_s` 指合并后时间格。**频域计算应读取 NPZ 内时间戳**。帧路径转成绝对路径。顶层 `model_id`、`model_revision`、`layer_index`、`pooling` 供后续加载器阻止跨模型/层/池化混用。

溯源记录含模型、requested/resolved revision、依赖版本、dtype、层索引、固定提示、像素/token 上限、输入 manifest 和逐帧 SHA-256，以及输入/隐状态采样率、`video_grid_thw`。模型冻结并使用 eval/inference 模式，不训练模型。

## 解释边界

固定指令为 `Describe the visible events.`，位于视频 token 之后。因果 decoder 的视觉位置不会读取其后的指令，不能称为“未来问题条件化”表示。视觉编码、temporal patch 合并及此前 decoder 上下文也使池化结果不等于独立原始帧表征。

完整事件提取不等于严格逐帧在线写入。若研究在线写入，只能在事件结束后使用该事件的已观察帧；更细粒度写入需要逐步前缀提取并检查视觉前端未来信息。

冻结因果 decoder 在保持原视觉编码/位置设置不变时，末尾追加文本不会让旧 token 状态被后文改写。静态 NPZ 无法直接测出“后文覆盖旧隐状态”。干扰实验应测真实更新的外部记忆、压缩状态或后续 readout 答案变化，并固定回答器、题目与预算。

本实现只有空间均值池化，不能保证保留物体关系与局部微动作。object/event-conditioned pooling、attention pooling、跨架构迁移和模型内部干预均需独立实现。

## 本次验证范围

- `--help` 成功，不加载 torch 或模型，不下载权重。
- 时间合并、Token 计数、空间池化和错误输入拒绝通过数值检查。
- 官方慢速 processor 处理 16 张合成帧，随机初始化的微型真实 Qwen2.5-VL 完成 CPU 前向：网格 `[8,4,4]`、3 项 hidden-state tuple、池化结果 `[8,32]` 且值有限。
- 另以临时保存的随机微型 checkpoint、真实 fast tokenizer、16 张 PNG 贯通整个 `extract` 入口：本地加载、三个分层 NPZ/manifest、metadata 透传和 complete 标记全部通过；无网络权重下载。
- 随机微型模型只验证接口与形状，不代表预训练模型能力、真实视频时序恢复或长期记忆收益。当前缓存没有 Qwen2.5-VL 权重，本次未下载或运行其 3B/7B 实验。

## Qwen3-VL 小模型路径：已运行真实权重

`spectral_memory.qwen3_backend` 与 `spectral_memory.extract_qwen3` 专用于固定版本 `Qwen/Qwen3-VL-2B-Instruct`，revision 为 `89644892e4d85e24eaac8bacfd4f463576704203`。本次使用本机缓存权重、CUDA 12.8 的 PyTorch 2.10.0、torchvision 0.25.0、Transformers 4.57.1，BF16 / SDPA / batch=1。Qwen3 使用官方 `Qwen3VLVideoProcessor`，需要 torchvision。

```powershell
.\.venv-cuda\Scripts\python.exe -m spectral_memory.extract_qwen3 --input data/perception_test_frames64/frames.manifest.jsonl --output-dir outputs/hidden_new --layers 0,14,28 --max-pixels 65536
```

Qwen3 的每个时间格有独立的视觉 Token 块，中间穿插时间戳文本；适配器逐块核对 delimiters、空间 Token 数和时间格数。预处理禁用内部采样，不补帧。时间戳来自相邻两帧中心，提示按官方格式显示一位小数，NPZ 保留未舍入数值。官方处理器的 `size.longest_edge` 表示全视频像素预算，因此传入帧数乘每帧像素上限，并核验最终空间网格。

Qwen3 层 0 是 decoder 输入，层 k<N 是第 k 个 decoder 层的输入，包含此前 DeepStack 注入，层 N 为最终 RMSNorm 输出。只为请求的层注册 hooks，跳过 LM logits、完整 hidden tuple 与 KV cache。微型真实架构的 hooks 与官方全部 hidden_states 已逐层对齐；官方 processor 的输入也独立通过逐元素比较。

真实提取已完成 8 视频 × 3 层，共 24 个 `[32,2048]` 矩阵，见 `outputs/qwen3_hidden_pilot/`。每个视频输入 64 帧，覆盖 25.54–35.03 秒；隐状态采样率约 0.899–1.233 Hz。不能对所有视频照用默认 0.5 Hz 频带上限；本次诊断固定为 `(0,0.1,0.2,0.4)` Hz。

上游最近邻时间重采样保留原始帧索引、实际 PTS 与最大重采样误差，未加时间低通；它不是无混叠采样。完整片段的视觉前端也不能视作逐帧因果写入。Qwen3 同样只使用空间均值池化；这些矩阵未构成长程记忆算法或可直接训练的效用数据集。
