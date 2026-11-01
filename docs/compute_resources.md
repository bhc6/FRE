# 算力与模型资源核查

核查日期：2026-11-01。工作区实际路径为 `D:\FRE`。本文区分 **observed（本机/官方元数据实测）**、**estimated（算术估算）**、**untested（尚未运行）**。本次仅只读检查硬件与 Hugging Face 模型缓存，访问公开资料；没有付费、启动云实例、安装 CUDA 环境或大规模下载权重。

## 结论与可执行路线

本机具备约 8 GiB 的 NVIDIA GPU，但当前 Python 安装的是 CPU-only PyTorch。现有环境能运行数据协议、特征提取、轻量 probe 和随机微型模型接口验证；完整 Qwen2.5-VL 隐状态提取需要更合适的运行环境。优先安排一次小规模外部 GPU pilot：3B 从 24 GiB 显存配置起步，7B 从 48 GiB 起步，batch=1、短事件、固定分辨率，并记录峰值显存与耗时后再估算完整实验。

这是起始配置建议，不是显存保证。尤其不能用“模型叫 3B”推断它必然适合本机 8 GiB：官方完整 3B 权重文件约 6.994 GiB，尚未包含运行时缓存和视觉/语言中间激活。

## 本机硬件与运行时：observed

| 项目 | 本次实测 | 意义 |
| --- | --- | --- |
| CPU | Intel Core Ultra 9 285H，系统报告 16 核/16 逻辑处理器 | 可做 CPU 数据处理和轻量 probe；未测真实视频提取吞吐 |
| 物理 RAM | 33,820,106,752 bytes，约 31.50 GiB | 不能全部视作模型可用内存 |
| 检查时空闲 RAM | 11,906,740 KiB，约 11.36 GiB | 随其他程序变化；3B float32 权重估算已高于这一数值 |
| NVIDIA GPU | GeForce RTX 5070 Laptop GPU | 有独立 GPU，不应将 torch.cuda=False 解释成没有 GPU |
| GPU 显存 | nvidia-smi：8,151 MiB 总量；一次快照 7,891 MiB free | Windows 显示/驱动会占用资源；总量与 free 不相等 |
| 驱动 / compute capability | 591.74 / 12.0 | GPU 可被驱动识别；CUDA PyTorch 仍需单独验证 |
| 当前 torch | 2.10.0+cpu；torch.version.cuda=None；cuda_available=False | 当前解释器无法进行 CUDA 模型前向 |
| 当前 Python / Transformers | Python 3.13.7 / Transformers 4.57.1 | 与现有适配器版本锁一致 |
| C 盘空闲 | 9,146,634,240 bytes，约 8.52 GiB | 不适合继续默认向 C 盘同时放模型与 CUDA 环境 |
| D 盘空闲 | 63,139,594,240 bytes，约 58.80 GiB | 可容纳小规模实验；100–300 小时原始帧与全部特征需另算 |

硬件来自 `Get-CimInstance`；显存来自 `nvidia-smi`。Windows `Win32_VideoController.AdapterRAM` 对这台机器显示不准确的 32 位数值，未用它作为显存依据。空闲 RAM、显存、磁盘是快照，运行前应重查。

### CUDA 环境：已完成独立安装与真实推理

前表记录初始全局 CPU 环境。随后已在 `.venv-cuda` 安装 `torch==2.10.0+cu128` 和 `torchvision==0.25.0+cu128`。官方 wheel 完整 SHA-256 为 `4d1b0b49c54223c7c04050b49eac141d77b6edbc34aea1dfc74a6fdb661baa8c`，验证记录位于 `.cache/wheels/torch-wheel-source.json`。CUDA 可用性、GPU 名称、compute capability 12.0 和 BF16 GPU 矩阵乘法均实测通过；真实 Qwen3-VL-2B 问答与分层提取也已完成。

安装、下载缓存与临时解压目录均放在项目所在 D 盘，全局 CPU 安装未改动。本次环境通过 `--system-site-packages` 复用已有 Transformers/NumPy 等依赖，因此不是完全隔绝的可移植锁定环境。固定 torch 2.10.0 是为减少迁移变量，不是声称它是最新版本。

来源：[PyTorch 官方历史版本安装页](https://pytorch.org/get-started/previous-versions/)、[CUDA 12.8 官方 wheel 索引](https://download.pytorch.org/whl/cu128/torch/)、[PyTorch 官方 Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)。

新机器的复现命令如下；本机安装阶段因单连接超时，实际采用官方 wheel 的可恢复分段下载并校验完整哈希，再安装本地 wheel。不要对现有环境重复安装：

```powershell
python -m venv --system-site-packages .venv-cuda
$env:PIP_CACHE_DIR = Join-Path (Get-Location).Path ".cache/pip"
$env:TEMP = Join-Path (Get-Location).Path ".cache/tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
.\.venv-cuda\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-cuda\Scripts\python.exe -m pip install -e ".[extract]"
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.ones(1, device='cuda'))"
```

Qwen3-VL-2B 的独立适配器已实现，并固定本地 snapshot revision；相关协议见 [提取指南](extraction_guide.md)。未来下载其他权重时再将该运行的 HF 缓存指向空间充足的盘。

## 模型可得性、大小与许可：observed

以下大小来自官方 Hugging Face API `?blobs=true`，对实际 `.safetensors` 文件逐个求和；不是按模型名称猜参数量。三个仓库本次均为 public、ungated。未下载这些远程权重。

| 模型 | 权重总 bytes / GiB | decoder 层数 / hidden dim | 许可文件或元数据 | 本地情况 |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | 7,509,337,976 / 6.994 GiB，2 个分片 | 36 / 2048 | Qwen Research License；正文将授权用途限定为 non-commercial research/evaluation，商业用途另需许可 | 无缓存权重 |
| Qwen2.5-VL-7B-Instruct | 16,584,414,560 / 15.445 GiB，5 个分片 | 28 / 3584 | Apache-2.0 | 无缓存权重 |
| Qwen3-VL-2B-Instruct | 4,255,140,312 / 3.963 GiB，1 个文件 | 28 / 2048 | Apache-2.0 | 已用真实权重完成 BF16 问答和 8 视频分层提取 |

3B 的模型卡元数据没有给出统一 `license` 字段，因此以实际 LICENSE 文件为依据，不能把字段空值当成无许可限制。这里记录许可文本事实；研究发表、再分发或商用前应按实际用途阅读该固定版本全文。

| 模型 | 本次核实 revision | 官方来源 |
| --- | --- | --- |
| 3B | `66285546d2b821cf421d4f5eb2576359d3770cd3` | [模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[API 文件元数据](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct?blobs=true)、[固定版 LICENSE](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE) |
| 7B | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | [模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)、[API 文件元数据](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct?blobs=true) |
| Qwen3 2B | `89644892e4d85e24eaac8bacfd4f463576704203` | [模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、[API 文件元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct?blobs=true) |

### 本地 Hugging Face 缓存核查

检查范围仅限当前用户的 `.cache/huggingface/hub`，未检查 token、密钥或 SSH 配置。

| 缓存条目 | 实测内容 | 可以做什么 |
| --- | --- | --- |
| Qwen3-VL-2B-Instruct | 仓库约 3.974 GiB；有 4,255,140,312-byte safetensors，revision 与上表一致 | 可作为之后独立验证的低成本架构候选；需新适配器，不能交给 Qwen2.5 提取器 |
| Qwen2-VL-2B-Instruct | 仅 1 个配置文件，1,196 bytes，无权重 | 不能称为已有可运行模型 |
| Qwen3-8B | 约 15 MiB 元数据/分词器，无权重 | 不能离线加载完整模型；且不是本项目主 VLM |
| all-MiniLM-L6-v2 | 有约 90.9 MB safetensors | 可供独立验证的文本语义基线，不可替代视觉隐状态 |
| GPT-2 / GPT-Neo-125M | 有完整权重文件 | 可做通用加载接口实验，不构成 VLM 科学结果 |
| CLIP / SigLIP2 条目 | 元数据存在，但本次未发现权重文件 | 不能把目录名当作模型已下载的证据 |

Qwen2.5-VL 仍只验证随机微型 CPU 模型；Qwen3-VL-2B 已完成预训练权重推理。缓存表中其他模型的可得性不能当作已验证运行的证据。

## 显存与存储估算：estimated

### 权重与中间状态必须分别计入

近似峰值显存由权重、视觉前端激活、语言层激活、保留的 hidden-state tuple、attention/kernel 工作空间和框架开销共同决定。`use_cache=False` 可以避免生成 KV cache，但不会消除上述其他开销。

当前提取器使用 `output_hidden_states=True`，会保留所有 decoder 层状态；`--layers` 仅决定最后保存哪些层，不会使运行时只计算/保留这几层。若序列长度为 S、隐藏维度 d、decoder 层数 L、每元素 b bytes，则仅该 tuple 的近似大小为：

`M_hidden ≈ (L + 1) × S × d × b`。

在 S=4096、bf16=2 bytes 时，3B 约 0.578 GiB，7B 约 0.793 GiB；S=8192 时翻倍。3B 完整 bf16 权重已约 6.994 GiB，叠加这部分就说明 8 GiB 设备缺少可靠余量，还没有计入其他激活。

CPU float32 权重存储粗估为 bf16 文件量的约两倍：3B 约 13.99 GiB，7B 约 30.89 GiB。文件包含小量非权重信息，这只是近似；加载峰值与运行峰值还会更高。当前空闲 RAM 约 11.36 GiB，不宜直接承诺 CPU 3B 可跑，更不能承诺 CPU 7B。

4-bit/8-bit 量化与 CPU offload 可能降低 GPU 压力，但当前提取器未实现，也未测试。量化会改变隐藏表示，必须在频谱研究中作为单独条件；不能把量化结果默默并入完整精度主结果。

### 输出存储不能忽略

空间池化后保存 float32、7 层、隐状态 1 Hz，不计压缩与元数据，每小时近似为：

`bytes_per_hour = 3600 × hidden_sampling_hz × saved_layers × hidden_dim × 4`。

| 模型 | 每小时池化隐状态 | 100 小时 | 300 小时 |
| --- | --- | --- | --- |
| 3B，d=2048 | 196.875 MiB | 19.23 GiB | 57.68 GiB |
| 7B，d=3584 | 344.531 MiB | 33.65 GiB | 100.94 GiB |

NPZ 压缩率必须测量，不能预先抵扣；模型、视频、PNG 帧、运行环境、标签与日志另计。D 盘当前约 58.80 GiB 空闲，不够无条件承诺 300 小时实验。优先保留原视频与帧索引、按需解码小批次，按视频分块存放特征；具体原视频读取由数据协议决定。

## 外部 GPU 候选与公开成本

2026-11-01 访问 [Runpod 官方价格页](https://www.runpod.io/pricing) 得到以下每 GPU 小时展示价。它们是公开页面报价快照，不是已预订价格；区域、库存、部署模式和后续价格变化可能影响实际可选项。未创建账号或实例。

| 官方页面配置 | 展示价 USD/GPU-hour | 本项目建议用途 | 状态 |
| --- | --- | --- | --- |
| NVIDIA L4，24 GB VRAM | $0.39 | 3B 短事件 pilot 的候选 | untested，无显存/吞吐保证 |
| NVIDIA A40，48 GB VRAM | $0.44 | 3B/7B 主提取及 paired readout 的候选 | untested，无库存保证 |
| NVIDIA RTX 4090，24 GB VRAM | $0.69 | 测得吞吐后比较单位有效样本成本 | untested，不能仅按小时价比较优劣 |

完整预算公式：

`Cost = hourly_GPU_price × (H_setup + H_extract + H_counterfactual + H_train + H_idle) + storage + egress + applicable_fees`。

以 A40 页面价为纯算术示例，10 GPU-hours=$4.40、50 GPU-hours=$22.00，均未含其他费用；**这些小时数不是完成本项目的耗时预测**。先在固定 batch、帧数、分辨率、层设置下测量单片段耗时和峰值，再用 `H_extract ≈ clip_count × seconds_per_clip / 3600` 外推，另计失败重试和数据搬运。

反事实标签需要相同题目/回答器的成对运行；若有 Q 个题目、K 个候选记忆、R 个重复种子，未经共享优化的额外调用规模可到 `2×Q×K×R`。因此不能仅用隐状态提取费用代表完整论文预算。

## 服务必须满足的接口

需要可执行 Python、可加载固定版本模型与权重、能调用 `output_hidden_states=True`、能读取视觉 Token/时间网格、能保存 NPZ 与逐次运行日志的 GPU 主机或自定义作业。SSH/Jupyter/容器作业只是可能的入口，不需要把账号或密钥写进项目。

只返回聊天答案的普通托管 API 不暴露本实验需要的逐层 hidden states 与视觉 Token 映射，因此不能替代表示提取环境。即使将来用托管模型辅助提出问题，也需另外验证标签生成、数据许可与评估一致性，不能将它等同于主实验 backbone。

## 分阶段资源决策

1. **现在，本机 CPU 已可做**：数据规范校验、合成 sanity checks、特征/轻量 probe、跨组拆分和配对日志处理。真实收益仍须真实视频与回答标签。
2. **先做 20–50 个事件 pilot**：外部候选 GPU 上固定 3B revision、batch=1、每事件 16–32 帧，记录 wall time、峰值显存、NPZ 大小、失败率；这一步不要求下载 100–300 小时数据。
3. **根据 pilot 决定扩展**：若 3B 下频域不能超过强时域/保序对照，先修正假设；有信号再扩到更多参与者和 7B。外部机器配置与账单须以真实测量更新。
4. **本机 GPU 的备选路线**：独立 CUDA 环境加已有 Qwen3-VL-2B 权重的专用适配器，只在时间对齐和隐藏层语义重新验证后运行；作为另一个模型条件报告，不能充当 Qwen2.5-VL 结果。

尚未发生的工作：CUDA wheel 安装、本机预训练 VLM 前向、云 GPU 实例启动、3B/7B 权重下载、真实视频吞吐 benchmark 和付费资源预订。所有相应性能/费用结论保持 untested 或 estimated。
