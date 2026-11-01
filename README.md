# Spectral Memory Probe

面向小型 VLM 内部时序证据传播与读取的研究项目。当前通过只读hook采集FFN、attention、残差与Q/K/V，检验时间频谱是否对失败定位及内部干预提供额外价值。频谱有效性、长期遗忘及跨外部记忆框架收益均为待检验问题；早期外部选帧实验作为独立历史记录保留。

GitHub归档包含代码、文档及选定报告/图表；原始数据、权重和完整激活/logit档案保留在本地。具体范围和文件哈希见[上传说明](docs/github_repository_contents.md)与[报告清单](docs/github_upload_manifest.json)。

## 阅读入口

- [学习式内部控制器：完整实验与论文结果稿](docs/controller_paper_results_v1.md)：12次训练、2522条配对记录全部完成；新测试频谱与时域均31/74，基线32/74，未修复端点问题。
- [完整逐seed主表与区间](outputs/controller_analysis_v1/report.md)、[LaTeX主表](outputs/controller_analysis_v1/main_table.tex)、[独立效果审计](outputs/controller_effects_audit_v1/report.md)及[完整性记录](outputs/controller_experiment_v1/final_integrity.json)：所有方法、原始分数、负结果及资源均可复核。
- [学习控制器数学推导](docs/learned_controller_mathematics.md)、[冻结协议](docs/learned_controller_protocol_v1.md)、[数据审计](docs/internal_controller_data_audit.md)与[当前最近邻文献](docs/controller_related_work_comparison.md)：区分实际内部读出干预、时域等价结构及未成立的记忆机制主张。
- [内部探针与研究决定](docs/internal_research_decision_20261101.md)：当前证据、否决的解释及后续可证伪问题。
- [完整空间/QKV被动采集](docs/passive_probe_protocol_v4.md)与[独立分析](outputs/passive_probe_v4/analysis/report.md)：8视频60条件，60次全词表等值、840次残差恒等式核验；空间池化与RoPE混杂诊断。
- [端点可读性对照](outputs/endpoint_control_v1/report.md)与[493项独立审计](outputs/endpoint_control_v1_audit/audit.md)：同一源视频的8静态QA全部正确，24混合QA仅12正确；尚不构成长期遗忘证据。
- [端点内部路由协议](docs/endpoint_routing_protocol_v1.md)与[160条件实测](outputs/endpoint_routing_v1/analysis/report.md)：target gate混合12/24→18/24，但静态8/8→6/8，未通过选择性门槛；不是频谱方法。
- [内部研究文献矩阵](docs/internal_probe_novelty_audit.md)与[Frame2Freq轴核查](docs/frame2freq_axis_audit.md)：正式会议、真实张量操作及直接先例。
- [频谱可辨识性推导](docs/probe_spectral_identifiability.md)、[47项基变换检查](outputs/probe_basis_checks/report.md)和[10项空间对齐反例](outputs/spatial_alignment_checks/report.md)：完整Fourier/ridge等价、相位丢失及对齐可能删除运动标签。
- [注意力—值的交叉谱推导](docs/attention_value_spectral_identity.md)：同功率可产生相反读出；32项CPU检查通过，尚非模型机制证据。
- [FFN 实测报告](outputs/ffn_development_report/report.md)、[假说与精确支路探针](docs/ffn_hypothesis.md)、[开发诊断协议](docs/ffn_development_protocol.md)：8 视频 GPU 诊断已完成；末层归一化占比增长与 DC 抵消有关，尚无记忆收益证据。
- [重启事件与资源记录](docs/resource_incident_20261101.md)：三次蓝屏的已保存证据；Smol 全量任务中断，后续 GPU 仅使用限额独立探针。
- [独立确认实验与当前结论](docs/confirmation_results_20261101.md)：40 视频、182 QA、固定字节记忆；频谱方法未通过预设主门槛。
- [完整方法与数学边界](docs/spectral_memory_method.md)：相位保留、多频带目标、时域等价式及实际读出。
- [冻结确认协议](docs/spectral_memory_protocol.md)与[短动作证据覆盖协议](docs/coverage_protocol.md)：区分开发、确认和后续探索。
- [已观察问题审计](docs/observed_problem_audit.md)：短动作漏帧、末态读出以及没有成立的机制假设。
- [逐篇文献矩阵](docs/related_work_matrix.md)：22 篇记忆方向正式顶会论文、独立6篇FFN机制分区，另列待确认、预印本与经典理论。
- [小模型研究定位](docs/small_model_research.md)：投稿价值、直接竞争、机制假设与停止条件。
- [真实小模型试点](docs/small_model_pilot.md)：36 题四组对照、真实隐状态、格式敏感性与下一阶段决定。
- [研究审计](docs/research_audit.md)：创新边界、数学前提及 Go/No-Go 条件。
- [数据协议](docs/data_protocol.md)：数据字段、反事实标签、时间划分与干扰设计。
- [实验执行协议](docs/experiment_protocol.md)：六组对照、指标、预算限制和阶段门槛。
- [真实模型提取指南](docs/extraction_guide.md)：独立 Qwen2.5-VL / Qwen3-VL 视频 token 对齐。
- [验证记录](docs/validation.md)：已运行的检查与尚未运行的真实实验。
- [资源调研与已获取样例](docs/resource_plan.md)：本机算力、模型缓存、公开数据及真实视频/问答包。

## 快速运行

在项目根目录使用 Python 3.10 或更新版本。核心实验只需 NumPy 和 SciPy：

```powershell
python -m pip install -e .
python -m spectral_memory demo --output-dir outputs/my_demo
python -m spectral_memory validate --manifest outputs/my_demo/manifest.jsonl
```

演示自动生成 24 个合成主体、192 条事件，运行 M1–M6、分组 bootstrap 和整数预算选择。输出目录必须是新目录，避免覆盖已有实验。

结果位于 `outputs/my_demo/run/report.md`，完整指标在 `report.json`，同时保存切分、预测、各组候选选择及拟合参数。**合成标签独立于隐状态；任何指标差异都不能证明研究假设成立。** 已运行的演示保存在 `outputs/demo_verified/run/`。

## 小模型真实试点

已在本机 RTX 5070 Laptop 上完成固定版本 `Qwen3-VL-2B-Instruct` 的 BF16 推理：36 题 × 4 条件，另提取 8 个真实视频的第 0、14、28 层隐状态。原始结果保存在 `outputs/qwen3_qa36_pilot/` 和 `outputs/qwen3_hidden_pilot/`；它们是小样本诊断，尚不支持长期记忆或频谱方法有效性的结论。

本次隔离环境为 `.venv-cuda`，使用 `torch==2.10.0+cu128`、`torchvision==0.25.0+cu128`、`transformers==4.57.1`。全局 CPU 环境保持原状。视频准备另需 OpenCV，可通过 `pip install -e ".[video]"` 安装。

```powershell
.\.venv-cuda\Scripts\python.exe -m spectral_memory.prepare_frames --manifest data/perception_test_sample/videos.jsonl --output-dir data/frames_new --frame-count 64 --max-side 512
.\.venv-cuda\Scripts\python.exe -m spectral_memory.evaluate_small --questions data/perception_test_sample/questions.jsonl --frames-manifest data/perception_test_frames64/frames.manifest.jsonl --output-dir outputs/qa_new --frames-per-condition 16 --conditions no_memory,first_half,last_half,uniform_full
.\.venv-cuda\Scripts\python.exe -m spectral_memory.extract_qwen3 --input data/perception_test_frames64/frames.manifest.jsonl --output-dir outputs/hidden_new --layers 0,14,28
```

Qwen3 命令默认固定缓存 revision，仅使用本地权重。三种视觉对照各输入 16 帧；无记忆组不输入视频。回答记录、解析、输入 token、耗时、峰值显存、载荷哈希与成对效用均单独保存。64 帧提取生成 32 个时间格，分析必须使用它们的实际物理采样率。

## 接入其他真实数据

先准备预采样 RGB 帧及时间戳 manifest，再安装模型提取的可选依赖：

```powershell
python -m pip install -e ".[extract]"
python -m spectral_memory.extract_qwen --input data/frames.jsonl --output-dir outputs/qwen --model Qwen/Qwen2.5-VL-3B-Instruct --layers auto --device cuda
```

默认只读取已缓存权重；确需下载时显式增加 `--allow-download`。发布实验使用固定 `--revision`。此适配器固定 `transformers==4.57.1`，不自动兼容 Qwen3-VL。每层单独生成 manifest，避免混层和混模型。

把固定读出模型的成对任务评估记录转成效用标签，再运行探针：

```powershell
python -m spectral_memory labels --manifest outputs/qwen/manifest.layer_0.jsonl --evaluations data/evaluations.jsonl --output outputs/qwen/labeled.layer_0.jsonl
python -m spectral_memory run --manifest outputs/qwen/labeled.layer_0.jsonl --output-dir outputs/probe --task-mode retrieve --group-key subject_id --budget 2048
```

`--budget` 与 manifest 的 `cost_unit` 一致；2048 只是示例，并非推荐预算。标签日志格式见数据协议；提取程序不自行生成任务收益。至少需要五个独立主体进行默认自动划分。`--task-mode write` 使用严格的未知未来查询写入协议。

## 实现范围

已实现物理 Hz 频段、Hann 全局/局部功率谱统计、三层 Haar 能量、强时域统计、保序/复频谱对照；训练集标准化、验证集选择 ridge 参数、按主体隔离测试、成对分组区间；反事实日志校验与固定预算的整数背包诊断。

```powershell
python -m spectral_memory intervene --manifest outputs/my_demo/manifest.jsonl --output-dir outputs/reversed --mode reverse
python -m spectral_memory run --manifest outputs/my_demo/manifest.jsonl --output-dir outputs/matched --projection-dim 64
```

干预命令产生新的候选版本并移除旧效用标签，必须重新评估任务。`--projection-dim 64` 提供等宽随机投影控制，不能替代充分训练的时域模型。

当前已实现并运行真实 Qwen3-VL 的多选 QA 自动评分、完整空间及Q/K/V采集、FFN/attention支路激活干预；没有训练出可泛化的内部记忆修复器，也没有完成完整流式记忆更新器或学习式压缩器。早期固定预算 probe 报告是候选单条效用之和的诊断，不能代替选中集合的真实问答收益。真实 QA 的候选日志没有伪造 hidden_path，也不能直接拼接成已满足独立划分的训练集。
