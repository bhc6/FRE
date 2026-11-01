# 本次验证记录

日期：2026-11-01。以下记录来自本工作区的实际执行。没有用合成结果支持研究有效性或顶会创新性结论。

## 环境

Windows / PowerShell；Python 3.13.7；NumPy 2.4.2；SciPy 1.18.0；PyTorch 2.10.0；Transformers 4.57.1；Pillow 11.3.0。未为本项目下载预训练模型权重。初次代码验证后已追加公开真实视频资源获取，见下文。

## 已通过

| 检查 | 结果 |
| --- | --- |
| `python -m compileall -q spectral_memory` | 全部模块可编译 |
| `python -m spectral_memory.extract_qwen --help` | 成功；提取模块导入不加载 torch/transformers |
| `demo --output-dir outputs/demo_verified` | 24 个合成主体、192 条事件；六模型完整输出 |
| 成对标签生成 | 384 行评估日志组成 192 对；产出 192 条效用标签 |
| `validate` | train/validation/test = 112/40/40 条；默认主体无交叉 |
| `run --projection-dim 64` | 六模型等宽投影控制完整运行；结果位于 `outputs/probe_matched/` |
| `intervene --mode reverse` | 生成 192 个新候选；移除原收益并要求重新评估 |
| 数值与背包性质 | 92 项断言通过，覆盖五种长度、Parseval、功率反转不变、相位随机化保功率、复频谱可逆/等能量、常量信号、60 组穷举背包对照与 bootstrap 方向 |
| 数据与标签拒绝/路径检查 | 19 项检查通过，覆盖错误哈希、协议混用、缺对、干预复用旧日志、未评估干预、载荷变化、混层、基础记忆变化、划分泄漏、重定位路径、写入分布和上下文限制、非均匀采样/Nyquist/空频带 |
| 写入模式全流程 | 临时合成双查询分布、成对标签、六模型 probe 全部通过；临时文件已自动清理 |
| 真实模型代码的微型前向 | 随机初始化的微型 Qwen2.5-VL，16 合成帧得到 8 时间格，三层 tuple，`[8,32]` 池化输出有限 |
| 提取器完整入口 | 临时随机微型 checkpoint + 真实 fast tokenizer + 16 PNG，成功产生三层 NPZ/manifest 与 complete 标记；无权重下载 |

上述数值/协议检查通过临时 Python 命令执行，没有为本项目引入测试框架。提取器的完整微型模型检查由并行代理独立运行；主流程、数值及协议检查由主代理运行。

## 现成输出

- `outputs/demo_verified/run/report.md` 与 `report.json`：默认六组探针。
- `outputs/demo_verified/paired.jsonl`：从配对日志重新计算的标签。
- `outputs/probe_matched/report.md`：等宽投影对照。
- `outputs/intervention_reverse/manifest.jsonl`：未标注的反转候选。
- 各运行目录另含 split manifest、预测、选中 ID/成本、回归参数。

所有合成样本标记 `synthetic=true`，报告标记 `synthetic_smoke_test`。其标签由独立随机数生成，与隐状态结构无关，模型之间的偶然差异不具有研究意义。

## 尚未验证

尚未运行预训练 Qwen2.5-VL-3B/7B、真实第一视角视频模型实验、长期记忆更新器、频谱选择后的候选集合收益或跨模型迁移。Qwen3-VL-2B 的真实 QA 和分层提取已在后续追加完成，产生相对空记忆的成对收益；详见末节。现有小样本仍不构成研究方法有效性的科学结论。

尚未复现矩阵中的第三方方法，未确认其全部代码、数据许可、权重和硬件可得性；论文核查范围明确标为一手摘要/元数据级。

## 追加：资源获取验证

`python -m spectral_memory.prepare_resources --output-dir data/perception_test_sample` 已完整执行：8 个官方视频、36 MCQA、71 动作标注；官方 `area=memory` 只有 1 题。视频和标注 archive 完整下载、ZIP 解压及内容哈希检查通过；8 视频各抽取首/中/末帧，共 24 帧均可解码，帧数与官方 metadata 全部一致。

下载/导入代码另通过三项检查：拒绝 ZIP 目录穿越、拒绝超过解压上限、正常嵌套路径可解压。资源 JSONL 保留未知主体/会话为 null，答案索引与选项一致，没有生成虚假的 utility。详细来源、哈希、已安装/未安装的算力状态和正式资源路线见 [资源计划](resource_plan.md)。

EPIC-KITCHENS 官方短片 `P03_15` 也已下载：39,548,685 bytes，原始标注明确主体为 `P03`，筛选得到 3 条动作区间。视频哈希及首/中/末 3 帧解码检查通过；来源 CSV 和精确 metadata 保留于 `data/epic_kitchens_p03_15/`。这些资源检查不构成 VLM 推理或长期记忆实验结果。

## 追加：真实 Qwen3-VL-2B 小模型试点

独立 CUDA 环境安装、官方 wheel SHA-256、GPU BF16 运算均通过；全局 CPU 环境未改动。专用适配器与官方 Processor 逐元素一致，禁止联网时仍可从本地快照加载，微型真实架构 hooks 与官方 hidden_states 逐层一致。

真实缓存权重已完成 36 题 × 4 条件，144 次生成全部完成；主规则下正确数依次为 13/20/21/19，格式失败 0/6/4/5。产出 108 条相对空记忆的实测 utility。格式敏感性另作事后分析，未覆盖原始评分。单道官方 memory 题四条件均答对，无法支持记忆收益。

真实分层提取完成 8 视频 × 3 层，24 个 `[32,2048]` NPZ 及全部五组特征有限，物理频带与各窗口 FFT bin 检查通过，数值不变量误差约 `1e-15`。提取峰值 allocated 约 4.827 GiB。没有训练真实 utility probe，也没有将不同载荷的 hidden 和标签合并。

评估器另通过无 GPU 的成功、异常、版本漂移、帧哈希变化与基线复用路径检查；全部源码编译通过。结果与限制见 [真实试点报告](small_model_pilot.md)。
# 2026-11-01 新增真实确认实验

本段更新当前实验状态；下方早期记录保留其当时的范围。主确认结果见 [完整报告](confirmation_results_20261101.md)。

- Qwen3-VL-2B：40 个独立于开发的视频，182 个官方 QA，9 个方法、3 个帧预算，以及全帧/无视频参考，共 5,642 条评分记录、3,863 次不同题目/载荷的真实前向。
- 冻结协议和源码快照在 outputs/spectral_memory_protocol_v1/。完整性审计已通过：无缺失/重复实验单元，合法选项 logits 重评分正确，真实 NPZ 内容和字节一致，原始输入及冻结源码哈希未变。
- 主结果未通过门槛：8 帧预算、77 个时序/状态问题、33 个原视频上，spectral − kcenter = −8.08pp，95% 视频配对区间 [−19.70, +2.02]。
- 官方动作覆盖：246 动作，其中 144 个短动作在原 64 帧中可观察。主预算 spectral − kcenter = −4.10pp，区间 [−14.46, +5.39]，没有支持短动作证据保留优势。60 组覆盖重新计算一致。
- 开发集完整尾部对照、三层动态诊断、固定首尾 B4 的 435 种锚点全枚举均已完成。更好的代理损失没有稳定转化为动作覆盖。
- SmolVLM2-500M 真实 GPU smoke 通过；同冻结记忆的迁移因系统重启中断，已保存 1,633/2,002 条记录，当前未恢复，不能给出完整迁移结论。资源记录见 `resource_incident_20261101.md`。

审计工具为 spectral_memory/audit_study.py；原始模型记录、实际选择集合、时间/显存和完整统计均保存于 outputs/，不以合成检查替代真实模型结果。

## 追加：FFN 有限资源 GPU 探针

按用户重新授权，已完成原8个开发视频各16帧、Qwen3-VL-2B BF16的单次前向，固定decoder 0/13/27层。一次仅一个模型进程，CUDA allocator fraction=0.65，CPU两线程，单次最长300秒；不恢复中断的Smol确认运行。最大allocated/reserved分别为4,363,261,440/4,410,310,656 bytes，整批启动时间未变，退出后GPU为空闲。重启原因尚未解决；资源采样不保证捕获所有短时峰值。

- 微型真实Qwen3 decoder在FP32和BF16下通过attention/FFN加法及模拟DeepStack原地更新检查；hook完整移除。记录为 `outputs/ffn_cpu_validation/hook_checks.json`。
- 资源守卫7项错误/边界检查通过，包括监测日志写入失败仍终止，以及结束阶段不会被监测竞态误杀；检查不调用GPU。
- 分析器19项数值/非法输入检查及合成NPZ入口检查通过，验证Parseval、交叉项、舍入分解和奇偶时间长度。汇总比值另有4项数值检查。
- 真实采集中48个token级、原dtype残差恒等检查通过；144条分支均值时间轨迹有限，全部8视频/128帧字节、模型revision、时间网格、NPZ与源码快照通过独立核对。汇总直接从NPZ重算能量，未只信任缓存报告。
- 本轮结束后全部源码编译通过，冻结v1的9个输入/源码哈希保持一致；结果保存在 `outputs/ffn_cpu_validation/frozen_source_check.json`。

真实描述性发现见 `outputs/ffn_development_report/report.md`：末层FFN占比比值中位数9.44，绝对动态能量比1.083，DC比0.110。不能将该发现视为QA提升或v1失败原因。尚未运行FFN因果干预、FFN选帧后的QA、新独立确认或长期记忆更新实验。

运行一个独立样例：

```powershell
.\.venv-cuda\Scripts\python.exe -m spectral_memory.probe_ffn --device cuda --video-id video_10993 --output-dir outputs/ffn_new_video_10993
.\.venv-cuda\Scripts\python.exe -m spectral_memory.analyze_ffn --input outputs/ffn_new_video_10993/branches.npz --output-dir outputs/ffn_new_video_10993/analysis
```

监测依赖 `psutil`，列于 `monitor` 可选依赖；GPU环境需可调用 `nvidia-smi`。输出目录必须新建或为空，失败时检查 `probe.json`、`resource_stop.json` 和退出码，不自动放宽额度重试。

