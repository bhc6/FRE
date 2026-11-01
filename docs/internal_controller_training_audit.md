# 内部读出 Controller 的数据契约与训练公平性审计

日期：2026-11-01。本审计只读取源代码、冻结数据协议与原始官方标注；没有导入 PyTorch、运行 GPU、读取缓存模型输出、训练日志、验证预测或测试预测。第3节已根据当前 cache/train/evaluate/runtime 源码完成最终复核，区分已修复检查、明确允许的近似训练以及外部执行契约的职责。

## 1. 数据契约和标签读取

数据与 split 数量已核：train 32 源 / 142 题，validation 8 源 / 40 题，test 16 源 / 74 题。新 test 与旧 48 源 ID / 内容哈希不重叠；所有同源 QA 留在同一 split。完整来源说明见 `internal_controller_data_audit.md`。

独立标准库校验器 `data/internal_controller_v1/evaluation_contract.py` 提供：

- `load_evaluation_cases(split)`：验证选源协议、split 协议、问题清单、帧清单和官方原始 JSON 的冻结 SHA256，核对全部源 ID、问题覆盖和 QA—帧来源关联。返回每题的 `question`、`source`、`official_question`、`subsets`。
- `official_question(annotations, question)`：把 `perception_test:video_N:mc:K` 映射为原始 JSON 的 `annotations[video_N]['mc_question']` 中 `id=K` 的唯一条目，逐字段核验题目、选项、选项顺序、答案、area、reasoning、tag。
- `subset_flags(question)`：固定报告子集，完全由原始标签派生，不接收预测。

命令 `python data/internal_controller_v1/evaluation_contract.py --split all` 只检查数据，不调用模型。该 helper 不替代推理时逐帧哈希核验；后者仍应在读取图像时检查。

### 评估输入字段

| 来源 | 字段与契约 |
| --- | --- |
| 问题清单 | `query_id` 唯一；`video_id` 与 query ID 一致；`split` 精确为 train / validation / test；`benchmark_split` 保持官方 valid |
| 原始题目 | `question: str`，`options: list[str]` 恰好 3 项，选项顺序保持原样 |
| 真实视频金标 | `answer_index: int` 为 0/1/2，`answer_text == options[answer_index]`；不可把 bool 当 index |
| 来源与类型 | `video_sha256`、`video_path` 与帧清单一致；`area`、`reasoning`、`tags` 与官方 JSON 一致 |
| 帧清单 | `frames`、`frame_sha256`、`timestamps_s` 均长 16；原始 PTS 为 `source_frame_timestamps_s`，原始帧号为 `source_frame_indices` |
| 时间解释 | `timestamps_s` 是均匀目标网格；不把源 FPS 误作 hidden sampling rate；不把不同宽高比的 token 网格统一假定为 60 个 patch |

### 建议的每方法每题输出字段

至少保留 `method, seed, checkpoint_sha256, protocol_sha256, split, query_id, video_id, condition, option_token_ids, option_logits, predicted_answer_index, gold_answer_index, correct`，以及帧清单 / 输入张量哈希、真实视觉 token 数、最后层插入位置、`delta_relative_l2` 和参考读出重建误差。原始 `option_logits` 指原选项顺序的 A/B/C 单 token logits；不要混称自由生成正确率或选项文本序列似然。

对 `condition=real` 才可使用官方 QA 金标。首帧/末帧重复的静态构造应将 `gold_answer_index`、`correct` 置 null，仅报告相对该构造自身 baseline 的 KL、logits/预测变化等指标。它们不是带原视频金标的准确率样本。

### 固定子集定义

| 子集 | 标签过滤 | train / validation / test |
| --- | --- | --- |
| all | 全部 real QA | 142 / 40 / 74 |
| official_memory | `area == 'memory'` | 23 / 9 / 18 |
| temporal | tags 与 sequencing / action counting / change detection 的交集非空 | 53 / 19 / 41 |
| descriptive_only | reasoning=descriptive，area≠memory，且不含以上任何时序标签 | 81 / 17 / 30 |

`descriptive_only` 不是已验证的静态可解题。它也不是旧字段 `descriptive_control_candidate_not_verified_static`，后者是另一组语义标签规则。不同子集之间可能重叠，分母必须逐项报告。

## 2. 已有的公平性措施

代码预设三组 paired seeds、每方法固定 200 steps、相同 AdamW 与学习率、相同每步 6 个真实 QA 加 2 个静态蒸馏条件的抽样规则；每方法重新初始化同一 NumPy 抽样 seed，使同 seed 的 minibatch 顺序可配对。最终 checkpoint 固定在 step 200，没有按新 test 结果挑选 epoch 的合法入口。

三个 MLP 版本有相同的网络形状；最后线性层零初始化，初始残差为零。全体方法共享每头 10% 原生 context 范数的残差上限与 ±0.25 gain 限制。constant_spectral 只有 8 个参数，是较小的常量增益消融，不能称为等参数主基线。

静态样本每个训练源两条，只使用基线蒸馏，没有沿用原视频 gold；这在目标语义上与真实 QA 监督分开。已缓存 backbone 特征、冻结末层尾部参数以及只训练 controller，使各方法的可训练对象明确。

## 3. 最终数据绑定复核

独立结论：**在消费已冻结且已通过独立契约检查的数据清单这一前提下，先前指出的训练行标签绑定、跨缓存 tail 身份、静态条件覆盖和原生评估路径均已补强，源码复核通过。** 这是实现审计，不是训练已完成、测试有效或无遗漏风险的证明。

| 检查 | 当前实现与复核结论 |
| --- | --- |
| 每个 cache 完整性 | `status=complete`、records SHA、各 NPZ SHA，以及本 cache 的 `frozen_tail.pt` SHA 均核对；不再只信任首个 tail 文件。 |
| 跨 cache tail 身份 | `save_tail` 写 canonical tensor manifest，包含输出投影、答案权重、MLP/norm 每张量的 shape/dtype/bytes SHA、text config 和 option token IDs；训练检查每 cache 的 manifest 自哈希与所有 chunk 的 manifest 哈希相同。检查的是实际 tail 配置与张量身份，而不只比较序列化文件名。 |
| 监督来源绑定 | 每个 cached query 必须属于传入训练清单且清单 `split=train`；cached question 完整对象、video ID、gold 与该冻结行相等。真实条件必须 supervised；静态条件必须 nonsupervised、gold=null，question 去除 answer_index/answer_text。完整 question 对象比较亦绑定清单中 video_sha256 等来源字段。 |
| 训练覆盖 | 真实 supervised query 集合必须等于全部允许训练 query；case ID 不得重复；每个源静态条件列表严格等于 `static_first, static_last`，不再只比较全局数量。 |
| 图像绑定 | cache 提取时对原16帧文件逐一计算 SHA，并与来源帧清单核对；静态条件同时保留 source-frame SHA 与真正重复输入的16项 SHA。 |
| 固定方法/seed/checkpoint | evaluate 要求完整 4 方法×3 seed 共12个声明checkpoint，逐文件校验哈希，检查训练完成、协议哈希、step=200及payload方法/seed一致。 |
| 主结果原生路径 | 对每个 case 重新运行完整模型，所有方法共用同一batch；零初始化 wrapper 须与native全词表逐元素相等，干预后native恢复也须相等，并检查输入张量哈希未改变。没有用缓存tail的得分代替主表。 |
| 干预边界与完成状态 | 评估检查每头相对delta≤0.100001、gain≤0.250001；资源中止、记录数不足、协议或源码在运行中改变均使该run失败，不能标complete。 |

### 近似 replay 是冻结设计，不是未修复的“精确性门槛”

`learned_controller_protocol_v1.md` 已明确：BF16 batch形状以及只保留3行输出权重使cached-tail训练目标是近似native目标。协议要求保存完整误差/flip，并要求前向和梯度有限；没有预设“精确replay”的误差门槛，也不在看到测试结果后增加门槛。全部validation、新test及seen endpoint主评估使用完整原生forward。

据此，撤回初审将“必须添加一个replay误差硬阈值”列为普遍阻塞项的表述。当前做法在诚实标明近似训练、如实报告全部数值差、最终以完整模型计分的条件下是明确的实验设计。不能把近似训练改写为精确等价，也不能隐去近似误差或答案变化。本审计未读取任何parity结果，因而不对其实际误差幅度作判断。

### 外部冻结契约仍承担的职责

Runner记录调用者传入的QA/帧清单哈希，但其自身没有把所有输入文件哈希逐项解析并比对协议正文中的固定哈希表。因此，独立 `evaluation_contract.py` 和测试前执行清单仍是整体数据绑定的一部分，不能声称单独运行任意参数的runner就完成全部官方来源核验。`--video-ids`支持资源分片；各chunk内部记录完整，不自动证明外部已跑完16源。最终汇总必须核对冻结74条real QA、32条static保持条件的全覆盖、无重复与全部12组checkpoint；不能把合法子分片误报为全测试完成。

本次复核的源码 SHA256：cache `c1086e2c7b06bf66d10482556384ad7e7973da98f35bad9d3e0b3529d139b291`；train `7d314ec27c5fdd96697a3296aedee8d8335db57ea77c42e5c140ce4f382c6f60`；evaluate `492ef9f3d130559ba2c965c0b2fd790f23438e30541776f4a273f118d4ef4663`；runtime `eccbbce89b75d289015e1affc2fa47f5a3486b6162c1747c3362b7cc8f8d0f4d`。协议 SHA256：`3954767ef2662e15eb4934f90eea08ca6d48eaaef39fd2d297e3d9a3df590748`。后续源码改变时，这些结论只绑定本次快照。

## 4. 频谱比较的解释边界

时域 basis 为 $\sum_p a_{tp}v_{tpd}$，谱 basis 来自 $\sum_p\overline{\widehat a_{kp}}\widehat v_{kpd}$。后者含跨时间的双线性交叉项，并不是前者的完整正交 Fourier 换基。因此，相同 MLP 参数量和相同输出范数上限是有价值的控制，但不能据此把差异唯一归因于“FFT 表示本身”。本项目已有的完整线性 Fourier 换基等价结论，也不能直接替代这一双线性特征比较。

`spectral_phaseblind` 当前只对 conditioner 使用聚合 cross-spectrum 的幅值与零虚部，实际干预 basis 仍为有相位谱 basis。并且 `abs(sum_p cross_p)` 保留由空间位置间相对相位干涉影响的幅值；它不是 `sum_p abs(A_p)*abs(V_p)`。应称“聚合交叉谱相位条件器消融”，不能声称整个模型不使用相位、只有两个功率谱或已隔离全部相位作用。

各 basis 的尺度不同会影响 gain 范围、饱和与裁剪频率。建议同时报告各方法实际残差范数与 cap-active 比例，而不只报告相同理论上限。主比较应在同一源、同一题、同一 seed 上配对；置信区间以源视频为重采样单位，不能把同视频多个 QA、多个 head 或三个 seeds 当作独立新视频。

## 5. AGENTS.md 路径审计

已检查实际工作区根目录及其祖先、用户提供的 C 盘项目映射路径及其祖先、环境出现的其他项目映射路径，以及各自 `docs`、`spectral_memory`、`data`、`data/internal_controller_v1` 路径，并对相关子树运行 `rg --files -g AGENTS.md`。本次未找到适用的 AGENTS.md。

当前工具运行时对实际工作区和已存在映射路径的 `is_junction()` 检查返回 false；因此没有把“此路径一定是 junction”当作已验证事实。无论盘符/映射显示方式如何，本轮编辑均用工作区相对路径，未通过删除或重建目录处理映射问题。
