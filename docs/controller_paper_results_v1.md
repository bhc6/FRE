# 小模型内部频谱控制器：完整实验与论文结果稿

日期：2026-11-01。状态：**本轮协议内实验全部完成，结果未显示频谱控制器收益。** 这是可复核的小规模对照实验，尚不能支撑“频谱修复内部记忆”或顶会主贡献的正面结论。

## 已完成范围与直接比较

冻结 Qwen3-VL-2B-Instruct 的 2,127,532,032 个去重基座参数，在末层最后文本位置的 attention 输出处加入小型残差控制器。比较 identity、同参数时域 MLP、交叉谱 MLP、仅条件器去相位的谱 MLP、八参数常量谱控制器。三个 MLP 均有 18,584 个参数。

32 个已看源视频的 142 道官方 QA 加 64 条无金标重复帧样本用于训练；8 个已看源的 40 道 QA 和 16 条重复帧样本用于验证。新测试使用与旧 48 源无 ID/内容哈希重叠的 16 个源，保留其全部 74 道官方三选题，另加 32 条无金标重复帧条件。原单源端点构造的 32 道有金标题单独列出。

四种可训练方法分别使用三个固定 seed、200 步 AdamW；同 seed 的方法逐步使用完全相同的训练样本。最后第 200 步检查点统一评估，未筛选最好的 seed 或验证检查点。所有方法使用相同 16 帧、8 个真实视觉时间格及完整原选项。控制的是输入预算与三种 MLP 的参数量；新增参数、临时激活及实测内存单独记账，不能笼统声称相对 identity 总存储完全相同。

全部准确率由完整模型前向计算，训练尾层重放分数不进入结果表。完成 12 次训练、194 个评估条件、2522 条方法/seed 记录。206 次训练特征采集保持原词表输出；194 个评估条件的零初始化干预及卸载恢复均与原生输出逐项相等。

## 新测试结果

seed 顺序固定为 20261101、20261102、20261103。百分比按三 seed 平均；identity 是一个共享基线。所有未聚合结果见 [完整逐 seed 主表](../outputs/controller_analysis_v1/test_main.md)。

| 方法 | 训练参数 | 每 seed 正确题数 / 74 | QA micro accuracy | 视频 macro accuracy |
| --- | ---: | --- | ---: | ---: |
| identity | 0 | 32（单基线） | 43.24% | 40.52% |
| 时域 MLP | 18,584 | 31 / 31 / 31 | 41.89% | 39.27% |
| 交叉谱 MLP | 18,584 | 31 / 31 / 31 | 41.89% | 39.27% |
| 条件器去相位谱 MLP | 18,584 | 31 / 31 / 32 | 42.34% | 39.69% |
| 常量谱控制器 | 8 | 32 / 31 / 31 | 42.34% | 39.69% |

预先声明的唯一主要比较是全部 74 题上三 seed 平均的 **spectral−time 视频宏平均正确率差**，实测为 **0.00 个百分点**。16 个源逐源及逐题的正确性对比均为零，因此固定 seed 的视频 bootstrap 和全局 seed×视频 bootstrap 都返回 `[0, 0]`。**这是观测样本没有任何正确性不一致所产生的退化经验区间，不能解释为总体等效性、无泛化不确定性或“频谱永远无效”。** 三个 seed 的准确率标准差为零也不表示训练随机性被充分估计。

交叉谱 MLP 相对 identity 的视频宏平均差为 −1.25 个百分点，两种 95% bootstrap 区间均为 `[−3.75, 0]`。该差异只来自同一题 `video_687:mc:3`：原三选项 logits 为 `[27.75, 27.875, 26.625]`，干预后为 `[27.75, 27.75, 26.625]`；A/B 精确平局按固定 argmax 次序选 A，而金标为 B。须保留这一回退，但不能将一个 BF16 边界上的平局包装成稳健的机制性损害。

temporal 标签子集为 41 题，identity 15/41，交叉谱 MLP 每 seed 14/41；memory 标签子集为 18 题，分别为 4/18 与 3/18。描述类非时序标签子集 30 题全部方法均为 16/30。子集重叠且标签不等于已验证的任务机制，不能把 memory 标签直接称为长期遗忘测试。

## 网络确实工作，但没有修复已知问题

独立从完整词表 NPZ 复算发现：三个 MLP 在全部 74 道真实题上产生不同的增益张量，128 个 head×coordinate 坐标都随输入变化；所有 12 个训练模型均改变了每道真实题至少一个词表 logit。常量谱增益在输入间保持固定。因而观察到的是有效执行、输入相关的内部干预。

这些变化未转化为任务收益：每个方法/seed 都没有修复任何新测试基线错误；所有回退最多涉及上述同一题。每个模型仅在 3–13 道真实题上改变了原三选项中的某个 logit，gold-margin 变化中位数均为零。完整变化统计与逐题审计见 [独立干预效果报告](../outputs/controller_effects_audit_v1/report.md)。输入相关的增益变化本身不能证明学到了有用的频谱模式。

实际最大 head 修正范数比仅约 1.994%，没有触发预设 10% context 限幅。由此不能把失败简单归因为“限幅截断了输出”，也不能推出放大幅度一定有效。该界限针对转换前 FP32 context 修正，不直接约束 BF16 投影后的残差或最终 logits。

测试集 32 条重复帧条件上，12 个模型均为 0 次 top-1 翻转；交叉谱 MLP 三个 seed 的平均 baseline→controller 三选项 KL 分别为 0.000234176、0.000242334、0.000151113。这些条件没有语义金标，只说明在这一构造上保留了基线选择。

原已看单源端点诊断中，全部 13 个设置均为静态 8/8、混合 12/24。功能保持门槛通过，**混合端点选择错误没有被修复**。该构造不能作为 32 个独立视频，也不能代替新源测试。

## 数值与资源说明

训练通过冻结最后层尾部计算图传播梯度，以节省显存。全部 206 个训练条件都公开了原生、batch1 和 batch8 尾层三选项分数：最大 logit 误差为 0.25；batch1 的 argmax 变化为 0/206，batch8 为 1/206，仅涉及一条无金标静态样本；真实 142 题均无变化。三 seed 共用相同缓存，这些重复数值记录不是独立确认。详见 [重放数值审计](../outputs/controller_replay_analysis_v1/report.md)。

本轮 15 个受限 GPU 进程累计 1070.32 秒，最长单进程 159.45 秒；CUDA allocator 已分配峰值 4,479,237,120 字节，已预留峰值 4,601,151,488 字节，采样最高 GPU 温度 71°C，资源停止次数为 0。保留两 CPU 线程、0.65 CUDA allocator 和 300 秒进程上限。两秒遥测可能漏掉短峰值；这是本轮运行成本，不是整个研究历史总成本。

按实际协议覆盖可核算 3355 次完整模型前向：412 次训练特征采集、2910 次验证/测试/端点评估、33 次工程冒烟；不含尾层训练和 CPU 数学测试。缓存、权重、原始分数和源码快照等本轮运行归档约 1.40 GB，不包含全部源视频数据。完整账目见 [最终完整性记录](../outputs/controller_experiment_v1/final_integrity.json)。

identity 的单次计时仅含原生前向。12 个训练控制器都计算全部 time/spectral/phase bases，时域和常量方法也承担 FFT 开销；zero/restoration 检查另计入总时长。这是研究实现的运行开销，不能作为各方法独立优化后的部署速度排名。硬件、驱动、共享参数计数及依赖来源见 [环境审计](../outputs/controller_experiment_v1/environment_audit.md)。

## 可直接用于论文的结果段落

> We evaluate a query-conditioned spectral readout controller in a frozen Qwen3-VL-2B model using a source-disjoint local test subset of 16 Perception Test validation videos and all 74 associated multiple-choice questions. Across three fixed training seeds, both the spectral and capacity-matched temporal controllers answer 31/74 questions correctly, compared with 32/74 for the unmodified model. Video-macro accuracy is 39.27% for either controller and 40.52% for the unmodified model. The spectral–temporal difference is zero on every observed source; the resulting degenerate bootstrap interval does not establish population-level equivalence. Learned gains vary across inputs and alter internal readout and vocabulary logits, but no baseline error on the new test subset is repaired. All controllers preserve the choices on 32 repeated-frame consistency conditions, while the previously observed mixed-endpoint failure remains unchanged at 12/24. These findings do not support a memory-repair claim for this intervention under the tested configuration.

本段必须与划分、训练预算、相位条件器消融边界和数值说明一起使用。视频为 14.362–34.910 秒，本地 test 来自官方 valid，不能写成官方隐藏测试集、长视频基准或全面复现第三方 SOTA。仅有一个基座、一个插入层、一个 uniform-full 输入配置和 16 个新源；没有多 harness 泛化证据。

## 研究决定与论文定位

本轮支持的是一个完整、受控的负结果：**在固定小型 VLM 的末层，输入相关的交叉谱读出修正没有优于同参数时域结构，也没有修复已观察的端点失败。** 这可用于论文中的诊断、消融或负结果部分，尚不能作为“频谱记忆能力涌现”的顶会主结果。

当前最后层控制器分支到此封存。后续机制研究应先在训练/开发源上识别“目标证据可被独立解码，但原问题读出失败”的具体层与子空间，再用目标证据恢复、无关证据恢复和对应时域干预验证选择性。只有这些控制显示可复现的修复，才有理由选择相应内部模块；FFN 仍是候选支路，尚未建立其导致遗忘的因果证据。新的调参或方法探索必须把本轮 test 视为已看数据，并另建未见确认集。

最近邻正式工作的重合与差异见 [当前内部方法文献对照](controller_related_work_comparison.md)；代数推导与可辨识边界见 [数学说明](learned_controller_mathematics.md)。经典交叉谱恒等式、已有谱 adapter、内部可读性与路径分析不作为首次贡献。

## 论文表格与复现入口

- [完整主报告](../outputs/controller_analysis_v1/report.md)：全部 seed、子集、静态保持、端点、成对区间和资源。
- [主表 LaTeX](../outputs/controller_analysis_v1/main_table.tex)及 [成对比较 LaTeX](../outputs/controller_analysis_v1/paired_comparisons.tex)：适用于带 `booktabs` 的论文模板。
- [数据协议](internal_controller_data_audit.md)、[冻结实验协议](learned_controller_protocol_v1.md)及 [分析复现命令](learned_controller_analysis.md)。
- [测试前封存记录](../outputs/controller_experiment_v1/pretest_freeze.json)：测试开始前保存数据、源码和 12 个检查点的哈希。
- 原始全词表分数及逐题记录位于 `outputs/controller_evaluation_v1/`；训练权重、优化器和逐步日志位于 `outputs/controller_training_v1/`。
