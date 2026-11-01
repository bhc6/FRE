# Spectral Memory Probe 实验执行协议

本项目首先交付冻结表征的效用探针与可控干预数据生成。其输出是决定是否值得继续研究的证据准备，不是完整长期记忆方法的实验结果。

## 1. 假设与可否证形式

H1：在固定数据划分和读出器下，频谱/局部时频特征相对语义与强时域统计，对边际效用预测有稳定增益。

H2：以事件方向、顺序、状态转移为目标时，保序表示或复频谱能够保留功率摘要不能区分的信息。保留完整复频谱只是可逆坐标变换，本身不会创造信息。

H3：在真正更新的记忆状态或明确读出干扰下，某些动态结构的保留改善未来任务恢复；该结论必须在相同成本、误差和模型容量控制下验证。

H4：基于这些结构的写入/检索策略，在固定真实记忆预算下改善所选记忆集合的下游任务收益，而不仅是提高单条效用拟合分数。

目前代码直接支持 H1 的离线实验、H2 的数值控制，以及 H3 的干预原语。H3/H4 的真实记忆更新和读出需要后续接入，不能由已有报告代替。

## 2. 已实现特征

| 分支 | 内容 | 重要限制 |
| --- | --- | --- |
| semantic | 时间均值 embedding、四个向量范数统计 | 冻结因果 VLM 中可能已含顺序信息，不是“纯静态语义” |
| temporal | 每维方差、一阶导数绝对/均方根、二阶导数、末首差、lag-1/2/4 相关 | lag 以时间格计；跨采样率时不能直接解释成相同物理时长 |
| spectral | 全局与局部 Hann PSD 的频带份额/熵/中心/平坦度/能量，局部均值/标准差/末首差，三层 Haar 能量 | 包含局部顺序；不能当作纯功率消融；Haar 是事件相对尺度 |
| ordered | 事件归一化时间上的 16 点线性插值序列 | 作为保序基线；插值降维并非抗混叠重采样，不能据此宣称保留所有原始动态 |
| phase | 同一 16 点序列的实/虚 rFFT 系数，含 DC 与单侧能量配重 | 与 ordered 信息等价，不能解释成新增独立信息 |

PSD 去均值、使用 Hann 窗，单位为 activation²/Hz；频段统计排除 DC。`power_spectrum(...,window="boxcar")` 可用于严格的功率守恒与反转控制。常量信号的动态能量等返回零。

默认频段为 `(0,0.125]`、`(0.125,0.25]`、`(0.25,0.5]` Hz，局部窗口 16 时间格、半窗步长。频段超出 Nyquist 或任何局部窗口内没有对应频点时直接报错。更改时间格采样率必须一并检查窗口时长、频率分辨率和频带。v1 不计算 coherence：单段 FFT 的朴素相干会退化，需独立实现有效窗口平均与不确定性估计。

不同事件长度仍可能留下长度、频率分辨率及窗口数的混杂。正式实验应报告等时长窗口和长度分层控制，跨采样率实验仅比较共同 Nyquist 支持区间，并固定物理窗口时长。

## 3. 六组探针与容量控制

| 模型 | 特征 |
| --- | --- |
| M1 | semantic + 合法 context |
| M2 | semantic + temporal + context |
| M3 | semantic + spectral + context |
| M4 | semantic + temporal + spectral + context |
| M5 | semantic + temporal + ordered + context |
| M6 | semantic + temporal + phase + context |

默认是带截距 ridge 回归；标准化仅拟合 train，alpha 在 `{0.1,1,10,100}` 中由 validation MSE 选择，测试集不参与选择。报告选定模型在 test 的表现，不再用 test 调参。参数按组独立选择；保存均值、尺度、系数、截距及可选投影。

M4 相对 M2 是主要增量对照。M3 与 M2 不是嵌套模型，不能直接归因。M5/M6 检查简单保序线性编码；充分的序列建模、复杂非线性特征或训练预算匹配仍需另外实现。

运行 `--projection-dim 64` 可以先按训练集标准化、固定随机投影，再使用相同宽度回归头。需要同时报告原宽度和等宽结果，并重复随机种子；该控制本身会损失信息，不能视为唯一公平性标准。进一步必须比较训练预算匹配的 TCN/GRU/小型时间 Transformer，以及原 hidden 序列的强非线性读出器。

## 4. 指标与统计

已报告 MSE、MAE、Spearman，以及同一 query/base-memory 决策组中的非负效用线性 NDCG。负效用截为零只用于 NDCG 的相关性增益定义；回归、选择和真实收益保留负值。目标或预测常量时 Spearman 为 null，而不是伪造为零。

成对比较使用同一 test 样本，先计算独立主体/分组内平均 MSE 差，再等权 bootstrap 分组。正值表示增强模型误差更低。小于 10 个独立组会标记探索性，少于两个不报区间。这个区间估计主体等权效应，而主 MSE 是事件等权；两者不可混为同一估计量。

不要对同视频的重叠窗口当作独立样本 bootstrap。层、频带、任务、干扰条件存在多重比较：先在 train/validation 固定一项主要假设与层，再测试；探索性图表单列，必要时预声明多重检验校正。bootstrap 区间本身不是显著性与实际收益的保证。

## 5. 固定预算选择

实现整数成本的 0/1 背包，对正预测分数进行加和优化。提供单条真实效用加和的 oracle，以及 random、recency 的同预算对照。保存选中候选 ID 和实际成本，负预测候选可不选。

真实集合效用一般不可加：重复信息、互补事件和干扰都会改变边际收益。因此 `sum_individual_utility_proxy` 仅是离线诊断；oracle 也只是“加性目标的 oracle”。正式实验必须拿选中集合重新评估 `R(q|selected_memory)`，并加入语义相似检索、运动/差分、均匀采样、事件覆盖等强基线。当前选择模块没有宣称次模性或最优真实任务收益。

## 6. 可运行顺序

```powershell
python -m spectral_memory demo --output-dir outputs/pilot_demo
python -m spectral_memory labels --manifest outputs/pilot_demo/unlabeled.jsonl --evaluations outputs/pilot_demo/evaluations.jsonl --output outputs/pilot_demo/paired.jsonl
python -m spectral_memory validate --manifest outputs/pilot_demo/paired.jsonl
python -m spectral_memory run --manifest outputs/pilot_demo/paired.jsonl --output-dir outputs/pilot_paired --projection-dim 64
python -m spectral_memory intervene --manifest outputs/pilot_demo/paired.jsonl --output-dir outputs/pilot_reverse --mode reverse
```

最后一步只生成未标注的干预表示。原来的 `evaluations.jsonl` 不能用于新候选；sample ID、candidate hash 已变化。真实模型提取与标注流程见数据协议和提取指南。

## 7. 阶段门槛

1. 可用性门槛：时间格、单位、候选/基础记忆哈希与源视频划分正确；常量、反转、相位随机化等数值控制通过；所选层能够解码目标状态/顺序。
2. 增量门槛：真实标签上 M4 对 M2 有跨主体稳定提升，且经等宽控制、强保序模型、长度/运动/动作类别控制后仍有实际意义。提升幅度阈值应在试点后、正式测试前预注册。
3. 机制门槛：受控表示或视频干预在真正消费该信息的读出器上改变恢复任务；identity 往返和等能量损伤控制不能解释全部现象。
4. 方法门槛：同真实成本下选中集合的真实任务表现改善，跨模型/用户/采样率保持，计算开销可接受。

如果仅 FFT 摘要优于均值池化，应定位为弱基线下的工程发现；如果收益完全由差分/时域模型解释，应转向时序结构保留问题。频谱为负结果仍有研究价值，但不能包装成确定的频域记忆机制。

## 8. 复现与计算范围

记录模型/处理器版本、权重 commit、输入和输出哈希、帧率/格率、层索引、prompt、池化、频段、窗口、所有种子、划分及预算单位。当前单机实现把事件特征载入内存，适合小规模探针；高维多层、大量事件应先缓存分层特征并改为分块训练。

提取器使用 `output_hidden_states=True`，即使只保存七层也可能暂存全部层，显存开销不能按七层计算。没有硬件实测前，不承诺 3B/7B 的显存或吞吐数字。先以短事件、小分辨率和一层输出验证数据对齐，再扩大规模。
