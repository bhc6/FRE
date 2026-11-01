# 编码后 `[T,H,W,D]` 功率谱的可辨识性与最小因果实验

日期：2026-11-01。本文是窄范围理论/文献审查，补充总矩阵，不声称新颖性、模型收益或规律不存在。所有新计算均为 CPU 上的 NumPy 合成张量代数；没有读取模型权重、预测、标签分数或新视频集。

## 结论先行

功率谱只保留复数 Fourier 系数的模平方。它适合做能量/周期性描述，却不是编码特征的充分统计量。对视觉编码后的 `X[T,H,W,D]`，至少要区分：逐空间位置时间谱、逐帧空间谱、联合 3D 时空谱、通道协方差谱和 attention 加权后的 context 谱。不同的汇聚顺序会产生不同不可辨识性：

- `|FFT_T|²` 对循环时间平移、严格时间反转不变，因此不能从它恢复起始时间或方向；但若保留跨空间位置的**联合** `|FFT_{T,H,W}|²`，时间反转一般会把 `k_T` 频率翻转，方向信息可能仍在时空耦合中。不能把“功率谱丢相位”简化成“所有反转都不可辨识”。
- 逐帧 `|FFT_{H,W}|²` 再对 `T` 平均会丢掉所有帧顺序；任意帧置换都给出相同的时间平均空间功率。若保留每帧的空间谱序列，时间顺序仍可存在于谱随帧的排列中。
- 空间平均后再做 `FFT_T` 会因不同位置相消而删除真实运动；先逐 patch 求功率再平均通常保留更多动态能量。
- 功率仍不唯一：共享的时间 all-pass 相位改变可以保持时间边缘谱和联合 3D 功率不变，同时改变实际特征；离散 phase retrieval 还存在平移、全局相位、共轭/反射和卷积等非平凡歧义。
- 沿 `D` 做 FFT 没有一般物理意义。通道旋转会保持跨通道总时间功率，却改变通道轴 FFT；通道协方差 eigen-spectrum 是另一种统计对象，不应叫时间频率。

合成张量检查见 `outputs/encoder_spectrum_identifiability_v2/algebra_checks.json`，脚本见同目录 `verify_algebra.py`；9项检查全部通过，误差约 `1e−16`，不涉及 VLM。

## 1. 具体不可辨识变换

设 `X[t,h,w,d]` 为实值编码特征，采用正交归一化 DFT。以下结论是代数性质，不是对某个模型的经验断言。

### 1.1 时间轴的边缘功率

对每个 `(h,w,d)` 计算

```text
F[k,h,w,d] = FFT_T X[:,h,w,d]
P[k,h,w,d] = |F[k,h,w,d]|²
```

循环平移 `X[t−τ]` 只给每个频点乘 `exp(−i2πkτ/T)`，所以 `P` 不变。严格时间反转 `X[T−1−t]` 给 `F[k]` 共轭并翻转频率索引；对实信号 `|F[k]|²=|F[−k]|²`，逐频点功率仍不变。于是时间边界、绝对起点和反向运动无法由该统计量单独恢复。

若只报告 `mean_(h,w,d) P[k,h,w,d]`，还会丢掉哪些 patch 在何时变化；若先做 `mean_(h,w)X` 再 FFT，空间异相运动可完全抵消。最小合成反例用两个相反空间位置的动态信号，full patch temporal power 非零而 pooled temporal power 为零。

### 1.2 空间谱与帧顺序

对每个 `t,d` 计算 `P_t[k_h,k_w,d]=|FFT_{H,W}X[t]|²`。若最后取 `mean_t P_t`，任意时间置换 `π` 都不改变结果。它可以描述总体纹理/空间尺度，却不能描述动作先后。若保留 `P_t` 序列或做 `FFT_T` 于复/幅值序列，顺序信息才有机会保留；这已经不是“时间平均空间功率”这一统计量。

### 1.3 联合 3D 功率并非万能

`P[k_t,k_h,k_w,d]=|FFT_{T,H,W}X|²` 保留了时空耦合，因此简单时间反转可能改变 `k_t` 与空间频率的配对；合成斜向运动的 3D 功率与其反转版本相对误差约1.22，而时间边缘功率完全相同。它仍丢掉复相位、全局时间/空间平移等歧义，并不能唯一重建 `[T,H,W]` 信号。

### 1.4 相位、平移与反射的先行证据

Beinert & Plonka 的离散 phase retrieval 综述把从 Fourier magnitude 恢复信号的 trivial ambiguities 列为 signal shift、全局旋转因子、conjugation/reflection，并讨论由卷积产生的 nontrivial ambiguities。它把 `|FFT|²` 等价于自相关信息，并明确问题即使理论上可唯一也可能病态。缓存与SHA见 `outputs/encoder_spectrum_identifiability_v2/phase_retrieval_ambiguities.{pdf,txt}`。

2026 预印本《The Importance of Phase in Neural Representations: An Internal Oppenheim–Lim Test of Image Classifiers》把幅值/相位交换变成逐层因果干预：对真实空间 feature map 使用 `F^{-1}(|H_A| H_B/|H_B|)`，并做 random-phase、unit-magnitude、batch-mean magnitude、DC-only 控制。作者在 ViT、GFNet、ResNet 和 PRISM2D 上报告身份读出可跟随 phase/sign donor；这说明“内部频谱干预”本身已有近邻工作，不能直接作为新颖性。论文范围是图像分类，不是 VLM 视频记忆。源码笔记固定 commit 见 `outputs/encoder_spectrum_identifiability_v2/internal_oppenheim_lim_author_notebook.ipynb`，未执行该 notebook。

Wadhwa et al. 的 phase-based video motion processing 则说明局部复小波的 phase 随平移变化，可用于运动估计/放大；这与功率只保留 magnitude 的不可辨识性互补。它不是 VLM hidden probe，但支持保留 phase 或相位差作为运动方向控制量。缓存 `phase_based_motion.{pdf,txt}`。

## 2. 与已有 Frame2Freq、LFM、NerVE 的边界

| 工作 | 真实轴/阶段 | 已有内容 | 本项目不能重复宣称 |
|---|---|---|---|
| Frame2Freq CVPR2026 | adapter 插入后的 `X[T,D]` 做 `FFT_T`；adapter 自身按 `[B·H·W·C_a,T]` 做 STFT；另有原视频 3D FFT | temporal spectral adapter、频段 ANOVA、时序运动诊断 | “首次把视觉 hidden 做时间频谱”或“频率模块没有内部谱分析” |
| LFM CVPR2026 | vision encoder 中间层 `Vhat_l` 组均值后 `FFT_{H,W}`；IFFT结果与 post-FFN `V_l` 融合，再进 projector | language-guided spatial frequency modulation | “首次在视觉编码后做空间 FFT”；不能把空间轴当时间轴 |
| NerVE ICLR2026 | FFN up/gate 的 pre/post activation，`[B,S,D]→[BS,D]` 后算 covariance eigen-spectrum | channel covariance spectral entropy/PR/JS | “沿D做FFT就是记忆频率”；协方差特征谱不是 Fourier phase/power |

因此可差异化的方向不是再画一张谱图，而是：在**小 VLM 的真实视觉注入边界**建立同轴、同布局的反事实干预，比较功率保留但 phase/order 改变的 native 输出，以及不改变谱的阶段/路径替换。

## 3. 为什么 attention-weighted context 可能误导

若每帧 context 是 `Y_t = a_t V_t`，时域相乘对应频域卷积：`FFT(Y)=FFT(a)*FFT(V)`。softmax 的 `a_t` 包络、跨帧集中、首帧/末帧偏置都会混入所谓“记忆频谱”。因此应并列保存：

1. 未加权 `V_t` 的 temporal power；
2. `a_t V_t` 的 power；
3. `a_t` 本身及其 autocorrelation/频谱；
4. 逐 patch 与 spatial-mean 两种汇聚。

旧 train32 中 attention 包络归一后 real/static AC 排序完全反转，是“测量对象被权重包络改变”的证据，不是“没有谱规律”的证据。跨模型 raw V 动态差异未随深度消失，也不能因为换层就保证 native 收益。先把混合点定位，再做干预。

## 4. 最小下一实验：功率保持、方向反事实、冻结原生输出

### 4.1 目标

回答一个窄问题：**当真实编码特征的 temporal power 保持不变、但 temporal phase/order 改变时，小 VLM 的原生输出是否改变；若改变，改变发生在视觉注入前后哪一个边界？** 该实验不是宣称时间反转标签正确，也不是新的训练方法。

### 4.2 固定输入与阶段

- 使用已审计的旧 train32 作为开发集；不读取或重调新159题分数。
- 每个视频保留真实 `T,H,W,D` 和 processor 时间采样，记录 PTS/帧索引、patch order、temporal patch size、DeepStack/merger 支路。
- 最少取三阶段：视觉 encoder 末端 `X_enc`、merger/connector 后 `X_conn`、写入 LLM 的视觉 token `X_llm0`。Qwen 的多支路视觉注入必须同步反事实；只改主支路要标为局部机制探针。

### 4.3 四个成对条件

对每个样本生成同一组 feature-level 条件，不重新编码视频：

1. **Identity**：原始 `X`，验证 hook round-trip 与无hook logits 完全一致。
2. **T-reverse**：`X[::-1]`，并同步帧时间顺序/位置元数据；对每个 patch channel 的 `|FFT_T|²` 保持不变，但改变顺序和 phase。若模型任务答案依赖动作方向，标签须单独定义，不能沿用原答案。
3. **3D-power-preserving all-pass**：对 `FFT_{T,H,W}X` 乘共享、共轭对称的非线性 phase mask，再逆变换；严格保持联合 3D power，改变相位。先做小 phase（例如固定 ±π/2 频带）和随机 phase 两种，避免一次破坏实值/布局。
4. **Spatial-mean control**：分别比较 `FFT_T(mean_{H,W}X)` 与 `mean_{H,W}|FFT_TX|²`，不作为模型干预，仅判断空间抵消是否解释旧谱结果。

每个非identity条件都保存：原生 logits 全词表差异、答案 token rank、视觉 token hidden cosine/RMS、下一个 block 的 Q/K/V、attention output、FFN output、最终解码文本（若安全）。不改变 tokenizer、prompt、generation seed、cache 或 temperature。

### 4.4 判别矩阵

| 观察 | 解释边界 |
|---|---|
| T-reverse 与原始 power 相同但 logits 明显变 | 功率统计量不足以解释 native 输出；phase/order 或位置编码参与。不能直接说“模型记住了方向”。 |
| 3D-power-preserving phase 改变也使 logits 变 | 联合功率仍非充分统计量；定位 phase-sensitive 路径。 |
| encoder 变而 connector/LLM0 恢复 | merger/connector 对反事实有投影不变性或低通效应；需检查 hidden，不要称模型无视 phase。 |
| 未加权V变、`aV`不变 | attention 包络可能抹平/重排动态，不是否定 encoder 规律。 |
| spatial-mean谱消失、fullpatch谱存在 | 先平均的统计口径不适合捕获局部运动；不证明模型使用它。 |
| 全部 native logits/答案不变 | 仅说明该 phase/order 方向在当前阶段未被读出；不是证明长期记忆不存在。 |

### 4.5 预注册的最小通过/失败标准

- Identity：全词表相对误差 `≤1e−6`（同 dtype 与相同缓存路径）；若不满足，停止解释并修 hook。
- 每个视频报告 power-preservation error（`≤1e−5`，按实际 dtype 放宽并记录）；同时报告 feature RMS/cosine，防止 norm 漂移冒充 phase 效应。
- 主要结果只看成对 logits/答案变化和阶段定位，不以一次准确率上涨作为成功标准。
- 若要声称 native 任务改善，必须再有未参与选择的冻结验证集、方向正确的反事实标签、identity/random/phase-null 对照。当前文献与本任务授权下不做这一步。

## 5. 新颖性与风险声明

该实验借鉴 phase retrieval、Oppenheim–Lim、phase-based motion processing，并与 Frame2Freq/LFM/NerVE 直接相邻；“做 FFT”“交换 phase”“检查谱熵”都已有先例。潜在贡献只能来自小 VLM 真实注入边界的严格可辨识性审计、attention 包络与未加权 V 的分离、以及经过预注册反事实的因果定位。没有新模型权重、没有新159题分数、没有跨模型 native 提升前，不应称为长期多模态记忆方法或顶会新理论。

## 6. 可追溯文件

- `outputs/encoder_spectrum_identifiability_v2/algebra_checks.json`：9项 NumPy 代数检查，全部通过。
- `outputs/encoder_spectrum_identifiability_v2/verify_algebra.py`：固定seed、无 Torch/GPU。
- `outputs/encoder_spectrum_identifiability_v2/sources.json`：phase retrieval survey、2026 Internal Oppenheim–Lim 论文、作者 notebook、phase-based video 论文的URL/SHA256。
- `docs/encoder_spectrum_source_alignment_v2.md`：Frame2Freq/LFM/NerVE 的阶段与谱轴对照。
- `docs/frame2freq_axis_audit.md`：作者实现轴/布局审计。

