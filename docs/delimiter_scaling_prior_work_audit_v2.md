# Delimiter residual缩放：先行工作与新颖性审计

检索截止2026-11-01；只读论文/作者代码，CPU与网络操作。数学命题由主任务维护在 `docs/delimiter_scaling_mathematics_v2.md`。本审计没有运行新模型、读取确认集分数或宣称新理论。

**结论：** RMSNorm的正尺度不变性、pre-norm残差大小与相对更新步长的关系，以及大范数对方向扰动的抑制都有明确先行工作。固定token集合的一次post-block缩放与后续attention/FFN更新乘倒数的等价，是这些性质的直接推论，适合作为机制核查/实现控制。当前检索没有找到完全同文表述的DTS专用命题，但这不足以支持理论优先权声明。潜在研究贡献应来自对已发表DTS解释的精确实证修正、跨模型可复现的故障定位与独立任务收益。

## 1. DTS原文是否已经给出此解释

核查了最终23页论文全文，并重点读取§5.1、Figure3、Appendix A.2/A.10；检索RMSNorm、LayerNorm、residual、scale invariance、damping及相关词。

- §5.1将缩放解释为delimiter收到更多attention，同时delimiter的value幅值增加，增强image tagging；Appendix A.2把hidden scaling与投影后的Q/K/V单独缩放联系起来。
- 检索到的normalization指softmax的概率归一化，而非下一层input RMSNorm。
- A.10讨论parallel context encoding、多sink分布与训练分布差异，没有给出post-block residual缩放与reciprocal update的等价。
- 作者代码的post-block residual hook见 `docs/delimiter_scaling_implementation_audit_v2.md`。它与投影后直接缩放Q/K/V是不同操作。已读材料没有解释这个归一化边界，但这不等于证明作者全部实验或最终收益无效。

来源：`https://proceedings.iclr.cc/paper_files/paper/2026/file/aebf6284fe85a8f44b4785d41bc8249a-Paper-Conference.pdf`。

## 2. 最接近的已发表/公开工作

| 文献 | 已有内容与确切位置 | 与固定token DTS命题的关系 |
|---|---|---|
| Root Mean Square Layer Normalization（2019） | §4、§4.1，Eq4–7：RMS正齐次及输入正尺度不变；§4.2讨论梯度尺度关系 | 基础数学已知；没有DTS/video修复主张 |
| GeoNorm: Unify Pre-Norm and Post-Norm with Geodesic Optimization（arXiv2601.22095v1，2026预印本） | Appendix B pp12–13，Eq10–11明确RMSNorm正尺度不变；Eq13给出residual状态重参数后的skip/update权重；Eq14最终norm输出等价 | 最直接的重参数与输出等价先例；不是选定token一次λ后全后缀1/λ的特定命题，但数学机制重叠显著 |
| nGPT: Normalized Transformer with Representation Learning on the Hypersphere（2025版本） | §2.2.1 Eq4–5普通pre-norm更新；§2.2.2 Eq10–11以可学习α_A/α_M控制attention/MLP在球面上的更新 | 已有“内部forward更新步长”解释；它改架构并训练，不是冻结模型中DTS的相同干预 |
| The Curse of Depth in Large Language Models（arXiv2502.05795v3） | §3–4：pre-LN residual方差随深度增长、深层作用降低，提出norm输出按深度缩放 | 支持相对更新与范数的研究背景；其理论假设与训练方案不证明任意token缩放能提高QA |
| How Attention Sinks Emerge in Large Language Models（arXiv2603.06591v2，PDF标Under review） | §3.1 Eq3–6 pp4–5：pre-RMSNorm移除幅值，正交方向扰动受1/r(x)抑制；MLP放大有助于P0方向稳定 | 直接近邻“高范数抵抗方向改变”；讨论主要是训练扰动与P0 sink，不是视频delimiter的原生推理收益 |
| The Structural Origin of Attention Sink（ICML2026） | §3.1–3.2/Fig6：value aggregation后的跨样本方差差异；mean-centered amplification与纯norm scaling的对照；§4追踪FFN super neurons与QK | 关键反例：范数大本身不保证形成sink，方向/方差结构与操作位置重要；不能把它的attention输出缩放等同post-block residual DTS |
| Spectral Filters, Dark Signals, and Attention Sinks（ACL2024） | §2说明pre-RMSNorm；§5/Fig9、Appendix D追踪早层MLP大范数dark component、后续QK及很小的value写入 | 更早的方向/子空间/sink功能解释；其“spectral”是embedding/unembedding SVD而非时间FFT，也不是DTS倒数更新命题 |

这七篇为额外先行工作，不用于凑足主矩阵的13篇2025/2026主会论文。预印本保持预印本身份，未核实的会议身份不从标题或仓库推断。

## 3. GeoNorm的具体重叠，避免误报

令 `x_k=x_(k−1)+phi_k(Norm(x_(k−1)))`，GeoNorm设 `z_k=x_k/sqrt(k+1)`，由尺度不变性得到：

```text
z_k = sqrt(k/(k+1)) z_(k−1)
      + 1/sqrt(k+1) phi_k(Norm(z_(k−1)))
Norm(z_k) = Norm(x_k)
```

这里phi可为attention或FFN。它已经把残差状态的缩放转写成skip与update系数，并显式讨论最终输出等价。不同点是该文使用深度相关全状态变换；本地控制是固定token子集、在某个block后施加λ、在后续所有sublayers乘1/λ。后者需要逐token归一化和attention跨token依赖的一致性，但证明仍是逐sublayer归纳。应引用前者并把后者称为针对作者hook的专门化推论。

## 4. 对抗性证据：什么不能被新解释推出

**范数并不足够。** ICML2026的Structural Origin以跨batch的每维方差为统计量，构造 `o'_k=mu+lambda*(o_k−mu)`，其中mu来自随机token的value全局均值。Fig6报告这种mean-centered操作可诱导sink，而单纯输出向量norm scaling对照不能。这个对照位于attention聚合输出，其原理不能直接替换DTS的post-block hook；但足以反驳“任何token放大都会更吸attention”的通用说法。

**抑制更新不必有益。** 高范数稳定旧方向可能保留已有摘要，也可能阻碍必要的新信息整合。Curse of Depth甚至把更新相对变弱视为深层能力受限。因此必须同时报告正/反向scale和真实任务后果，不能先把“更稳定”命名为“更好的记忆”。

**sink不总是有害。** ACL2024与主矩阵VAR/RAR均提示sink承担特定功能；norm、attention质量、value贡献和答案logit贡献需要分别测量。attention升高或熵下降都不是充分的任务改善指标。

**等价不等于恒等。** 路径A=DTS与路径B=reciprocal updates可等价，但二者通常都不同于native基线。下一次attention可相同；其后的residual相加改变方向后，后续QKV和attention仍可变化。不能据RMSNorm说整次干预全无作用，也不能声称第一层不变意味着最终不变。

**数学条件有实际边界。** ε>0、低精度、额外视觉残差注入、postnorm、动态token选择、缓存或不同generation输入均需检查。多层连续缩放需累积scale，不能用一次全后缀1/λ控制冒充作者层0–3各乘λ的完全复现。普通投影bias并不破坏“归一化输入已相同→投影相同”的推论。

## 5. 建议的研究表述

可表述为：对已发表DTS作者实现进行归一化感知审计；在明确pre-norm条件下，将token residual缩放解释为后续相对更新的重新加权，并通过匹配的update-damping路径检验该解释；进一步检查是否在独立视频数据和多个模型中改善原生预测。

不应表述为：首次发现RMSNorm尺度不变、首次提出residual更新步长、已证明放大delimiter能强化记忆、或数学等价证明保证模型性能提升。若只有数值等价而无跨模型任务改进，这仍是有价值的复现诊断，但还不足以独立支持强方法论文。

## 6. 来源与可追溯性

- RMSNorm：`https://arxiv.org/abs/1910.07467`。
- GeoNorm固定v1：`https://arxiv.org/abs/2601.22095v1`。
- nGPT：`https://arxiv.org/abs/2410.01131`（本地PDF显示v2，2025-04-23）。
- Curse of Depth固定v3：`https://arxiv.org/abs/2502.05795v3`。
- P0 sink固定v2：`https://arxiv.org/abs/2603.06591v2`。
- Structural Origin正式页面：`https://proceedings.mlr.press/v306/li26bs.html`；PDF由该页面链接至MLResearch官方GitHub资源，已缓存。
- Dark Signals正式页面：`https://aclanthology.org/2024.acl-long.263/`。

七篇PDF、提取文本、SHA256与来源记录位于 `outputs/literature_internal_mechanism_v2/scaling_prior_sources.json`，ID14–20。阅读范围为上列公式/方法/相关实验，不声称七篇全部细节逐字读完。检索未命中完全相同命题不构成全领域不存在先例的证明。
