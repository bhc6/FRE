# 从末层小残差转向内部路由：三篇最近主会近邻

核查日期：2026-11-01。本轮只核三篇最接近的原论文正文，未按摘要中的 improvement 或 parameter-free 字样推定可复现收益。当前本地基座是固定 Qwen3-VL-2B；以下没有一篇直接证明本地模型与这批 Perception Test 问题上的收益。

| 论文与正文依据 | 实际干预及评估 | 相对 v1 的实质变化与限制 |
| --- | --- | --- |
| **See What You Are Told: Visual Attention Sink in Large Multimodal Models**，ICLR 2025，§5、Eq.4、表1，方法 VAR [1] | 对 image-centric head 的各文本 query，把 sink attention 的比例 `p` 汇总为 `Ω`，按原比例分配给非 sink 视觉 key，整行质量不变。默认 `p=0.6`，sink 阈值 `τ=20`。表1 GQA：Qwen2-VL-7B 64.5→64.7，LLaVA-1.5-7B 62.0→63.5。 | 直接改变 attention key 路由，可作用多层、多 query，区别于 v1 的末层末 query context 小残差。无直接 2B 结果；sink hidden dimensions 和 image-centric 阈值有架构/任务依赖，不能原样移植到 Qwen3。 |
| **RAR: Reversing Visual Attention Re-Sinking for Unlocking Potential in Multimodal Large Language Models**，ICLR 2026，§4.1、§5、§6.1、表2 [2] | SADS 用两个二分量 GMM 区分视觉强度与非视觉 attention entropy，保留 vision、广域 sink_G 及 head0，裁去其余 head。主结果经过 **670k 数据微调**：GQA Qwen2.5-VL-3B 60.4→64.2，InternVL2-2B 55.6→59.0。 | 后部多层 head 选择改变可用信息通路；parameter-free 不等于零训练。不能把主表收益当成直接冻结干预效果；正文另列末5层训练免调 reallocation 的 +0.74pp 诊断，量级和实验角色不同。 |
| **Reallocating Attention Across Layers to Reduce Multimodal Hallucination**，CVPR 2026，§3.1–3.2、Eq.6–10、表1–2 [3] | 先按模态 attention 质量比给 head 分类，再按层边界区别浅层 perception 与深层 reasoning，缩放整个 head 输出：`O'_h = α_h(A_h V_h)`。模型为 Kimi-VL-A3B 与两个 7B 系统，按基准小子集校准超参。 | 修改多层整 head context，**A 本身不变**；不是视觉概率重分配，也不是 v1 的受限加性残差。A3B 为激活参数标签，不等于总参数3B。正文公式中的 query 聚合口径需实现级对齐，不能仅凭文字写成已严格复现。 |

## VAR 公式核对

对选中层/head 的文本 query `i`，令 `S` 为已判定 sink key，`V` 为非 sink 视觉 key，原 softmax 全行权重为 `a`：

```text
Ω_i = p * Σ_{j∈S} a_ij
a'_ij = (1-p) * a_ij                         j∈S
a'_ij = a_ij + Ω_i*a_ij/(Σ_{k∈V} a_ik)      j∈V
a'_ij = a_ij                                其余合法key
```

image-centric 选择比为 `r=非sink视觉mass/全部视觉mass`，按文本 query/head 计算。原文 §6.1 不干预最后层；表4取消 head 选择、对全部 head 干预时三个评测指标均降为0。因此任意全 head 视觉增强不能称 VAR 的安全等价实现。[1]

分母为0时必须采用预声明无操作策略，不能产生 NaN。所有计算仍遵守 causal/padding mask。`p=0` 应恢复原路径，非零干预也应核整行质量和非法位置为0。sink 是否包含文本key、检测所取 residual stream、GQA head 映射都必须与所声称的方法一致。[1]

## 本地下一步判断

以下是本项目基于原文的推断，不是论文已经证明的结论：优先检查早/中/后层、多个文本 query 的视觉质量守恒重分配，并以整 head 输出缩放作机制不同的对照。先用已看64源检验可实现性和静态副作用，再冻结方法、强度及主指标；新增32源只用于之后的独立源确认。不能把新 test 标签子集上的事后最优层当预声明收益。

只有观察到本地 head 的视觉质量不足或 sink 退化，才进一步引入对应识别器。Qwen3 的 DeepStack、时间分组、RoPE 与 Qwen2/InternVL 不同；识别失败或无效也是预期可检验结果。完整 RAR 训练规模超出这一低成本局部试验。

## 原文来源

1. ICLR 2025 正式论文：`https://proceedings.iclr.cc/paper_files/paper/2025/file/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Paper-Conference.pdf`
2. ICLR 2026 正式记录：`https://proceedings.iclr.cc/paper_files/paper/2026/hash/42a1f9ef76af656f3fe33580ab794351-Abstract-Conference.html`；作者机构发表版正文：`https://eprints.gla.ac.uk/382078/1/800_RAR_Reversing_Visual_Atten.pdf`
3. CVPR 2026 正式正文：`https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Reallocating_Attention_Across_Layers_to_Reduce_Multimodal_Hallucination_CVPR_2026_paper.pdf`；正式页与PDF本轮均已取得HTTP200。作者版本：`https://arxiv.org/html/2510.10285v3`。

本次另检索到分隔符缩放论文，但没有纳入已核三篇的方法清单，避免将未完成公式核验的候选写成已验证实现依据。现有外部harness保持不变不等于一定有涨点；这里仅锁定可区别于 v1 的内部机制。
