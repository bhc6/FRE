# 注意力与值的时间谱：固定头读出恒等式和相位反例

日期：2026-11-01。本文给出有限维线性代数推导和纯 NumPy 合成验证，不联网、不使用 GPU、不读取真实 QA 结果。这是 Parseval 内积恒等式在注意力读出上的应用，不宣称新定理、性能改善或已发现模型的失败机制。

## 1. 固定 query、固定 head，保留全序列归一化

固定某层、某个 query 位置和某个注意力头。令视觉 key/value 位置能对应到同一网格 $t=0,\ldots,T-1$、$p=1,\ldots,P$，每个 value 为 $v_{tp}\in\mathbb R^{D_h}$。文本和其他非视觉位置统一记为集合 $\mathcal S_{other}$。本文讨论推理时无 dropout 的 softmax 权重，并把不可访问位置的权重置零：

$$
a_i=\frac{\exp(\ell_i)}{\sum_{j\in\mathcal S_{visual}\cup\mathcal S_{other}}\exp(\ell_j)},
\qquad \ell_i=q^\top k_i/\sqrt{D_h}+b_i.
$$

位置编码、实际 mask 和其他注意力偏置均应已包含在 logits 中；这里不把不同模型的注意力计算简化成未验证的实现。视觉部分的权重 $a_{tp}$ 没有再单独 softmax：

$$
m_{visual}=\sum_{t,p}a_{tp}\le1,\qquad
c_{visual,d}=\sum_{t,p}a_{tp}v_{tpd},\qquad
c_{total,d}=c_{visual,d}+\sum_{i\in\mathcal S_{other}}a_i v_{id}.
$$

当 $m_{visual}>0$ 时，视觉条件分布 $\widetilde a_{tp}=a_{tp}/m_{visual}$ 的读出满足 $c_{visual}=m_{visual}\widetilde c_{visual}$。只在视觉 logits 上重新 softmax 会得到条件读出，并丢掉视觉总注意力质量；它一般不是原 head 的视觉贡献。特殊 token 若不属于视觉网格，也必须计入 other。

## 2. 时间 DFT 对读出的精确分解

对每个固定空间位置 $p$ 和 value 通道 $d$，沿长度 $T$ 的时间轴使用单位归一化 DFT：

$$
\widehat a_{kp}=\frac{1}{\sqrt T}\sum_t a_{tp}e^{-2\pi i kt/T},\qquad
\widehat v_{kpd}=\frac{1}{\sqrt T}\sum_t v_{tpd}e^{-2\pi i kt/T}.
$$

由 Fourier 矩阵 $F^*F=I$，以及实数 $a_{tp}$，得到逐通道恒等式：

$$
\boxed{c_{visual,d}=\sum_{k=0}^{T-1}\sum_p\overline{\widehat a_{kp}}\,\widehat v_{kpd}.}
$$

这不是“将注意力平均后再看 value 的谱”，也不是两个功率谱相乘。空间位置必须逐一匹配后求和；分别先池化空间再相乘，会多出不同位置间的交叉项。NumPy 对应 `fft(..., axis=0, norm="ortho")`。若使用默认非单位归一化前向 FFT，右侧必须再除以 $T$。

对实输入只保存 rFFT，可定义

$$
w_k=\begin{cases}
1,&k=0,\\
1,&T\text{ 为偶数且 }k=T/2,\\
2,&\text{其他非负频率},
\end{cases}\qquad
\Gamma_{kd}=w_k\operatorname{Re}\sum_p\overline{\widehat a_{kp}}\,\widehat v_{kpd}.
$$

于是 $c_{visual,d}=\sum_{k=0}^{\lfloor T/2\rfloor}\Gamma_{kd}$。奇数 $T$ 的最后一项仍须乘 2；$T=1$ 只有 DC，$T=2$ 只有 DC 与 Nyquist。$\Gamma_{kd}$ 是**有符号的向量贡献**，可以为负，不能解释为非负能量或频带正确率。

若进一步使用预先固定的线性读出方向 $u\in\mathbb R^{D_h}$，$u^\top\Gamma_k$ 可以重建该方向上的读出。它不自动等于最终答案 logit 的因果贡献：多头输出投影、残差、后续归一化和非线性仍须处理。对所有头分别分解并经过同一线性输出投影，可重建该注意力子层的线性输出部分；不能由此跳过后续网络。

上述等式是规则网格上的代数身份，即使各时间点对应同一空间槽位而非同一物体也成立。物理解读则需额外条件：时间采样应等间隔，且保持实际 token 映射。时间不等间隔时仍可按序列索引算 DFT，但不能直接把频点称为物理 Hz。变长帧、不同空间网格、丢帧或池化都应先明确对齐操作。

## 3. 相同注意力功率和 value 功率，读出仍可相反

取 $T=8,P=D_h=1$，同一条 value 轨迹为

$$
v=(1,0,0,0,0,0,0,-1),\qquad
a^{first}=(1,0,0,0,0,0,0,0),\qquad
a^{last}=(0,0,0,0,0,0,0,1).
$$

两条 attention 是循环平移，故每个频点均有

$$
|\widehat a^{first}_k|^2=|\widehat a^{last}_k|^2=1/T.
$$

两种情形使用完全相同的 $v$，value 功率当然也相同，但 $c^{first}=1$、$c^{last}=-1$。差异保留在交叉谱的相对相位：

$$
\operatorname{Re}(\overline{\widehat a_k}\widehat v_k)
=|\widehat a_k|\,|\widehat v_k|\cos(\arg\widehat v_k-\arg\widehat a_k).
$$

严格 one-hot 可由 mask 或 softmax 极限得到。为避免把反例限制在极限，另取 $\epsilon=0.1$，令 $a'=(1-\epsilon)a+\epsilon\mathbf1/T$。两条新 attention 均严格为正，可由有限 logits $\log a'$ 精确产生；功率依旧相同，读出变为 $+0.9$ 与 $-0.9$。因此，“同一个固定 head 的读出可由 attention power 和 value power 单独确定”已被反例否定。此结论不否定功率统计在某些任务上的预测价值。

## 4. Gate 的局部恒等式及适用边界

固定当前 query/head 的 $Q,K,V$ 与 logits，给每个可访问位置一个非负 gate $g_i$，并要求 $\sum_i a_i g_i>0$。若 gate 的操作定义为 softmax 前加 $\log g_i$，则

$$
a'_i=\frac{a_i g_i}{\sum_j a_j g_j}.
$$

$g_i=0$ 对应该位置 logits 置为 $-\infty$。仅 gate 视觉位置时，其他位置取 $g_i=1$，分母仍包含其他位置，且文本权重通常也会因重新归一化而改变。不能在视觉子集归一化后直接宣称重建了完整 attention。若操作定义为 softmax 后仅乘 gate、不重新归一化，输出则是 $\sum_i a_i g_i v_i$，属于另一个干预。

在这个固定状态的局部操作下，新的视觉读出仍可由 $\widehat a'$ 与同一 $\widehat v$ 按上一节重建。时间域逐点 gate 一般会混合频率；不能把它未经推导地替换成对 $\widehat a$ 的独立逐频缩放。

真实模型从较早位置或层插入 gate 后，后续 hidden states、query、key、value 和注意力分布都可能改变。这时重新前向得到的是新的状态，不能把冻结 $QKV$ 的局部公式当作整个多层网络的精确反事实。若要测试跨层因果作用，应区分“当前层固定状态的局部重算”和“完整模型重跑”，分别记录它们的输入和测量位置。

## 5. 最小验证与可用范围

代码：`spectral_memory/attention_spectral_checks.py`，命令：

```powershell
python -m spectral_memory.attention_spectral_checks --output-dir outputs/attention_spectral_checks_v1
```

脚本只使用 Python 标准库和 NumPy，单进程并限制数学库线程数为 2；拒绝已存在输出目录。固定种子 20261101，使用 float64、容差 $10^{-12}$，保存源码 SHA256、版本、每项误差与反例数组。

检查涵盖 $T\in\{1,2,3,7,8,31,32\}$，每种长度均验证完整 FFT、正确加权 rFFT、默认 FFT 的 $1/T$ 因子、视觉质量与文本贡献、含文本分母的 gate 归一化及零 gate。另检查 one-hot/有限 softmax 相位反例和奇偶端点权重。输出 `checks.json` 和 `examples.npz`。

2026-11-01 本地运行：**32 项全部通过**，`torch_imported=false`。完整 FFT 重建最大误差为 $1.945\times10^{-16}$，单边加权 rFFT 为 $1.943\times10^{-16}$；这只是上述合成输入下的数值误差。源码 SHA256：`a6e6d7c7f6b4f87198b070efd84e85fd2f1b9e35b341bdc5f5f344e02739f151`。实际结果保存在 `outputs/attention_spectral_checks_v1/checks.json`。

这些检查只能确认公式与这份小实现一致；不能证明真实模型 hook 正确、注意力权重已捕获完整、空间对齐可靠，或该分析可以解释已观察到的问答错误。当前最直接的实验用途是：捕获同一 query/head 的实际 $a,V$ 后先验证读出重建，再讨论频带贡献与相位；重建未通过时不解释频谱。
