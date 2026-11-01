# 学习式内部读出干预：数学对象、等价式与可辨识边界

日期：2026-11-01。对应冻结协议 `learned_controller_protocol_v1.md` 与 `spectral_memory/learned_readout.py`。本文推导计算对象，不预设实验能提高正确率。

## 1. 改变的是当前问题的内部读出

对最后 decoder 层的最后 prefill 文本位置，令各 query head 的实际注意力输出为

\[
c_h=\sum_{j\in\mathcal K}a_{hj}V_{hj},\qquad
a_{hj}=\operatorname{softmax}_{j\in\mathcal K}
\left(q_h^{\mathsf T}K_{hj}/\sqrt{d_h}+M_{hj}\right).
\]

集合 \(\mathcal K\) 包含全部因果可见文本和视觉 key。实验使用实际 head normalization、RoPE 与 GQA 对应关系；没有把视觉权重重新归一化。FP32 重建的 \(a\) 只用于控制器特征，原模型 SDPA 计算得到的 \(c_h\) 保留为基线。

视觉位置按处理器的真实时间格分为 \((t,p)\)，其中 \(T=8\)，每格对应输入的两帧，\(p\) 是空间位置。定义

\[
B^{\rm time}_{htd}=\sum_p a_{htp}V_{htpd},\qquad
c^{\rm visual}_{hd}=\sum_t B^{\rm time}_{htd}.
\]

这里没有先平均空间 token，也没有把隐状态通道当作时间轴。固定空间位置不一定跟随同一个移动物体；该限制不会使下面的代数恒等式失效，但限制物理解释。

本轮只给最后文本位置的 attention 输出增加一个残差，随后经过原有 FFN、最终归一化及输出层。它检验的是**问题条件化的证据读取修正**，没有改变视觉前缀存储方式，也没有证明更早层形成了新的记忆或建立了跨请求的持久记忆。

## 2. 交叉谱保存了什么

沿时间维采用正交归一化 DFT：

\[
\widehat a_{hkp}=T^{-1/2}\sum_ta_{htp}e^{-2\pi i kt/T},\qquad
\widehat V_{hkpd}=T^{-1/2}\sum_tV_{htpd}e^{-2\pi i kt/T}.
\]

逐空间位置形成交叉谱再求和：

\[
C_{hkd}=\sum_p\overline{\widehat a_{hkp}}\widehat V_{hkpd}.
\]

Parseval 恒等式给出 \(c^{\rm visual}_{hd}=\sum_{k=0}^{T-1}C_{hkd}\)。实数输入满足共轭对称；当 \(T=8\)，代码保存八个实坐标

\[
B^{\rm spec}=
[\Re C_0,2\Re C_1,-2\Im C_1,2\Re C_2,-2\Im C_2,
2\Re C_3,-2\Im C_3,\Re C_4].
\]

原视觉读出的重建系数是 \([1,1,0,1,0,1,0,1]\)，不是全部取 1。\(|\widehat a|^2\) 和 \(|\widehat V|^2\) 不能决定 \(C\)：相对相位可以让相同功率产生相反的读出。这是经典交叉谱性质，不作为新定理。

令完整 Hermitian 频域增益为 \(\gamma_k\)，正频率对应的两个实参数是它的实部、虚部，则

\[
u^{\rm spec}_{hd}=\sum_{k=0}^{T-1}\gamma_{hk}C_{hkd}
=\sum_{\ell=0}^{T-1}w_{h\ell}R_{h\ell d},
\quad
R_{h\ell d}=\sum_{t,p}a_{htp}V_{h,(t+\ell)\bmod T,p,d},
\]

其中 \(w=\operatorname{ifft}(\overline\gamma)\)，此处 inverse FFT 使用常见的 \(1/T\) 归一化；\(u\) 是下节限幅前的修正。Hermitian 对称保证结果为实数。因而谱控制器对应**循环时间位移相关项的加权组合**；时域控制器 \(\sum_tg_tB^{\rm time}_t\) 对应逐时间位置的加权。两者参数量相同，但允许的算子结构不同；频域写法本身不构成不可替代的表达能力。

一个直接反例：若 \(a_{htp}\) 对时间完全均匀，则 \(\widehat a_{hkp}=0\) 对所有非零频点成立，交叉谱只剩 DC。此时值轨迹的许多排列或循环位移具有相同交叉谱，任何只读取该交叉谱的控制器都无法从中恢复被消去的顺序。频谱并不自动保存所有时序信息。

## 3. 小网络的作用及限制

共享于各 head 的控制器使用 \(1152\to16\to8\) 的两层 MLP 和 SiLU，输入为实际 post-head-norm、post-RoPE query 与按原生 context RMS 缩放的 basis：

\[
g_h=0.25\tanh\left(f_\theta\left[q_h;
\operatorname{vec}(B_h)/s_h\right]\right),\qquad
s_h=\max(\|c_h\|_2/\sqrt{128},10^{-8}).
\]

先计算 \(u_h=\sum_rg_{hr}B_{hr}\)，再令

\[
\Delta c_h=u_h\min\left(1,
\frac{0.10\|c_h\|_2}{\max(\|u_h\|_2,10^{-8})}\right).
\]

这保证控制器返回的 FP32 修正在每 head 上不超过原生 context 范数的 10%（允许浮点计算误差）；后续 BF16 转换的舍入仍可能略微改变这个比率。零 context 不能生成非零修正。共享增益不按通道独立学习，形成明确的低容量约束。三个 MLP 都有 18,584 个参数，常量谱控制器只有八个增益参数。

`spectral_phaseblind` 只把**条件器**的谱输入换成逐复频点幅值。实际修正仍使用有符号的完整交叉谱 basis，query 与基座也仍可携带顺序信息。该消融只能检验条件器使用相位/符号的额外作用，不能证明全模型相位的必要性。

增益可随输入变化，是引入网络的具体假设。是否必要必须由 `spectral_mlp` 对 `constant_spectral` 的结果判断；频谱结构是否有额外价值还必须比较同参数 `time_mlp`。任何完整可逆 FFT 后的无限制稠密线性映射都可把换基吸收入权重，故不能只凭加入 FFT 宣称新能力。

## 4. FFN 与输出误差为何必须实际测量

忽略浮点舍入时，末层的映射可写为

\[
x'=x+A+W_O\Delta c,\quad
h=x'+\operatorname{FFN}(\operatorname{RMSNorm}_{\rm post}(x')),
\quad z=W_U\operatorname{RMSNorm}_{\rm final}(h).
\]

在原生基线对应位置计算各 Jacobian，小扰动的一阶传播是

\[
\Delta z\approx W_UJ_{\rm final}
\left(I+J_{\rm FFN}J_{\rm post}\right)W_O\Delta c.
\]

因此 FFN 的局部 Jacobian 可能放大、削弱或改变修正方向。context 中的 10% 范数界不直接成为 residual、logit 或正确率的 10% 界；不能用平均频谱能量变化代替任务效应，也不能仅凭本实验归因“FFN 导致遗忘”。需要另一个有控制的层/支路定位实验才能建立这种机制归因。

实际 BF16 实现先计算 `linear(BF16(delta), W_O)`，再与原 attention 输出相加，然后走原残差与 FFN。尾层缓存训练保留这个算序；batch 形状和选取三个 unembedding 行仍可能导致数值差。训练公开全部 batch1/batch8 重放误差，论文准确率只来自启用相同 checkpoint 的完整模型前向。

## 5. 对论文结论的约束

CPU 数学检查验证交叉谱恒等式、循环位移等价、退化反例、限幅与梯度；实际零初始化检查验证包装没有改变基线。它们都不证明记忆问答收益。因果干预改变分数证明此计算支路影响输出；只有独立视频上的完整配对比较才能支持任务增益。

本轮主问题是冻结的三 seed 平均 `spectral_mlp - time_mlp` 视频宏平均正确率差。与 identity、常量谱、phaseblind 的比较及静态保持是补充证据。结果无论正负，都只对应这一模型、位置、采样预算和短视频划分；不能外推到所有外部上下文方法或长期记忆涌现。
