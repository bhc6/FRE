# Qwen3-VL 隐状态频谱的结构数学：机制、可否证推论与实验边界

2026-11-01。本文件补充现有 [记忆方法](spectral_memory_method.md)、[FFN 假说](ffn_hypothesis.md) 和 [开发协议](ffn_development_protocol.md)。以下区分：由源码与数学保证的结构性质、仅在给定条件下成立的局部近似、需实验检验的任务假说。没有加载模型、运行 GPU、重训练或修改冻结方法；CPU 合成检查不构成模型效果证据。

当前研究主线已明确为**模型内部前向传播中的时序证据保持与可学习干预**。第8节及第9节的外部记忆定义仅解释历史实验的数学边界，不是继续扩展外部选帧 harness 的建议；当前最小方案见第11节。

**最先影响研究方向的结构事实：在标准 decoder 输出路径中，只改最后一层的过去视觉 token 的 FFN 输出，不可能改变末端答案位置的 logits。** 最后层视觉轨迹即使有很大的频谱变化，也不能直接当作当前答案的 FFN 瓶颈。中间层、答案位置及外部记忆读出是不同的因果对象。

## 1. 实际实现与符号

核查的是已安装的 transformers 4.57.1 的 modeling_qwen3_vl.py，以及本地 Qwen/Qwen3-VL-2B-Instruct revision 89644892e4d85e24eaac8bacfd4f463576704203 的 config.json。源码 SHA-256 为 dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da；配置 SHA-256 为 bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b。校验脚本记录实际绝对路径和哈希，不把 config 中历史 transformers_version 字段当作当前安装版本。

| 对象 | 本地配置或源码事实 |
| --- | --- |
| 语言 decoder | 28 层，零起始索引 0–27；hidden 2048，FFN intermediate 6144 |
| 注意力 | 16 个 query heads、8 个 KV heads，每头 128；GQA 每两头共享一个 KV head |
| 语言 FFN | 无偏置 gate/up/down 投影；SiLU 门控乘积，即 SwiGLU |
| 归一化 | 两处 pre-RMSNorm、query/key 每头 RMSNorm、终端 RMSNorm；epsilon 为 1e-6 |
| MRoPE | 默认 RoPE，interleaved sections [24,20,20]，rope_theta 5,000,000 |
| 视觉前端 | temporal patch=2、spatial patch=16、merge=2；视觉 MLP 是 GELU 系列，不能混称语言 SwiGLU |
| DeepStack | 视觉编码器零起始层 5/11/17 的 merger 输出，依次在语言 decoder 0/1/2 **返回后**加到视觉位置 |
| 推理模式 | 本项目 eval、SDPA；attention_dropout=0，当前机制探针 use_cache=False |

官方固定版本源码为 [S1]，Qwen3-VL 结构来源为 [S2]。以下公式以实数运算表达；BF16 实现还包含投影、加法与归约舍入。源码中的 RMSNorm 先转 float32 计算均方和归一化，再转回输入 dtype 并乘 learned weight。不能把实数恒等式当作跨运算重排的逐 bit 恒等保证。

记 token 序列位置为 \(i\)，层为 \(\ell\)，隐藏维数为 \(d\)。视觉位置另有映射 \(i(t,p)\)，其中 \(t\) 为相邻两输入帧构成的时间格，\(p\) 为该格的空间位置；它们与完整序列中的时间戳、边界和文字 token 交错。DFT 沿 \(t\) 做，绝不是沿完整 tokenizer 序号或隐藏通道做。

## 2. 精确的残差结构

### 2.1 RMSNorm 与注意力

对列向量 \(x\in\mathbb R^d\)，定义

\[
N_{\gamma,\epsilon}(x)
=\Gamma x/r(x),\qquad
\Gamma=\operatorname{diag}(\gamma),\quad
r(x)=\sqrt{\|x\|_2^2/d+\epsilon}.
\tag{1}
\]

它不减去通道均值，也不减去时间均值。不同位置各自归一化。

令 \(h_{\ell,i}\) 为层输入，\(s_i=N_{\ell,\mathrm{attn}}(h_{\ell,i})\)。对 query head \(a\) 及其共享 KV head \(g(a)\)：

\[
q_i^a=R(p_i)\,N_{\ell,q}(W_{\ell,q}^{a}s_i),\qquad
k_j^{g(a)}=R(p_j)\,N_{\ell,k}(W_{\ell,k}^{g(a)}s_j),
\]
\[
v_j^{g(a)}=W_{\ell,v}^{g(a)}s_j,\qquad
\alpha_{ij}^{a}
=\frac{\exp((q_i^a)^\top k_j^{g(a)}/\sqrt{128}+M_{ij})}
{\sum_r\exp((q_i^a)^\top k_r^{g(a)}/\sqrt{128}+M_{ir})}.
\tag{2}
\]

\(M\) 包含因果及 padding 屏蔽，不可见位置取负无穷；\(R(p_i)\) 表示该配置的 interleaved MRoPE 旋转，不能简写成“只由物理时间决定的正弦”。每层的 query RMSNorm 与 key RMSNorm 各有一组 128 维 learned gamma，各自在相应的所有头之间共享；不是每头独立一组参数。

\[
a_{\ell,i}=W_{\ell,o}\operatorname{Concat}_a
  \left(\sum_j\alpha_{ij}^{a}v_j^{g(a)}\right),\qquad
y_{\ell,i}=h_{\ell,i}+a_{\ell,i}.
\tag{3}
\]

在有缓存的前向中，key/value 在 attention 内从**本层输入经 pre-norm**计算并写入 cache，早于本层 FFN。当前代码无 attention 投影偏置。

### 2.2 SwiGLU、DeepStack 与输出

\[
u_{\ell,i}=N_{\ell,\mathrm{ffn}}(y_{\ell,i}),\qquad
f_\ell(u)=W_{\ell,d}
  \left[\phi(W_{\ell,g}u)\odot(W_{\ell,u}u)\right],
\quad \phi(z)=z\,\operatorname{sigmoid}(z),
\]
\[
m_{\ell,i}=f_\ell(u_{\ell,i}),\qquad
v_{\ell,i}=y_{\ell,i}+m_{\ell,i}.
\tag{4}
\]

FFN 是逐 token 映射；它不会在自己内部再读别的时间位置。其输入已经包含 attention 汇集的上下文，所以“逐 token”不等于“不处理时间相关信息”。给定输入矩阵，FFN 与任意 token 置换可交换；真实视频置换还会改变前面的 attention、位置、时间戳和视觉两帧 patch，不能直接套用这一等变性。

令 \(D_\ell\) 为经相应视觉 merger 投影并散射到完整序列的 DeepStack 残差；非视觉位置为零。精确边界是

\[
h_{\ell+1}=v_\ell+D_\ell,\quad
D_\ell=0\ \text{for }\ell\notin\{0,1,2\},
\]
\[
z_i=N_{\mathrm{final}}(h_{28,i}),\qquad
\operatorname{logits}_i=W_{\mathrm{vocab}}z_i.
\tag{5}
\]

视觉层 5/11/17 不是语言层 5/11/17。当前 block output hook 得到 \(v_\ell\)，下一层 pre-hook 得到 \(v_\ell+D_\ell\)，终端返回值又多了 final RMSNorm。因此相邻 hidden cache 的差不能一般地标成“FFN 增量”。

DeepStack 的 merger 也不是简单池化。对视觉层 \(j\in\{5,11,17\}\) 的四个相邻空间 token 输出 \(\xi^{[j]}_1,\ldots,\xi^{[j]}_4\in\mathbb R^{1024}\)，按实现的空间 merge 顺序拼接 \(c^{[j]}\in\mathbb R^{4096}\)，则相应注入为

\[
D_\ell=\operatorname{Scatter}_{\mathcal V}
\left\{W_{2,\ell}\operatorname{GELU}
\left[W_{1,\ell}\operatorname{LayerNorm}_{4096}(c^{[j]})+b_{1,\ell}\right]
+b_{2,\ell}\right\},\quad (j,\ell)=(5,0),(11,1),(17,2).
\tag{5a}
\]

其中第一线性层 4096→4096、第二层 4096→2048，均含偏置；LayerNorm 使用 epsilon=1e-6，并在拼接后作用。最终视觉主 merger 则先对每个 1024 维 token 做 LayerNorm，再按同一空间规则拼接后过两层线性与 GELU。二者的归一化位置不同，不能用同一个“先池化再投影”公式替代。

当前探针的对应量为 decoder_input=\(h\)、attention_delta=\(a\)、pre_ffn=\(y\)、ffn_input=\(u\)、ffn_delta=\(m\)、post_ffn=\(v\)。现有旧缓存的 layer0 是 decoder 入口；layer28 是终端归一化后，不能拿二者推断某一个 FFN 的独立作用。

## 3. 最后一层视觉 FFN 的无路径命题

设 \(\mathcal V\) 为过去视觉位置集合，答案读出位置为 \(q\notin\mathcal V\)。干预只替换第 27 层的 \(m_{27,i}\)，\(i\in\mathcal V\)，保留本层输入、attention、其他位置的 MLP 输出、所有权重和序列布局。

由式(4)，\(v_{27,q}=y_{27,q}+m_{27,q}\) 不变。由式(5)，最后层没有 DeepStack 或下一层 attention；final RMSNorm 和 vocab projection 都逐位置作用。因此

\[
\boxed{
\frac{\partial\,\operatorname{logits}_q}
{\partial m_{27,i}}=0,\quad i\in\mathcal V,\ i\ne q,
\qquad
\operatorname{logits}^{\mathrm{do}(m_{27,\mathcal V}\leftarrow r)}_q
=\operatorname{logits}_q.
}
\tag{6}
\]

第二式是任意合法有限替换的精确结构结论，不只是“小梯度”近似。zero、打乱、放大这些视觉 MLP 输出都不提供合法的答案因果路径。若此类实现改变答案 logits，应先检查 mask、hook 的共享张量别名、干预位置、非有限数或额外后处理，而不是先宣称发现机制。

**后续 decode 也不例外。** 所有层的已缓存 KV 均在其 attention 中计算；第 27 层视觉 FFN 干预发生在相关 KV 生成之后，且没有下一层需要为这些被改状态生成 KV。基线与干预的缓存、首个输出分布相同；在相同后续 token 或耦合随机种子下，逐步归纳得到下一步输入、各层缓存增量及 logits 相同。独立随机采样可能产生不同 token，那是采样噪声，不是这条干预的因果效应。

命题的边界：

- 若干预答案读出位置本身，式(6)不适用。
- 若改第 27 层输入、attention 的 Q/K/V、共享 MLP 权重或更早层，式(6)不适用。
- 若外部记忆选择器读取这些末层状态，改变选帧后重新输入模型，则增加了“状态→选择→新输入→答案”的外部路径，必须作为不同实验研究。
- 若使用跨 token 的额外输出池化、非标准 cache、训练损失/梯度更新或重排序列，应重新画计算图。权重绑在 embedding 与 lm_head 上，不等于一次激活干预会修改权重。
- 中间层 13 的视觉 FFN 输出可进入层 14 的 K/V 并到达后续答案；“存在路径”只表明效应可能非零，不保证该效应大、任务特异或有益。

原开发探针第 27 层的频谱占比现象属于**没有直接答案读出路径的视觉分支几何**。不能把它据为“末层视觉 FFN 阻碍当前答案”的证据，也不能从该分支零干预效应推断所有 FFN 无关。

## 4. 非线性为什么改变时间频谱

### 4.1 精确的卷积恒等式

暂固定一个空间位置，沿 \(T\) 个时间格取归一化后的 FFN 输入 \(u_t\)。采用单位 DFT：

\[
\widehat x_k=T^{-1/2}\sum_{t=0}^{T-1}x_t e^{-2\pi \mathrm{i}kt/T}.
\]

设 \(g_t=W_gu_t,\ b_t=W_uu_t,\ a_t=\phi(g_t)\)。对每个 intermediate 通道，逐点乘积满足

\[
\widehat{a\odot b}_k
=T^{-1/2}\sum_{r=0}^{T-1}\widehat a_r
 \odot\widehat b_{(k-r)\bmod T},\qquad
\widehat m_k=W_d\,\widehat{a\odot b}_k.
\tag{7}
\]

这是有限序列上的**循环卷积**，与跨时间执行卷积网络是不同概念。通道投影本身与时间 DFT 可交换，门控乘积和 SiLU 的非线性使和频、差频与谐波出现；超出 Nyquist 的成分会按离散索引折叠。输入采样无抗混叠滤波时，不能把这些成分直接解释成真实运动频率。

在零附近，

\[
\phi(g)=\tfrac12g+\tfrac14g^2-\tfrac1{48}g^4+\cdots,
\quad
\phi(g)b=\tfrac12gb+\tfrac14g^2b-\tfrac1{48}g^4b+\cdots.
\tag{8}
\]

二次项已可产生 DC 与二倍频；更高项产生进一步混频。这是局部展开，不能对任意大激活截断后声称精确。无需假设 Taylor 无穷级数全局收敛：下节只使用有限阶 Taylor 和明示的余项界。

一个标量构造 \(u_t=0.2\sin(2\pi3t/64)\)、\(f(u)=\operatorname{SiLU}(u)u\)，在输入没有的 bin 6 产生非零输出。它没有读取额外视频信息；高频出现只是重编码的结果。反之，如果所有 \(u_t\) 完全相同，逐 token FFN 输出必然相同，不会自行凭空制造时间变化。

### 4.2 Jacobian 与局部线性解释

令 \(g=W_gu,\ b=W_uu\)，则

\[
J_f(u)=W_d\left[
 \operatorname{diag}(b\odot\phi'(g))W_g+
 \operatorname{diag}(\phi(g))W_u
\right].
\tag{9}
\]

对方向 \(\delta\)，

\[
D^2f(u)[\delta,\delta]=W_d\left[
 b\odot\phi''(g)\odot(W_g\delta)^2
 +2\phi'(g)\odot(W_g\delta)\odot(W_u\delta)
\right].
\tag{10}
\]

令 \(u_t=\bar u+\delta_t\)。若包含所有线段 \(\bar u+s\delta_t,\ s\in[0,1]\) 的区域上二阶导数的双线性算子范数不超过 \(M\)，则

\[
f(u_t)=f(\bar u)+J_f(\bar u)\delta_t+r_t,\quad
\|r_t\|_2\le \tfrac12M\|\delta_t\|_2^2,
\]
\[
\sum_k\|\widehat r_k\|_2^2
=\sum_t\|r_t\|_2^2
\le\tfrac14M^2\sum_t\|\delta_t\|_2^4.
\tag{11}
\]

对非 DC 频点，若余项足够小，\(\widehat m_k\approx J_f(\bar u)\widehat\delta_k\)。**同一个**线性 Jacobian 只混合通道，不混合时间频率；各频率携带的方向不同，能量仍可被不同程度放大。若分别在每个时间点使用不同 \(J_t\)，则 \(J_t\delta_t\) 的 DFT 又包含矩阵系数与信号的循环卷积，不能称为一个全局固定频率响应。

例如在半径 \(\rho\) 的局部球内，取
\(G=\|W_g\|_2,\ U=\|W_u\|_2,\ D=\|W_d\|_2\)，
\(A=\|W_g\bar u\|_\infty+\rho G\)，
\(B=\|W_u\bar u\|_\infty+\rho U\)。
由 sigmoid 导数的基本界，有一项保守上界

\[
M\le D\left[B(1/2+A/4)G^2+
                   2(1+A/4)GU\right].
\tag{12}
\]

它说明余项可被检查，不保证真实模型中的半径或常数足够小。完整 FFN 残差映射 \(G_\ell(y)=y+f_\ell(N_\ell(y))\) 的 Jacobian 是

\[
J_{G_\ell}(y)=I+J_{f_\ell}(N_\ell(y))J_{N_\ell}(y).
\tag{13}
\]

只测 \(u\to m\) 不等于分析 \(y\to y+m\)；前者没有包含归一化及残差支路。

## 5. RMSNorm：尺度、方向与精度

由式(1)直接求导：

\[
J_N(x)=\Gamma\left[
 r^{-1}I-\frac{xx^\top}{d\,r^3}
\right],\qquad
J_N(x)x=\Gamma x\,\epsilon/r^3.
\tag{14}
\]

当 \(\Gamma=I\) 时，径向与切向导数分别为 \(\epsilon/r^3\) 和 \(1/r\)。learned gamma 非均匀时还会按通道重新缩放，不能继续把上述数值当成完整 Jacobian 的一般特征值。

对正数 \(c\)，

\[
N(cx)=
\frac{c\sqrt{\|x\|^2/d+\epsilon}}
{\sqrt{c^2\|x\|^2/d+\epsilon}}N(x).
\tag{15}
\]

只有 \(\epsilon=0\) 时正尺度不变性严格成立；实际 \(\epsilon>0\) 时通常近似成立。若 \(\Gamma=I\)，每个非零 token 的方向不变；若 \(\Gamma\) 非均匀，方向会按同一个对角映射改变，时间变化还叠加了每个 token 不同的标量分母。因此 RMSNorm 可以改变时间频谱，不可当作跨时间不变的常数归一化。

还需避免另一个过强说法：“RMSNorm 数学上必定删掉全部尺度信息”。若 \(\epsilon>0\)、所有 \(\gamma_j\ne0\)，设 \(z=N(x)\)、\(w=\Gamma^{-1}z\)，则

\[
s=\|w\|^2/d<1,\qquad
x=\sqrt{\epsilon/(1-s)}\,w.
\tag{16}
\]

故在理想实数域内它可逆。问题是径向导数可能极小、逆映射病态，且 BF16/浮点舍入会破坏这种理想可恢复性；\(\epsilon=0\) 或 gamma 有零坐标时则不同。不能把精确可逆性、有限精度可恢复性与固定 reader 易读性混为一谈。

最低成本检查应同时报告 token RMS、单位方向的余弦/距离及原幅度轨迹，明确是在每个 token 归一化后池化，还是池化后归一化；两者通常不交换。终端 RMSNorm 需与末层 FFN 分开，不可把 final normalized states 的变化全部归因于 FFN。

## 6. 残差能量与空间池化

### 6.1 交叉项决定增减

对同一中心化规则和频带投影 \(P_bF\)，由 \(v=y+m\) 得

\[
E_b(v)=E_b(y)+E_b(m)
 +2\operatorname{Re}\langle P_bFy,P_bFm\rangle.
\tag{17}
\]

增量自身能量大不表示输出能量大：反向增量可抵消原信号。若实际保存数组满足 \(v=y+m+e\)，还要加入
\(E_b(e)+2\operatorname{Re}\langle P_bFy,P_bFe\rangle+
2\operatorname{Re}\langle P_bFm,P_bFe\rangle\)。
\(e\) 可同时包含 BF16 残差加法及空间均值归约舍入，不能全部命名为模型机制。

记时间方差能量为 \(V\)、DC 能量为 \(D\)，占比 \(r=V/(V+D)\)，则

\[
\frac{r_{\mathrm{post}}}{r_{\mathrm{pre}}}
=\frac{V_{\mathrm{post}}/V_{\mathrm{pre}}}
{(V_{\mathrm{post}}+D_{\mathrm{post}})/(V_{\mathrm{pre}}+D_{\mathrm{pre}})}.
\tag{18}
\]

合成例 \(y_t=10+\sin(2\pi t/64)\)、\(m_t=-9\)，时间能量完全不变而时间占比增为 67 倍。这是指标反例，未声称该常量增量来自实测 Qwen 权重。DC 只意味着沿时间的均值，不能自动语义化为“背景”“噪声”或“无用信息”。

现有 v1 记忆目标先去时间均值、排除 DC。式(18)揭示的占比分母问题不能直接解释该已去 DC 目标的失败，更不能据此宣称已修复它。

### 6.2 非线性与池化的 Jensen gap

对一个时间格，空间均值 \(\bar u=P^{-1}\sum_pu_p\) 一般满足

\[
\Delta_{\mathrm{pool}}
=P^{-1}\sum_pf(u_p)-f(\bar u)\ne0.
\]

若 \(f\) 在这些点的凸包内二阶导数范数不超过 \(M_f\)，Taylor 的一阶项在求均值时相消，所以

\[
\|\Delta_{\mathrm{pool}}\|_2
\le\frac{M_f}{2P}\sum_p\|u_p-\bar u\|_2^2.
\tag{19}
\]

这是向量范数界；SwiGLU 一般既不凸也不凹，不能给各输出坐标统一的 Jensen 不等式方向。若比较的是未归一化 residual，应把 \(f\) 换成 \(f\circ N\)，并使用该复合函数的曲率界。

平方范数则有严格的二阶矩分解：

\[
P^{-1}\sum_p\|u_p\|^2
=\|\bar u\|^2+P^{-1}\sum_p\|u_p-\bar u\|^2.
\tag{20}
\]

对时间对齐的空间序列，中心化和空间均值都是线性算子，故池化后的时间能量不超过各空间序列时间能量的平均；但当前只保存空间均值和逐时刻二阶矩，不能恢复每个 patch 的时间均值与跨时刻协方差，也就不能精确恢复被池化抹去的时间能量。

反例：两个 patch 始终为 \((+1,-1)\)，与每帧交替 \((+1,-1),(-1,+1)\)，在每个时刻的均值均为 0、二阶矩均为 1；前者每个 patch 的时间方差为 0，后者为 1。现有汇总字段无法区分它们。对池化向量重新执行 MLP 不能“还原真实 FFN 输出”，也不能由均值图证明局部短动作不存在。

## 7. 静态视频也能有非零 hidden 时间频谱

decoder 的输入不只是像素。即使每个视觉时间格的嵌入相同，它们的可见前缀长度、时间戳文本、序列位置、MRoPE 坐标以及前面 token 的上下文都可能不同。

一个不需要位置编码的反例：单头 attention 的所有 query/key 为 0，前缀 token 的 value 为 0，每个重复视觉 token 的 value 为 1，因果 mask 保留所有过去位置。第 \(t\) 个视觉位置的输出为

\[
a_t=\frac{t}{t+1},\qquad t=1,\ldots,T.
\tag{21}
\]

视觉内容完全相同，输出却随可见前缀比例变化，非 DC 能量非零。反之，若所有可见 value 都为同一个向量，任何归一化 attention 权重的加权和都相同；因果结构**允许而不必然造成**静态漂移。

本地 Qwen3-VL 的 get_rope_index 将视频按时间格分开，格内 temporal index 为 0，再加不断推进的序列偏移；物理时间通过类似 <1.2 seconds> 的文本编码。项目模板保留相邻两帧的时间中心，显示一位小数。因此不能把 decoder 的某条正弦形 hidden 轨迹直接称为“纯视觉运动频率”，也不能把 Qwen3 的实现误写成物理秒数直接进入一个连续时间 RoPE。

最低成本的真实模型对照应使用一段既有开发视频：原内容与重复同一帧的静态内容，保持目标时间网格、文字提示、token 数、视觉槽位和位置 ID 一致。静态对照若保留明显 hidden 时间谱，就否定“该频谱全部来自视觉动态”的强解释。若另改时间戳，必须记录实际 token IDs、长度及 position IDs；时间戳文字改变而 token 长度也改变，无法单独识别“钟表时间”作用。更强的固定位置/固定长度时间戳干预是专门的合成机制控制，不应冒充自然视频 QA。

由于模型非线性，\(H_{\mathrm{real}}-H_{\mathrm{static}}\) 是配对差，不是已识别的纯运动因果分量；静态基准选哪一帧也会影响结果。重复首帧与重复末帧可作预声明敏感性检查，不能事后挑最有利的基准。

此外，有限窗、端点不连续、目标网格与源 PTS 的差异、每条视频不同的采样间隔、两帧时间 patch 与无抗混叠重采样都会影响谱。当前 FFN 探针 \(T=8\) 只有 4 个非 DC rFFT bin；其 equal-count 三段分组与 v1 记忆的固定相对频率三段规则不同，不能混用频带名称得出同一目标的结论。

## 8. 频谱重建等价于输入条件化的时域二次型

采用原方法符号：\(X\in\mathbb R^{T\times d}\)，选中集合 \(S\)，固定时间戳下线性插值算子 \(A_S\)，残差 \(R_S=(I-A_S)X\)，去时间均值矩阵 \(C=I-\mathbf1\mathbf1^\top/T\)。\(A_S\mathbf1=\mathbf1\)，所以减去第一行不改变插值残差。

令 \(F\) 为单位完整 DFT，\(P_b\) 为对称正负频率带投影，DC 排除，偶数长度的 Nyquist 只计一次。设 \(E_b=\|P_bFCX\|_F^2,\ E=\sum_bE_b\)。已有目标是

\[
L(S;X)=\frac13\sum_b
\frac{\|P_bFCR_S\|_F^2}{E_b+0.01E}
=\operatorname{tr}(R_S^\top Q_XR_S),
\]
\[
Q_X=CF^*\left[\frac13\sum_b
\frac{P_b}{E_b+0.01E}\right]FC.
\tag{22}
\]

\(Q_X\) 为实对称半正定矩阵，\(Q_X\mathbf1=0\)；常量轨迹 \(E=0\) 时按实现取目标与 \(Q_X\) 为 0。一侧 rFFT 与此等价需要通常正频率乘 2、Nyquist 乘 1，不能把单边幅值平方直接相加。

式(22)在给定原轨迹后对 residual 是二次型；分母依赖 \(X\)，故不是对所有输入固定的全局二次函数。非零整体缩放时，分子和分母同乘平方，目标可以不变；这也说明“整体输入的二次损失”描述不准确。

等权非 DC 目标只等价于 \(\|CR_S\|_F^2\)，完整时域误差还包括

\[
\|R_S\|_F^2=\|CR_S\|_F^2+
T\|\overline R_S\|_2^2.
\tag{23}
\]

完整复数 DFT 是可逆换坐标，不增加信息；丢相位的幅值谱一般不可逆，循环位移等可产生同幅值。频带加权改变的是失真偏好。只要时域实现使用完全相同的 \(Q_X\)、候选集与确定性 tie-break，精确算术下目标和理想选择相同；浮点近并列需另核查。

更关键的是：本项目用 \(A_SX\) 重建观察时的 hidden 作为选帧代理，但 reader 实际重新编码选中的 RGB 和时间戳。一般有

\[
H_{\mathrm{reader}}(\mathrm{frames}_S,\mathrm{query})
\ne A_SH_{\mathrm{observer}}(\mathrm{all\ frames},\mathrm{neutral\ prompt}).
\tag{24}
\]

上下文、位置、时间戳文字、attention 和 query 均可能不同。因此即使频谱重建误差严格降低，也没有自动的 QA 改善保证。若要把 hidden 误差转成答案保证，必须另外建立同一个读出函数的扰动界、margin 条件，以及式(24)两端的联系；当前没有这些条件。

## 9. 信息、任务充分性与可读性

固定模型权重 \(\theta\)，FFN 残差是其输入的确定函数。对目标 \(Y\) 和问题 \(Q\)，数据处理关系给出

\[
I(Y;v_\ell\mid Q,\theta)
\le I(Y;y_\ell\mid Q,\theta).
\tag{25}
\]

它允许重新编码、突出方向或令某种受限 reader 更容易读出，禁止把新谐波解释成输入里原本没有的新信息。跨 DeepStack 比较时须把额外视觉注入 \(D_\ell\) 也纳入输入；不能将单一分支的不等式错误用于有新侧输入的整层。

“表示含有信息”与“给定 reader 能读到”是两回事。构造两个独立均匀比特 \(x_1,x_2\in\{-1,+1\}\)，任务为 \(Y=x_1x_2\)。原表示包含完整答案但不能被一个线性阈值完全分开；可逆重编码 \(z=(x_1x_2,x_2)\) 的逆为 \(x=(z_1z_2,z_2)\)，信息量未变，而答案可由第一坐标直接读出。这说明非线性可能改善**某类读出器的可读性**，没有证明 Qwen 当前 FFN 或任何频带真的如此。

对 query-free 记忆 \(M=g(V)\)，任务充分性是针对特定分布的条件独立：

\[
Y\perp V\mid(M,Q),\quad\text{即}\quad
p(Y\mid V,Q)=p(Y\mid M,Q).
\tag{26}
\]

它不要求逐像素重建原视频；反之，视觉上相近或 hidden 的均方误差小也不保证任务充分。假设未来问题可以随机索引视频中任意一个独立比特事件，固定小预算无法普遍保持全部事件。所谓“长期充分记忆”必须声明任务族、延迟、视频长度和预算，不能从一个能量保留率外推。

从完整输入到确定性记忆，\(I(Y;M\mid Q,\theta)\le I(Y;V\mid Q,\theta)\)；同预算如何保留更相关的部分是统计决策问题，不是 FFT 的信息创造能力。冻结推理也没有把这条新视频写进 FFN 权重。

## 10. 从结构约束到可否证实验

| 结构结论/候选假说 | 最省算力的检查 | 何时应否定当前解释 |
| --- | --- | --- |
| 最后视觉 FFN 无答案路径 | 先用本文 CPU toy；真实模型固定一题作 visual-only zero/sham，比较全部选项或全 logits；扩展 36 个开发 QA 属于实现复核 | 非零差先否定 hook/位置/实现符合命题前提，不把零差包装成普遍 FFN 无效 |
| 中间层视觉 FFN 可能影响答案 | 预固定第 13 层，对同一题 baseline、sham、有限缩放或 zero；记录 logits、margin 与 actual delta 范数 | 无差只能限制这组层/位置/输入；有差却不特异于时间题，不能称时间记忆机制 |
| FFN 变化具有任务方向性 | 与同层 attention、相同位置同范数扰动及静态题比较；先少量预声明合成对 | 任意扰动都同样改变答案，或所有题普遍变差，就缺少特异性 |
| 频谱变化并非纯尺度/DC | 先对既有 NPZ 重算式(17)–(18)，并报告 raw、centered、方向指标 | 只有占比改变、DC 抵消或整体缩放，不能声称动态信息增益 |
| hidden 动态不是位置/前缀伪影 | 一个固定原视频加静态复制对照，保持文本、槽位、grid、position IDs 一致 | 静态仍有类似谱，就否定“纯视觉动态”解释；配对差仍需谨慎解释 |
| 局部空间证据被均值抵消 | CPU 合成反例；如需真实量只在少量层/视频保存 token 级或预声明区域轨迹 | 仅有空间均值和二阶矩，不能给出具体 patch 时间信息损失 |
| 时域/频域诊断的数值口径一致 | CPU 比较同 \(Q_X\) 的两种实现，只核对已有数学定义 | 同 \(Q_X\) 得不同值先查实现；不从换坐标直接推导内部证据增益 |
| 内部干预可以学习到有益作用 | 固定中间层与视觉位置，零初始化轻量残差；先检查答案损失梯度路径，再在训练数据拟合 | 梯度结构为零、只改尺度/静态题也同样受损，或独立样本无增益，就否定当前版本 |

无路径命题意味着应优先把末层视觉 FFN 留作**负对照**，而不是花大量 GPU 在该位置搜索“有益 alpha”。末层表示若被外部选择器读取会产生另一条路径，但这属于历史外部记忆问题，不是当前内部前向研究主线。

中间层干预能改变答案仍不等于形成了更好的内部证据保持机制。最小方法链是：先固定有答案路径的取点，排除位置、范数和池化的替代解释；再构造满足原因果顺序的轻量前向干预，使用训练数据学习；最后在未见源视频上检验时序任务、静态能力、上下文长度与额外计算成本。当前 8 个开发视频及已看过结果的 40 个确认视频不能重新包装为新方法的独立确认集。

频谱变化、因果 logits 改变、受控任务可读性、QA 准确率与随延迟/干扰增加的证据保持分别是不同证据层。不能用其中一项替代整条链；本文没有预设新方法能够成功。

## 11. 一页方案：内部时序证据保持与可学习前向干预

**研究对象。** 对预声明层 \(\ell\)，分别取
\(H_\ell[t,p,d]\)、attention 后的 \(Y_\ell[t,p,d]\)、FFN 输入 \(U_\ell[t,p,d]\)、增量 \(M_\ell[t,p,d]\) 和输出 \(V_\ell[t,p,d]\)。
attention 路径另区分 RoPE 前后的 Q/K 与 V；Q/K 只因位置旋转发生的变化不能被写成新增视觉证据。完整 token 序列中答案位置的表示和 logits 是行为终点。空间 \(p\) 是固定网格索引，不等于被跟踪的同一物体。

对时间 DFT 后的 \(\widehat H_{\ell,k}\in\mathbb C^{P\times d}\)，令 \(u_0=\mathbf1/\sqrt P\)，则逐频率精确分解

\[
E_{\ell,k}^{\mathrm{full}}
=P^{-1}\|\widehat H_{\ell,k}\|_F^2
=P^{-1}\|u_0^\top\widehat H_{\ell,k}\|_2^2
 +P^{-1}\|(I-u_0u_0^\top)\widehat H_{\ell,k}\|_F^2
=E_{\ell,k}^{\mathrm{mean}}+E_{\ell,k}^{\mathrm{spatial}}.
\tag{27}
\]

式(27)隔离全空间均值与空间非均匀分量，避免把池化抵消误认成层间丢失。它仍只描述能量；任务信息可能位于任意频带、相位、跨通道关系或低能量方向，**不预设高频有用，也不预设 DC 无用**。跨层能量下降不直接等于遗忘，升高也不等于保留。

**合法干预。** 首先固定有路径的中间层，例如现有探针的13层；末层27的仅视觉 FFN 是零效应、零答案梯度负对照。使用同一问题的配对视觉顺序/短暂状态控制，人工确认标签；对“恢复原事实”的 activation patching 与“新视频应答”的准确率采用不同、明确的目标，不能把改变后的故事仍按旧标签计为自然 QA。比较 FFN、attention 的同位置同范数干预、sham 与静态问题，不用已见确认答案挑层。

全轨迹 FFT 后重写每个早期视觉 token 会使用未来时间格，通常破坏原 decoder 因果性；它至多是标明非因果的离线诊断。若要学习兼容标准自回归前向的结构，可在 FFN 残差输出加入一个极小的因果低秩时间支路：

\[
\widetilde v_{\ell,t,p}
=y_{\ell,t,p}+m_{\ell,t,p}+\delta_{\ell,t,p},\qquad
\delta_{\ell,t,p}
=A\sum_{j=0}^{R}a_j B\,u_{\ell,t-j,p},
\tag{28}
\]

其中 \(A\in\mathbb R^{d\times r}\)、\(B\in\mathbb R^{r\times d}\)、\(r\ll d\)，只读当前 token 和较早时间格的同网格位置，缺失前缀置零并单列启动边界。初始化 \(A=0\) 可精确恢复基线。冻结原模型，使用训练集答案损失加小扰动正则学习新参数；不在推理中读取答案，不预设系数必须是高通，也不先删除 DC。式(28)是待否证候选，不是已实现或已证明有效的方法。

这个支路的状态是模型前向中的有限历史 buffer，需计入额外内存、延迟和跨视频重置；它不是选择 RGB 帧的外部 harness。相同视频在不同 prefill 分块/逐步推进下应给出一致的合法状态更新，后续纯文本 decode 不得重新消费旧视觉格。有限长度零填充 FIR 也不能用带末端回绕的循环 FFT 直接替代，否则会引入未来泄漏。

**独立证据与停止条件。** 先做 CPU 的因果性和公式检查，再做少量真实 forward 的 baseline/sham/负对照与梯度连通性检查；之后才训练单层小支路。预先固定训练/开发/独立测试源视频，报告时序标签准确率或正确答案 margin、静态题副作用、随证据延迟/干扰增长的曲线、参数与运行成本。对照应包含同位置同参数量的纯通道支路，以及没有额外时序混合的版本。若收益只在静态题、只在重标定幅度或仅在已见视频存在，不能声称改善内部时序记忆；若原模型从采样开始就没看到事件，任何后层支路都不能宣称恢复了未输入的证据。

## 12. CPU 数值检查及复现

独立脚本位于 outputs/structure_math_checks/validate_structure.py，使用 NumPy float64、固定种子 20261101、单进程、线程上限 2；不 import torch、不实例化 transformer、不读取模型权重。只读取模型源码和配置文本做 provenance，全部数值为合成小矩阵。

**本次 35 项检查全部通过。** checks.json 保存每项断言与误差，synthetic_examples.npz 保存小型反例轨迹，均位于 outputs/structure_math_checks/。

| 检查 | 本次结果 |
| --- | --- |
| SwiGLU 精确频域卷积 | 最大绝对误差 6.96e-17 |
| SwiGLU Jacobian 方向有限差分 | 最大绝对误差 7.87e-13 |
| 局部 Taylor 半径依次减半 | 余项缩小 4.0300、4.0151、4.0076 倍；保守界均满足 |
| RMSNorm Jacobian | 最大绝对误差 1.25e-10 |
| epsilon>0 的理想逆映射 | 最大绝对误差 3.00e-15 |
| 相同时间能量、不同占比分母 | 时间能量保持 0.5，占比增 67 倍 |
| 谱损失与同 \(Q_X\) 时域二次型，T=7/8/32 | 最大损失差 1.11e-16 |
| toy 最后层视觉 FFN 置零 | 视觉 hidden 改变范数 1.3625，答案 logits 差严格为 0，所有已计算 KV 相同 |
| 同一 toy 的前一层视觉 FFN 置零 | 答案 logits 最大差 0.0887，展示“有路径可产生效应” |
| 静态视觉、非相同前缀 value | 因果平均产生 0.5 到 0.8889 的时间漂移 |
| 内部干预新增检查 | 逐频率空间能量分解成立；末层视觉缩放的答案有限差分梯度为0；因果 FIR 对未来输入修改保持前缀不变；整段 FFT 滤波可把未来修改传播到前缀 |

toy 只复现所需计算拓扑，不是 Qwen 权重缩小版；其非零差不能外推为真实模型效果。浮点恒等检查是实现核对，理论依据仍是本文推导；有限示例通过不证明全部输入上的数值稳定性。XOR 有限网格搜索只是线性不可分的示例核对，不替代一般超平面的解析论证。

复现命令：

~~~powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:NUMEXPR_NUM_THREADS='2'
.\.venv-cuda\Scripts\python.exe .\outputs\structure_math_checks\validate_structure.py
~~~

## 13. 原始资料

- [S1] Hugging Face Transformers，固定 [v4.57.1 Qwen3-VL modeling source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)。本地核对：RMSNorm 338 行起，attention 385 行起、cache update 435 行起，MLP 460 行起，decoder 476 行起，DeepStack 862 行起，final norm 869 行，位置构建 923 行起，lm_head 应用 1362 行。
- [S2] Qwen Team，[Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)。这里用来核查模型设计身份，实际运行公式与边界仍以固定安装源码和本地 config 为准。
- [S3] Noam Shazeer，[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。SwiGLU 名称与门控结构的原始来源；不作为本文视频假说的实证支持。
- [S4] Biao Zhang, Rico Sennrich，[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。RMSNorm 的原始来源；本文显式保留 epsilon 与有限精度边界。

