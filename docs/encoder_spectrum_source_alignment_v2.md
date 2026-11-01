# Encoder谱探针与三篇论文的精确对齐

2026-11-01。此文只固定文献定义与测量口径，不读取本地模型分数，不把“此前未观察到规律”解释为“规律不存在”，也不保证换一个层就会得到规律或收益。

## 1. Frame2Freq：时间谱，但取谱阶段须区分

CVPR2026正式正文和补充材料已缓存为 `outputs/literature_internal_mechanism_v2/09_frame2freq{,_supp}.{pdf,txt}`。补充材料来源：`https://openaccess.thecvf.com/content/CVPR2026/supplemental/Ponbagavathi_Frame2Freq_Spectral_Adapters_CVPR_2026_supplemental.pdf`。

**诊断谱。** Supplement A.1 / Algorithm1读取backbone在adapter插入之后的temporal embedding `X_i[T,D]`，沿T做一维FFT，然后对D平均功率：

```text
X_tilde_i[f,d] = FFT_T(X_i)[f,d]
P_i[f] = mean_d |X_tilde_i[f,d]|²
```

再按真实动作类别作between/within ANOVA频段区分性并归一化。这里输入已无空间patch轴，不能声称论文直接对每个原始patch均做相同诊断，也不能把after-adapter样本当作完全未适配的encoder证据。

**adapter算子。** Supplement A.2明确逐空间位置、逐channel沿时间STFT：重排为`[B·H·W·C_a,T]`，Hann窗口，`nfft=min(32,T/2)`、hop=`nfft/4`，保留实/虚部，做卷积再iSTFT。Diving48配置每个ViT block在MHSA之后插adapter。主文多尺度结构及其他数据集插入数量需分别读取；作者代码布局疑点在已有 `docs/frame2freq_axis_audit.md`，本探针不是该adapter的逐行复现。

**原视频谱是第三条路径。** A.1另对16帧灰度224×224原视频做3D FFT_(H,W,T)及切片可视化；不能把该图直接称作hidden-state频谱。

## 2. LFM：视觉内部层的二维空间谱，位于projector之前

CVPR2026正式正文：`https://openaccess.thecvf.com/content/CVPR2026/papers/Ouyang_Language-guided_Frequency_Modulation_for_Large_Vision-Language_Models_CVPR_2026_paper.pdf`。缓存为 `10_lfm.pdf/.txt`。

正文§3.2.2 Eq6–7区分两种vision特征：

```text
Vhat_l = LN(V_(l−1) + MSA(V_(l−1)))
V_l = LN(Vhat_l + FFN(Vhat_l))
```

这是论文记号；不能据此假定所有真实CLIP实现都是同样postnorm结构。Eq8把一组邻层的`Vhat_l[P²,D]`平均；Eq9对应前文二维Fourier定义，沿空间H,W变换。Eq10–11用额外language encoder特征对频域信号调制；Eq12对IFFT回空间的特征与同组post-FFN `V_l`均值相加；Eq13各组沿channel拼接后才进入两层MLP projector。它不是decoder token时间谱，不是沿channel排列做FFT，也不是只采encoder最后一层。

## 3. NerVE：FFN协方差特征谱，完全不同于Fourier频率

ICLR2026正式正文：`https://proceedings.iclr.cc/paper_files/paper/2026/file/2764bbdbd9f36d9f1ec63f4ad22b24ff-Paper-Conference.pdf`。缓存 `12_nerve.pdf/.txt`，§2.1。

非门控FFN取up projection后、非线性前的`W_up x+b`，与非线性后、down projection前的`σ(W_up x+b)`。SwiGLU取`W_gate x`，与`σ(W_gate x)⊙W_up x`。它把`[B,S,D]`展成`[B·S,D]`，有意丢弃序列顺序，再算D×D协方差的eigenvalues和entropy/participation ratio。不能把该eigenspectrum称作时间FFT，也不能将FFN channel索引当作具有邻接物理意义的空间轴。

## 4. 本地encoder谱的可解释口径

Qwen merger前后、Smol vision末/connector后及LLM第0层输入是合理的阶段对照，但必须保存真实T/H/W、patch顺序、特殊token排除、temporal patch size、pixel shuffle、crop/thumbnail映射。对同一张图只应解释空间谱；跨帧表示才有时间轴。拼成序列的patch索引不能整体当时间。

建议并列记录：

```text
fullpatch_temporal_power = mean_(h,w,d) |rFFT_T(X[T,H,W,D])|²
pooled_temporal_power = mean_d |rFFT_T(mean_(h,w) X)|²
spatial_power = mean_(t,d) |FFT_(H,W)(X)|²
```

先平均再求功率与先求功率再平均通常不同；空间相位抵消可使pooled谱漏掉运动。单边rFFT汇总总能量要明确DC/Nyquist及中间频率倍数约定。跨视频若采样间距不同，频率先报告cycles/clip或cycles/sample；只有通过PTS与重采样明确间距后才能报告Hz。去均值、窗口、norm与DC分开记录，不应按结果临时换口径。

对于attention加权的`a_t V_t`，时域乘法对应频域卷积：谱同时携带权重包络和V内容，不能直接等同V的时间信息。因此未加权encoder/V谱和attention-weighted context谱应并列，前者有变化而后者无变化只定位了潜在混合环节，尚未证明native推理损失或记忆瓶颈。

频谱集中、方向可分、跨层衰减都属于表征证据。后续仍需独立冻结样本、可证伪干预和原生答案改善，才能进一步主张模型实际利用该信息。
