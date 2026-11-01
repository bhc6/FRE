# 读出频谱的attention包络混淆：CPU缓存审计V2

日期：2026-11-01。只读旧V1 train cache，不运行GPU、不改变缓存、不读取新确认集。

## 取谱对象

缓存来自Qwen3-VL-2B最后decoder层、最后文本query。每个query head有8个时间组，每组空间token数按视频实际网格核验，最多64。原始time basis是 B[h,t,:]=sum_p a[h,t,p] V[h,t,p,:]，其中a来自全部keys的softmax，未在视觉部分重新归一。它是attention加权context，不是未加权视觉隐状态。

本次由存档的attention_weights与真实visual_positions恢复alpha[h,t]=sum_p a[h,t,p]，并计算mu=B/max(alpha,1e-12)。mu仍是逐时间空间attention条件加权的V，仍含空间权重、语言层上下文与位置效应，绝不是纯未加权V或视觉encoder输出。

对B、mu、alpha分别沿时间轴做orthonormal rFFT；DC和Nyquist权重1、内部频点权重2，不沿channel变换。先在每head汇总channel能量，再对head的非DC占比作均值，得到每源每条件一个统计量；另外保存跨head能量合并后的占比作为敏感性口径。Query heads不是独立样本，也不能当真实KV heads。

## 配对、零质量与能量检查

206个缓存条目的哈希均验证；只选择32个旧训练源按query_id排序的第一题，匹配real/static_first/static_last共96条件，其余真实问题不进入配对统计。静态均无gold，不计算静态准确率。每源内静态先对两个端点平均，然后32源等权；10000次配对source bootstrap，seed 20261101，探索性percentile 95%区间。

alpha最小值为4.5444312681e-08；alpha<=1e-12的head-time共0，排除head共0。策略为mu按max(alpha,epsilon)计算，任何存在低质量时间点的head从该条件全部信号的谱对比中一起排除；若alpha=0，要求B严格为0。

B=alpha×mu相对闭合误差最大5.8e-17；Parseval误差最大6.98e-16；非DC能量与直接时域去均值平方和误差最大4.23e-16。因此总非DC能量没有超出时间方差基线的额外信息。

## 配对结果

| 信号 | real均值 | static均值 | real−static [95% CI] | 正/负源数 |
|---|---:|---:|---|---:|
| weighted_B | 0.531677 | 0.596253 | -0.064576 [-0.073004, -0.055976] | 0/32 |
| conditional_mu | 0.206602 | 0.139994 | +0.066608 [0.056038, 0.077854] | 32/0 |
| mass_alpha | 0.431945 | 0.549962 | -0.118017 [-0.132065, -0.103752] | 0/32 |
| envelope_only_B | 0.431945 | 0.549962 | -0.118017 [-0.132065, -0.103752] | 0/32 |

同源静态输入的B减mu非DC占比平均为+0.456259，CI [0.435655,0.475034]；32/32源为正。此对照直接检验取谱时的时间attention质量包络，但并未将mu内部的空间attention因素排除。

## 包络与内容项的有符号分解

固定每个head的mu时间均值，定义E_t=alpha_t×mean_t(mu)、R_t=B_t−E_t。则AC(B)=AC(E)+AC(R)+2<demean(E),demean(R)>。保存三项与有符号交叉项，不能忽略相互抵消而把两项单独百分比称为因果贡献。

| static端点平均的能量比 | 均值 | 95% source CI |
|---|---:|---|
| envelope_over_B | 0.812427 | [0.773476, 0.851923] |
| remainder_over_B | 0.394212 | [0.379247, 0.409026] |
| signed_interaction_over_B | -0.206640 | [-0.248302, -0.168187] |

上述分解的相对闭合误差最大为4.11e-16。固定内容包络E的每head非DC占比与alpha的非DC占比相同，恒等检查通过。

## 可以与不能得出的结论

这些只读对照说明：必须区分视觉encoder空间谱、原始V时间谱、条件视觉context谱与包含attention质量的加权context谱。对最后一种对象的非DC直接赋予运动或事件含义存在混淆。mu提供一个剥离时间质量包络的诊断控制，但它没有排除空间attention变化、上下文或位置因素。

这里没有训练或部署新控制器，没有测得准确率收益，也未证明该谱混淆导致某道题错误。取法修正与任务修复是不同证据层级；如要开发方法，需与时域方差/去均值、同参数控制以及未见源确认一起检验。

分析脚本SHA256：8e85466b2ab33d89c77219b819375ce07c9ae4219f4feaffd5995ef3c9c869a4。检查数：2832。完整逐head频率能量、源配对与图位于outputs/readout_spectrum_confounds_v2。
