# Frame2Freq张量轴审计：论文、作者代码与本地Qwen3-VL探针

获取与核查日期：2026-11-01。本轮通过官方HTTP读取CVF主会页面、论文、补充材料及作者仓库，另用NumPy做索引布局检查、读取既有Qwen3-VL存盘数组的header与中性输入元数据。没有导入或执行作者模型，没有运行GPU、读取新端点QA结果或访问40个旧确认视频结果。

**结论：用户关于“沿时间轴独立做FFT”的解释是正确的通用模型；不能把示例中的所有数值和STFT参数直接当成Frame2Freq的真实配置。** 作者代码的rFFT确实沿窗口内时间轴，但去掉CLS、采用不同bottleneck宽度，并存在需要核实的复数打包/滤波轴问题。论文还单独做过内部时间谱的类别判别性分析，不能把它只归为adapter。以下核查用于澄清事实，不预先决定本项目必须沿用它的方法。

## 1. 正式会议身份已核实

[CVF正式页面](https://openaccess.thecvf.com/content/CVPR2026/html/Ponbagavathi_Frame2Freq_Spectral_Adapters_for_Fine-Grained_Video_Understanding_CVPR_2026_paper.html)明确列出：

- 标题：**Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding**。
- 作者：Thinesh Thiyakesan Ponbagavathi、Constantin Seibold、Alina Roitberg。
- 论文集：**Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026**，pp.24073–24083。
- [CVPR 2026主会总目录](https://openaccess.thecvf.com/CVPR2026?day=all)含同名论文并指向上述页面；不是仅依赖仓库自称录用，也不是CVPRW条目。

本轮正式页、主会总目录、PDF及作者代码均实际取得HTTP 200并核对内容。结论是CVPR 2026主会论文，不应仅标成预印本或泛称“可能CVPR”。

## 2. 用户维度示例：哪些成立，哪些只是示意

| 说明 | 核查判断 |
| --- | --- |
| `X[B,T,N,D]`，例`[8,16,197,768]` | 合理的CLIP ViT-B/16示意；197含1个CLS和196个patch。作者adapter实际先去掉CLS，故谱支路处理196个patch。batch=8是示例，不是论文唯一batch。 |
| `FC_down`把768降到96 | 作为假设bottleneck完全成立；本轮作者B/16工厂配置是`adapter_width=384`，不是96。MS默认两支路各192通道。 |
| `W_down[D,d]`，写作`W_down X` | 对末维为D的行向量约定，应写`Z=X W_down+b`。PyTorch `Linear`的存储weight形状为`[d,D]`，实际计算`X weight.T+b`。 |
| 沿`dim=1`作时间FFT，各`B×N×d`条序列独立 | 对`[B,T,N,d]`正确。作者MS已重排为`[B,chunks,window,H,W,C_f]`，故用`dim=2`；它仍是时间轴，不是改成了空间轴。 |
| 完整FFT输出T频点，实输入rFFT输出`T//2+1` | 正确；完整FFT含负频率冗余，单边能量求和需正确处理非DC/Nyquist的权重。 |
| T=16，STFT window=8、hop=4、center=True得到5个频点、5个窗口 | 这一通用例子成立；作者代码实际是window=8、hop=2、center=False，也恰好得到5×5。相同shape不表示参数、边界或窗口中心相同。 |
| 固定patch不等于跟踪同一物体 | 正确。目标移位、相机运动和遮挡均可改变固定空间位置的轨迹；谱不能自动分离这些因素。 |

投影只作用最后的特征维度，通常会混合通道。时间FFT本身不混合不同patch或通道；这不等于整个adapter或其上游视觉Transformer都不做空间/通道混合。

## 3. 作者MS实现的真实轴

依据`models_adapter.py`中`TemporalFreqAdapter`及B/16工厂函数。以下将用户示例的B=8、T=16代入作者默认宽度：

    adapter入口       [B*T, 197, 768] = [128, 197, 768]
    去CLS             [128, 196, 768]
    fc1降维           [128, 196, 384]
    还原时间/空间      [8, 16, 14, 14, 384]
    默认两支路拆分      temporal=192，frequency=192

对频率支路，T=16时窗口固定为16、8、4，各窗口不重叠切块：

| 窗长 | rFFT前`[B,chunks,window,H,W,C_f]` | rFFT后 |
| --- | --- | --- |
| 16 | `[8,1,16,14,14,192]` | `[8,1,9,14,14,192]` |
| 8 | `[8,2,8,14,14,192]` | `[8,2,5,14,14,192]` |
| 4 | `[8,4,4,14,14,192]` | `[8,4,3,14,14,192]` |

源码第140–150行明确使用`rfft(..., dim=2)`与`irfft(..., n=window, dim=2)`。滤波后的每个尺度先各自逆变换、拼回长度T，再在时间域平均；随后与时间支路拼接，经`fc2`回到D维并残差加回patch。不能把实现描述成不同长度频谱未经对齐就直接平均。正文式(4)以“平均后iFFT”示意，多种频谱尺寸的对齐未在该式明确；本轮以实际代码区分这两个层次。

默认B/16工厂明确传入`kernel_size=(3,1,1)`。在**轴正确打包**的意图下，时间支路沿T作depthwise卷积，频率支路沿F作depthwise卷积，H/W核大小为1。通道混合来自前后的全连接投影；depthwise组数等于输入通道数，卷积自身不跨组混合。构造函数的一般默认值却是`(3,3,3)`，直接使用它会涉及空间邻域，不能脱离配置概括所有变体。

默认插入点在视觉block的attention残差之后、原MLP之前；作者也提供attention前插入选项。其对象是视觉骨干内部表示，不是Qwen语言decoder的FFN输出或Q/K/V。

## 4. STFT实现与布局核对项

`STFT_dual_adapter`对所有adapter通道处理，不像默认MS那样拆为两个半宽分支。去CLS、降维后，先整理为`[B*H*W*C_a,T]`，沿每行时间做STFT。T=16时：

- `n_fft=min(32,T//2)=8`；`hop_length=max(1,n_fft//4)=2`。
- Hann窗，`center=False`；输出每行`[frequency=5, window_index=5]`。
- 不进行时间补中心的前向窗起点为0、2、4、6、8；用户示例的`center=True,hop=4`是另一套边界与窗口布置。
- 逆变换代码未传`center`。作者依赖版本所对应的PyTorch 2.7官方文档中，`istft`默认`center=True`；因此前/逆变换的中心设置不一致，需在复现前验证。

### 4.1 不能忽视的真实打包问题

本轮读取的代码文件SHA256为`63bb1052f16fee7ffc681b1a39164a19ef8f3888db8f7718878a67297c286f5b`。它是可变`main`分支快照，以下结论限于这一版本。

**MS段**：`view_as_real`后布局为`[B,chunks,F,H,W,C_f,2]`，代码第145行直接`view`为`[B*chunks,2*C_f,F,H,W]`，没有先将通道和实虚轴换到前面。`view`只重新解释线性存储，不等于转置。因此后续卷积中名为F的轴不再保证对应原来的频率，名为空间/通道的轴也不能按注释直接解释。

CPU用坐标标签数组核查：源shape`[1,1,3,2,2,2,2]`，实际打包与正确轴置换的目标shape相同，但48个元素中46个位置不一致。例如此小例中，沿所谓卷积频率轴移动一步可对应原空间位置移动。MS打包后立即原样reshape返回仍可恒等，因此仅测试无卷积的往返无法发现滤波轴问题。

**STFT段**：代码先形成`[BHW,C_a,F,windows,2]`，再直接`view`成`[BHW,2*C_a,F,windows,1]`。实虚轴并未显式移到通道旁边；对shape`[1,3,5,2,2]`的CPU标签数组，实际与预期打包60个元素中56个不同，按当前解包流程进行恒等卷积替身的往返也有56个位置不同。

此外，即使先修正实虚打包，代码名为`conv3d_temporal`的`(3,1,1)`核面对的布局是`[BHW,2*C_a,F,windows,1]`，其第一卷积轴是F，不是时间窗口；`(3,3,1)`分支才覆盖F和窗口两个轴。命名不能替代实际维度核查。

这些是静态源码与NumPy索引验证结果，没有运行或修改作者网络，也不证明论文实验成绩无效。其含义是：可以确认rFFT本身沿时间；不能无保留地声称当前仓库整个滤波段严格实现了独立patch/通道的频带处理。复现需要另外确认所用提交、打包方式及边界设置。

## 5. 论文实际上有三种不同频谱操作

**更正此前概括：把Frame2Freq仅当作前向adapter相关工作是不完整的。** 官方补充材料还明确包含提取骨干内部时间embedding、进行时间FFT并用类别标签计算判别性的诊断；这一部分与当前只读频谱研究更接近，必须纳入查重。

补充材料A.1明确区分：

1. **原始视频的3D谱图**：16张224×224灰度帧构成`V[H,W,T]`，对x/y/t三个轴作FFT，随后shift、对数幅值/功率处理，取中央时间频率切片并作类别均值。这是原始像素的时空谱可视化，不是adapter的时间FFT。若中央切片对应时间DC，它不能单独识别帧顺序；不能以这一图证明方向信息已被保留。
2. **时间表征的类别判别性诊断**：取插入adapter后的骨干时间embedding `X_i[T,D]`，作1D时间FFT，计算`P_i(f)=mean_d |FFT_t(X_i)[f,d]|²`。按动作类别金标分组，用between/within方差比形成`D(f)`，再跨频率归一化。其Algorithm 1是明确的“内部时间谱＋标签判别性”先例。
3. **前向中的谱适配器**：降维、时域/频域变换、逆变换、残差注入，用动作分类监督训练adapter与分类头。这改变前向，与只读诊断不同。

第2项未在本轮核实到精确的空间聚合代码及采集hook位置，不能擅自断言它就是某一层FFN或逐patch轨迹。作者可执行文件中的`return_features`路径返回patch均值特征，但未核实其与补充材料分析脚本的对应关系。功率聚合也不保留完整相位：不能从动作分类判别性曲线推导所有倒序/方向任务可解。

## 6. 与本地Qwen3-VL的精确对应

本轮只读检查`outputs/passive_probe_v4/video_1580/real.npz`的数组header及`records.jsonl`首条中性条件，核实：

| 项目 | Frame2Freq默认B/16示例 | 本地Qwen3-VL-2B真实探针 |
| --- | --- | --- |
| 原始展示图像数 | 16 | 16 |
| 内部时间轴长度 | 16个按帧视觉表示 | 8个相邻两帧temporal patch格 |
| 空间格 | 14×14=196；原序列另含CLS | 前端grid为12×20，经merge=2为6×10=60，无同类CLS入口 |
| 对象/维度 | 视觉骨干D=768；adapter宽384 | decoder分支D=2048；当前只读probe无新增降维 |
| 单视频FFN轨迹 | 不能直接对应当前语言FFN | `layer_13_ffn_delta: [8,60,2048]` |
| Q轨迹 | 不是作者adapter的采集对象 | `[8,60,16,128]` |
| K/V轨迹 | 同左 | `[8,60,8,128]`，不能统称每位置2048维 |
| 全轨迹时间rFFT频点数 | T=16时9；MS另含更短窗 | T=8时5，含DC与Nyquist，仅4个非DC频点 |

对本地`H[T,P,D]=[8,60,2048]`，纯时间分析对应：

    spectrum = np.fft.rfft(H, axis=0, norm="ortho")
    spectrum.shape == (5, 60, 2048)

这相当于对`60×2048=122880`条长度8的标量轨迹独立作变换。若另加batch轴成为`[B,T,P,D]`，时间才是`axis=1`。Q/K/V额外的head维度应保留，仍沿T变换，不对head或通道序号赋予物理时间含义。

旧空间均值`[8,2048]`与完整`[8,60,2048]`不是同一个测量。空间均值与FFT的线性运算可交换，但“先平均再平方幅值”与“先平方幅值再平均”一般不同；局部相反变化可能在前者抵消。

作者当前adapter假设`H=W=sqrt(L-1)`，且先删除CLS。直接传本地60个视觉位置会丢一个真实patch并违反平方格断言；补一个假CLS后60也不是平方数。因此概念上的时间轴对应，不意味着作者代码可直接接入Qwen。

### 6.1 物理频率不可照搬

video_1580中性探针的8个时间中心约为`[1.1397,5.6983,10.2569,14.8156,19.3742,23.9328,28.4915,33.0501]`秒，间隔约4.558632秒。按该展示网格计算rFFT频点为：

    [0, 0.0274205, 0.0548410, 0.0822615, 0.1096820] Hz

论文正文对其特定16帧/30fps设置使用的约1–10Hz“中频”范围不能搬到这里。原视频编码约30fps，不代表本地稀疏采样hidden也有30Hz采样率。当前两个视觉帧组成一个非线性时间patch；该粗网格的谱无法恢复采样前未保留的高速动作，也未提供抗混叠保证。不均匀选帧时还需另行处理时间轴，不能仅对序号FFT后直接报告相同Hz。

## 7. 对独立研究方向的影响

可以采用用户建议中的通用时间轴理解，同时保持研究问题独立。Frame2Freq的监督对象是动作类别；当前关注同一输入下的decoder证据传播与端点/时序读出失败。其类别判别性指标是重要相邻基线，但不自动解释当前FFN的因果作用，也不保证频谱比等容量时域诊断更好。

最需要避免的两种推断是“既然它用时间FFT，我们也应直接插adapter”和“既然已经有它，内部频谱诊断没有可研究的问题”。合理做法是明确对象、时间网格、标签、读出与因果对照，允许结果显示频谱没有额外价值。当前端点可读性检测及后续只读probe按独立协议进行，不因这份参考提前选频带或改写结果。

## 8. 一手来源与获取边界

- [CVF正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Ponbagavathi_Frame2Freq_Spectral_Adapters_for_Fine-Grained_Video_Understanding_CVPR_2026_paper.html)与[主会总目录](https://openaccess.thecvf.com/CVPR2026?day=all)：会场、标题、作者和页码已核。
- [正式PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ponbagavathi_Frame2Freq_Spectral_Adapters_for_Fine-Grained_Video_Understanding_CVPR_2026_paper.pdf)：方法、谱分析和训练范围；SHA256 `ad7299c331e2cca5c80146dce364f50dd91a6d4727c17f988bd141406ce24be2`。
- [正式补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Ponbagavathi_Frame2Freq_Spectral_Adapters_CVPR_2026_supplemental.pdf)：A.1与Algorithm 1；SHA256 `ecb399bb9bb9b6f43bf31d6813dde78b13f0e206dfa07e5c9f650e1bc637f708`。
- [作者代码](https://github.com/th-nesh/Frame2Freq/blob/main/models_adapter.py)及[原始文件](https://raw.githubusercontent.com/th-nesh/Frame2Freq/main/models_adapter.py)：本轮核查第20–333、351–438、582–620行等；代码哈希见第4节。没有下载权重或执行训练。
- [作者README](https://github.com/th-nesh/Frame2Freq/blob/main/README.md)：确认官方实现说明与PyTorch依赖；README不替代正式会议证据。
- PyTorch 2.7官方[stft](https://docs.pytorch.org/docs/2.7/generated/torch.stft.html)和[istft](https://docs.pytorch.org/docs/2.7/generated/torch.istft.html)：核实center默认值。未将stable文档跳转到的其他版本混作作者依赖。

所有网络获取均为2026-11-01；网址可能更新，因此同时记录关键文件SHA256。未确认作者论文运行使用的精确代码提交、全部实验配置及分析脚本，本轮不宣称完整复现。
