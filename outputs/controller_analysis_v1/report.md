# 学习式控制器：独立CPU审计与完整结果

全部准确率来自完整普通模型forward；从原始全词表NPZ独立复算。identity只有一个实际基线。
三个训练seed固定为20261101/20261102/20261103；以下准确率及差值单位为百分比/百分点。
本地test来自官方valid中16个新源视频，14.362–34.910秒；不能称官方隐藏测试或长期记忆结论。

## test：全部方法和seed

all: 16源/74题; temporal: 16源/41题; memory: 8源/18题; descriptive_only: 15源/30题。标签子集有重叠。

| 方法 | seed | all micro | all macro | temporal micro | temporal macro | memory micro | memory macro | descriptive micro | descriptive macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | — | 43.24 | 40.52 | 36.59 | 35.94 | 22.22 | 16.67 | 53.33 | 57.78 |
| time_mlp | 20261101 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| time_mlp | 20261102 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| time_mlp | 20261103 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| spectral_mlp | 20261101 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| spectral_mlp | 20261102 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| spectral_mlp | 20261103 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| spectral_phaseblind | 20261101 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| spectral_phaseblind | 20261102 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| spectral_phaseblind | 20261103 | 43.24 | 40.52 | 36.59 | 35.94 | 22.22 | 16.67 | 53.33 | 57.78 |
| constant_spectral | 20261101 | 43.24 | 40.52 | 36.59 | 35.94 | 22.22 | 16.67 | 53.33 | 57.78 |
| constant_spectral | 20261102 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |
| constant_spectral | 20261103 | 41.89 | 39.27 | 34.15 | 33.85 | 16.67 | 12.50 | 53.33 | 57.78 |

### seed均值与样本标准差

| 方法 | all micro | all macro | temporal macro | memory macro | descriptive macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 43.24 (单基线) | 40.52 (单基线) | 35.94 (单基线) | 16.67 (单基线) | 57.78 (单基线) |
| time_mlp | 41.89 ± 0.00 | 39.27 ± 0.00 | 33.85 ± 0.00 | 12.50 ± 0.00 | 57.78 ± 0.00 |
| spectral_mlp | 41.89 ± 0.00 | 39.27 ± 0.00 | 33.85 ± 0.00 | 12.50 ± 0.00 | 57.78 ± 0.00 |
| spectral_phaseblind | 42.34 ± 0.78 | 39.69 ± 0.72 | 34.55 ± 1.20 | 13.89 ± 2.41 | 57.78 ± 0.00 |
| constant_spectral | 42.34 ± 0.78 | 39.69 ± 0.72 | 34.55 ± 1.20 | 13.89 ± 2.41 | 57.78 ± 0.00 |

### test：重复帧无金标保持诊断

| 方法 | seed | 源/条件 | KL均值 | top1改变 | 平均绝对margin变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | — | 16/32 | 0 | 0/32 | 0 |
| time_mlp | 20261101 | 16/32 | 0.000237905 | 0/32 | 0.03125 |
| time_mlp | 20261102 | 16/32 | 0.000202916 | 0/32 | 0.0234375 |
| time_mlp | 20261103 | 16/32 | 0.000130144 | 0/32 | 0.0195312 |
| spectral_mlp | 20261101 | 16/32 | 0.000234176 | 0/32 | 0.0351562 |
| spectral_mlp | 20261102 | 16/32 | 0.000242334 | 0/32 | 0.0195312 |
| spectral_mlp | 20261103 | 16/32 | 0.000151113 | 0/32 | 0.0351562 |
| spectral_phaseblind | 20261101 | 16/32 | 0.000234022 | 0/32 | 0.0273438 |
| spectral_phaseblind | 20261102 | 16/32 | 0.000243107 | 0/32 | 0.0234375 |
| spectral_phaseblind | 20261103 | 16/32 | 0.000151113 | 0/32 | 0.0351562 |
| constant_spectral | 20261101 | 16/32 | 0.000150388 | 0/32 | 0.0078125 |
| constant_spectral | 20261102 | 16/32 | 0.000135887 | 0/32 | 0.0078125 |
| constant_spectral | 20261103 | 16/32 | 0.000150388 | 0/32 | 0.0078125 |

这些重复帧条件不计算正确率，也不替代有金标的seen静态8题门槛。

## validation：全部方法和seed

all: 8源/40题; temporal: 7源/19题; memory: 3源/9题; descriptive_only: 8源/17题。标签子集有重叠。

| 方法 | seed | all micro | all macro | temporal micro | temporal macro | memory micro | memory macro | descriptive micro | descriptive macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | — | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| time_mlp | 20261101 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| time_mlp | 20261102 | 45.00 | 45.27 | 47.37 | 55.95 | 44.44 | 44.44 | 47.06 | 44.79 |
| time_mlp | 20261103 | 45.00 | 45.27 | 47.37 | 55.95 | 44.44 | 44.44 | 47.06 | 44.79 |
| spectral_mlp | 20261101 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| spectral_mlp | 20261102 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| spectral_mlp | 20261103 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| spectral_phaseblind | 20261101 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| spectral_phaseblind | 20261102 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| spectral_phaseblind | 20261103 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| constant_spectral | 20261101 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| constant_spectral | 20261102 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |
| constant_spectral | 20261103 | 42.50 | 43.48 | 42.11 | 51.19 | 44.44 | 44.44 | 47.06 | 44.79 |

### seed均值与样本标准差

| 方法 | all micro | all macro | temporal macro | memory macro | descriptive macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 42.50 (单基线) | 43.48 (单基线) | 51.19 (单基线) | 44.44 (单基线) | 44.79 (单基线) |
| time_mlp | 44.17 ± 1.44 | 44.67 ± 1.03 | 54.37 ± 2.75 | 44.44 ± 0.00 | 44.79 ± 0.00 |
| spectral_mlp | 42.50 ± 0.00 | 43.48 ± 0.00 | 51.19 ± 0.00 | 44.44 ± 0.00 | 44.79 ± 0.00 |
| spectral_phaseblind | 42.50 ± 0.00 | 43.48 ± 0.00 | 51.19 ± 0.00 | 44.44 ± 0.00 | 44.79 ± 0.00 |
| constant_spectral | 42.50 ± 0.00 | 43.48 ± 0.00 | 51.19 ± 0.00 | 44.44 ± 0.00 | 44.79 ± 0.00 |

### validation：重复帧无金标保持诊断

| 方法 | seed | 源/条件 | KL均值 | top1改变 | 平均绝对margin变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | — | 8/16 | 0 | 0/16 | 0 |
| time_mlp | 20261101 | 8/16 | 0 | 0/16 | 0 |
| time_mlp | 20261102 | 8/16 | 0 | 0/16 | 0 |
| time_mlp | 20261103 | 8/16 | 1.7787e-07 | 0/16 | 0 |
| spectral_mlp | 20261101 | 8/16 | 0 | 0/16 | 0 |
| spectral_mlp | 20261102 | 8/16 | 1.7787e-07 | 0/16 | 0 |
| spectral_mlp | 20261103 | 8/16 | 0 | 0/16 | 0 |
| spectral_phaseblind | 20261101 | 8/16 | 0 | 0/16 | 0 |
| spectral_phaseblind | 20261102 | 8/16 | 0 | 0/16 | 0 |
| spectral_phaseblind | 20261103 | 8/16 | 0 | 0/16 | 0 |
| constant_spectral | 20261101 | 8/16 | 0 | 0/16 | 0 |
| constant_spectral | 20261102 | 8/16 | 3.36297e-08 | 0/16 | 0 |
| constant_spectral | 20261103 | 8/16 | 0 | 0/16 | 0 |

这些重复帧条件不计算正确率，也不替代有金标的seen静态8题门槛。

## 预声明主要比较

全部74题video-macro，spectral−time = 0.00百分点；固定seed的视频bootstrap 95%区间 [0.00, 0.00]；video×seed区间 [0.00, 0.00]。

每次按两个选源层各抽8个源，所有方法共用抽样。seed全局抽取；identity不伪造独立seed。区间是小样本描述，不能保证总体覆盖率。

| 子集 | 方法−参考 | 指标 | 差值 | 视频95%区间 | 视频×seed95%区间 | 有效重复 |
| --- | --- | --- | ---: | --- | --- | ---: |
| all | time_mlp - identity | micro_accuracy | -1.35 | [-4.23, 0.00] | [-4.23, 0.00] | 10000 |
| all | time_mlp - identity | video_macro_accuracy | -1.25 | [-3.75, 0.00] | [-3.75, 0.00] | 10000 |
| all | spectral_mlp - identity | micro_accuracy | -1.35 | [-4.23, 0.00] | [-4.23, 0.00] | 10000 |
| all | spectral_mlp - identity | video_macro_accuracy | -1.25 | [-3.75, 0.00] | [-3.75, 0.00] | 10000 |
| all | spectral_phaseblind - identity | micro_accuracy | -0.90 | [-2.82, 0.00] | [-3.42, 0.00] | 10000 |
| all | spectral_phaseblind - identity | video_macro_accuracy | -0.83 | [-2.50, 0.00] | [-3.33, 0.00] | 10000 |
| all | constant_spectral - identity | micro_accuracy | -0.90 | [-2.82, 0.00] | [-3.39, 0.00] | 10000 |
| all | constant_spectral - identity | video_macro_accuracy | -0.83 | [-2.50, 0.00] | [-3.33, 0.00] | 10000 |
| all | spectral_mlp - time_mlp | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| all | spectral_mlp - time_mlp | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| all | spectral_mlp - spectral_phaseblind | micro_accuracy | -0.45 | [-1.41, 0.00] | [-2.47, 0.00] | 10000 |
| all | spectral_mlp - spectral_phaseblind | video_macro_accuracy | -0.42 | [-1.25, 0.00] | [-2.50, 0.00] | 10000 |
| all | spectral_mlp - constant_spectral | micro_accuracy | -0.45 | [-1.41, 0.00] | [-2.35, 0.00] | 10000 |
| all | spectral_mlp - constant_spectral | video_macro_accuracy | -0.42 | [-1.25, 0.00] | [-2.08, 0.00] | 10000 |
| temporal | time_mlp - identity | micro_accuracy | -2.44 | [-7.69, 0.00] | [-7.69, 0.00] | 10000 |
| temporal | time_mlp - identity | video_macro_accuracy | -2.08 | [-6.25, 0.00] | [-6.25, 0.00] | 10000 |
| temporal | spectral_mlp - identity | micro_accuracy | -2.44 | [-7.69, 0.00] | [-7.69, 0.00] | 10000 |
| temporal | spectral_mlp - identity | video_macro_accuracy | -2.08 | [-6.25, 0.00] | [-6.25, 0.00] | 10000 |
| temporal | spectral_phaseblind - identity | micro_accuracy | -1.63 | [-5.13, 0.00] | [-6.12, 0.00] | 10000 |
| temporal | spectral_phaseblind - identity | video_macro_accuracy | -1.39 | [-4.17, 0.00] | [-5.56, 0.00] | 10000 |
| temporal | constant_spectral - identity | micro_accuracy | -1.63 | [-5.13, 0.00] | [-6.20, 0.00] | 10000 |
| temporal | constant_spectral - identity | video_macro_accuracy | -1.39 | [-4.17, 0.00] | [-5.56, 0.00] | 10000 |
| temporal | spectral_mlp - time_mlp | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| temporal | spectral_mlp - time_mlp | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| temporal | spectral_mlp - spectral_phaseblind | micro_accuracy | -0.81 | [-2.56, 0.00] | [-4.44, 0.00] | 10000 |
| temporal | spectral_mlp - spectral_phaseblind | video_macro_accuracy | -0.69 | [-2.08, 0.00] | [-4.17, 0.00] | 10000 |
| temporal | spectral_mlp - constant_spectral | micro_accuracy | -0.81 | [-2.56, 0.00] | [-4.26, 0.00] | 10000 |
| temporal | spectral_mlp - constant_spectral | video_macro_accuracy | -0.69 | [-2.08, 0.00] | [-3.47, 0.00] | 10000 |
| memory | time_mlp - identity | micro_accuracy | -5.56 | [-16.67, 0.00] | [-16.67, 0.00] | 10000 |
| memory | time_mlp - identity | video_macro_accuracy | -4.17 | [-12.50, 0.00] | [-12.50, 0.00] | 10000 |
| memory | spectral_mlp - identity | micro_accuracy | -5.56 | [-16.67, 0.00] | [-16.67, 0.00] | 10000 |
| memory | spectral_mlp - identity | video_macro_accuracy | -4.17 | [-12.50, 0.00] | [-12.50, 0.00] | 10000 |
| memory | spectral_phaseblind - identity | micro_accuracy | -3.70 | [-11.11, 0.00] | [-13.33, 0.00] | 10000 |
| memory | spectral_phaseblind - identity | video_macro_accuracy | -2.78 | [-8.33, 0.00] | [-11.11, 0.00] | 10000 |
| memory | constant_spectral - identity | micro_accuracy | -3.70 | [-11.11, 0.00] | [-13.33, 0.00] | 10000 |
| memory | constant_spectral - identity | video_macro_accuracy | -2.78 | [-8.33, 0.00] | [-11.11, 0.00] | 10000 |
| memory | spectral_mlp - time_mlp | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| memory | spectral_mlp - time_mlp | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| memory | spectral_mlp - spectral_phaseblind | micro_accuracy | -1.85 | [-5.56, 0.00] | [-9.09, 0.00] | 10000 |
| memory | spectral_mlp - spectral_phaseblind | video_macro_accuracy | -1.39 | [-4.17, 0.00] | [-8.33, 0.00] | 10000 |
| memory | spectral_mlp - constant_spectral | micro_accuracy | -1.85 | [-5.56, 0.00] | [-9.09, 0.00] | 10000 |
| memory | spectral_mlp - constant_spectral | video_macro_accuracy | -1.39 | [-4.17, 0.00] | [-6.94, 0.00] | 10000 |
| descriptive_only | time_mlp - identity | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | time_mlp - identity | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_mlp - identity | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_mlp - identity | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_phaseblind - identity | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_phaseblind - identity | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | constant_spectral - identity | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | constant_spectral - identity | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_mlp - time_mlp | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_mlp - time_mlp | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_mlp - spectral_phaseblind | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_mlp - spectral_phaseblind | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_mlp - constant_spectral | micro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |
| descriptive_only | spectral_mlp - constant_spectral | video_macro_accuracy | 0.00 | [0.00, 0.00] | [0.00, 0.00] | 10000 |

## 已看单源endpoint附录

| 方法 | seed | 静态正确 | mixed正确 | 静态8/8门槛 |
| --- | ---: | ---: | ---: | --- |
| identity | — | 8/8 | 12/24 | True |
| time_mlp | 20261101 | 8/8 | 12/24 | True |
| time_mlp | 20261102 | 8/8 | 12/24 | True |
| time_mlp | 20261103 | 8/8 | 12/24 | True |
| spectral_mlp | 20261101 | 8/8 | 12/24 | True |
| spectral_mlp | 20261102 | 8/8 | 12/24 | True |
| spectral_mlp | 20261103 | 8/8 | 12/24 | True |
| spectral_phaseblind | 20261101 | 8/8 | 12/24 | True |
| spectral_phaseblind | 20261102 | 8/8 | 12/24 | True |
| spectral_phaseblind | 20261103 | 8/8 | 12/24 | True |
| constant_spectral | 20261101 | 8/8 | 12/24 | True |
| constant_spectral | 20261102 | 8/8 | 12/24 | True |
| constant_spectral | 20261103 | 8/8 | 12/24 | True |

此处32条件来自一个已看源，不加入新test样本数或bootstrap。


## 资源与复现

| 运行目录 | 状态 | 过程秒数 | CUDA allocated MiB | sampled RSS MiB | sampled GPU °C |
| --- | --- | ---: | ---: | ---: | ---: |
| endpoint | complete | 134.37 | 4187.64 | 3864.48 | 70.00 |
| test00 | complete | 93.45 | 4191.45 | 4470.34 | 68.00 |
| test01 | complete | 128.64 | 4192.04 | 4252.30 | 71.00 |
| test02 | complete | 117.69 | 4203.29 | 4271.38 | 71.00 |
| test03 | complete | 159.45 | 4198.20 | 4252.57 | 69.00 |
| validation00 | complete | 122.09 | 4197.50 | 2963.07 | 70.00 |
| validation01 | complete | 126.74 | 4192.88 | 3860.48 | 71.00 |
| controller_smoke_v1 | complete | 15.69 | — | 4371.77 | 55.00 |
| chunk00 | complete | 36.80 | 4194.77 | 3584.99 | 65.00 |
| chunk01 | complete | 33.56 | 4185.61 | 4407.03 | 66.00 |
| chunk02 | complete | 34.85 | 4190.38 | 3749.37 | 69.00 |
| chunk03 | complete | 35.05 | 4195.36 | 2863.79 | 69.00 |
| seed20261101 | complete | 10.27 | 193.73 | 2195.72 | 54.00 |
| seed20261102 | complete | 9.99 | 193.73 | 2197.53 | 52.00 |
| seed20261103 | complete | 11.68 | 193.73 | 2209.97 | 56.00 |

资源遥测为两秒采样，可能遗漏短峰值；CUDA allocator峰值另列。资源清单仅覆盖传入目录，失败proof及旧实验是否齐全由scope字段明确标识。

审计2522条记录、194个完整13方法/seed条件；原始文件、源码、checkpoint、抽样表与分析器SHA见report.json及audit_runs.json。
