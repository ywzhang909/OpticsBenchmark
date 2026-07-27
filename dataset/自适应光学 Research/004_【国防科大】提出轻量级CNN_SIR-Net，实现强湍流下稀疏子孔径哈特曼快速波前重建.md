---
title: "【国防科大】提出轻量级CNN SIR-Net，实现强湍流下稀疏子孔径哈特曼快速波前重建"
author: "ao_cas"
date: Tue, 21 Jul 2026 07:50:47 +0800
source: https://mp.weixin.qq.com/s/1iPbRMoz69QINxMGKwsLpA

# 【国防科大】提出轻量级CNN SIR-Net，实现强湍流下稀疏子孔径哈特曼快速波前重建

> 国防科大提出轻量级CNN SIR-Net，实现强湍流下稀疏子孔径哈特曼快速波前重建开篇自适应光学通过实时探测和

# 国防科大提出轻量级CNN SIR-Net，实现强湍流下稀疏子孔径哈特曼快速波前重建

## 开篇

自适应光学通过实时探测和校正波前畸变来提升光学系统性能，广泛应用于天文观测、激光通信和生物显微成像。Shack-Hartmann波前传感器（SHWFS）因结构紧凑、重建速度快而成为最主流的波前探测手段。然而，强湍流条件下近场强度起伏剧烈，导致子孔径光斑动态缺失；弱信标信号和高背景噪声则限制了子孔径密度，要求在低信噪比和低空间采样率下重建高空间频率像差。国防科技大学的研究团队提出了一种轻量级卷积神经网络SIR-Net，通过子孔径图像并行处理与特征置零压缩，在强湍流（d/r₀≈3）下实现了0.191 ms的高速高精度波前重建。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvib9dHX4r1JpyBplSIG7Ir3lf35eI1Xzjwtq3KP85E6sdiaibtib2aYc5jP6Df3I9E3o3S2yIkjf8fpugzeH9vF8eIozSrSKu4EaYQ/640?wx_fmt=png&from=appmsg)

图1 SIR-Net网络架构。（a）整体结构；（b）SIR-Net1核心模块。

## 技术创新

SIR-Net（Sub-aperture Images Reconstruction Network）采用两阶段级联设计：SIR-Net1基于Inception-ResNet架构进行多尺度并行特征提取，通过1×1、3×3、5×5并行卷积层同步捕捉子孔径光斑的局部微观结构和全局波前相位关联；SIR-Net2则为基于多层感知机（MLP）的回归网络，负责将提取的特征映射到Zernike系数。所有卷积模块遵循"卷积→批归一化→激活函数"的标准序列，确保与TensorRT加速框架兼容。

模型的关键创新在于子孔径参数的物理共享策略：由于同一湍流条件下所有子孔径具有统计均匀性，SIR-Net1的参数在所有子孔径间共享，通过裁剪后的子孔径图像在批次维度上并行处理，大幅减少了参数量。此外，研究团队提出了特征置零法（FZM）用于模型压缩：逐一将SIR-Net1提取的每个特征置零后评估波前重建RMS误差的变化，识别并移除非关键特征。当特征数从12压缩至8时，验证集RMS仅从0.082λ微增至0.084λ，但推理时间从2.4 ms降至2.1 ms；进一步压缩至7则误差跃升至0.176λ，超出容忍阈值。因此，8特征配置成为最优平衡点。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pv9xf0SsgV2icWBdrpJm9icMia7PRSqOwfVs2OhvFKOjrmjj6KtB3lfNlml9g3GRLSlJEmb4FdibIsjCILTNjNmKGK8g8bFccQjs8T8/640?wx_fmt=png&from=appmsg)

图2 SIR-Net仿真流程图。

## 实验结果

仿真基于Kolmogorov湍流模型，采用d/r₀≈4的湍流强度，4×4微透镜阵列配置，子孔径图像分辨率60×60像素。在102,000组样本上训练后，在2,000组独立测试样本上验证。与传统模态法（RMS高达1.4λ）和ResNet-50（RMS约0.049λ）相比，SIR-Net的RMS约为0.080λ，Strehl比稳定在0.89-0.97，接近理想光学系统极限。

在TensorRT部署优化中，研究团队通过NVIDIA Nsight Systems定位到c5+b5模块消耗98.38%推理时间（39.45 ms），将其移除并重构邻层连接后，推理速度提升至0.381 ms，相比基线提升105倍。最终模型参数量486,700，仅为ResNet-50的2.1%；模型大小1.772 MB，压缩率达91.36%。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv81XVya4BaujIoUIBULQ0KRIBKrOJwozTcm98kAia40SLLpev7NAZkdfBv33h97sT62bPf2aWIE44ibfMyg1FaIwia11nZSxF9wlg/640?wx_fmt=png&from=appmsg)

图3 SIR-Net训练曲线。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv8peS5TVsRsWaicp1ibZM92UUQJjEayBGMbBBVDGCmmWOy4nI78pRibbYyTpT1CX3CQibaicRibUJ7CqSG8icLXaTegbgTbksWMJ2ZBKQ/640?wx_fmt=png&from=appmsg)

图4 三种方法在三个随机测试案例上的波前重建结果对比。上排：SHWFS图像、真实湍流相位、重建波前及残差；中排：补偿前后点扩散函数；下排：PSF中心强度剖面对比。

实验系统采用635 nm激光、空间光调制器（SLM）生成湍流相位屏（d/r₀≈2.5-3.3），由6×6稀疏SHWFS记录。72,000组数据用于训练，14,000组用于测试。SIR-Net的实验RMS误差为0.085λ、0.090λ和0.066λ，与ResNet-50（0.060λ、0.065λ、0.053λ）存在约0.02λ的精度差距，但仍远低于自适应光学系统典型容差阈值0.1λ。在FP16模式下，SIR-Net单帧推理时间仅0.191 ms，为构建>5 kHz自适应光学闭环校正系统提供了算法基础。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pv8qZXhmWn0JZ7dgYNDg8cv6vJqKCc7vyTdWh0TgIQibJ2cUr8cNib4ILIjPuzicyCyF80xwPHqPf9vsYF4TRib7KXXfF15xpqDXhS8/640?wx_fmt=png&from=appmsg)

图5 三组随机实验样本的波前恢复结果。（a）样本1；（b）样本2；（c）样本3。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv9hQVpQF34CtOBX9sQFh319AOZLTnKMIicYwTqtS7arRXjCPicRARyHR9FprBUnO7Apib0fHYg5oPthJh8w97zqpCym44WCaEgcBc/640?wx_fmt=png&from=appmsg)

图6 模型加速性能对比。FP16模式下SIR-Net推理速度0.191 ms/帧，达到ResNet-50的3.7倍实时处理能力。

## 应用价值

论文指出，SIR-Net适用于强湍流条件下信标光能量较弱的自适应光学场景，如天文观测中的暗弱导星跟踪、激光通信中的大气信道补偿等。其极小的模型尺寸（1.772 MB）和极快的推理速度（0.191 ms）使其特别适合资源受限的边缘计算部署，为自适应光学系统的小型化和实时化提供了技术支撑。

## 结语

这项工作提出了轻量级卷积神经网络SIR-Net，通过并行多尺度特征提取和FZM特征筛选机制，在强湍流稀疏子孔径条件下实现了高效高精度的波前重建。数值仿真和实验均验证了该方法的可行性，FP16模式下单帧延迟0.191 ms、模型仅1.772 MB的性能指标展示了其在高速闭环自适应光学中的潜力。作者指出，未来将进一步提升模型可解释性，并研究极端噪声条件下的鲁棒性。

## 参考文献

Yang, Shuwei, Yulong He, Yu Ning, Jun Li, Wenjing Zhang, Naiting Gu, Quan Sun, Fengjie Xi, and Xiaojun Xu. "Lightweight convolutional neural network for wavefront reconstruction via a Shack-Hartmann sensor with spatially downsampled microlens." Optics Express 33, no. 19 (2025): 40948-40959.