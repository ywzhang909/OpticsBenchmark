---
title: "【西安工业大学】基于TransUNet迁移学习实现横向剪切干涉剪切参数高精度自动计算"
author: "ao_cas"
date: Sun, 19 Jul 2026 11:49:13 +0800
source: https://mp.weixin.qq.com/s/TeqDFgm2mam2Q1MHvAUOoQ

# 【西安工业大学】基于TransUNet迁移学习实现横向剪切干涉剪切参数高精度自动计算

> 西安工业大学基于TransUNet迁移学习实现横向剪切干涉剪切参数高精度自动计算开篇横向剪切干涉仪通过将原始波

# 西安工业大学基于TransUNet迁移学习实现横向剪切干涉剪切参数高精度自动计算

## 开篇

横向剪切干涉仪通过将原始波前与其横向剪切副本叠加产生干涉条纹，具有结构简单、共光路干涉的优点，广泛应用于高精度波前测量。然而，剪切干涉图的离散边缘分布和弱梯度变化往往导致剪切参数计算不准确，进而降低波前重建精度。西安工业大学等单位的研究团队提出了一种基于TransUNet的迁移学习方法，通过先在大规模仿真数据上预训练、再用少量实验数据微调的策略，实现了剪切干涉图的精确四分类分割，并据此完成剪切参数的高精度计算。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv8o7j9hibUcy1gJTH5BzVRQia2Kta17RHQ6uib6ibCxB6M4eA4XHswFYsJDHOReEBGPKt0w5JZP4XmCj2g3L4ribTe23ZMJWvf32ib7w/640?wx_fmt=png&from=appmsg)

图1 基于偏振光栅的同步相移横向剪切干涉系统实验装置。

## 技术创新

该方法的核心是TransUNet网络架构，其结合了CNN的局部特征提取能力和Transformer的全局上下文建模优势。编码器采用混合CNN-Transformer结构，通过自注意力机制捕获干涉条纹的长程依赖关系；解码器通过级联上采样模块和跳层连接，逐步恢复高分辨率分割结果。训练采用复合损失函数（Dice损失+交叉熵损失+边缘正则化项），以mIoU作为分割精度评价指标。

为应对真实实验数据稀缺的问题，研究团队采用两阶段迁移学习策略：第一阶段在大规模仿真剪切干涉图（10,000对）上预训练，使模型学习干涉条纹的通用形成机制；第二阶段仅用130对实验干涉图进行微调，保持网络结构和超参数不变，让模型自适应真实环境的光学噪声和振动干扰。这种策略仅需约1/8的训练数据即可达到优于直接训练的效果。

在分割结果基础上，采用最小二乘圆拟合算法确定波前中心位置，并结合几何关系计算剪切量、剪切角和波前半径。该方法实现了从干涉图到剪切参数再到波前重建的完整自动化流程。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pvicI3E3QB3CyC1RQSC5ia9TSag6YOoUVbQCL8c4mW97merfXpClj4ribFaxKmWswH8jadVk0rJEYibmUZWMnU8DWZllYdibBzyHE4f4/640?wx_fmt=png&from=appmsg)

图2 两阶段迁移学习训练策略流程图。

## 实验结果

预训练阶段，模型在20个周期内迅速收敛，验证集mIoU从0.981稳步提升至0.9996，损失值降至0.01以下。仿真测试表明，剪切量测量误差约为0.0106像素，剪切角误差约为0.0038度，半径误差不超过0.02像素，均达到亚像素级精度。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv84ovbhpnOpsbRlNTWCcG1mrrvxVCcYnUwHRQI8Sj9yF62KTaMBto11bsortbSqTjcwDt425AHjC38V68g6Zd4Yu70Y3icyF1F0/640?wx_fmt=png&from=appmsg)

图3 不同剪切参数仿真干涉图的处理结果。（I）原始图；（II）网络分割掩模；（III）分割结果的最小二乘圆拟合；（IV）真值圆拟合。

在迁移学习阶段，与直接训练（1,040张旋转增强图像，无预训练）的对照实验显示：迁移模型初始训练损失0.18，20个周期内收敛至0.008；验证损失从0.095降至0.002，而直接训练仅从0.596降至0.009。分割精度方面，迁移模型初始mIoU达0.94，10个周期内稳定在0.99以上，充分证明了迁移学习在小样本条件下的优越性。

对四组真实实验干涉图的处理结果表明，剪切量测量误差在0.31-0.72像素之间，剪切角误差在0.03-0.37度之间，半径误差在0.21-0.47像素之间。与对照组直接训练方法相比，所有几何参数的测量误差均显著降低。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pv8icXKdcMO1lib0Zg67Bf0FKomUIKwBo0lDmIw432KQS2Jr5kXGJsxA6tUAmCMByslRoEg2qEeekhUtUWiaiauvr3vUqkasR9OfMrk/640?wx_fmt=png&from=appmsg)

图4 不同剪切参数实验干涉图的处理结果。（I）原始图；（II）网络分割掩模；（III）分割结果的最小二乘圆拟合；（IV）真值圆拟合。

波前重建验证采用球面光学元件的两组横向剪切干涉图。以ZYGO干涉仪测量结果为参考（PV=0.116λ，RMS=0.015λ），所提方法计算的剪切参数用于波前重建后，第一组PV=0.128λ（偏差0.012λ），RMS=0.025λ（偏差0.010λ）；第二组PV=0.138λ（偏差0.022λ），RMS=0.026λ（偏差0.011λ）。两种情况下均优于传统边缘检测法、最小矩形法和Radon变换法，且满足自适应光学系统小于0.1λ的工程阈值。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvibW7W5swuX7bN2pffPp354CFuZb740HZB5icicmqHJWGQyPaHPwY7wAib3BuSXcfMicRRy64IN8hC6pFFQX0IysicxiadmYsV7cCkViaw/640?wx_fmt=png&from=appmsg)

图5 第一组实验剪切干涉图的不同方法波前重建结果对比。（a）边缘检测法；（b）最小矩形法；（c）Radon变换法；（d）本文方法。

## 应用价值

论文指出，该方法为横向剪切干涉仪提供了一种智能化、高精度和高效的参数计算解决方案。在光学元件面形检测、自适应光学波前传感等需要精确剪切参数的应用中，该方法可显著降低人工干预，提升自动化水平。同时，其迁移学习策略有效缓解了真实实验数据获取困难的问题，为深度学习在光学测量领域的实际部署提供了可行路径。

## 结语

这项工作创新性地将TransUNet和迁移学习引入横向剪切干涉参数计算，通过仿真预训练与实验微调的协同策略，实现了亚像素级剪切参数测量，并有效提升了后续波前重建精度。实验验证了该方法在噪声和弱梯度条件下的鲁棒性。作者指出，未来可进一步探索网络架构优化和更丰富的数据增强策略，以提升模型在更广泛场景中的泛化能力。

## 参考文献

Zhou, Bei, Ailing Tian, Liansheng Sui, Bingcai Liu, Hongjun Wang, Siqi Wang, Jiaming Su, and Peifeng Liu. "High-precision shear parameters calculation method in lateral shearing interferometry via transfer learning-enhanced TransUNet." Applied Physics B 131, no. 10 (2025): 193.