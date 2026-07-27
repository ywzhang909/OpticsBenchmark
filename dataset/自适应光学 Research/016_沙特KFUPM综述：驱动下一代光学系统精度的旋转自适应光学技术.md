---
title: "沙特KFUPM综述：驱动下一代光学系统精度的旋转自适应光学技术"
author: "ao_cas"
date: Sat, 04 Jul 2026 07:50:05 +0800
source: https://mp.weixin.qq.com/s/0PToMi7fbiCvxoH5FTJHww

# 沙特KFUPM综述：驱动下一代光学系统精度的旋转自适应光学技术

> 沙特KFUPM综述：驱动下一代光学系统精度的旋转自适应光学技术开篇导读自适应光学（AO）技术自诞生以来，已成为

# 沙特KFUPM综述：驱动下一代光学系统精度的旋转自适应光学技术

## 开篇导读

自适应光学（AO）技术自诞生以来，已成为修正大气湍流和光学系统像差、提升成像分辨率的关键手段。然而，传统 AO 主要应对标量像差，在面对具有角向或旋转特性的复杂像差时往往力不从心。为此，研究者将旋转调节机制引入自适应光学系统，形成了驱动旋转自适应光学（ARAO）。沙特法赫德国王石油矿业大学（KFUPM）的研究团队近期发表综述，系统梳理了 ARAO 从传统 AO 中演化而来的技术脉络、核心原理及其在天文、医学、国防等领域的前沿应用。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvib7NAZpyPS4yzf62EMRVdekNL0YdSRF6JIyD3ZZKo3QiaBZygf0icKpn44AictShZTlkZOmiaezV8EoOYoOpmNkDbfxXXYLqXicouXM/640?wx_fmt=png&from=appmsg)

图1 传统自适应光学系统中的可变形镜及其执行器结构示意。

## 研究进展综述

ARAO 的发展源于对大气湍流等复杂像差的深入认识。早期自适应光学的概念可追溯到1953年 Horace Babcock 提出的动态修正望远镜像差的设想，但直到1970年代实时波前传感器与可变形镜成熟后，AO 才真正进入实用阶段。随着应用场景的拓展，研究者发现仅对标量像差进行修正不足以应对所有场景，特别是当入射光波前存在角向变化时，需要引入旋转自由度的校正机制。

ARAO 的核心在于旋转校正器（rotational corrector），它能够在动态调整镜面形变的同时，补偿角相关的像差。综述详细讨论了压电材料与微机电系统（MEMS）两类主要驱动机制：压电执行器利用压电效应实现快速、精密的形变控制，无电磁干扰，适合高精度场景；MEMS 执行器则通过微电子与微机械融合，实现了自适应光学系统的微型化与低功耗，适用于便携式和空间受限设备。此外，液晶相位调制和磁驱动等新型执行方式也在探索之中。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pv9JeZ3qbZLHGA2R0fj7YKMwkKsNiadQYian0td7yicsxE6UvPLBpBhHqBOjXpA5dguqUxwK3bh68KJ1ichyllyg20uktXgOHBrkUWo/640?wx_fmt=png&from=appmsg)

图2 压电驱动可变形镜结构示意图。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvibqPzKZsIwy4hxv0iaj9z5zuIE9lFX53FrOn8EYAnAPcoQkgxMC5Zia3TTmlGCeuvGuu5HeL9ia6fT3wRqkYJev1o1ichRL9OvWRno/640?wx_fmt=png&from=appmsg)

图3 MEMS 驱动可变形镜结构示意图。

在控制算法方面，综述介绍了最小均方（LMS）自适应滤波、随机并行梯度下降（SPGD）以及模型预测控制（MPC）等经典算法，并指出深度学习与神经网络正逐步融入波前预测与控制环节，有望降低计算负载并提升系统响应能力。波前传感器方面，Shack-Hartmann 传感器因其鲁棒性和简洁性得到广泛应用，而曲率传感器、金字塔波前传感器等也在特定场景中展现出优势。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pv8mnVNAJyQVUIfBsWcEHBhhkbKpfFY8tf6ibAIvIbtqIsJ3JTrArWCkLfK90lJKqvPk4aia528MVajl0CR1ic6PS2R7nRIjtMUNWM/640?wx_fmt=png&from=appmsg)

图4 Shack-Hartmann 波前传感器工作原理。

在天文观测领域，ARAO 显著提升了地基望远镜的成像能力。综述回顾了 Keck 天文台、欧洲南方天文台甚大望远镜（VLT）、Gemini 天文台和 Palomar 天文台等利用自适应光学取得的里程碑成果，包括直接成像系外行星、解析恒星表面细节以及观测银河系中心恒星运动等。在医学成像中，自适应光学被用于眼底视网膜细胞级分辨成像，以及结合光学相干断层扫描（OCT）诊断黄斑变性、糖尿病视网膜病变等疾病。在国防与安全领域，ARAO 可应用于激光通信、机载光学系统像差修正和先进目标监视系统。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pvicnvRplhcAULjDwKwqvIhhGdYWOUWpLmOicibysAtAWHHiczJjyFkCbRjsaicc8X7gZhH7Ww4zZliayGpldfd1xjWnTxmU0PmiaN2ZicM/640?wx_fmt=png&from=appmsg)

图5 欧洲南方天文台甚大望远镜（VLT）。

## 核心争议与挑战

综述指出，ARAO 系统仍面临若干挑战。首先，旋转像差的精确检测需要高灵敏度的波前传感器，而现有传感器在复杂动态环境下的测量精度仍有提升空间。其次，实时反馈控制对系统响应速度提出了极高要求，如何在快速变化的湍流或振动环境中保持稳定性，是控制算法设计中的难点。此外，不同驱动机制各有优劣：压电执行器可能存在迟滞和非线性，MEMS 系统的制造一致性和长期可靠性仍需验证，而电磁驱动在速度精度上不及压电方案。

## 未来趋势与展望

作者认为，ARAO 的未来发展将与多项新兴技术深度融合。一是机器学习，尤其是深度神经网络在波前预测和实时控制中的集成，可提升系统对非线性像差的适应能力。二是量子光学领域，自适应光学可用于缓解湍流对量子纠缠信号的扰动，保障量子通信与量子成像的可靠性。三是微型化与集成化，MEMS 技术的进步将使 ARAO 适用于手持设备、无人机载荷和空间天文台等体积功耗受限平台。此外，水下成像、虚拟现实/增强现实（VR/AR）中的实时视觉校正，以及自主导航系统的环境扰动补偿，也被视为 ARAO 的潜在拓展方向。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pvicMRJqeL7QtmLJDruUdibGPgKkOzicOibuU47g8Qd9hCNnRQ7F918fgc85guyCmBuKxDjdrsORXBaZhUzUt2DqVUWSibGLoVInKiat0/640?wx_fmt=png&from=appmsg)

图6 用于视网膜成像的眼科自适应光学仪器示意。

## 结语

这篇综述从 ARAO 的历史演进、物理原理、驱动机制、控制算法到多领域应用进行了系统梳理，为研究者理解旋转自适应光学如何突破传统 AO 的标量像差校正局限提供了全景视角。随着材料科学、微纳制造和人工智能的持续推进，ARAO 有望在更多极端和动态环境中实现更高精度的光学校正，为天文观测、医学诊断和量子技术等领域提供关键支撑。

## 参考文献

Alzaydi, A., Sarhan, A. A., & Khalifa, M. (2025). Actuated rotational adaptive optics in shaping the next generation of optical systems: an in-depth analysis. Arabian Journal for Science and Engineering, 50(17), 13469-13501.