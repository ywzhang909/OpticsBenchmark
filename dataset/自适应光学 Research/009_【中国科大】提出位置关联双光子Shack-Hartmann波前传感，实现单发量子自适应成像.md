---
title: "【中国科大】提出位置关联双光子Shack-Hartmann波前传感，实现单发量子自适应成像"
author: "ao_cas"
date: Tue, 14 Jul 2026 07:54:00 +0800
source: https://mp.weixin.qq.com/s/TpCiXd_bXSoF6uRei-OVSA

# 【中国科大】提出位置关联双光子Shack-Hartmann波前传感，实现单发量子自适应成像

> 【中国科大】提出位置关联双光子Shack-Hartmann波前传感，实现单发量子自适应成像开篇量子成像利用空间

# 【中国科大】提出位置关联双光子Shack-Hartmann波前传感，实现单发量子自适应成像

## 开篇

量子成像利用空间纠缠光子对可获得超越经典极限的空间分辨率、抗噪声能力和反直觉成像效应。然而，大气湍流或光学元件缺陷引入的相位像差会同时降低经典和量子成像的性能。自适应光学技术通过实时测量和校正波前像差来应对这一挑战。中国科学技术大学的研究团队提出了一种名为位置关联双光子Shack-Hartmann波前传感（PCB-SHWS）的新方法，可在单发测量中直接重建加载于纠缠光子对上的相位分布，为实现量子自适应光学提供了更直接的途径。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvibtYluU5PfmrFHla9qicaUQtSUPoKibCA3iaq5exoksh0uvbumQJH0WjhJ1osNdDxDrP3lpajObfUF52o9fmC7e50LQN4iaFT2W1nI/640?wx_fmt=png&from=appmsg)

图1 经典Shack-Hartmann波前传感器（a）与PCB-SHWS（b）原理对比。经典SHWS测量单光子的焦点位移；PCB-SHWS测量纠缠光子对的位置质心分布，倾斜相位导致质心位移。

## 技术创新

PCB-SHWS的核心思想源于爱因斯坦-波多尔斯基-罗森（EPR）佯谬中的位置关联概念。对于具有强位置关联的双光子场，若在微透镜阵列前施加相位分布，光子对在微透镜后焦面的联合概率分布将呈现特定的质心分布。通过测量双光子位置质心分布，即可提取每个子孔径内的平均相位梯度，进而利用与经典Shack-Hartmann相同的数据处理方法重建完整相位。

理论分析表明，在理想位置关联近似下，双光子质心分布的峰值位置对应于平均相位梯度，与经典SHWS一致，但峰宽仅为经典结果的一半。这意味着PCB-SHWS在保持相同空间分辨率的同时，具有更高的灵敏度潜力。该方法不需要经典参考光束、不需要偏振纠缠，也无需扫描校正相位，仅需单次测量即可完成相位重建，效率显著优于此前需要逐次扫描Zernike系数的方法。

实验装置采用404 nm激光泵浦BBO晶体产生简并共线I型自发参量下转换（SPDC）光子对，经4f系统和空间光调制器（SLM）后，由微透镜阵列和EMCCD相机进行符合计数测量。通过多帧法估算联合概率分布（JPD），再计算质心分布来提取相位梯度信息。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvibT7WLxCRlD2h2SN5I4uOyic4YficRUHuJiavDuf3O187HOJXeZgt0KPl5vIoUYO4icicJ7xvp7d7tYic2Lng0y9AkpbXMXkSeAhSyEc/640?wx_fmt=png&from=appmsg)

图2 PCB-SHWS实验装置图。BBO晶体产生的下转换光子经透镜、SLM和微透镜阵列后，由EMCCD探测。

## 实验结果

研究团队测量了五种情况下的相位分布：无相位、双曲抛物面（鞍形）相位、多个Legendre模式叠加、塑料薄膜引入的随机像差、以及校正后的薄膜像差。图3展示了计算的Legendre系数和重建的相位分布。在鞍形相位情况下，测得的L₂₀和L₀₂系数分别接近理论值10和-10；在Legendre模式叠加和薄膜校正情况下，相位分布的均方根误差分别为0.0623λ和0.0502λ，与SLM设计值吻合良好。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv8rLxjD0N0hXFwoq7CKRPWYxMOexUAhDmVyZnRxPMdStnHVTQwaAKqeYDHhbEcjeEEqgl3LaT2OckaaazV1Yq0ycN1giaVTSvpo/640?wx_fmt=png&from=appmsg)

图3 双光子相位测量结果。从左至右依次为：无相位、鞍形相位、Legendre模式叠加、塑料薄膜、薄膜校正后的系数和相位分布。

自适应成像实验进一步验证了该方法的有效性。在SLM前放置塑料薄膜引入像差，分别采集无薄膜、有薄膜和校正后三种情况的图像。图4显示，有薄膜时直接图像和反关联对分布中的目标图案完全无法辨认；经PCB-SHWS测量相位并由SLM加载共轭相位校正后，目标图案重新可见，质心分布也恢复为单峰。由于薄膜略有弯曲且不完全处于SLM平面，校正后的图像相比无薄膜情况略模糊，但改善效果显著。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv85ABZnPInrvz6RSMOgrvaVX2qlKGR3gQIRhDaKQiaUJVPzYk594nTLrJhgQeyT3e07Q9RTBm9BZHufVTnb8SqgBkJSWibiaUgpYY/640?wx_fmt=png&from=appmsg)

图4 自适应成像结果。上行：无薄膜情况；中行：有薄膜情况；下行：薄膜校正后的情况。从左至右依次为：直接图像、条件概率分布、质心分布和反关联对联合概率分布。

## 应用价值

论文指出，PCB-SHWS为量子显微镜、量子远程成像和量子通信中的自适应像差校正提供了一种高效工具。与传统的无传感器方法相比，PCB-SHWS仅需一次测量而非多次扫描，大幅缩短了像差校正时间。若结合时间标记相机等更先进的双光子JPD测量技术，还有望实现实时自适应校正，应对时变像差场景。

## 结语

这项工作将经典Shack-Hartmann波前传感原理拓展至量子光学领域，提出了位置关联双光子质心测量的新型波前传感方法，并通过实验验证了双光子相位测量和自适应成像。作者指出，受限于微透镜孔径和相机像素尺寸，该方法的空间分辨率与经典SHFS类似；未来可通过缩小微透镜尺寸或放大光束来进一步提升分辨率。结合量子光学中的高阶关联测量，微透镜阵列在探测多光子光场高阶关联性质方面也具有广阔潜力。

## 参考文献

Zheng, Yi, Zhao-Di Liu, Jian-Shun Tang, Jin-Shi Xu, Chuan-Feng Li, and Guang-Can Guo. "Position-correlated biphoton wavefront sensing for quantum adaptive imaging." Light: Science & Applications 14, no. 1 (2025): 311.