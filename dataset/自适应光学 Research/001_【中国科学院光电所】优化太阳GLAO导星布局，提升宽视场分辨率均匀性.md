---
title: "【中国科学院光电所】优化太阳GLAO导星布局，提升宽视场分辨率均匀性"
author: "ao_cas"
date: Fri, 24 Jul 2026 07:53:56 +0800
source: https://mp.weixin.qq.com/s/yODMtgRPzfUK1oq7G8iY3Q

# 【中国科学院光电所】优化太阳GLAO导星布局，提升宽视场分辨率均匀性

> 中国科学院光电所优化太阳GLAO导星布局，提升宽视场分辨率均匀性开篇地面层自适应光学（GLAO）通过多个导星（

# 中国科学院光电所优化太阳GLAO导星布局，提升宽视场分辨率均匀性

## 开篇

地面层自适应光学（GLAO）通过多个导星（GS）探测地面层湍流并对广视场进行校正，是太阳望远镜等宽视场高分辨率观测的关键技术。然而，GLAO不仅需要高空间分辨率，还要求在宽视场内保持均匀的补偿效果，这对太阳磁场拓扑测量和斑点成像等应用至关重要。中科院光电技术研究所与成都工业学院的研究团队提出了宽视场分辨率均匀性评价指标，系统优化了导星布局，并通过室内实验和云南天文台一米新真空太阳望远镜（NVST）实测进行了验证。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvicn42vTIavK91dBxDyJ12DFYrSr1KTlpTkVQuFyUsGUzFuln6YStdbJWVBthWZmkKR4ibSaHSVuN0R0jpAS6HdHJFiaibUADZdT0c/640?wx_fmt=png&from=appmsg)

图1 三种导星布局示意图。（a）9颗导星单环布局；（b）中心+环（1+8）布局；（c）三环布局。

## 技术创新

研究团队提出了综合成像锐度因子E和PSF空间稳定性因子V的宽视场性能评价指标CEF = E + V，其中E反映全视场平均成像质量，V反映PSF在全视场的空间变化程度。基于该指标，研究团队在仿真中对多种导星布局进行了系统评估，包括单环、中心+环、双环和三环等多种对称配置。

仿真基于NVST的GLAO系统参数（望远镜口径0.98 m，视场60″，DM 13×13促动器，9个导星），采用Fuxian湖太阳观测站实测的7层湍流剖面。结果表明，1+8GS中心+环布局可实现更高的平均分辨率，而9GS单环布局则在均匀性方面表现最优。与1+8GS相比，9GS布局在仅牺牲2.2%平均分辨率的情况下，将分辨率均匀性提升了41.4%。其物理原因在于：单环布局从相同视场半径的不同方向探测湍流，对于各向同性大气湍流可获得更均匀的检测；而中心+环或多环布局在波前平均过程中会中和不同半径的信息，反而降低均匀性。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv9xLK2aKMu36qfasL67zUKAr2qPiczNywuPmBtf2xaE7Xvd1IicUYm97871XiaTnebZSbF8BibvresuSHZq6R0zbjialZFHTXs8Iv0M/640?wx_fmt=png&from=appmsg)

图2 1+8GS（左）与9GS（中）布局的FWHM分布及差异（右）。9GS在35″外视场具有更小的FWHM。

## 实验结果

为验证仿真结果，研究团队设计并搭建了室内太阳GLAO平台。平台采用Sugar Cube绿光LED、积分球和太阳米粒组织切片模拟扩展源，在150 m等效高度放置湍流发生器（r₀=6 cm）。平台配置9×8微透镜阵列的多方向Shack-Hartmann波前传感器，帧率高达1800 Hz，科学相机视场80″×60″。

室内实验对比了5颗导星下的1+4GS与5GS布局，以及9颗导星下的1+8GS与9GS布局。结果显示，对于5颗导星，1+4GS的平均r₀为30.0 cm（标准差2.0 cm），5GS为25.3 cm（标准差1.1 cm），即5GS以15.7%的平均分辨率损失换取了45.0%的均匀性提升。对于9颗导星，1+8GS平均r₀为31.9 cm（标准差1.9 cm），9GS为27.8 cm（标准差1.3 cm），均匀性提升31.6%（分辨率损失12.8%）。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvibA251Lg6psj2qe85EwYcfkzj69Ityia6ynjjxZzGgaz8FtFib6yTaItmSm2zcMPlyBxuXIicb0g9Lw4icppJ97hn5QVVY28J6bbG8/640?wx_fmt=png&from=appmsg)

图3 9颗导星GLAO图像对比。（a）1+8GS；（b）9GS；（g-h）广义Fried参数r₀分布。9GS布局均匀性更优。

NVST实测于2023年11月28日开展，对比了1+8GS和9GS两种布局对太阳黑子的GLAO校正。200帧长曝光图像显示，GLAO校正后米粒组织对比度从1.7%提升至2.1%（1+8GS）和2.2%（9GS）。广义Fried参数分布显示，1+8GS平均r₀为16.0 cm（标准差1.6 cm），9GS为13.4 cm（标准差1.3 cm），9GS布局将均匀性提升了18.7%，同时平均分辨率降低16.25%。此外，9GS布局的参考PSF在全视场中心与边缘更接近，有利于简化图像重建过程。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv9JMERgGIc51snGFbgibf5mqjDiahqkeG5ib3dGwK4Hicpw2v9T1XLibXNxO220Mibt8hlsEgwb0fCgN9PEMXud1qyia0asFB2nluQtMQ/640?wx_fmt=png&from=appmsg)

图4 NVST实测广义Fried参数r₀分布。（a）未校正；（b）1+8GS；（c）9GS。

## 应用价值

论文指出，该研究为太阳GLAO系统的导星布局优化提供了理论依据和实验验证。对于需要均匀宽视场成像的应用（如太阳斑点成像重建），单环导星布局是更优选择；而对于追求中心视场极限分辨率的应用，中心+环布局更为合适。该评价指标和方法也可推广至其他宽视场自适应光学系统的设计中。

## 结语

这项工作提出了宽视场分辨率均匀性评价指标CEF，系统比较了不同导星布局对GLAO性能的影响，并通过仿真、室内实验和NVST实测三级验证确认了单环布局在均匀性方面的优势。研究表明，导星布局的选择应根据科学观测需求在分辨率与均匀性之间做出权衡。作者指出，下一步将研究导星区域大小与分辨率一致性之间的关系。

## 参考文献

Yang, Ying, Lanqiang Zhang, Nanfei Yan, Dingkang Tong, Xian Ran, Libo Zhong, and Changhui Rao. "Wide-field resolution uniformity for solar ground-layer adaptive optics." Optics and Lasers in Engineering 193 (2025): 109053.