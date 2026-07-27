---
title: "【印度TIFR与耶路撒冷希伯来大学】提出对称遗传算法实现纠缠光子快速波前校正"
author: "ao_cas"
date: Sun, 05 Jul 2026 07:38:57 +0800
source: https://mp.weixin.qq.com/s/hLNqhPWMM5OnapRQVGo_hg

# 【印度TIFR与耶路撒冷希伯来大学】提出对称遗传算法实现纠缠光子快速波前校正

> 印度TIFR与耶路撒冷希伯来大学提出对称遗传算法实现纠缠光子快速波前校正开篇空间纠缠是量子通信、量子成像和量子

# 印度TIFR与耶路撒冷希伯来大学提出对称遗传算法实现纠缠光子快速波前校正

## 开篇

空间纠缠是量子通信、量子成像和量子计算中的关键资源，但光子在复杂介质中传播时，其空间关联会受到扰动，导致纠缠退化。为恢复这种关联，通常需要自适应光学进行波前校正。然而量子信号本身非常微弱，常规方法往往需要辅助经典光束，增加了实验复杂度。印度塔塔基础研究所（TIFR）与耶路撒冷希伯来大学的研究团队近期在 APL Photonics 发表研究，提出一种对称遗传算法（sGA），在不依赖辅助经典光束的情况下，实现了对空间纠缠光子的高效波前校正。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv8LIfV8BehwOrGjNfFYFiaHZkKqKBkg3tuYwibibSNVpLqR389PibuMnVSZALJpMjgZz4yuCEQREDFRIYZvxnw7QB2e4wRYfldh1h4/640?wx_fmt=png&from=appmsg)

图1 实验装置示意图：PPKTP 晶体产生纠缠光子对，经散射片扰动后由空间光调制器进行波前校正。

## 技术创新

该研究的核心物理洞察是：对于放置在自发参量下转换（SPDC）晶体远场的散射片，只有其偶宇称（even-parity）分量会对双光子关联产生扰动，而奇宇称分量不影响关联。基于这一特性，研究团队设计了对称遗传算法（sGA），在优化时仅校正波前的偶宇称部分，将优化参数空间减半。

具体实现上，系统使用空间光调制器（SLM）加载相位掩模，上半部分像素独立优化，下半部分通过旋转对称约束自动确定。实验采用高斯泵浦光通过 PPKTP 晶体产生中心波长的纠缠光子对，Schmidt 数约 300。散射片引入波前扰动后，系统通过测量相关点处的符合计数作为反馈信号，驱动 SLM 进行优化。每个超像素由个独立像素组成，相位离散为 16 个等间隔值，种群大小为 15，突变率 0.1，每代保留 3 个精英解。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv9nvSFkLWic3Gqgbamb7ibPPPialxEU62ICmn7axusTIVmTXwagJBc2YSKicGWXDBLPfRsLsSL08dELUgnQbVDZv7YdncziaXCSWtWw/640?wx_fmt=png&from=appmsg)

图2 GA 与 sGA 校正后的 SLM 相位图及对应的二维关联图对比，以及两种算法随代数演化的符合计数增强曲线。

## 实验结果

实验对比了标准遗传算法（GA）和对称遗传算法（sGA）的校正效果。在 100 代优化后，sGA 的关联对比度达到 7.6，高于 GA 的 5.5。sGA 在 100 代末的增强幅度比 GA 高出 38%。更为显著的是，GA 需要 100 代才能达到增强因子 5.9，而 sGA 仅需 25 代即可达到同等增强水平，收敛速度提升约 4 倍。

研究团队还考察了探测器积分时间对校正效果的影响。积分时间分别为 0.2 s、0.5 s 和 1 s 时， longer 积分时间带来更高的信噪比和增强效果，但更新速度更慢。在低光子通量环境中，延长积分时间有助于稳定反馈信号；在高光子通量条件下，缩短积分时间可在不显著牺牲性能的情况下加速收敛。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvicjOTNIV6HBUyxlAC53vQQ9SZlBzSmZxSJl9MQTuI5FkaSOjl4EUMxw4VeUfY3xE0NFEAj8flw3NGWuzgBNoX4CsphiasdoIokQ/640?wx_fmt=png&from=appmsg)

图3 不同积分时间下 sGA 的归一化符合计数演化曲线，以及 100 代后 GA 与 sGA 的增强对比。

此外，当 SLM 优化中心与光束中心存在偏移时，sGA 对中心对准误差表现出一定敏感性，偏移 20 像素时最终增强下降约 40%。而 GA 不受此影响。这说明 sGA 在实际应用中需要精确确定光束中心，研究团队采用奇 Zernike 多项式扫描的方法实现了这一对准。

## 应用价值

论文指出，该方法可应用于量子成像、安全量子通信和量子传感等需要空间纠缠的复杂环境。由于无需辅助经典光束，sGA 降低了实验对准复杂度，并避免了经典光束与量子信号分离困难的问题。更快的收敛速度也意味着该方法有望用于中等动态环境中的实时波前校正。

## 结语

这项工作从双光子关联的宇称对称性出发，通过对称遗传算法将波前校正参数空间减半，在有限代数内实现了更快的收敛和更高的增强效果。研究也揭示了反馈信噪比与探测器积分时间对优化性能的影响规律，为低光条件下的直接量子反馈波前校正提供了实验验证。

## 参考文献

Kiran Bajar, Ronen Shekel, Vikas S. Bhat, Rounak Chatterjee, Yaron Bromberg, and Sushil Mujumdar, "Rapid and efficient wavefront correction for spatially entangled photons using symmetrized optimization," APL Photonics 10, 090802 (2025). https://doi.org/10.1063/5.0276544