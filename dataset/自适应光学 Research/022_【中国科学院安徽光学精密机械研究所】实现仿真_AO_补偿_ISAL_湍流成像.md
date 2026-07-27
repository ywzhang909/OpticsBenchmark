---
title: "【中国科学院安徽光学精密机械研究所】实现仿真 AO 补偿 ISAL 湍流成像"
author: "ao_cas"
date: Sun, 28 Jun 2026 08:58:16 +0800
source: https://mp.weixin.qq.com/s/KS8PBaGBjNRb9Yj_wO0Ikw

# 【中国科学院安徽光学精密机械研究所】实现仿真 AO 补偿 ISAL 湍流成像

> 中国科学院安徽光学精密机械研究所实现仿真 AO 补偿 ISAL 湍流成像Inverse synthetic a

# 中国科学院安徽光学精密机械研究所实现仿真 AO 补偿 ISAL 湍流成像

Inverse synthetic aperture ladar（ISAL）依赖目标回波的相干相位历史生成高分辨图像。大气湍流会降低 heterodyne efficiency 和 spatial mode matching，并引入随时间变化的相位不稳定，使成像质量下降。Azezigul Abdukerim 等研究 real-time adaptive optics compensation 对 ISAL 成像的影响，并把 AO 与 phase gradient autofocus（PGA）和 range-Doppler（R-D）成像算法结合，评估强湍流下的补偿边界。

## 用动态相位屏描述湍流回波

论文使用 infinitely long phase screen（ILPS）模拟动态大气湍流，而不是只使用静态 random phase screen。回波相位效应用投影到`LG00`模式上的 projection phase shift`φ00`表示，这个量直接影响相干接收和后续相位历史稳定性。AO 部分采用 Shack-Hartmann 风格波前传感、phase unwrapping 和 Zernike reconstruction； realistic AO 中 correction order 设置为 30th order，也展示了 15th-order AO correction 对`φ00`的影响。

系统假设 beacon wavelength 比 signal 短 100 nm，并与信号同轴传播。ISAL 参数包括 laser wavelength`1024 nm`、PRF`17987 Hz`、range sampling number 64、pulse number 512、target distance 1000 m、range resolution 0.29、cross-range resolution 0.05 和 pulse duration`5.12 ms`。目标采用 MiG-25 scattering point model。

ISAL 成像对相位历史连续性很敏感。目标散射点在脉冲序列中的相位变化包含横向运动信息，一旦湍流引入额外相位抖动，R-D 成像所需的相干积累就会受损。论文用`φ00`描述回波投影到基模后的相位变化，是为了把湍流造成的光场畸变和相干接收效率联系起来。AO 校正的直接目标不是生成图像，而是先让接收光场更稳定、更接近可相干处理的状态。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvicZvnMnIUdKPYhxicU0YBDjDH3wRPufnWwI9A7XhnvqWjAqM1aGsMfh6w3RpECWF1JLuTO5OuibTIT8ARTya4MXuKBFfrSBibawWY/640?wx_fmt=png&from=appmsg)

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv9GNk7IycbRzd70LAFLRibTcnoUWC42fibzuM6Osnu608sD3K8LsJeDUdvib1ttDgQdWKw6AhvocHqALOWqrTjKMPrluueHeszyxs/640?wx_fmt=png&from=appmsg)

## AO 稳定 φ00，PGA 改善慢时间相位

仿真显示，在不同`r0`和 wind speeds 下，AO 可以减小 echo signal 的 phase fluctuations，使`φ00`更稳定。没有 AO 时，强湍流会使 ISAL 成像失败或目标结构难以辨认；加入 realistic AO 后，即使在强湍流和不同风速条件下，也能生成可识别的 ISAL 图像。随后 PGA 进一步校正 slow-time domain 中的 piston phase error，配合 R-D 算法改善图像分辨率。

这里的流程可以理解为两级校正：AO 先在光学接收端提高相干接收和模式匹配稳定性，PGA 再在信号处理端补偿残余相位误差。论文强调二者不是互相替代，而是共同作用。AO 减弱湍流造成的快速波前扰动，PGA 则处理成像数据中的慢时间相位偏差。

这一点也解释了为什么仅靠后端成像算法并不充分。PGA 可以估计并校正慢时间 piston phase error，但如果湍流已经显著降低 heterodyne signal 或让相位变化过于剧烈，进入算法的数据本身就会缺乏稳定相干信息。AO 在光学前端提高回波质量后，PGA 才更容易把剩余相位误差整理成清晰图像。论文的成像对比显示，强湍流下加入 realistic AO 后，目标散射点结构才重新变得可辨。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvicw3b0bK3icm1rz4rVYUq5wa1Iib7FZzib3mpU8abE1XAicEFWTcH7t7AAPkIia9TPlCN42DZkrr12lEvlESe2LLnOq64l5LDm7CpjM/640?wx_fmt=png&from=appmsg)

## AO 带宽决定强湍流下的成像边界

论文重点分析了 AO bandwidth。对于`r0=0.03 m`、`ν=5 m/s`的强湍流条件，Greenwood frequency 为`fG=71.67 Hz`。作者测试`fAO=200 Hz`、`100 Hz`、`60 Hz`和`35 Hz`。结果显示，`200 Hz`、`100 Hz`、`60 Hz`下图像仍较清晰；当`35 Hz`或更低时，强湍流下成像失败。风速增加到`15 m/s`时，AO bandwidth 仍需至少约`60 Hz`才能保持有效，`35 Hz`在`r0=0.03m`、`ν=15m/s`下失败。

图像质量还通过 PSLR 评估。论文指出 AO correction 后 PSLR 降低，较低 PSLR 对应更好的图像质量；随着 AO bandwidth 在强湍流下降低，PSLR 增大，图像退化。这些结果把“是否使用 AO”进一步细化为“AO 的实时响应是否足够快”。在强湍流下，带宽不足的 AO 无法及时跟随相位变化，补偿效果会明显下降。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pv8CKWxb9k6aicy9bevSmc218B8zrjc4dPGwmewVpe1BvyNfcj5LEXc2boiaZDicZhS1QDL4ypiasQFoZmdlAMZSAaGMJa4wxgMrG1c/640?wx_fmt=png&from=appmsg)

## 研究意义与边界

这项研究把 AO 引入 ISAL 大气湍流补偿问题，并用动态湍流、相干接收相位和成像算法建立了一条仿真链路。它说明，在强湍流条件下，realistic AO 能提高 ISAL 成像成功率，但其效果受到 AO bandwidth、WFS、重建算法和 actuator 响应限制。论文结论中指出，在该仿真设置下，强湍流中 AO bandwidth 需要超过`60 Hz`。

需要注意的是，本文主要是仿真研究，未来仍需硬件 AO correction 和实际链路实验验证。它的贡献在于给出了 ISAL 成像与 AO 带宽之间的定量关系，为后续设计实时补偿系统提供依据。

带宽结论也应放在论文给定参数下理解。`60 Hz`以上有效，是在目标距离、`1024 nm`波长、MiG-25 scattering point model、给定`r0`和风速组合下得到的仿真结果。实际系统还会受到 WFS 信噪比、计算延迟、变形镜行程、闭环稳定性和 beacon 与 signal 非完全同路等因素影响。尽管如此，论文清楚展示了一个趋势：强湍流下 AO 不是只要存在就足够，实时响应必须接近或超过湍流变化速度。

这对 ISAL 系统设计有直接启发。若只根据静态波前误差选择校正阶数，可能会忽略风速和 Greenwood frequency 对闭环速度的要求；若只提高后端成像算法，也无法弥补前端相干接收已经损失的信息。论文把`fAO=200 Hz`、`100 Hz`、`60 Hz`和`35 Hz`放在同一组强湍流条件下比较，使带宽阈值变成可观察的成像退化过程，而不是抽象控制指标。

同时，PSLR 的变化提供了比视觉图像更量化的判断。AO correction 后 PSLR 降低，说明目标主瓣相对旁瓣更突出；当带宽下降时，PSLR 上升，图像中的散射点结构变得不稳定。把`φ00`时间序列、ISAL 图像和 PSLR 三者对应起来，是本文较完整的地方：它把湍流相位、光学补偿和最终成像质量连成了一条可分析链路。

## 参考文献

Azezigul Abdukerim et al., “Mitigation of atmospheric turbulence effects on ISAL imaging by real-time adaptive optics compensation,” 2025.Vol. 33, No. 17 / 25 Aug 2025 / Optics Express 37130