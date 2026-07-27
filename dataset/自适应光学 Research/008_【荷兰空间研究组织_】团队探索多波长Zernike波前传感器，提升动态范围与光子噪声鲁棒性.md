---
title: "【荷兰空间研究组织 】团队探索多波长Zernike波前传感器，提升动态范围与光子噪声鲁棒性"
author: "ao_cas"
date: Wed, 15 Jul 2026 06:08:00 +0800
source: https://mp.weixin.qq.com/s/ScsaKvbgxrio287lppvzuw

# 【荷兰空间研究组织 】团队探索多波长Zernike波前传感器，提升动态范围与光子噪声鲁棒性

> SRON团队探索多波长Zernike波前传感器，提升动态范围与光子噪声鲁棒性开篇直接成像类地行星需要达到10⁻

# SRON团队探索多波长Zernike波前传感器，提升动态范围与光子噪声鲁棒性

## 开篇

直接成像类地行星需要达到10⁻⁸至10⁻¹⁰的对比度，这对波前传感与控制提出了纳米级甚至皮米级的精度要求。Zernike波前传感器（ZWFS）因其高灵敏度有望触及量子信息极限，成为极端自适应光学（XAO）系统的有力候选。然而，其高度非线性响应和有限的动态范围限制了实际应用。荷兰空间研究组织（SRON）与莱顿大学的研究团队通过数值仿真系统研究了多波长测量策略对ZWFS性能的改善，涵盖标量与矢量两种配置，在动态范围、光子噪声鲁棒性和相位解包裹三方面展示了多波长方法的优势。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pvibkXicfT4bR9n6TOQaPIxzlnWB9eae9Vf9jLnSDtxpJgYqpxKV1Lw7W5iarqQBU1EhpntgoQricuO8tbAhALBWa1rl02prGCAibEhs/640?wx_fmt=png&from=appmsg)

图1 基于加速梯度下降的多波长波前重建流程。输入波前由模态系数描述，通过ZWFS模型传播产生各波长预期输出，与实际测量比较后通过反向传播计算梯度并优化。

## 技术创新

ZWFS本质上是一种自参考干涉仪，其响应呈余弦形式，导致相位反转仅限π区间。标量ZWFS通过物理凹坑引入光程差，矢量ZWFS则利用液晶或超表面在正交偏振态上施加相反相移，天然实现相位分集。研究团队采用了一种加速梯度下降重建算法，将多波长信息纳入统一的优化框架：每个波长有独立的代价函数，通过反向传播计算梯度后加权平均，利用Newton-CG优化器迭代求解最佳模态系数。

在动态范围方面，不同波长对相同光程差（OPD）具有不同相位响应，这种"天然相位分集"可帮助突破单波长的2π限制。在光子噪声方面，多波长测量增加了可用光子总数，且当各波长独立重建时，可避免单色重建器处理宽带信号时引入的色散误差。在相位解包裹方面，双波长测量可产生等效合成波长（拍频波长），用于重建超出2π范围的大幅度不连续像差，如分段望远镜的"花瓣误差"（petal errors）。

仿真设置以600 nm为中心波长，对标量ZWFS采用5π/2相移设计，第二波长1000 nm对应3π/2相移；点直径设为2λ/D。矢量ZWFS则在同一波长上施加±π/2的偏振相关相移。

## 实验结果

动态范围仿真在500个随机波前（RMS 0-250 nm）上测试。结果显示：对于标量ZWFS，单波长600 nm和1000 nm各有有限的重建范围，而多波长联合重建可显著扩展动态范围，成功重建更大RMS的波前。对于矢量ZWFS，单偏振已提供足够的相位分集，多波长并未进一步改善动态范围，冗余信息甚至偶尔导致算法在大幅像差下收敛失败。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvicnsgiaeyicz8FvhBqDAEFjEl67h5TMgcuhEVFY1l2FNWLkjzukicb9oqdhNQWkG3ibqQsDHXm7DRcrODZqQfqCqVXNTvvIhZG4yZ8/640?wx_fmt=png&from=appmsg)

图2 标量（上排）与矢量（下排）ZWFS动态范围对比。（a,b,d,e）单色重建残余RMS；（c,f）多波长重建残余RMS。多波长明显扩展了标量ZWFS的动态范围。

光子噪声仿真以20 nm RMS波前为输入，比较单色与多波长配置。对于标量ZWFS，增加428 nm或1000 nm辅助波长均降低了重建误差，但不同波长组合的提升幅度不同。对于矢量ZWFS，在带宽超过50%后，经典宽带重建的误差因色散效应开始上升，而多波长独立重建的误差则持续下降，因为多波长策略有效利用了更多光子而不受色散误差惩罚。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pv9370ploO11KmeDr3RLHxP2frnYLMhQtiaqm27qye8XqrNiaHwibr5yp84XibSVPKsYwojFDxR5NghwjXX83tttpyGZzAicxQJ8TTa8/640?wx_fmt=png&from=appmsg)

图3 矢量ZWFS在不同带宽下的重建误差。经典方法（vZWFS）存在最优带宽，多波长方法（mw-vZWFS）在大带宽下仍持续改善。

相位解包裹仿真在六足蜘蛛支撑的36段六边形分段瞳孔上测试。对于单花瓣位移±2.5 μm的情况，单色700 nm重建被限制在±350 nm内，而600 nm与700 nm双波长解包裹（等效波长4.2 μm）可重建全范围。对于多个花瓣同时激励（总RMS 1.4 μm）的复杂场景，梯度下降结合双波长解包裹仍能成功重建。然而，解包裹带来的噪声放大效应显著：相同光子数下，解包裹方法的方差明显高于单色重建，因此建议仅在大幅误差下切换使用。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvichEupQyO30KrwSdBRrUWwuicAWrUrEHmrEtLzkcTlqiaqGiahsdn0u06TrsFuXiaic6XIBkNhTENpYmfQykat08mIP6wiaD1KVnpV4I/640?wx_fmt=png&from=appmsg)

图4 多花瓣模式同时激励（总RMS 1.4 μm）的重建结果。（a）输入波前；（b）估计波前；（c）残余误差。

## 应用价值

论文指出，多波长ZWFS方案可借助现有技术（如分色镜、积分场单元）实现，而微波动能电感探测器（MKID）等新兴技术将提供更优的实施方案——MKID可单次测量同时分辨光子能量，实现无额外光机结构的多波长波前传感，且具有零读出噪声和极低暗计数。SRON正在搭建结合可见光多波长、变形镜、ZWFS和MKID阵列的实验平台，以验证上述方法。

## 结语

这项工作通过数值仿真全面评估了多波长测量对ZWFS性能的改善。研究表明，多波长策略可扩展标量ZWFS动态范围、提升光子噪声鲁棒性，并实现大不连续像差的相位解包裹。矢量ZWFS与多波长测量的结合被认为是最具前景的方案，可同时发挥所有优势。作者指出，未来还可进一步优化掩模设计和代价函数，以进一步提升性能。

## 参考文献

Darcis, M., Haffert, S. Y., Chambouleyron, V., Doelman, D. S., de Visser, P. J., & Kenworthy, M. A. (2025). Adding colour to the Zernike wavefront sensor: Advantages of including multi-wavelength measurements for wavefront reconstruction. Astronomy & Astrophysics, 701, A157.