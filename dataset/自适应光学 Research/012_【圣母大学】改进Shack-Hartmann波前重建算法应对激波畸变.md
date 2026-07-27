---
title: "【圣母大学】改进Shack-Hartmann波前重建算法应对激波畸变"
author: "ao_cas"
date: Wed, 08 Jul 2026 06:35:43 +0800
source: https://mp.weixin.qq.com/s/LIhPlkyrmLA0h53a10M_nw

# 【圣母大学】改进Shack-Hartmann波前重建算法应对激波畸变

> 圣母大学改进Shack-Hartmann波前重建算法应对激波畸变开篇Shack-Hartmann 波前传感器（

# 圣母大学改进Shack-Hartmann波前重建算法应对激波畸变

## 开篇

Shack-Hartmann 波前传感器（SHWFS）是自适应光学、激光系统与非接触流场测量中广泛使用的波前诊断工具。然而，当光束穿过强密度梯度区域（如超音速激波）时，透镜阵列后的衍射光斑会出现畸变或分裂，导致传统的质心算法给出错误的局部倾斜估计，进而严重劣化重建波前的精度。圣母大学（University of Notre Dame）的研究团队近期通过实验对比 SHWFS 与数字全息波前传感器（DHWFS），提出了一种结合高阶统计量与样条插值的改进重建算法，使 SHWFS 在激波环境下的测量误差平均降低约 30%。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pv8x8mGPtv92bibpYUSLloKt1JDMIus5ZhkDww3mDolNJTP00l32czyEvJH0DH5qFb63YclCdIfJiaQtdowpxTlpA7VdNiaelIrI2o/640?wx_fmt=png&from=appmsg)

图1 跨音速流场中局部激波的 CFD 仿真（左）与纹影图像（右）。

## 技术创新

研究的核心思路是：在 SHWFS 图像中识别并剔除被激波畸变的光斑，然后利用剩余的正常光斑重建波前，最后通过插值填补被剔除区域。论文评估了三种识别畸变光斑的指标：标准差（standard deviation）、峰度（kurtosis）和斜率差异（slope discrepancy）。

标准差和峰度从光斑强度分布的二阶和四阶统计矩出发，衡量光斑的弥散程度和尾部特征。激波导致的非衍射极限光斑通常具有更大的弥散和更显著的异常尾部，因此这两项指标能够有效区分正常光斑与畸变光斑。斜率差异则通过比较 SHWFS 测得的局部斜率与重建波前再投影后的斜率之间的偏差，来定位异常测量点。由于最小二乘重建的全局平滑特性，单个激波区域的错误斜率会向外扩散，导致斜率差异在激波附近形成较大区域。

剔除畸变光斑后，论文采用 Southwell 重建算法对剩余数据点进行波前重建，再利用样条插值对被剔除区域进行填补，最终得到完整的连续波前。这种"识别-剔除-插值"策略避免了错误斜率对全局波前的污染。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/7rsSvDn8pvibonH3d4brUPJbSP6kIdNCFdWvPx0kia0kzqYrwVWAvS8icJWtgdvhO1XlAvDia0RDmZfBBocmCAVwyPFlWPiaMibjyYic5tbc778OibI/640?wx_fmt=png&from=appmsg)

图2 SHWFS 与 DHWFS 对比流程：将 DHWFS 波前降采样并配准后，与 SHWFS 重建波前逐像素比较。

## 实验结果

实验在圣母大学 Hessert 航空航天实验室的吸入式风洞中进行。测试段为，局部圆柱模型使气流在其顶部加速至超音速并形成局部激波。自由流马赫数在 0.43–0.48 之间。激光束沿激波方向传播，同时使用 SHWFS 和 DHWFS 进行同步测量，以干涉测量为基础的 DHWFS 被作为"地面真值"。

直接对比显示，SHWFS 显著低估了激波引起的相位跳变。DHWFS 测得波前的峰谷值平均约为，而 SHWFS 仅为。当将波前差传播到远场时，未修正的 SHWFS 波前在衍射极限桶内的归一化功率随激波强度增加而非线性下降，最强激波情况下仅约 30%。

在三种识别指标中，标准差和峰度对激波光斑的识别较为一致，而斜率差异由于全局平滑效应会选中激波周围更大的区域。实验表明，剔除约 2.4% 的数据点时，三种指标均能获得较好的误差改善。进一步经过样条插值后，斜率差异方法的均方根误差（）略低于统计量方法，但三种方法的最终误差均显著高于无激波时的亚音速基准（约）。

![](https://mmbiz.qpic.cn/mmbiz_png/7rsSvDn8pvibSRKGics2qrpRNBeqZVFSVBzjwuH09pVExAA21T8nTHFb8ZibVP72M2F65nGxfKQ2iaR5uI3UgQM01zzltgo4kibUNNMEqeva60Qc/640?wx_fmt=png&from=appmsg)

图3 标准差（a）、峰度（b）和斜率差异（c）的空间分布图，数值越高的区域对应畸变光斑。

## 应用价值

论文面向机载定向能系统和航空光学测量中对激波环境波前感知的需求，提出了一种可显著提升 SHWFS 在局部激波条件下重建精度的算法。由于 SHWFS 结构紧凑、成本低、鲁棒性好，在航空航天和高速流场光学诊断中应用广泛，而激波是不可避免的常见现象。该改进算法无需更换硬件，仅通过后处理即可提升测量精度，具有较强的工程实用性。

## 结语

这项工作通过实验系统揭示了激波对 SHWFS 测量精度的影响机理，并开发了基于高阶统计量和斜率差异的激波光斑识别与插值重建算法。虽然插值后的误差仍高于无激波情形，但研究为激波容忍的相位重建算法发展提供了重要参考。作者指出，当激波贯穿整个孔径时，现有方法的精度仍面临挑战，未来需要进一步探索激波区域的精确校正策略。

## 参考文献

Chu, Ethan D., Timothy J. Bukowski, and Stanislav Gordeyev. "Improved wavefront reconstruction for Shack–Hartmann wavefront sensor in the presence of shock-related distortions." Optics & Laser Technology 187 (2025): 112764.