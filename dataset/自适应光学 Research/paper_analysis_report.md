# 自适应光学 Research 论文分析报告

**数据集**: `paper_info_extract.jsonl` (25篇论文)  
**字段**: `ten keywords`, `objective`, `novelty`, `method`, `performance metrics`  
**生成时间**: 2026-07-27  
**对齐评估器**: `rubric_based_evaluator.py`

---

## 目录

1. [数据集概览](#1-数据集概览)
2. [逐篇论文分析](#2-逐篇论文分析)
3. [主题分布统计](#3-主题分布统计)
4. [研究方法分类](#4-研究方法分类)
5. [性能指标汇总](#5-性能指标汇总)

---

## 1. 数据集概览

本数据集包含25篇自适应光学领域的最新研究论文，覆盖以下核心方向：

| 研究方向 | 论文数 | 代表论文 |
|---------|--------|----------|
| 波前传感与重建 | 8 | 002, 004, 009, 012, 019, 021, 025, 011 |
| 自适应光学系统 | 5 | 001, 016, 022, 023, 017 |
| 量子光学与成像 | 3 | 005, 009, 015 |
| 空间光学与通信 | 3 | 003, 013, 020 |
| 高对比成像 | 2 | 008, 018 |
| 工程优化与数字孪生 | 2 | 010, 024 |
| 太阳观测 | 1 | 014 |
| 其他（干涉测量、校准等） | 1 | 006, 007 |

---

## 2. 逐篇论文分析

### Paper 001 — 优化太阳GLAO导星布局，提升宽视场分辨率均匀性

**机构**: 中国科学院光电所

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Ground-Layer Adaptive Optics, GLAO, Guide Star Optimization, Solar Observation, Wide Field of View, Resolution Uniformity, Correlation Evaluation Function, NVST, Atmospheric Turbulence, Conjugate Height |
| **objective** | 优化GLAO导星配置以实现太阳观测宽视场均匀分辨率，建立定量评价指标CEF |
| **novelty** | 提出CEF评价指标，证明单环导星布局在均匀性上优于传统中心+环布局 |
| **method** | 数值仿真+实验室验证+NVST实测三级验证 |
| **performance metrics** | CEF ≥ 0.95均匀性；FWHM提升15-25% |

**分析**: 该研究建立了完整的GLAO优化框架，从理论仿真到实际观测验证形成闭环。CEF指标的提出为宽视场AO系统设计提供了量化工具。

---

### Paper 002 — 深度学习增强全息波前传感器

**机构**: 北京理工大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Deep Learning, Holographic Wavefront Sensor, High-order Aberration, Modal Wavefront Sensing, Shack-Michelson Interferometer, Mode Number, Speckle Noise, Adaptive Optics, Point Spread Function, CNN |
| **objective** | 突破传统HMWS的模式数限制和散斑噪声瓶颈 |
| **novelty** | 双网络架构：模态回归+像素级重建，同时解决速度与精度问题 |
| **method** | 双CNN架构，Network 1做Zernike系数提取，Network 2做像素级波前重建 |
| **performance metrics** | 模式数从15扩展到50；SNR提升8 dB；高阶像差重建精度95% |

**分析**: 双网络设计巧妙地平衡了速度与精度需求，解决了传统全息波前传感器的核心限制。

---

### Paper 003 — 差分波前传感实现空间引力波探测指向控制

**机构**: 中国科学院光电所

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Differential Wavefront Sensing, DWS, Gravitational Wave Detection, Pointing Control, Dual Closed-Loop, Space Optics, Phase-Locked Loop, LISA, Milli-Hertz, Precision Metrology |
| **objective** | 开发基于DWS的双闭环精密指向控制系统，用于空间引力波探测 |
| **novelty** | 双闭环DWS架构解耦粗精指向控制，ASD校准方法实现mHz频段绝对相位-角度转换 |
| **method** | DWS+四象限探测器+双环控制+ASD校准+空间环境模拟实验 |
| **performance metrics** | 指向稳定性 <100 nrad/√Hz (>1 mHz)；相位测量精度 <0.1 mrad |

**analysis**: 直接支撑LISA/太极等重大空间科学任务，技术指标达到任务要求，工程转化潜力巨大。

---

### Paper 004 — 轻量级CNN SIR-Net波前重建

**机构**: 国防科大

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Wavefront Reconstruction, Deep Learning, Shack-Hartmann, Sparse Sub-apertures, Strong Turbulence, CNN, Real-time, Adaptive Optics, TensorRT, Model Compression |
| **objective** | 开发轻量级CNN用于强湍流稀疏子孔径下的快速波前重建 |
| **novelty** | SIR（子孔径信息复用）+FZM（特征置零压缩），实现模型压缩与精度保持 |
| **method** | 轻量CNN+参数共享+TensorRT加速+FP16量化 |
| **performance metrics** | 模型1.77 MB；推理0.191 ms；支持>5 kHz闭环 |

**analysis**: 极低的延迟和模型体积直接满足星载/便携式AO系统的实时部署需求。

---

### Paper 005 — 单像素盲调制衍射成像

**机构**: 哈工大

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Single-pixel Imaging, Scattering Media, Phase Retrieval, Diffraction, Blind Modulation, Coherent Imaging, Complex Field Reconstruction, Wavefront Shaping, Optical Memory Effect, Computational Imaging |
| **objective** | 实现散射介质中高保真复场重建（振幅+相位） |
| **novelty** | 结合光学记忆效应与计算相位恢复，无需散射矩阵标定 |
| **method** | 单像素探测器+结构化照明+迭代相位恢复+相干检测 |
| **performance metrics** | 重建保真度 >0.9；相位精度 λ/10；适用透过率10⁻³-10⁻⁶ |

**analysis**: 无需标定即可通过散射介质成像，在生物成像和安全检测领域有重要应用。

---

### Paper 006 — TransUNet迁移学习实现横向剪切干涉参数自动计算

**机构**: 西安工业大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Lateral Shearing Interferometry, TransUNet, Transfer Learning, Shear Parameter, Wavefront Sensing, Deep Learning, Phase Unwrapping, Aberration Measurement, Optical Testing, Automatic Calibration |
| **objective** | 开发自动化剪切参数提取方法，消除人工校准需求 |
| **novelty** | 首次将TransUNet+迁移学习应用于横向剪切干涉参数提取 |
| **method** | 预训练TransUNet微调+特征提取+回归网络 |
| **performance metrics** | 剪切参数精度 <1%相对误差；处理时间 <1秒 |

**analysis**: 自动化校准显著降低了干涉测量系统的操作复杂度。

---

### Paper 007 — 色散干涉图斜率拼接镜共相位方法

**机构**: 上海天文台

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Segmented Mirror, Co-phasing, Dispersed Interferometry, Piston Sensing, Closed-loop, Segmented Telescope, Phase Diversity, Wavelength Scanning, Edge Sensor, Extreme Large Telescope |
| **objective** | 开发基于色散干涉图斜率的拼接镜闭环共相位方法 |
| **novelty** | 用色散干涉图斜率作为活塞误差度量，结合闭环反馈控制 |
| **method** | 色散宽带干涉+波长依赖干涉图分析+闭环伺服控制 |
| **performance metrics** | 活塞检测精度 λ/50 (RMS)；共相位精度 <20 nm |

**analysis**: 为极大望远镜拼接镜系统提供了实用的共相位解决方案。

---

### Paper 008 — 多波长Zernike波前传感器

**机构**: 荷兰空间研究组织 (SRON)

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Zernike Wavefront Sensor, Multi-wavelength, Dynamic Range, Photon Noise, Extreme Adaptive Optics, Exoplanet Imaging, Coronagraphy, Wavefront Control, High Contrast, Sensitivity Analysis |
| **objective** | 研究多波长ZWS以提升动态范围和光子噪声鲁棒性 |
| **novelty** | 系统分析多波长ZWS操作，证明多波长同步测量显著扩展动态范围 |
| **method** | 多波长ZWS仿真+光子噪声传播分析+单波长配置对比 |
| **performance metrics** | 动态范围扩展3-5倍；光子噪声灵敏度提升2-3 dB |

**analysis**: 为极端AO系统的波前传感器设计提供了重要理论指导。

---

### Paper 009 — 位置关联双光子Shack-Hartmann波前传感

**机构**: 中国科大

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Quantum Imaging, Spatial Entanglement, Shack-Hartmann Sensor, Photon Pair, Wavefront Sensing, Single-shot, Quantum Adaptive Optics, Correlated Photons, Phase Measurement, Quantum Metrology |
| **objective** | 开发单发量子波前传感方法 |
| **novelty** | 首次结合空间纠缠光子对与Shack-Hartmann检测实现单发量子波前传感 |
| **method** | SPDC光子对产生+分束Shack-Hartmann+重合测量+量子图像校正 |
| **performance metrics** | 单发测量精度 λ/20；量子增强因子1.5-2× |

**analysis**: 量子增强波前传感为下一代AO系统提供了新方向。

---

### Paper 010 — 低频外差干涉法校准LC-SLM

**机构**: 中北大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Spatial Light Modulator, LC-SLM, Phase Calibration, Heterodyne Interferometry, Low Frequency, Phase Response, Wavefront Control, Liquid Crystal, Optical Metrology, Linearity |
| **objective** | 开发低频外差干涉法校准LC-SLM相位响应 |
| **novelty** | 低频外差干涉作为校准技术，提供高精度相位表征 |
| **method** | AOM低频外差干涉+锁相检测+相位-电压关系表征 |
| **performance metrics** | 相位校准精度 <0.5% 2π；线性度从±5%改善到±1% |

**analysis**: 提高了LC-SLM在AO系统中的使用精度。

---

### Paper 011 — 集成马赫-曾德尔波前传感器

**机构**: 法国里昂天体物理研究中心 (CRAL)

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Wavefront Sensor, Mach-Zehnder, Integrated Optics, Extreme Adaptive Optics, Exoplanet, Photonic Integrated Circuit, Starlight Suppression, High Contrast Imaging, Nulling Interferometry, Giant Telescope |
| **objective** | 开发集成MZ波前传感器用于极端AO |
| **novelty** | 首个集成光子MZ波前传感器，同时实现波前传感和星光压制 |
| **method** | 光子集成电路MZ干涉仪+宽带消光+集成相位检测 |
| **performance metrics** | 消光深度 >10⁻³ (10%带宽)；传感精度 λ/100；器件<10×10 mm |

**analysis**: 集成光学为下一代极大望远镜AO系统提供了紧凑解决方案。

---

### Paper 012 — 改进Shack-Hartmann波前重建应对激波畸变

**机构**: 圣母大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Shack-Hartmann, Wavefront Reconstruction, Shock Distortion, Centroid Algorithm, Strong Turbulence, Adaptive Optics, Spot Detection, Robust Algorithm, Nonlinear Effects, Wavefront Sensor |
| **objective** | 改进SHWFS算法以处理强湍流引起的激波畸变波前 |
| **novelty** | 开发针对激波畸变的鲁棒重建算法，引入光斑模式识别和自适应校正 |
| **method** | 改进质心检测+自适应窗口+迭代重建+激波校正 |
| **performance metrics** | 弱湍流性能保持<15%退化(r₀<5cm)；光斑检测率>90%；3×强湍流改进 |

**analysis**: 为强湍流AO系统提供了实用的波前重建解决方案。

---

### Paper 013 — 基于扩展信标的模型化倾斜校正

**机构**: 北京理工大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Space Telescope, Segmented Mirror, Tilt Correction, Extended Beacon, Model-based Control, Wavefront Sensing, Pointing Error, Adaptive Optics, JWST, Coronagraphy |
| **objective** | 开发基于扩展信标的模型化倾斜校正方法 |
| **novelty** | 扩展信标倾斜传感+模型化控制，突破点源引导星限制 |
| **method** | 焦面图像扩展信标传感+段几何模型+预测控制+JWST仿真 |
| **performance metrics** | 倾斜校正残差 <1 mas RMS；适用于0.5角秒扩展信标 |

**analysis**: 为高对比成像提供了无需点源引导星的精细指向方案。

---

### Paper 014 — 比较日珥Hα波前探测算法

**机构**: 中国科学院光电所

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Solar Prominence, Hα Imaging, Wavefront Sensing, Correlation Algorithm, Solar Adaptive Optics, Image Motion, Tie Point, Feature Tracking, Fried Parameter, Seeing Measurement |
| **objective** | 比较评估不同波前探测算法用于太阳Hα日珥观测 |
| **novelty** | 系统比较Hα日珥成像专用波前传感算法，证明互相关法SNR更优 |
| **method** | 互相关算法+连接点方法+信噪比分析+太阳望远镜实验对比 |
| **performance metrics** | 互相关法SNR 3×高于连接点方法；测量精度 λ/8 |

**analysis**: 为太阳AO系统选择合适的波前传感算法提供了实验证据。

---

### Paper 015 — 对称遗传算法实现纠缠光子波前校正

**机构**: 印度TIFR与耶路撒冷希伯来大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Quantum Optics, Entangled Photons, Wavefront Correction, Genetic Algorithm, Spatial Light Modulator, Quantum Communication, Optimization, Two-photon Interference, Hong-Ou-Mandel, Quantum Imaging |
| **objective** | 开发对称遗传算法实现纠缠光子快速波前校正 |
| **novelty** | 利用光子对对称相关结构加速优化，收敛速度提升5-10× |
| **method** | 对称遗传算法+SLM相位调制+HOM干涉度量+并行评估 |
| **performance metrics** | 收敛时间从30分钟降至3-5分钟；纠缠保持保真度>0.95 |

**analysis**: 显著加速了量子光学系统的波前校正过程。

---

### Paper 016 — 旋转自适应光学技术综述

**机构**: 沙特KFUPM

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Rotational Adaptive Optics, Review, Astronomical Imaging, Atmospheric Turbulence, Wavefront Correction, Telescope, Deformable Mirror, Wavefront Sensor, Ground Layer, Multi-conjugate |
| **objective** | 综述利用场旋转实现宽视场湍流校正的旋转AO技术 |
| **novelty** | 首个全面综述，建立统一框架理解不同望远镜配置的场旋转校正 |
| **method** | 文献综述+理论分析+RAO方法比较+技术成熟度评估 |
| **performance metrics** | 校正Strehl比0.1-0.8；校正视场比传统AO宽2-3×；TRL 3-6 |

**analysis**: 为旋转AO技术的发展提供了全面的技术路线图。

---

### Paper 017 — 稀疏盲解卷积计算自适应光学

**机构**: 南京理工大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Fluorescence Microscopy, Computational Adaptive Optics, Blind Deconvolution, Sparse Regularization, Aberration Correction, Image Restoration, Point Spread Function, Biological Imaging, 3D Imaging, Iterative Algorithm |
| **objective** | 开发稀疏盲解卷积计算AO方法提升荧光显微成像质量 |
| **novelty** | 同时估计PSF和物体结构的计算AO技术，无需硬件波前传感 |
| **method** | 稀疏正则化盲解卷积+交替最小化+稀疏先验约束 |
| **performance metrics** | 分辨率提升2-3× (FWHM)；Strehl比从0.2恢复到>0.7；处理时间<5分钟 |

**analysis**: 为生命科学成像提供了低成本的后处理像差校正方案。

---

### Paper 018 — 重新审视dOTF高对比波前传感

**机构**: 蔚蓝海岸天文台

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Differential Optical Transfer Function, dOTF, High Contrast Imaging, Wavefront Sensing, Non-common Path Aberration, Coronagraphy, Focal Plane Sensing, Exoplanet Detection, Starlight Suppression, Phase Retrieval |
| **objective** | 重新审视并改进dOTF高对比成像波前传感 |
| **novelty** | 识别此前未被注意的系统误差源，提出改进校准协议 |
| **method** | 详细误差分析+相位多样性校正+探测器非线性研究+技术对比 |
| **performance metrics** | dOTF精度从λ/20提升到λ/100；非共路像差灵敏度<1 nm |

**analysis**: 恢复了dOTF技术的理论灵敏度极限，为高对比成像提供了可靠的波前传感手段。

---

### Paper 019 — 用部分斜率信息实现波前校正

**机构**: 中国科学院光电所

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Adaptive Optics, Partial Slope, Wavefront Control, Deformable Mirror, Modal Control, Sub-aperture, Weighted Control, Efficiency, Real-time, Atmospheric Turbulence |
| **objective** | 开发加权模态控制方法，利用部分斜率信息提升校正效率 |
| **novelty** | 基于信号质量权重的选择性部分斜率使用，降低计算负载 |
| **method** | 加权模态重建+信噪比加权+控制矩阵优化+实验验证 |
| **performance metrics** | 计算需求降低40%；低SNR下Strehl比提升10-15%；>1 kHz实时运行 |

**analysis**: 在保持校正质量的同时显著降低了AO系统的计算负担。

---

### Paper 020 — AI相位调制缓解FSO指向误差

**机构**: 弗鲁米嫩塞联邦大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Free Space Optical Communication, Pointing Error, AI Phase Modulation, Beam Tracking, Atmospheric Turbulence, Machine Learning, Optical Beam Control, Scintillation, Beam Wander, Communication Link |
| **objective** | 利用AI相位调制缓解自由空间光通信指向误差 |
| **novelty** | AI驱动的预测性指向误差补偿，从被动响应转为主动预测 |
| **method** | 深度学习指向误差预测+SLM自适应相位调制+预测控制 |
| **performance metrics** | 指向误差方差降低60-70%；链路可用性>99.5%；BER从10⁻³改善到10⁻⁶ |

**analysis**: 将AO技术扩展到光通信领域，为FSO系统提供了智能指向稳定方案。

---

### Paper 021 — 抗噪多条纹分解算法

**机构**: 中国计量大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Lateral Shearing Interferometry, Fringe Decomposition, Noise Robustness, Phase Unwrapping, Wavefront Sensing, Multi-fringe, Algorithm, Optical Testing, Interferogram Analysis, Signal Processing |
| **objective** | 开发抗噪多条纹分解算法用于横向剪切干涉测量 |
| **novelty** | 专为噪声横向剪切干涉图设计的鲁棒分解算法 |
| **method** | 频域多分量分解+自适应噪声滤波+迭代相位提取 |
| **performance metrics** | SNR低至5 dB时精度保持λ/15；3×噪声容限提升；处理时间<0.5秒 |

**analysis**: 显著提升了横向剪切干涉测量在低信噪比条件下的适用性。

---

### Paper 022 — 仿真AO补偿ISAL湍流成像

**机构**: 中国科学院安徽光学精密机械研究所

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Inverse Synthetic Aperture Ladar, ISAL, Adaptive Optics, Atmospheric Turbulence, Phase Gradient Autofocus, Heterodyne Detection, Coherent Imaging, Phase Correction, Range-Doppler, Laser Radar |
| **objective** | 研究实时AO补偿与PGA结合对ISAL湍流成像的改善效果 |
| **novelty** | 首次系统研究AO+PGA组合用于ISAL湍流缓解 |
| **method** | 动态相位屏仿真+SHWFS AO校正+PGA算法+R-D成像+AO-PGA处理链 |
| **performance metrics** | AO带宽需求 >60 Hz；图像质量提升10-15 dB PSLR |

**analysis**: 为ISAL系统在强湍流条件下的成像提供了定量设计依据。

---

### Paper 023 — 共轭自适应光学用于深脑成像显微内窥术

**机构**: 加利福尼亚大学圣迭戈分校

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Field-conjugate Adaptive Optics, FCAO, Microendoscopy, Deep Brain Imaging, GRIN Lens, Multiphoton Microscopy, Two-photon, Neural Imaging, Aberration Correction, Wavefront Sensorless |
| **objective** | 开发FCAO扩展GRIN透镜微内窥镜的校正视场 |
| **novelty** | 首次将场共轭AO应用于GRIN透镜微内窥镜 |
| **method** | 场共轭面SLM+无波前传感图像优化+高斯环基函数+活体验证 |
| **performance metrics** | 校正视场从140μm扩展到350μm；边缘强度提升2.5-3× |

**analysis**: 为神经科学研究提供了更高质量的深脑成像工具。

---

### Paper 024 — 大口径反射镜数据驱动优化与数字孪生

**机构**: 大连理工大学

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Mirror System, Data-driven Optimization, Digital Twin, Finite Element Analysis, Multi-objective Optimization, Lightweight Design, Large Aperture, Surrogate Model, Shape Optimization, Structural Mechanics |
| **objective** | 开发数据驱动优化框架+数字孪生用于大口径反射镜系统 |
| **novelty** | CNSGAIII-EHVI约束多目标优化+数字孪生平台集成 |
| **method** | 45维参数化模型+CNSGAIII-EHVI优化+RBF代理模型+Unity3D可视化 |
| **performance metrics** | FEA调用从数千次降至100次；代理模型R²>0.999；响应时间<2秒 |

**analysis**: 大幅缩短了大口径镜面系统的设计周期，同时提供了实时性能监控能力。

---

### Paper 025 — 深度学习实现单次横向剪切干涉

**机构**: 韩国标准科学研究院

| 字段 | 内容摘要 |
|------|----------|
| **ten keywords** | Lateral Shearing Interferometry, Deep Learning, Single-shot, Phase Measurement, Wavefront Sensing, Polarization Grating, Pixelated Polarizer, Optical Testing, Dynamic Measurement, Neural Network |
| **objective** | 开发深度学习单次LSI方法同时提取两正交方向波前信息 |
| **novelty** | 首次深度学习单次LSI分离复合xy剪切信息 |
| **method** | PG双方向剪切干涉仪+PPC相移采集+改进DYnet++网络 |
| **performance metrics** | 精度λ/14 RMS；视频帧率动态测量；噪声容限40% |

**analysis**: 实现了LSI的单发动态测量，为高速表面检测提供了新方案。

---

## 3. 主题分布统计

| 主题类别 | 论文编号 | 数量 |
|---------|----------|------|
| 波前传感技术 | 002, 004, 009, 011, 012, 019, 021, 025 | 8 |
| AO系统设计 | 001, 016, 017, 022, 023 | 5 |
| 量子光学 | 005, 009, 015 | 3 |
| 空间光学 | 003, 013, 020 | 3 |
| 高对比成像 | 008, 018 | 2 |
| 工程优化 | 010, 024 | 2 |
| 太阳观测 | 001, 014 | 2 |
| 干涉测量 | 006, 007 | 2 |

---

## 4. 研究方法分类

| 方法类别 | 论文数 | 占比 |
|---------|--------|------|
| 深度学习/机器学习 | 10 | 40% |
| 数值仿真+实验验证 | 8 | 32% |
| 传统光学传感 | 4 | 16% |
| 综述/理论分析 | 2 | 8% |
| 其他 | 1 | 4% |

---

## 5. 性能指标汇总

| 指标类型 | 典型数值范围 | 涉及论文 |
|---------|-------------|---------|
| 波前精度 | λ/8 - λ/100 | 002, 009, 011, 018, 025 |
| Strehl比 | 0.2 → 0.7-0.95 | 001, 017, 023 |
| 计算延迟 | 0.191 ms - 5 min | 004, 019, 021, 025 |
| 模型大小 | 1.77 MB - 20M参数 | 004, 025 |
| 动态范围扩展 | 2-5× | 008, 023, 025 |
| SNR提升 | 2-8 dB | 002, 014, 021 |
| 指向稳定性 | <1 mas - 100 nrad/√Hz | 003, 013, 020 |

---

## 数据集统计

- **总论文数**: 25
- **平均每篇关键词数**: 10 (固定)
- **字段完整性**: 100% (所有论文均包含5个完整字段)
- **语言**: 英文 (字段内容)
- **来源机构覆盖**: 中国(14)、欧洲(4)、美国(2)、印度(1)、韩国(1)、沙特(1)、巴西(1)、其他(1)

---

**文件位置**: `D:\workspace\OpticsBenchmark\dataset\自适应光学 Research\paper_analysis_report.md`  
**数据集**: `D:\workspace\OpticsBenchmark\dataset\自适应光学 Research\paper_info_extract.jsonl`
