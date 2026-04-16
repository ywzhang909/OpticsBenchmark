# 光学设计基础

> 本文档介绍光学设计的基本概念，为理解和评测光学智能体提供必要的领域知识。

---

## 📑 目录

1. [几何光学基础](#1-几何光学基础)
2. [光学系统组成](#2-光学系统组成)
3. [像差理论](#3-像差理论)
4. [光学性能指标](#4-光学性能指标)
5. [光学设计流程](#5-光学设计流程)

---

## 1. 几何光学基础

### 1.1 基本概念

| 术语 | 英文 | 定义 |
|------|------|------|
| **光线** | Ray | 描述光传播方向的直线 |
| **焦距** | Focal Length | 透镜到焦点的距离 |
| **光轴** | Optical Axis | 通过系统中心的光线 |
| **孔径** | Aperture | 限制光束的开口 |
| **视场** | Field of View | 系统能成像的范围 |

### 1.2 折射定律

```
n₁ sin(θ₁) = n₂ sin(θ₂)
```

其中：
- `n₁, n₂`: 两种介质的折射率
- `θ₁`: 入射角
- `θ₂`: 折射角

### 1.3 透镜成像公式

```
1/f = 1/u + 1/v
```

其中：
- `f`: 焦距
- `u`: 物距
- `v`: 像距

### 1.4 近轴光学

在近轴近似下 (小角度近似)，光学系统可以用矩阵光学描述：

```python
# 自由空间传播
def propagate(distance: float, angle: float) -> tuple[float, float]:
    """
    光线在自由空间传播
    
    Args:
        distance: 传播距离
        angle: 初始角度 (弧度)
    
    Returns:
        (y, angle): 新的 y 位置和角度
    """
    y = angle * distance
    return y, angle

# 薄透镜折射
def thin_lens(focal_length: float, y: float, angle: float) -> tuple[float, float]:
    """
    薄透镜折射
    
    Args:
        focal_length: 焦距
        y: 光线高度
        angle: 入射角度
    
    Returns:
        (y, angle): 新的角度
    """
    new_angle = angle - y / focal_length
    return y, new_angle
```

---

## 2. 光学系统组成

### 2.1 典型镜头结构

```
┌─────────────────────────────────────────────────────────────┐
│                        镜头系统                              │
├─────────────────────────────────────────────────────────────┤
│  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐                       │
│  │ 1 │→ │ 2 │→ │ 3 │→ │ 4 │→ │ 5 │→ ...                 │
│  └───┘  └───┘  └───┘  └───┘  └───┘                       │
│   ↓       ↓       ↓       ↓       ↓                        │
│  面1    面2     面3     面4     面5    ...                   │
│                                                             │
│  OBJ ──────────────────────────────────→ IMAGE             │
│  (物体)                                         (像面)        │
│                                                             │
│  EPD ────────────────────→ (入瞳)                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 镜片类型

| 类型 | 描述 | 典型用途 |
|------|------|----------|
| **平凸透镜** | 一面平，一面凸 | 准直、聚焦 |
| **平凹透镜** | 一面平，一面凹 | 发散光束 |
| **双凸透镜** | 两面都凸 | 成像、放大 |
| **双凹透镜** | 两面都凹 | 发散光束 |
| **消色差透镜** | 两片或多片组合 | 校正色差 |
| **非球面透镜** | 曲面非球面 | 减少像差 |

### 2.3 玻璃材料

常用光学玻璃：

```python
# Schott 玻璃系列
OPTICAL_GLASSS = {
    "N-BK7": {        # 常用冕牌玻璃
        "nd": 1.5168,    # d 光折射率 (587.6nm)
        "vd": 64.1,      # 阿贝数
        "density": 2.51,
    },
    "N-SF11": {      # 重火石玻璃
        "nd": 1.7847,
        "vd": 25.8,
        "density": 3.78,
    },
    "F2": {          # 火石玻璃
        "nd": 1.6200,
        "vd": 36.4,
        "density": 3.61,
    },
    "N-SSK8": {     # 重冕玻璃
        "nd": 1.6880,
        "vd": 53.2,
        "density": 3.31,
    },
}
```

---

## 3. 像差理论

### 3.1 七种像差

| 像差类型 | 描述 | 影响 |
|----------|------|------|
| **球差** | 边缘光线与近轴光线焦点不一致 | 图像模糊 |
| **彗差** | 离轴点成像呈彗星状 | 图像变形 |
| **像散** | 子午/弧矢焦线分离 | 图像不清晰 |
| **场曲** | 平面物体成像在曲面上 | 边缘模糊 |
| **畸变** | 放大率随视场变化 | 图像变形 |
| **色差** | 不同波长焦点位置不同 | 彩色边缘 |
| **像差** | 波前与理想球面的偏差 | 综合影响 |

### 3.2 波像差

光学系统像差可以用 Zernike 多项式描述：

```python
import numpy as np

# Zernike 多项式索引 (Noll 约定)
ZERNIKE_INDICES = {
    1: ("Z1", " piston"),
    2: ("Z2", " tilt-x"),
    3: ("Z3", " tilt-y"),
    4: ("Z4", " defocus"),
    5: ("Z5", " astigmatism 45°"),
    6: ("Z6", " astigmatism 0°"),
    7: ("Z7", " coma-x"),
    8: ("Z8", " coma-y"),
    # ... 更多项
}

def calculate_wavefront_error(zernike_coeffs: list[float], rho: float, theta: float) -> float:
    """
    计算波前误差
    
    Args:
        zernike_coeffs: Zernike 系数列表
        rho, theta: 归一化极坐标
    
    Returns:
        波前误差值
    """
    # 简化实现
    error = 0.0
    for i, c in enumerate(zernike_coeffs):
        # 这里应该是完整的 Zernike 多项式计算
        error += c * (rho ** i) * np.cos(i * theta)
    return error
```

### 3.3 色差校正

消色差透镜通过组合不同玻璃材料校正色差：

```python
def design_achromat(
    focal_length: float,
    glass1: str = "N-BK7",
    glass2: str = "F2",
) -> dict[str, float]:
    """
    设计简单的消色差双合透镜
    
    Args:
        focal_length: 系统焦距 (mm)
        glass1: 正透镜材料
        glass2: 负透镜材料
    
    Returns:
        透镜参数
    """
    # 简化的设计计算
    # 实际设计需要迭代优化
    
    return {
        "focal_length": focal_length,
        "lens1_material": glass1,
        "lens2_material": glass2,
        "lens1_power": 2.0 / focal_length,  # diopters
        "lens2_power": -1.0 / focal_length,
    }
```

---

## 4. 光学性能指标

### 4.1 MTF (调制传递函数)

MTF 描述光学系统传递对比度的能力：

```
MTF(f) = M_image(f) / M_object(f)
```

其中：
- `f`: 空间频率 (lp/mm 或 cycles/mm)
- `M`: 对比度 = (Imax - Imin) / (Imax + Imin)

```python
def calculate_mtf(
    spatial_frequency: float,
    wavelength: float = 0.55,
    fnumber: float = 2.0,
) -> float:
    """
    简化 MTF 计算 (圆孔径衍射极限)
    
    Args:
        spatial_frequency: 空间频率 (lp/mm)
        wavelength: 波长 (μm)
        fnumber: F 数
    
    Returns:
        MTF 值 [0, 1]
    """
    # 光学系统的 MTF 受衍射极限和像差的共同影响
    # 这里使用简化的 sinc 函数模型
    
    cutoff_frequency = 1 / (1.22 * wavelength * fnumber)  # 衍射截止频率
    
    if spatial_frequency > cutoff_frequency:
        return 0.0
    
    # 简化的 sinc 模型
    normalized_freq = spatial_frequency / cutoff_frequency
    mtf = np.sinc(normalized_freq)
    
    return abs(mtf)
```

### 4.2 RMS 点列图

```python
def calculate_spot_rms(
    rays: list[tuple[float, float]],
) -> float:
    """
    计算 RMS 点列半径
    
    Args:
        rays: 光线交点列表 [(x, y), ...]
    
    Returns:
        RMS 半径
    """
    import math
    
    n = len(rays)
    if n == 0:
        return 0.0
    
    # 计算质心
    cx = sum(r[0] for r in rays) / n
    cy = sum(r[1] for r in rays) / n
    
    # 计算 RMS 半径
    sum_sq = sum((r[0] - cx)**2 + (r[1] - cy)**2 for r in rays)
    rms = math.sqrt(sum_sq / n)
    
    return rms
```

### 4.3 畸变

```python
def calculate_distortion(
    ideal_height: float,
    actual_height: float,
) -> float:
    """
    计算畸变百分比
    
    Distortion(%) = (y_actual - y_ideal) / y_ideal × 100%
    
    Args:
        ideal_height: 理想像高
        actual_height: 实际像高
    
    Returns:
        畸变百分比 (正=桶形, 负=枕形)
    """
    if ideal_height == 0:
        return 0.0
    
    distortion = (actual_height - ideal_height) / ideal_height * 100
    return distortion
```

### 4.4 透射率

```python
def calculate_transmission(
    surface_reflectance: list[float],
    glass_thickness: list[float],
    glass_absorption: float = 0.01,
    n_surfaces: int = 10,
) -> float:
    """
    计算系统透射率
    
    Args:
        surface_reflectance: 每个表面的反射率
        glass_thickness: 每片玻璃厚度 (mm)
        glass_absorption: 玻璃吸收系数 (per mm)
        n_surfaces: 表面数量
    
    Returns:
        系统透射率 [0, 1]
    """
    transmission = 1.0
    
    # 表面反射损失
    for R in surface_reflectance:
        transmission *= (1 - R)
    
    # 玻璃吸收损失
    for thickness in glass_thickness:
        transmission *= math.exp(-glass_absorption * thickness)
    
    return transmission
```

### 4.5 性能指标汇总

| 指标 | 符号 | 单位 | 典型目标 |
|------|------|------|----------|
| MTF @ 50 lp/mm | MTF | % | > 0.7 |
| RMS 点列半径 | RMS Spot | μm | < 10 |
| 畸变 | Distortion | % | < 2% |
| 透射率 | T | % | > 85% |
| RMS 波前误差 | WFE | λ (波长) | < 0.07λ |
| 渐晕 | Vignetting | % | < 30% |

---

## 5. 光学设计流程

### 5.1 设计流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        光学设计流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  需求分析 │ → │  初步设计 │ → │   优化   │ → │  公差分析 │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │                                                   │      │
│       ▼                                                   ▼      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ 规格定义 │ ← │ 技术评审 │ ← │ 性能验证 │ ← │ 生产移交 │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 需求分析

```python
@dataclass
class LensRequirements:
    """镜头设计需求"""
    focal_length: float           # 焦距 (mm)
    fnumber: float                # F 数
    field_of_view: float          # 视场角 (°)
    working_distance: float       # 工作距离 (mm)
    wavelength_range: tuple[float, float]  # 波长范围 (μm)
    
    # 性能要求
    mtf_target: float = 0.7       # MTF 目标 @ 50 lp/mm
    distortion_max: float = 2.0   # 最大畸变 (%)
    transmission_min: float = 0.85 # 最小透射率
    
    # 约束
    max_elements: int = 10        # 最大透镜数量
    max_length: float = 100.0     # 最大长度 (mm)
    max_diameter: float = 50.0    # 最大直径 (mm)
    
    # 成本
    max_cost: float = 1000.0       # 最大成本 ($)
```

### 5.3 优化变量

```python
# 典型优化变量
OPTIMIZATION_VARIABLES = {
    # 曲率半径
    "curvature": {
        "type": "continuous",
        "bounds": (-1.0, 1.0),  # 曲率范围 (1/mm)
        "description": "每个面的曲率半径的倒数",
    },
    
    # 厚度
    "thickness": {
        "type": "continuous",
        "bounds": (0.5, 50.0),   # mm
        "description": "透镜中心厚度或间隔",
    },
    
    # 非球面系数
    "aspheric": {
        "type": "continuous",
        "bounds": (-1e-3, 1e-3),
        "description": "非球面系数 (4th order)",
    },
    
    # 材料
    "material": {
        "type": "discrete",
        "options": ["N-BK7", "N-SF11", "F2", ...],
        "description": "透镜材料",
    },
}
```

### 5.4 优化算法

```python
from scipy.optimize import minimize

def optimize_lens(
    initial_lens: dict,
    requirements: LensRequirements,
    algorithm: str = "Damped Least Squares",
) -> dict:
    """
    优化镜头设计
    
    Args:
        initial_lens: 初始设计
        requirements: 设计需求
        algorithm: 优化算法
    
    Returns:
        优化后的镜头参数
    """
    
    def merit_function(params):
        """
        评价函数 (越小越好)
        """
        # 计算各项指标
        mtf = calculate_mtf_from_lens(params)
        distortion = calculate_distortion(params)
        
        # 加权求和
        merit = (
            1.0 * max(0, requirements.mtf_target - mtf) +
            0.5 * max(0, distortion - requirements.distortion_max) +
            0.3 * thickness_penalty(params)
        )
        
        return merit
    
    # 执行优化
    result = minimize(
        merit_function,
        initial_lens["params"],
        method=algorithm,
        options={"maxiter": 1000}
    )
    
    return {"params": result.x, "merit": result.fun}
```

---

## 📚 参考资料

1. Born, M., & Wolf, E. (2019). *Principles of Optics* (7th ed.). Cambridge University Press.
2. Hecht, E. (2016). *Optics* (5th ed.). Pearson.
3. Kidger, M. J. (2002). *Fundamental Optical Design*. SPIE Press.
4. Zemax OpticStudio Documentation

---

*最后更新: 2024-XX-XX*
