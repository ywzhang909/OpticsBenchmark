# Optical Design Basics

> This document introduces the fundamental concepts of optical design and provides the domain knowledge needed to understand and evaluate optical agents.

---

## 📑 Table of Contents

1. [Geometric Optics Basics](#1-geometric-optics-basics)
2. [Optical System Composition](#2-optical-system-composition)
3. [Aberration Theory](#3-aberration-theory)
4. [Optical Performance Metrics](#4-optical-performance-metrics)
5. [Optical Design Workflow](#5-optical-design-workflow)

---

## 1. Geometric Optics Basics

### 1.1 Basic Concepts

| Term | English | Definition |
|------|------|------|
| **Light ray** | Ray | A straight line describing the propagation direction of light |
| **Focal length** | Focal Length | The distance from a lens to its focal point |
| **Optical axis** | Optical Axis | The ray passing through the centre of the system |
| **Aperture** | Aperture | The opening that limits the light beam |
| **Field of view** | Field of View | The range over which the system can form images |

### 1.2 Law of Refraction

```
n₁ sin(θ₁) = n₂ sin(θ₂)
```

where:
- `n₁, n₂`: refractive indices of the two media
- `θ₁`: angle of incidence
- `θ₂`: angle of refraction

### 1.3 Thin Lens Imaging Equation

```
1/f = 1/u + 1/v
```

where:
- `f`: focal length
- `u`: object distance
- `v`: image distance

### 1.4 Paraxial Optics

Under the paraxial approximation (small-angle approximation), an optical system can be described using matrix optics:

```python
# Free-space propagation
def propagate(distance: float, angle: float) -> tuple[float, float]:
    """
    Propagate a ray through free space.

    Args:
        distance: propagation distance
        angle: initial angle (radians)

    Returns:
        (y, angle): the new y position and angle
    """
    y = angle * distance
    return y, angle

# Thin lens refraction
def thin_lens(focal_length: float, y: float, angle: float) -> tuple[float, float]:
    """
    Refraction at a thin lens.

    Args:
        focal_length: focal length
        y: ray height
        angle: angle of incidence

    Returns:
        (y, angle): the new angle
    """
    new_angle = angle - y / focal_length
    return y, new_angle
```

---

## 2. Optical System Composition

### 2.1 Typical Lens Structure

```
┌───────────────────────────────────────────────────────────────────┐
│                         Lens System                               │
│                                                                   │
│  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐                                │
│  │ 1 │─►│ 2 │─►│ 3 │─►│ 4 │─►│ 5 │ ─► ...                         │
│  └───┘  └───┘  └───┘  └───┘  └───┘                                │
│      ↓      ↓      ↓      ↓      ↓                                │
│   Surf.1 Surf.2 Surf.3 Surf.4 Surf.5                              │
│                                                                   │
│  OBJ ──────────────────────► IMAGE                                │
│  (object)                                    (image plane)        │
│                                                                   │
│  EPD ──────────────► (entrance pupil)                             │
└───────────────────────────────────────────────────────────────────┘
```

### 2.2 Lens Types

| Type | Description | Typical Use |
|------|------|----------|
| **Plano-convex lens** | One flat side, one convex side | Collimation, focusing |
| **Plano-concave lens** | One flat side, one concave side | Diverging a beam |
| **Biconvex lens** | Both sides convex | Imaging, magnification |
| **Biconcave lens** | Both sides concave | Diverging a beam |
| **Achromatic lens** | A combination of two or more elements | Correcting chromatic aberration |
| **Aspheric lens** | Surface is an asphere | Reducing aberration |

### 2.3 Glass Materials

Commonly used optical glasses:

```python
# Schott glass catalogue
OPTICAL_GLASSS = {
    "N-BK7": {        # common crown glass
        "nd": 1.5168,    # d-line refractive index (587.6nm)
        "vd": 64.1,      # Abbe number
        "density": 2.51,
    },
    "N-SF11": {      # dense flint glass
        "nd": 1.7847,
        "vd": 25.8,
        "density": 3.78,
    },
    "F2": {          # flint glass
        "nd": 1.6200,
        "vd": 36.4,
        "density": 3.61,
    },
    "N-SSK8": {     # dense crown glass
        "nd": 1.6880,
        "vd": 53.2,
        "density": 3.31,
    },
}
```

---

## 3. Aberration Theory

### 3.1 The Seven Aberrations

| Aberration | Description | Effect |
|----------|------|------|
| **Spherical aberration** | Marginal rays and paraxial rays do not focus at the same point | Blurred image |
| **Coma** | Off-axis point images form a comet shape | Image distortion |
| **Astigmatism** | Tangential and sagittal focal lines are separated | Unsharp image |
| **Field curvature** | A flat object is imaged onto a curved surface | Blurred edges |
| **Distortion** | Magnification varies with field height | Image distortion |
| **Chromatic aberration** | Different wavelengths focus at different positions | Coloured fringes |
| **Aberration** | Deviation of the wavefront from the ideal sphere | Combined effect |

### 3.2 Wavefront Aberration

The aberration of an optical system can be described with Zernike polynomials:

```python
import numpy as np

# Zernike polynomial indices (Noll convention)
ZERNIKE_INDICES = {
    1: ("Z1", " piston"),
    2: ("Z2", " tilt-x"),
    3: ("Z3", " tilt-y"),
    4: ("Z4", " defocus"),
    5: ("Z5", " astigmatism 45°"),
    6: ("Z6", " astigmatism 0°"),
    7: ("Z7", " coma-x"),
    8: ("Z8", " coma-y"),
    # ... more terms
}

def calculate_wavefront_error(zernike_coeffs: list[float], rho: float, theta: float) -> float:
    """
    Calculate the wavefront error.

    Args:
        zernike_coeffs: list of Zernike coefficients
        rho, theta: normalized polar coordinates

    Returns:
        the wavefront error value
    """
    # Simplified implementation
    error = 0.0
    for i, c in enumerate(zernike_coeffs):
        # This should be a full Zernike polynomial evaluation
        error += c * (rho ** i) * np.cos(i * theta)
    return error
```

### 3.3 Chromatic Aberration Correction

An achromatic lens corrects chromatic aberration by combining different glass materials:

```python
def design_achromat(
    focal_length: float,
    glass1: str = "N-BK7",
    glass2: str = "F2",
) -> dict[str, float]:
    """
    Design a simple achromatic doublet.

    Args:
        focal_length: system focal length (mm)
        glass1: material of the positive lens
        glass2: material of the negative lens

    Returns:
        the lens parameters
    """
    # Simplified design calculation
    # A real design requires iterative optimization

    return {
        "focal_length": focal_length,
        "lens1_material": glass1,
        "lens2_material": glass2,
        "lens1_power": 2.0 / focal_length,  # diopters
        "lens2_power": -1.0 / focal_length,
    }
```

---

## 4. Optical Performance Metrics

### 4.1 MTF (Modulation Transfer Function)

MTF describes the ability of an optical system to transfer contrast:

```
MTF(f) = M_image(f) / M_object(f)
```

where:
- `f`: spatial frequency (lp/mm or cycles/mm)
- `M`: contrast = (Imax - Imin) / (Imax + Imin)

```python
def calculate_mtf(
    spatial_frequency: float,
    wavelength: float = 0.55,
    fnumber: float = 2.0,
) -> float:
    """
    Simplified MTF calculation (circular-aperture diffraction limit).

    Args:
        spatial_frequency: spatial frequency (lp/mm)
        wavelength: wavelength (μm)
        fnumber: F-number

    Returns:
        MTF value in [0, 1]
    """
    # The MTF of an optical system is jointly determined by the
    # diffraction limit and the aberrations.
    # A simplified sinc-function model is used here.

    cutoff_frequency = 1 / (1.22 * wavelength * fnumber)  # diffraction cutoff frequency

    if spatial_frequency > cutoff_frequency:
        return 0.0

    # Simplified sinc model
    normalized_freq = spatial_frequency / cutoff_frequency
    mtf = np.sinc(normalized_freq)

    return abs(mtf)
```

### 4.2 RMS Spot Diagram

```python
def calculate_spot_rms(
    rays: list[tuple[float, float]],
) -> float:
    """
    Calculate the RMS spot radius.

    Args:
        rays: list of ray intersection points [(x, y), ...]

    Returns:
        the RMS radius
    """
    import math

    n = len(rays)
    if n == 0:
        return 0.0

    # Calculate the centroid
    cx = sum(r[0] for r in rays) / n
    cy = sum(r[1] for r in rays) / n

    # Calculate the RMS radius
    sum_sq = sum((r[0] - cx)**2 + (r[1] - cy)**2 for r in rays)
    rms = math.sqrt(sum_sq / n)

    return rms
```

### 4.3 Distortion

```python
def calculate_distortion(
    ideal_height: float,
    actual_height: float,
) -> float:
    """
    Calculate the distortion percentage.

    Distortion(%) = (y_actual - y_ideal) / y_ideal × 100%

    Args:
        ideal_height: ideal image height
        actual_height: actual image height

    Returns:
        distortion percentage (positive = barrel, negative = pincushion)
    """
    if ideal_height == 0:
        return 0.0

    distortion = (actual_height - ideal_height) / ideal_height * 100
    return distortion
```

### 4.4 Transmission

```python
def calculate_transmission(
    surface_reflectance: list[float],
    glass_thickness: list[float],
    glass_absorption: float = 0.01,
    n_surfaces: int = 10,
) -> float:
    """
    Calculate the transmission of the system.

    Args:
        surface_reflectance: reflectance of each surface
        glass_thickness: thickness of each glass element (mm)
        glass_absorption: absorption coefficient of the glass (per mm)
        n_surfaces: number of surfaces

    Returns:
        the transmission of the system in [0, 1]
    """
    transmission = 1.0

    # Loss due to surface reflection
    for R in surface_reflectance:
        transmission *= (1 - R)

    # Loss due to absorption in the glass
    for thickness in glass_thickness:
        transmission *= math.exp(-glass_absorption * thickness)

    return transmission
```

### 4.5 Summary of Performance Metrics

| Metric | Symbol | Unit | Typical Target |
|------|------|------|----------|
| MTF @ 50 lp/mm | MTF | % | > 0.7 |
| RMS spot radius | RMS Spot | μm | < 10 |
| Distortion | Distortion | % | < 2% |
| Transmission | T | % | > 85% |
| RMS wavefront error | WFE | λ (wavelength) | < 0.07λ |
| Vignetting | Vignetting | % | < 30% |

---

## 5. Optical Design Workflow

### 5.1 Design Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   Optical Design Workflow                                       │
│                                                                                 │
│  ┌───────────────┐  →┌───────────────┐  →┌───────────────┐  →┌───────────────┐  │
│  │Requirements   │  →│Preliminary    │  →│Optimize       │  →│Tolerance      │  │
│  │Analysis       │  →│Design         │  →│               │  →│Analysis       │  │
│  └───────────────┘  →└───────────────┘  →└───────────────┘  →└───────────────┘  │
│         │                                                     │                 │
│         ▼                                                     ▼                 │
│  ┌───────────────┐  ←┌───────────────┐  ←┌───────────────┐  ←┌───────────────┐  │
│  │Spec           │  ←│Technical      │  ←│Performance    │  ←│Production     │  │
│  │Definition     │  ←│Review         │  ←│Validation     │  ←│Transfer       │  │
│  └───────────────┘  ←└───────────────┘  ←└───────────────┘  ←└───────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Requirements Analysis

```python
@dataclass
class LensRequirements:
    """Lens design requirements."""
    focal_length: float           # focal length (mm)
    fnumber: float                # F-number
    field_of_view: float          # field of view (°)
    working_distance: float       # working distance (mm)
    wavelength_range: tuple[float, float]  # wavelength range (μm)

    # Performance requirements
    mtf_target: float = 0.7       # MTF target @ 50 lp/mm
    distortion_max: float = 2.0   # maximum distortion (%)
    transmission_min: float = 0.85 # minimum transmission

    # Constraints
    max_elements: int = 10        # maximum number of lens elements
    max_length: float = 100.0     # maximum length (mm)
    max_diameter: float = 50.0    # maximum diameter (mm)

    # Cost
    max_cost: float = 1000.0       # maximum cost ($)
```

### 5.3 Optimization Variables

```python
# Typical optimization variables
OPTIMIZATION_VARIABLES = {
    # Curvature
    "curvature": {
        "type": "continuous",
        "bounds": (-1.0, 1.0),  # curvature range (1/mm)
        "description": "reciprocal of the radius of curvature of each surface",
    },

    # Thickness
    "thickness": {
        "type": "continuous",
        "bounds": (0.5, 50.0),   # mm
        "description": "centre thickness of a lens or a spacing",
    },

    # Aspheric coefficients
    "aspheric": {
        "type": "continuous",
        "bounds": (-1e-3, 1e-3),
        "description": "aspheric coefficient (4th order)",
    },

    # Material
    "material": {
        "type": "discrete",
        "options": ["N-BK7", "N-SF11", "F2", ...],
        "description": "lens material",
    },
}
```

### 5.4 Optimization Algorithm

```python
from scipy.optimize import minimize

def optimize_lens(
    initial_lens: dict,
    requirements: LensRequirements,
    algorithm: str = "Damped Least Squares",
) -> dict:
    """
    Optimize a lens design.

    Args:
        initial_lens: initial design
        requirements: design requirements
        algorithm: optimization algorithm

    Returns:
        the optimized lens parameters
    """

    def merit_function(params):
        """
        Merit function (smaller is better).
        """
        # Evaluate each metric
        mtf = calculate_mtf_from_lens(params)
        distortion = calculate_distortion(params)

        # Weighted sum
        merit = (
            1.0 * max(0, requirements.mtf_target - mtf) +
            0.5 * max(0, distortion - requirements.distortion_max) +
            0.3 * thickness_penalty(params)
        )

        return merit

    # Run the optimization
    result = minimize(
        merit_function,
        initial_lens["params"],
        method=algorithm,
        options={"maxiter": 1000}
    )

    return {"params": result.x, "merit": result.fun}
```

---

## 📚 References

1. Born, M., & Wolf, E. (2019). *Principles of Optics* (7th ed.). Cambridge University Press.
2. Hecht, E. (2016). *Optics* (5th ed.). Pearson.
3. Kidger, M. J. (2002). *Fundamental Optical Design*. SPIE Press.
4. Zemax OpticStudio Documentation

---

*Last updated: 2024-XX-XX*
