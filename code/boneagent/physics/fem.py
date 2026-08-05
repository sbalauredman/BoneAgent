from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MeshSpecification:
    porosity: float
    pore_diameter_micrometer: float
    strut_diameter_micrometer: float
    interconnectivity: float
    minimum_elements: int
    characteristic_length_micrometer: float


@dataclass(frozen=True)
class MaterialLaw:
    elastic_modulus_gpa: float
    poisson_ratio: float
    yield_strength_mpa: float
    density_kg_m3: float


@dataclass(frozen=True)
class CompressionProtocol:
    displacement_rate_mm_min: float = 1.0
    total_strain: float = 0.2
    increments: int = 200
    bottom_fixed: bool = True


@dataclass(frozen=True)
class FEMResult:
    compressive_strength_mpa: float
    effective_modulus_gpa: float
    maximum_von_mises_mpa: float
    failure_strain: float
    converged: bool
    element_count: int


def mesh_specification(
    porosity: float,
    pore_diameter_micrometer: float,
    strut_diameter_micrometer: float,
    interconnectivity: float,
) -> MeshSpecification:
    if not 0.3 <= porosity <= 0.8:
        raise ValueError("porosity must be within [0.3, 0.8]")
    minimum = 100000 if porosity > 0.7 else 50000
    characteristic = min(strut_diameter_micrometer / 4.0, pore_diameter_micrometer / 8.0)
    characteristic *= 0.75 + 0.25 * (1.0 - interconnectivity)
    return MeshSpecification(
        porosity,
        pore_diameter_micrometer,
        strut_diameter_micrometer,
        interconnectivity,
        minimum,
        characteristic,
    )


def isotropic_stiffness(material: MaterialLaw) -> NDArray[np.float64]:
    modulus = material.elastic_modulus_gpa * 1000.0
    poisson = material.poisson_ratio
    if not -1.0 < poisson < 0.5:
        raise ValueError("Poisson ratio is outside the stable isotropic interval")
    factor = modulus / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    stiffness = np.zeros((6, 6), dtype=np.float64)
    normal_diagonal = factor * (1.0 - poisson)
    normal_off_diagonal = factor * poisson
    shear = factor * 0.5 * (1.0 - 2.0 * poisson)
    stiffness[:3, :3] = normal_off_diagonal
    np.fill_diagonal(stiffness[:3, :3], normal_diagonal)
    np.fill_diagonal(stiffness[3:, 3:], shear)
    return stiffness


def von_mises(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    if stress.shape[-1] != 6:
        raise ValueError("stress tensors must use six-component Voigt form")
    sx, sy, sz, txy, tyz, txz = np.moveaxis(stress, -1, 0)
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy**2 + tyz**2 + txz**2)
    )


def effective_modulus(
    strain: NDArray[np.float64], stress_mpa: NDArray[np.float64], maximum_strain: float = 0.01
) -> float:
    selected = (strain >= 0.0) & (strain <= maximum_strain)
    if np.sum(selected) < 2:
        raise ValueError("insufficient elastic-region samples")
    design = np.column_stack((strain[selected], np.ones(np.sum(selected))))
    slope = np.linalg.lstsq(design, stress_mpa[selected], rcond=None)[0][0]
    return float(slope / 1000.0)


def compressive_strength(
    strain: NDArray[np.float64], stress_mpa: NDArray[np.float64]
) -> tuple[float, float]:
    if strain.shape != stress_mpa.shape:
        raise ValueError("strain and stress shapes must match")
    index = int(np.argmax(stress_mpa))
    return float(stress_mpa[index]), float(strain[index])


def gibson_ashby_modulus(solid_modulus_gpa: float, porosity: float, exponent: float = 2.0) -> float:
    relative_density = 1.0 - porosity
    return solid_modulus_gpa * relative_density**exponent


def gibson_ashby_strength(
    solid_strength_mpa: float, porosity: float, exponent: float = 1.5, coefficient: float = 0.3
) -> float:
    relative_density = 1.0 - porosity
    return coefficient * solid_strength_mpa * relative_density**exponent
