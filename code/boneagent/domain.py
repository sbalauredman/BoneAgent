from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Fidelity(str, Enum):
    ML = "ml"
    FEM = "fem"
    DFT = "dft"
    CLINICAL = "clinical"


class FailureMode(str, Enum):
    SLOW_DEGRADATION = "slow_degradation"
    INSUFFICIENT_OSSEOINTEGRATION = "insufficient_osseointegration"
    MECHANICAL_COLLAPSE = "mechanical_collapse"
    INFLAMMATORY_RESPONSE = "inflammatory_response"
    INFECTION = "infection"
    PARTICLE_MIGRATION = "particle_migration"


class AgentKind(str, Enum):
    LITERATURE = "literature"
    SIMULATION = "simulation"
    SYNTHESIS = "synthesis"
    CLINICAL = "clinical"
    EVALUATION = "evaluation"
    ORCHESTRATOR = "orchestrator"


@dataclass(frozen=True)
class Composition:
    calcium_phosphorus_ratio: float
    hydroxyapatite_fraction: float
    beta_tcp_fraction: float
    dopant_atomic_number: int
    dopant_fraction: float

    def __post_init__(self) -> None:
        if not 1.0 <= self.calcium_phosphorus_ratio <= 2.0:
            raise ValueError("calcium to phosphorus ratio must be within [1, 2]")
        if not 0.0 <= self.hydroxyapatite_fraction <= 1.0:
            raise ValueError("hydroxyapatite fraction must be within [0, 1]")
        if not 0.0 <= self.beta_tcp_fraction <= 1.0:
            raise ValueError("beta TCP fraction must be within [0, 1]")
        if abs(self.hydroxyapatite_fraction + self.beta_tcp_fraction - 1.0) > 1e-6:
            raise ValueError("phase fractions must sum to one")
        if not 0.0 <= self.dopant_fraction <= 0.2:
            raise ValueError("dopant fraction must be within [0, 0.2]")

    def vector(self) -> FloatArray:
        return np.asarray(
            [
                self.calcium_phosphorus_ratio,
                self.hydroxyapatite_fraction,
                self.beta_tcp_fraction,
                float(self.dopant_atomic_number),
                self.dopant_fraction,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Scaffold:
    porosity: float
    pore_diameter_micrometer: float
    strut_diameter_micrometer: float
    interconnectivity: float

    def __post_init__(self) -> None:
        if not 0.3 <= self.porosity <= 0.8:
            raise ValueError("porosity must be within [0.3, 0.8]")
        if not 100.0 <= self.pore_diameter_micrometer <= 600.0:
            raise ValueError("pore diameter must be within [100, 600]")
        if not 50.0 <= self.strut_diameter_micrometer <= 300.0:
            raise ValueError("strut diameter must be within [50, 300]")
        if not 0.0 <= self.interconnectivity <= 1.0:
            raise ValueError("interconnectivity must be within [0, 1]")

    def vector(self) -> FloatArray:
        return np.asarray(
            [
                self.porosity,
                self.pore_diameter_micrometer,
                self.strut_diameter_micrometer,
                self.interconnectivity,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Processing:
    sintering_temperature_celsius: float
    oxygen_fraction: float
    holding_hours: float

    def __post_init__(self) -> None:
        if not 500.0 <= self.sintering_temperature_celsius <= 1600.0:
            raise ValueError("sintering temperature must be within [500, 1600]")
        if not 0.0 <= self.oxygen_fraction <= 1.0:
            raise ValueError("oxygen fraction must be within [0, 1]")
        if not 0.1 <= self.holding_hours <= 24.0:
            raise ValueError("holding time must be within [0.1, 24]")

    def vector(self) -> FloatArray:
        return np.asarray(
            [self.sintering_temperature_celsius, self.oxygen_fraction, self.holding_hours],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Candidate:
    identifier: str
    composition: Composition
    scaffold: Scaffold
    processing: Processing

    def vector(self) -> FloatArray:
        return np.concatenate(
            [self.composition.vector(), self.scaffold.vector(), self.processing.vector()]
        )


@dataclass(frozen=True)
class PropertyPrediction:
    formation_energy_ev_atom: float
    surface_energy_j_m2: float
    dissolution_barrier_ev: float
    compressive_strength_mpa: float
    elastic_modulus_gpa: float
    degradation_weeks: float
    cell_viability: float
    osteogenic_potential: float
    clinical_survival: float
    bone_ingrowth_fraction: float
    uncertainty: FloatArray = field(default_factory=lambda: np.ones(10, dtype=np.float64))

    def objectives(self) -> FloatArray:
        return np.asarray(
            [
                self.compressive_strength_mpa,
                self.elastic_modulus_gpa,
                self.degradation_weeks,
                self.cell_viability,
                self.osteogenic_potential,
                self.clinical_survival,
                self.bone_ingrowth_fraction,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Evaluation:
    candidate: Candidate
    fidelity: Fidelity
    prediction: PropertyPrediction
    cbmd: float
    objective_vector: FloatArray
    feasible: bool
    constraint_values: Mapping[str, float]
    cycle: int


@dataclass(frozen=True)
class ClinicalOutcome:
    candidate_vector: FloatArray
    implant_survival_24m: float
    bone_ingrowth_fraction: float
    revision_rate: float
    followup_months: float
    center: str
    sample_size: int
    failure_modes: Sequence[FailureMode]


@dataclass(frozen=True)
class ConstraintBoundary:
    name: str
    coefficients: FloatArray
    intercept: float
    threshold: float

    def evaluate(self, vector: FloatArray) -> float:
        return float(self.coefficients @ vector + self.intercept - self.threshold)


@dataclass(frozen=True)
class Message:
    sender: AgentKind
    receiver: AgentKind
    message_type: str
    cycle: int
    payload: object
    source_ids: tuple[str, ...]
    timestamp: float


@dataclass(frozen=True)
class CampaignSummary:
    cycles: int
    evaluations: tuple[Evaluation, ...]
    pareto_front: tuple[Evaluation, ...]
    hypervolume_history: tuple[float, ...]
    converged: bool
