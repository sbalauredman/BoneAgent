from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from boneagent.agents.base import Agent, AgentResponse, MessageBus
from boneagent.domain import AgentKind, Candidate, Fidelity, PropertyPrediction


class PropertyBackend(Protocol):
    def predict(
        self, candidates: list[Candidate], fidelity: Fidelity
    ) -> list[PropertyPrediction]: ...


@dataclass(frozen=True)
class SimulationRequest:
    candidates: tuple[Candidate, ...]
    fidelity: Fidelity


@dataclass(frozen=True)
class SimulationResult:
    candidates: tuple[Candidate, ...]
    predictions: tuple[PropertyPrediction, ...]
    fidelity: Fidelity


class AnalyticPropertyBackend:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = {
            Fidelity.ML: 0.045,
            Fidelity.FEM: 0.018,
            Fidelity.DFT: 0.006,
            Fidelity.CLINICAL: 0.025,
        }

    def _prediction(self, candidate: Candidate, fidelity: Fidelity) -> PropertyPrediction:
        composition = candidate.composition
        scaffold = candidate.scaffold
        processing = candidate.processing
        ratio = composition.calcium_phosphorus_ratio
        hap = composition.hydroxyapatite_fraction
        porosity = scaffold.porosity
        pore = scaffold.pore_diameter_micrometer
        temperature = processing.sintering_temperature_celsius
        deviation = ratio - 1.56
        densification = 1.0 / (1.0 + np.exp(-(temperature - 1000.0) / 120.0))
        strength = 180.0 * (1.0 - porosity) ** 2.2 * (0.75 + 0.25 * hap) * densification
        modulus = 20.0 * (1.0 - porosity) ** 2.0 * (0.7 + 0.3 * hap)
        degradation = 17.0 + 32.0 * (ratio - 1.5) + 8.0 * hap - 5.0 * porosity
        viability = 0.92 - 0.7 * deviation**2 - 0.1 * composition.dopant_fraction
        osteogenic = 0.86 - 0.9 * deviation**2 + 0.08 * scaffold.interconnectivity
        survival = 0.88 - 0.45 * abs(degradation - 18.0) / 18.0
        ingrowth = 0.35 + 0.45 * porosity + 0.12 * scaffold.interconnectivity
        formation = -3.1 - 0.8 * hap + 0.4 * deviation**2
        surface = 0.7 + 0.5 * abs(deviation) + 0.1 * composition.dopant_fraction
        barrier = 0.4 + 0.9 * hap + 0.2 * (pore / 600.0)
        scale = self.noise[fidelity]
        perturbation = self.rng.normal(0.0, scale, 10)
        uncertainty = np.full(10, scale**2, dtype=np.float64)
        return PropertyPrediction(
            formation_energy_ev_atom=float(formation + perturbation[0]),
            surface_energy_j_m2=float(max(surface + perturbation[1], 0.0)),
            dissolution_barrier_ev=float(max(barrier + perturbation[2], 0.0)),
            compressive_strength_mpa=float(max(strength * (1.0 + perturbation[3]), 0.0)),
            elastic_modulus_gpa=float(max(modulus * (1.0 + perturbation[4]), 0.0)),
            degradation_weeks=float(max(degradation * (1.0 + perturbation[5]), 0.1)),
            cell_viability=float(np.clip(viability + perturbation[6], 0.0, 1.0)),
            osteogenic_potential=float(np.clip(osteogenic + perturbation[7], 0.0, 1.0)),
            clinical_survival=float(np.clip(survival + perturbation[8], 0.0, 1.0)),
            bone_ingrowth_fraction=float(np.clip(ingrowth + perturbation[9], 0.0, 1.0)),
            uncertainty=uncertainty,
        )

    def predict(self, candidates: list[Candidate], fidelity: Fidelity) -> list[PropertyPrediction]:
        return [self._prediction(candidate, fidelity) for candidate in candidates]


class SimulationAgent(Agent[SimulationRequest, SimulationResult]):
    kind = AgentKind.SIMULATION

    def __init__(self, bus: MessageBus, backend: PropertyBackend) -> None:
        super().__init__(bus)
        self.backend = backend

    def act(self, request: SimulationRequest, cycle: int) -> AgentResponse[SimulationResult]:
        started = time.perf_counter()
        candidates = list(request.candidates)
        predictions = self.backend.predict(candidates, request.fidelity)
        if len(predictions) != len(candidates):
            raise RuntimeError("property backend returned an inconsistent prediction count")
        result = SimulationResult(
            candidates=request.candidates,
            predictions=tuple(predictions),
            fidelity=request.fidelity,
        )
        self.notify(AgentKind.EVALUATION, "simulation_result", cycle, result)
        return AgentResponse(result, (), time.perf_counter() - started)
