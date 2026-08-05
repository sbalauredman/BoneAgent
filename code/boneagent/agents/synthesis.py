from __future__ import annotations

import time
from dataclasses import dataclass

from boneagent.agents.base import Agent, AgentResponse, MessageBus
from boneagent.domain import AgentKind, Candidate


@dataclass(frozen=True)
class SynthesisRequest:
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class SynthesisOperation:
    ordinal: int
    name: str
    temperature_celsius: float | None
    duration_hours: float | None
    atmosphere_oxygen_fraction: float | None
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SynthesisProtocol:
    candidate_id: str
    precursor_molar_ratio: tuple[float, float]
    operations: tuple[SynthesisOperation, ...]
    quality_targets: tuple[tuple[str, float], ...]


class SynthesisAgent(Agent[SynthesisRequest, tuple[SynthesisProtocol, ...]]):
    kind = AgentKind.SYNTHESIS

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(bus)

    def _protocol(self, candidate: Candidate) -> SynthesisProtocol:
        composition = candidate.composition
        processing = candidate.processing
        scaffold = candidate.scaffold
        operations = (
            SynthesisOperation(
                ordinal=1,
                name="precursor_weighing",
                temperature_celsius=None,
                duration_hours=None,
                atmosphere_oxygen_fraction=None,
                details=(
                    ("calcium_phosphorus_ratio", f"{composition.calcium_phosphorus_ratio:.5f}"),
                ),
            ),
            SynthesisOperation(
                ordinal=2,
                name="aqueous_precipitation",
                temperature_celsius=37.0,
                duration_hours=2.0,
                atmosphere_oxygen_fraction=0.21,
                details=(("target_phase_hap", f"{composition.hydroxyapatite_fraction:.5f}"),),
            ),
            SynthesisOperation(
                ordinal=3,
                name="scaffold_forming",
                temperature_celsius=25.0,
                duration_hours=1.0,
                atmosphere_oxygen_fraction=0.21,
                details=(
                    ("porosity", f"{scaffold.porosity:.5f}"),
                    ("pore_diameter_um", f"{scaffold.pore_diameter_micrometer:.3f}"),
                ),
            ),
            SynthesisOperation(
                ordinal=4,
                name="sintering",
                temperature_celsius=processing.sintering_temperature_celsius,
                duration_hours=processing.holding_hours,
                atmosphere_oxygen_fraction=processing.oxygen_fraction,
                details=(("ramp_celsius_per_minute", "5.0"),),
            ),
        )
        return SynthesisProtocol(
            candidate_id=candidate.identifier,
            precursor_molar_ratio=(composition.calcium_phosphorus_ratio, 1.0),
            operations=operations,
            quality_targets=(
                ("porosity", scaffold.porosity),
                ("pore_diameter_um", scaffold.pore_diameter_micrometer),
                ("strut_diameter_um", scaffold.strut_diameter_micrometer),
            ),
        )

    def act(
        self, request: SynthesisRequest, cycle: int
    ) -> AgentResponse[tuple[SynthesisProtocol, ...]]:
        started = time.perf_counter()
        result = tuple(self._protocol(candidate) for candidate in request.candidates)
        self.notify(AgentKind.ORCHESTRATOR, "synthesis_result", cycle, result)
        return AgentResponse(result, (), time.perf_counter() - started)
