from __future__ import annotations

import logging

from boneagent.agents.clinical import ClinicalFeedbackAgent, ClinicalRequest
from boneagent.agents.evaluation import EvaluationAgent, EvaluationRequest
from boneagent.agents.literature import LiteratureAgent, LiteratureRequest
from boneagent.agents.simulation import SimulationAgent, SimulationRequest
from boneagent.agents.synthesis import SynthesisAgent, SynthesisRequest
from boneagent.domain import CampaignSummary, Candidate, ClinicalOutcome, Fidelity
from boneagent.search.space import CandidateSpace
from boneagent.settings import CampaignSettings

logger = logging.getLogger(__name__)


class CampaignEngine:
    def __init__(
        self,
        settings: CampaignSettings,
        literature: LiteratureAgent,
        simulation: SimulationAgent,
        clinical: ClinicalFeedbackAgent,
        evaluation: EvaluationAgent,
        synthesis: SynthesisAgent,
        space: CandidateSpace,
    ) -> None:
        self.settings = settings
        self.literature = literature
        self.simulation = simulation
        self.clinical = clinical
        self.evaluation = evaluation
        self.synthesis = synthesis
        self.space = space

    def _rank(
        self,
        candidates: tuple[Candidate, ...],
        fidelity: Fidelity,
        count: int,
        constraints: tuple,
        cycle: int,
    ) -> tuple[Candidate, ...]:
        simulation = self.simulation.act(SimulationRequest(candidates, fidelity), cycle).value
        result = self.evaluation.act(EvaluationRequest(simulation, constraints), cycle).value
        feasible = [item for item in result.evaluations if item.feasible]
        feasible.sort(key=lambda item: item.cbmd, reverse=True)
        return tuple(item.candidate for item in feasible[:count])

    def run(self, outcomes: tuple[ClinicalOutcome, ...] = ()) -> CampaignSummary:
        cascade = self.settings.cascade
        initial = self.literature.act(
            LiteratureRequest("calcium phosphate bone scaffold", cascade.candidate_count), 0
        ).value
        candidates = initial.candidates
        constraints = tuple()
        history = []
        hypervolumes = []
        stable_cycles = 0
        converged = False
        completed_cycles = 0
        for cycle in range(1, cascade.maximum_cycles + 1):
            completed_cycles = cycle
            if outcomes and (cycle == 1 or cycle % cascade.clinical_interval == 0):
                clinical = self.clinical.act(ClinicalRequest(outcomes, candidates), cycle).value
                constraints = clinical.constraints
            fem_candidates = self._rank(
                candidates,
                Fidelity.ML,
                cascade.fem_count,
                constraints,
                cycle,
            )
            dft_candidates = self._rank(
                fem_candidates,
                Fidelity.FEM,
                cascade.dft_count,
                constraints,
                cycle,
            )
            final_simulation = self.simulation.act(
                SimulationRequest(dft_candidates, Fidelity.DFT), cycle
            ).value
            final_evaluation = self.evaluation.act(
                EvaluationRequest(final_simulation, constraints), cycle
            ).value
            history.extend(final_evaluation.evaluations)
            hypervolumes.append(final_evaluation.hypervolume)
            if len(hypervolumes) > 1:
                previous = hypervolumes[-2]
                change = abs(hypervolumes[-1] - previous) / max(abs(previous), 1e-12)
                stable_cycles = stable_cycles + 1 if change < cascade.convergence_fraction else 0
            if cycle >= cascade.minimum_cycles and stable_cycles >= cascade.convergence_cycles:
                converged = True
                break
            parents = [item.candidate for item in final_evaluation.pareto_front]
            candidates = tuple(self.space.perturb(parents, cascade.candidate_count, cycle + 1))
        front = tuple(self.evaluation.archive.evaluations)
        self.synthesis.act(
            SynthesisRequest(tuple(item.candidate for item in front[:10])), completed_cycles
        )
        return CampaignSummary(
            cycles=completed_cycles,
            evaluations=tuple(history),
            pareto_front=front,
            hypervolume_history=tuple(hypervolumes),
            converged=converged,
        )
