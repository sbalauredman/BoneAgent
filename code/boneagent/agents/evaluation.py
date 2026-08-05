from __future__ import annotations

import time
from dataclasses import dataclass

from boneagent.agents.base import Agent, AgentResponse, MessageBus
from boneagent.agents.simulation import SimulationResult
from boneagent.domain import AgentKind, ConstraintBoundary, Evaluation
from boneagent.science.pareto import ParetoArchive
from boneagent.science.scoring import CompositeScorer


@dataclass(frozen=True)
class EvaluationRequest:
    simulation: SimulationResult
    constraints: tuple[ConstraintBoundary, ...]


@dataclass(frozen=True)
class EvaluationResult:
    evaluations: tuple[Evaluation, ...]
    pareto_front: tuple[Evaluation, ...]
    hypervolume: float


class EvaluationAgent(Agent[EvaluationRequest, EvaluationResult]):
    kind = AgentKind.EVALUATION

    def __init__(
        self,
        bus: MessageBus,
        scorer: CompositeScorer,
        archive: ParetoArchive,
        seed: int = 42,
    ) -> None:
        super().__init__(bus)
        self.scorer = scorer
        self.archive = archive
        self.seed = seed

    def act(self, request: EvaluationRequest, cycle: int) -> AgentResponse[EvaluationResult]:
        started = time.perf_counter()
        evaluations = []
        for candidate, prediction in zip(
            request.simulation.candidates,
            request.simulation.predictions,
            strict=True,
        ):
            values = {
                boundary.name: boundary.evaluate(candidate.vector())
                for boundary in request.constraints
            }
            feasible = all(value <= 0.0 for value in values.values())
            score = self.scorer.evaluate(prediction)
            evaluations.append(
                Evaluation(
                    candidate=candidate,
                    fidelity=request.simulation.fidelity,
                    prediction=prediction,
                    cbmd=score.composite,
                    objective_vector=score.vector(),
                    feasible=feasible,
                    constraint_values=values,
                    cycle=cycle,
                )
            )
        front = self.archive.update(evaluations)
        result = EvaluationResult(
            evaluations=tuple(evaluations),
            pareto_front=tuple(front),
            hypervolume=self.archive.hypervolume(self.seed + cycle),
        )
        self.notify(AgentKind.ORCHESTRATOR, "evaluation_result", cycle, result)
        return AgentResponse(result, (), time.perf_counter() - started)
