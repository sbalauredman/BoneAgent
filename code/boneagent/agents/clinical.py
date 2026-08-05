from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from boneagent.agents.base import Agent, AgentResponse, MessageBus
from boneagent.domain import (
    AgentKind,
    Candidate,
    ClinicalOutcome,
    ConstraintBoundary,
    FailureMode,
)


@dataclass(frozen=True)
class ClinicalRequest:
    outcomes: tuple[ClinicalOutcome, ...]
    active_candidates: tuple[Candidate, ...]
    minimum_failures: int = 5


@dataclass(frozen=True)
class ClinicalResult:
    constraints: tuple[ConstraintBoundary, ...]
    center_scores: dict[str, tuple[float, float]]
    failure_prevalence: dict[FailureMode, float]


class ClinicalFeedbackAgent(Agent[ClinicalRequest, ClinicalResult]):
    kind = AgentKind.CLINICAL

    def __init__(self, bus: MessageBus, regularization: float = 1e-3) -> None:
        super().__init__(bus)
        self.regularization = regularization

    def _failure_prevalence(
        self, outcomes: tuple[ClinicalOutcome, ...]
    ) -> dict[FailureMode, float]:
        weighted = {mode: 0.0 for mode in FailureMode}
        denominator = 0.0
        for outcome in outcomes:
            weight = float(max(outcome.sample_size, 1))
            denominator += weight
            for mode in outcome.failure_modes:
                weighted[mode] += weight
        if denominator == 0.0:
            return weighted
        return {mode: count / denominator for mode, count in weighted.items()}

    def _center_scores(
        self, outcomes: tuple[ClinicalOutcome, ...]
    ) -> dict[str, tuple[float, float]]:
        grouped: dict[str, list[tuple[float, float]]] = {}
        for outcome in outcomes:
            score = 0.5 * (outcome.implant_survival_24m + outcome.bone_ingrowth_fraction)
            grouped.setdefault(outcome.center, []).append((score, float(outcome.sample_size)))
        estimates = {}
        for center, observations in grouped.items():
            values = np.asarray([item[0] for item in observations], dtype=np.float64)
            weights = np.asarray([max(item[1], 1.0) for item in observations])
            mean = float(np.average(values, weights=weights))
            variance = float(np.average((values - mean) ** 2, weights=weights))
            estimates[center] = (mean, np.sqrt(variance / max(len(values), 1)))
        return estimates

    def _fit_boundary(
        self,
        outcomes: tuple[ClinicalOutcome, ...],
        mode: FailureMode,
        minimum_failures: int,
    ) -> ConstraintBoundary | None:
        if not outcomes:
            return None
        labels = np.asarray([mode in item.failure_modes for item in outcomes], dtype=np.float64)
        if int(labels.sum()) < minimum_failures or int((1.0 - labels).sum()) < minimum_failures:
            return None
        features = np.asarray([item.candidate_vector for item in outcomes], dtype=np.float64)
        center = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        normalized = (features - center) / scale
        design = np.column_stack((normalized, np.ones(normalized.shape[0])))
        weights = np.zeros(design.shape[1], dtype=np.float64)
        for _ in range(100):
            logits = design @ weights
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            gradient = design.T @ (probability - labels)
            curvature = probability * (1.0 - probability)
            hessian = design.T @ (curvature[:, None] * design)
            hessian += self.regularization * np.eye(hessian.shape[0])
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if np.linalg.norm(step) < 1e-8:
                break
        coefficients = weights[:-1] / scale
        intercept = float(weights[-1] - coefficients @ center)
        return ConstraintBoundary(
            name=mode.value,
            coefficients=coefficients,
            intercept=intercept,
            threshold=0.0,
        )

    def act(self, request: ClinicalRequest, cycle: int) -> AgentResponse[ClinicalResult]:
        started = time.perf_counter()
        constraints = []
        for mode in FailureMode:
            boundary = self._fit_boundary(request.outcomes, mode, request.minimum_failures)
            if boundary is not None:
                constraints.append(boundary)
        result = ClinicalResult(
            constraints=tuple(constraints),
            center_scores=self._center_scores(request.outcomes),
            failure_prevalence=self._failure_prevalence(request.outcomes),
        )
        self.notify(AgentKind.ORCHESTRATOR, "clinical_result", cycle, result)
        return AgentResponse(result, (), time.perf_counter() - started)
