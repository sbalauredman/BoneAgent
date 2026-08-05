from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from boneagent.domain import Evaluation, Fidelity

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Throughput:
    fidelity: Fidelity
    candidates: int
    elapsed_seconds: float
    candidates_per_hour: float


@dataclass(frozen=True)
class ConvergenceRecord:
    cycle: int
    hypervolume: float
    relative_change: float
    stable: bool


@dataclass(frozen=True)
class CampaignCost:
    gpu_hours: float
    cpu_hours: float
    wall_hours: float
    estimated_currency_cost: float


@dataclass(frozen=True)
class FidelityAgreement:
    lower_fidelity: Fidelity
    higher_fidelity: Fidelity
    spearman: float
    kendall: float
    top_k_overlap: float
    sample_count: int


def throughput(fidelity: Fidelity, candidates: int, elapsed_seconds: float) -> Throughput:
    if candidates < 0:
        raise ValueError("candidate count must not be negative")
    if elapsed_seconds <= 0.0:
        raise ValueError("elapsed time must be positive")
    return Throughput(
        fidelity=fidelity,
        candidates=candidates,
        elapsed_seconds=elapsed_seconds,
        candidates_per_hour=3600.0 * candidates / elapsed_seconds,
    )


def convergence_records(
    hypervolumes: FloatArray,
    threshold: float = 0.01,
) -> list[ConvergenceRecord]:
    if hypervolumes.ndim != 1:
        raise ValueError("hypervolumes must be one-dimensional")
    records = []
    for index, value in enumerate(hypervolumes):
        if index == 0:
            change = float("inf")
        else:
            change = abs(float(value - hypervolumes[index - 1])) / max(
                abs(float(hypervolumes[index - 1])), 1e-12
            )
        records.append(
            ConvergenceRecord(
                cycle=index + 1,
                hypervolume=float(value),
                relative_change=change,
                stable=change < threshold,
            )
        )
    return records


def converged_cycle(
    hypervolumes: FloatArray,
    threshold: float = 0.01,
    consecutive: int = 2,
) -> int | None:
    run = 0
    for record in convergence_records(hypervolumes, threshold):
        run = run + 1 if record.stable else 0
        if run >= consecutive:
            return record.cycle
    return None


def estimate_campaign_cost(
    gpu_hours: float,
    cpu_hours: float,
    wall_hours: float,
    gpu_hour_price: float,
    cpu_hour_price: float = 0.0,
) -> CampaignCost:
    values = (gpu_hours, cpu_hours, wall_hours, gpu_hour_price, cpu_hour_price)
    if any(value < 0.0 for value in values):
        raise ValueError("cost inputs must not be negative")
    return CampaignCost(
        gpu_hours=gpu_hours,
        cpu_hours=cpu_hours,
        wall_hours=wall_hours,
        estimated_currency_cost=gpu_hours * gpu_hour_price + cpu_hours * cpu_hour_price,
    )


def ranking_agreement(
    lower_identifiers: list[str],
    lower_scores: FloatArray,
    higher_identifiers: list[str],
    higher_scores: FloatArray,
    lower_fidelity: Fidelity,
    higher_fidelity: Fidelity,
    top_k: int = 10,
) -> FidelityAgreement:
    if len(lower_identifiers) != lower_scores.size:
        raise ValueError("lower identifiers and scores must match")
    if len(higher_identifiers) != higher_scores.size:
        raise ValueError("higher identifiers and scores must match")
    lower_map = dict(zip(lower_identifiers, lower_scores, strict=True))
    higher_map = dict(zip(higher_identifiers, higher_scores, strict=True))
    common = sorted(set(lower_map) & set(higher_map))
    if len(common) < 2:
        raise ValueError("at least two shared candidates are required")
    lower = np.asarray([lower_map[identifier] for identifier in common])
    higher = np.asarray([higher_map[identifier] for identifier in common])
    spearman = stats.spearmanr(lower, higher).statistic
    kendall = stats.kendalltau(lower, higher).statistic
    effective_k = min(top_k, len(common))
    lower_top = {common[index] for index in np.argsort(lower)[-effective_k:]}
    higher_top = {common[index] for index in np.argsort(higher)[-effective_k:]}
    overlap = len(lower_top & higher_top) / effective_k
    return FidelityAgreement(
        lower_fidelity=lower_fidelity,
        higher_fidelity=higher_fidelity,
        spearman=float(spearman),
        kendall=float(kendall),
        top_k_overlap=overlap,
        sample_count=len(common),
    )


def evaluations_by_fidelity(
    evaluations: list[Evaluation],
) -> dict[Fidelity, list[Evaluation]]:
    grouped = {fidelity: [] for fidelity in Fidelity}
    for evaluation in evaluations:
        grouped[evaluation.fidelity].append(evaluation)
    return grouped


def best_per_cycle(evaluations: list[Evaluation]) -> dict[int, Evaluation]:
    selected: dict[int, Evaluation] = {}
    for evaluation in evaluations:
        current = selected.get(evaluation.cycle)
        if current is None or evaluation.cbmd > current.cbmd:
            selected[evaluation.cycle] = evaluation
    return selected


def composition_shift(before: list[Evaluation], after: list[Evaluation]) -> dict[str, float]:
    if not before or not after:
        raise ValueError("both evaluation groups are required")
    before_ratios = np.asarray(
        [item.candidate.composition.calcium_phosphorus_ratio for item in before]
    )
    after_ratios = np.asarray(
        [item.candidate.composition.calcium_phosphorus_ratio for item in after]
    )
    result = stats.mannwhitneyu(after_ratios, before_ratios, alternative="two-sided")
    return {
        "before_mean": float(np.mean(before_ratios)),
        "before_standard_deviation": float(np.std(before_ratios, ddof=1)),
        "after_mean": float(np.mean(after_ratios)),
        "after_standard_deviation": float(np.std(after_ratios, ddof=1)),
        "mean_shift": float(np.mean(after_ratios) - np.mean(before_ratios)),
        "mann_whitney_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def failure_fraction(evaluations: list[Evaluation]) -> dict[str, float]:
    totals: dict[str, int] = {}
    violations: dict[str, int] = {}
    for evaluation in evaluations:
        for name, value in evaluation.constraint_values.items():
            totals[name] = totals.get(name, 0) + 1
            violations[name] = violations.get(name, 0) + int(value > 0.0)
    return {name: violations[name] / count for name, count in totals.items() if count > 0}
