from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class PairedComparison:
    statistic: float
    raw_p_value: float
    corrected_p_value: float
    effect_size: float
    significant: bool


def percentile_bootstrap(
    values: FloatArray,
    statistic: Callable[[FloatArray], float] = lambda sample: float(np.mean(sample)),
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 42,
) -> ConfidenceInterval:
    if values.size == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[index] = statistic(sample)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return ConfidenceInterval(statistic(values), float(lower), float(upper), confidence)


def bonferroni(p_values: FloatArray) -> FloatArray:
    return np.minimum(p_values * p_values.size, 1.0)


def holm(p_values: FloatArray) -> FloatArray:
    count = p_values.size
    ordering = np.argsort(p_values)
    sorted_values = p_values[ordering]
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, value in enumerate(sorted_values):
        candidate = min((count - rank) * value, 1.0)
        running = max(running, candidate)
        adjusted[ordering[rank]] = running
    return adjusted


def rank_biserial(left: FloatArray, right: FloatArray) -> float:
    differences = left - right
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = np.sum(ranks[nonzero > 0.0])
    negative = np.sum(ranks[nonzero < 0.0])
    return float((positive - negative) / (positive + negative))


def paired_wilcoxon(
    baseline: FloatArray,
    method: FloatArray,
    comparisons: int,
    alpha: float = 0.05,
) -> PairedComparison:
    if baseline.shape != method.shape:
        raise ValueError("paired arrays must have matching shapes")
    result = stats.wilcoxon(method, baseline, alternative="two-sided", zero_method="wilcox")
    raw = float(result.pvalue)
    corrected = min(raw * comparisons, 1.0)
    return PairedComparison(
        statistic=float(result.statistic),
        raw_p_value=raw,
        corrected_p_value=corrected,
        effect_size=rank_biserial(method, baseline),
        significant=corrected < alpha,
    )


def seed_summary(values: FloatArray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("values must not be empty")
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "median": float(np.median(values)),
    }
