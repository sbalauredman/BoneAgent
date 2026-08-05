from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import roc_auc_score

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RegressionMetrics:
    mean_absolute_error: float
    root_mean_squared_error: float
    coefficient_of_determination: float
    spearman_correlation: float


@dataclass(frozen=True)
class ClassificationMetrics:
    area_under_curve: float
    accuracy: float
    sensitivity: float
    specificity: float
    positive_predictive_value: float
    negative_predictive_value: float


def mean_absolute_error(target: FloatArray, prediction: FloatArray) -> float:
    _matching(target, prediction)
    return float(np.mean(np.abs(target - prediction)))


def root_mean_squared_error(target: FloatArray, prediction: FloatArray) -> float:
    _matching(target, prediction)
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def coefficient_of_determination(target: FloatArray, prediction: FloatArray) -> float:
    _matching(target, prediction)
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - np.mean(target)) ** 2)
    if total <= 0.0:
        return 0.0
    return float(1.0 - residual / total)


def spearman_correlation(target: FloatArray, prediction: FloatArray) -> float:
    _matching(target, prediction)
    statistic = stats.spearmanr(target, prediction).statistic
    return float(0.0 if np.isnan(statistic) else statistic)


def regression_metrics(target: FloatArray, prediction: FloatArray) -> RegressionMetrics:
    return RegressionMetrics(
        mean_absolute_error=mean_absolute_error(target, prediction),
        root_mean_squared_error=root_mean_squared_error(target, prediction),
        coefficient_of_determination=coefficient_of_determination(target, prediction),
        spearman_correlation=spearman_correlation(target, prediction),
    )


def classification_metrics(
    target: NDArray[np.int64], probability: FloatArray, threshold: float = 0.5
) -> ClassificationMetrics:
    _matching(target, probability)
    prediction = probability >= threshold
    positive = target == 1
    negative = target == 0
    true_positive = int(np.sum(prediction & positive))
    true_negative = int(np.sum(~prediction & negative))
    false_positive = int(np.sum(prediction & negative))
    false_negative = int(np.sum(~prediction & positive))
    sensitivity = _safe_ratio(true_positive, true_positive + false_negative)
    specificity = _safe_ratio(true_negative, true_negative + false_positive)
    ppv = _safe_ratio(true_positive, true_positive + false_positive)
    npv = _safe_ratio(true_negative, true_negative + false_negative)
    auc = float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else 0.5
    return ClassificationMetrics(
        area_under_curve=auc,
        accuracy=_safe_ratio(true_positive + true_negative, len(target)),
        sensitivity=sensitivity,
        specificity=specificity,
        positive_predictive_value=ppv,
        negative_predictive_value=npv,
    )


def expected_calibration_error(
    target: NDArray[np.int64], probability: FloatArray, bins: int = 15
) -> float:
    _matching(target, probability)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if upper == 1.0:
            selected = (probability >= lower) & (probability <= upper)
        else:
            selected = (probability >= lower) & (probability < upper)
        if not np.any(selected):
            continue
        confidence = float(np.mean(probability[selected]))
        accuracy = float(np.mean(target[selected]))
        error += float(np.mean(selected)) * abs(confidence - accuracy)
    return error


def brier_score(target: NDArray[np.int64], probability: FloatArray) -> float:
    _matching(target, probability)
    return float(np.mean((target - probability) ** 2))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _matching(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError("arrays must have matching shapes")
    if left.size == 0:
        raise ValueError("arrays must not be empty")
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("arrays must contain finite values")
