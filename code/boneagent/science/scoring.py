from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from boneagent.domain import PropertyPrediction
from boneagent.settings import ScoreSettings


@dataclass(frozen=True)
class ScoreBreakdown:
    mechanics: float
    biology: float
    degradation: float
    clinical: float
    composite: float

    def vector(self) -> NDArray[np.float64]:
        return np.asarray(
            [self.mechanics, self.biology, self.degradation, self.clinical],
            dtype=np.float64,
        )


def minmax(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("upper bound must exceed lower bound")
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def interval_gaussian(
    value: float,
    lower: float,
    upper: float,
    standard_deviation: float | None = None,
) -> float:
    if lower > upper:
        raise ValueError("lower endpoint must not exceed upper endpoint")
    if lower <= value <= upper:
        return 1.0
    center = 0.5 * (lower + upper)
    sigma = standard_deviation or 0.5 * (upper - lower)
    if sigma <= 0.0:
        raise ValueError("standard deviation must be positive")
    distance = min(abs(value - lower), abs(value - upper))
    center_distance = abs(value - center)
    effective_distance = max(distance, center_distance - 0.5 * (upper - lower))
    return float(np.exp(-0.5 * (effective_distance / sigma) ** 2))


class CompositeScorer:
    def __init__(self, settings: ScoreSettings) -> None:
        self.settings = settings
        total = (
            settings.mechanics_weight
            + settings.biology_weight
            + settings.degradation_weight
            + settings.clinical_weight
        )
        if not np.isclose(total, 1.0):
            raise ValueError("CBMD weights must sum to one")

    def mechanics(self, prediction: PropertyPrediction) -> float:
        strength = minmax(
            prediction.compressive_strength_mpa,
            self.settings.strength_min_mpa,
            self.settings.strength_max_mpa,
        )
        modulus = minmax(
            prediction.elastic_modulus_gpa,
            self.settings.modulus_min_gpa,
            self.settings.modulus_max_gpa,
        )
        return 0.5 * (strength + modulus)

    def biology(self, prediction: PropertyPrediction) -> float:
        viability = float(np.clip(prediction.cell_viability, 0.0, 1.0))
        osteogenic = float(np.clip(prediction.osteogenic_potential, 0.0, 1.0))
        return 0.5 * (viability + osteogenic)

    def degradation(self, prediction: PropertyPrediction) -> float:
        return interval_gaussian(
            prediction.degradation_weeks,
            self.settings.ingrowth_start_weeks,
            self.settings.ingrowth_end_weeks,
        )

    def clinical(self, prediction: PropertyPrediction) -> float:
        survival = float(np.clip(prediction.clinical_survival, 0.0, 1.0))
        ingrowth = float(np.clip(prediction.bone_ingrowth_fraction, 0.0, 1.0))
        return 0.5 * (survival + ingrowth)

    def evaluate(self, prediction: PropertyPrediction) -> ScoreBreakdown:
        mechanics = self.mechanics(prediction)
        biology = self.biology(prediction)
        degradation = self.degradation(prediction)
        clinical = self.clinical(prediction)
        composite = (
            self.settings.mechanics_weight * mechanics
            + self.settings.biology_weight * biology
            + self.settings.degradation_weight * degradation
            + self.settings.clinical_weight * clinical
        )
        return ScoreBreakdown(
            mechanics=mechanics,
            biology=biology,
            degradation=degradation,
            clinical=clinical,
            composite=composite,
        )


def score_matrix(
    predictions: list[PropertyPrediction], settings: ScoreSettings
) -> NDArray[np.float64]:
    scorer = CompositeScorer(settings)
    if not predictions:
        return np.empty((0, 5), dtype=np.float64)
    return np.asarray(
        [
            [
                result.mechanics,
                result.biology,
                result.degradation,
                result.clinical,
                result.composite,
            ]
            for result in (scorer.evaluate(prediction) for prediction in predictions)
        ],
        dtype=np.float64,
    )
