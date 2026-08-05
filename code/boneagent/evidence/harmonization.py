from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CenterObservations:
    name: str
    values: FloatArray
    standard_errors: FloatArray
    sample_sizes: NDArray[np.int64]


@dataclass(frozen=True)
class HarmonizationEstimate:
    global_mean: float
    global_standard_deviation: float
    center_means: dict[str, float]
    center_standard_deviations: dict[str, float]
    heterogeneity: float
    effective_sample_size: float


class HierarchicalHarmonizer:
    def __init__(
        self,
        prior_mean: float = 0.5,
        prior_standard_deviation: float = 0.25,
        between_center_scale: float = 0.1,
    ) -> None:
        self.prior_mean = prior_mean
        self.prior_variance = prior_standard_deviation**2
        self.between_center_scale = between_center_scale

    def _center_estimate(self, observations: CenterObservations) -> tuple[float, float]:
        variance = np.maximum(np.square(observations.standard_errors), 1e-10)
        precision = 1.0 / variance
        mean = float(np.sum(precision * observations.values) / np.sum(precision))
        standard_error = float(np.sqrt(1.0 / np.sum(precision)))
        return mean, standard_error

    def fit(self, centers: list[CenterObservations]) -> HarmonizationEstimate:
        if not centers:
            raise ValueError("at least one center is required")
        estimates = {center.name: self._center_estimate(center) for center in centers}
        means = np.asarray([estimates[center.name][0] for center in centers])
        errors = np.asarray([estimates[center.name][1] for center in centers])
        fixed_weights = 1.0 / np.maximum(np.square(errors), 1e-10)
        fixed_mean = np.sum(fixed_weights * means) / np.sum(fixed_weights)
        q = float(np.sum(fixed_weights * np.square(means - fixed_mean)))
        degrees = max(len(centers) - 1, 1)
        correction = np.sum(fixed_weights) - np.sum(np.square(fixed_weights)) / np.sum(
            fixed_weights
        )
        tau_square = max(0.0, (q - degrees) / max(correction, 1e-10))
        tau_square = max(tau_square, self.between_center_scale**2)
        random_weights = 1.0 / (np.square(errors) + tau_square)
        data_precision = float(np.sum(random_weights))
        prior_precision = 1.0 / self.prior_variance
        posterior_variance = 1.0 / (data_precision + prior_precision)
        posterior_mean = posterior_variance * (
            np.sum(random_weights * means) + prior_precision * self.prior_mean
        )
        posterior_sd = float(np.sqrt(posterior_variance))
        center_means = {}
        center_sds = {}
        for center, raw_mean, error in zip(centers, means, errors, strict=True):
            center_precision = 1.0 / max(error**2, 1e-10)
            population_precision = 1.0 / tau_square
            variance = 1.0 / (center_precision + population_precision)
            mean = variance * (center_precision * raw_mean + population_precision * posterior_mean)
            center_means[center.name] = float(mean)
            center_sds[center.name] = float(np.sqrt(variance))
        i_squared = max(0.0, (q - degrees) / max(q, 1e-10))
        effective = float(np.sum(random_weights) ** 2 / np.sum(np.square(random_weights)))
        return HarmonizationEstimate(
            global_mean=float(posterior_mean),
            global_standard_deviation=posterior_sd,
            center_means=center_means,
            center_standard_deviations=center_sds,
            heterogeneity=i_squared,
            effective_sample_size=effective,
        )


def virtual_center(country: str | None, source: str) -> str:
    if source.lower() == "clinicaltrials.gov":
        return "ClinicalTrials.gov"
    if country is None:
        return "PubMed Americas"
    normalized = country.strip().lower()
    europe = {
        "austria",
        "belgium",
        "denmark",
        "finland",
        "france",
        "germany",
        "greece",
        "ireland",
        "italy",
        "netherlands",
        "norway",
        "poland",
        "portugal",
        "spain",
        "sweden",
        "switzerland",
        "united kingdom",
    }
    asia_pacific = {
        "australia",
        "china",
        "india",
        "indonesia",
        "japan",
        "malaysia",
        "new zealand",
        "singapore",
        "south korea",
        "taiwan",
        "thailand",
    }
    if normalized in europe:
        return "PubMed Europe"
    if normalized in asia_pacific:
        return "PubMed Asia-Pacific"
    return "PubMed Americas"
