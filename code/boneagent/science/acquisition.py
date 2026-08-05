from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from boneagent.domain import Fidelity


@dataclass(frozen=True)
class AcquisitionChoice:
    candidate_index: int
    fidelity: Fidelity
    information_gain: float
    cost: float
    value: float


def gaussian_entropy(variance: NDArray[np.float64]) -> NDArray[np.float64]:
    safe = np.maximum(variance, np.finfo(np.float64).tiny)
    return 0.5 * np.log(2.0 * np.pi * np.e * safe)


def posterior_variance(
    prior_variance: NDArray[np.float64], observation_variance: float
) -> NDArray[np.float64]:
    prior_precision = 1.0 / np.maximum(prior_variance, np.finfo(np.float64).tiny)
    observation_precision = 1.0 / max(observation_variance, np.finfo(np.float64).tiny)
    return 1.0 / (prior_precision + observation_precision)


def expected_information_gain(
    prior_variance: NDArray[np.float64], observation_variance: float
) -> NDArray[np.float64]:
    posterior = posterior_variance(prior_variance, observation_variance)
    return gaussian_entropy(prior_variance) - gaussian_entropy(posterior)


class MultiFidelityAcquisition:
    def __init__(self, costs: dict[Fidelity, float], noise: dict[Fidelity, float]) -> None:
        self.costs = costs
        self.noise = noise
        if set(costs) != set(noise):
            raise ValueError("cost and noise fidelity sets must match")
        if any(value <= 0.0 for value in costs.values()):
            raise ValueError("costs must be positive")
        if any(value <= 0.0 for value in noise.values()):
            raise ValueError("noise variances must be positive")

    def score(self, variances: NDArray[np.float64], fidelity: Fidelity) -> NDArray[np.float64]:
        gains = expected_information_gain(variances, self.noise[fidelity])
        return gains / self.costs[fidelity]

    def choices(
        self,
        variances: NDArray[np.float64],
        budget: float,
    ) -> list[AcquisitionChoice]:
        if variances.ndim != 1:
            raise ValueError("variances must be one-dimensional")
        candidates: list[AcquisitionChoice] = []
        for fidelity, cost in self.costs.items():
            if cost > budget:
                continue
            gains = expected_information_gain(variances, self.noise[fidelity])
            values = gains / cost
            for index, (gain, value) in enumerate(zip(gains, values, strict=True)):
                candidates.append(
                    AcquisitionChoice(index, fidelity, float(gain), cost, float(value))
                )
        candidates.sort(key=lambda item: item.value, reverse=True)
        chosen: list[AcquisitionChoice] = []
        remaining = budget
        used: set[tuple[int, Fidelity]] = set()
        for item in candidates:
            key = (item.candidate_index, item.fidelity)
            if key in used or item.cost > remaining:
                continue
            chosen.append(item)
            used.add(key)
            remaining -= item.cost
        return chosen
