from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]


@dataclass(frozen=True)
class AssimilationResult:
    parameters: FloatVector
    covariance: FloatMatrix
    gain: FloatMatrix
    residual: FloatVector


class KalmanAssimilator:
    def __init__(self, parameters: FloatVector, covariance: FloatMatrix) -> None:
        if covariance.shape != (parameters.size, parameters.size):
            raise ValueError("parameter covariance shape is inconsistent")
        self.parameters = parameters.astype(np.float64, copy=True)
        self.covariance = covariance.astype(np.float64, copy=True)

    def update(
        self,
        observations: FloatVector,
        predictions: FloatVector,
        observation_matrix: FloatMatrix,
        observation_covariance: FloatMatrix,
    ) -> AssimilationResult:
        if observations.shape != predictions.shape:
            raise ValueError("observation and prediction shapes must match")
        residual = observations - predictions
        innovation = (
            observation_matrix @ self.covariance @ observation_matrix.T + observation_covariance
        )
        projected = self.covariance @ observation_matrix.T
        gain = np.linalg.solve(innovation.T, projected.T).T
        parameters = self.parameters + gain @ residual
        identity = np.eye(self.parameters.size, dtype=np.float64)
        left = identity - gain @ observation_matrix
        covariance = left @ self.covariance @ left.T + gain @ observation_covariance @ gain.T
        covariance = 0.5 * (covariance + covariance.T)
        self.parameters = parameters
        self.covariance = covariance
        return AssimilationResult(parameters, covariance, gain, residual)

    def neighborhood_correction(
        self,
        active_vectors: FloatMatrix,
        observation_vectors: FloatMatrix,
        residuals: FloatVector,
        bandwidth: float | None = None,
    ) -> FloatVector:
        if observation_vectors.shape[0] != residuals.size:
            raise ValueError("observation count must match residual count")
        if bandwidth is None:
            bandwidth = median_pairwise_distance(observation_vectors)
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        distances = pairwise_squared_distances(active_vectors, observation_vectors)
        weights = np.exp(-distances / (2.0 * bandwidth**2))
        weight_sum = weights.sum(axis=1)
        corrections = weights @ residuals
        return corrections / np.maximum(weight_sum, np.finfo(np.float64).tiny)


def pairwise_squared_distances(left: FloatMatrix, right: FloatMatrix) -> FloatMatrix:
    left_norm = np.sum(np.square(left), axis=1, keepdims=True)
    right_norm = np.sum(np.square(right), axis=1, keepdims=True).T
    distance = left_norm + right_norm - 2.0 * left @ right.T
    return np.maximum(distance, 0.0)


def median_pairwise_distance(values: FloatMatrix) -> float:
    if values.shape[0] < 2:
        return 1.0
    distances = pairwise_squared_distances(values, values)
    upper = distances[np.triu_indices(values.shape[0], k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.median(np.sqrt(positive)))
