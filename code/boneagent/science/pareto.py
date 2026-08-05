from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from boneagent.domain import Evaluation


def dominates(left: NDArray[np.float64], right: NDArray[np.float64]) -> bool:
    if left.shape != right.shape:
        raise ValueError("objective vectors must have matching shapes")
    return bool(np.all(left >= right) and np.any(left > right))


def non_dominated_indices(values: NDArray[np.float64]) -> NDArray[np.int64]:
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    count = values.shape[0]
    selected = np.ones(count, dtype=bool)
    for index in range(count):
        if not selected[index]:
            continue
        for other in range(count):
            if index == other:
                continue
            if dominates(values[other], values[index]):
                selected[index] = False
                break
    return np.flatnonzero(selected).astype(np.int64)


def fast_non_dominated_sort(values: NDArray[np.float64]) -> list[NDArray[np.int64]]:
    count = values.shape[0]
    domination_count = np.zeros(count, dtype=np.int64)
    dominated: list[list[int]] = [[] for _ in range(count)]
    fronts: list[list[int]] = [[]]
    for left in range(count):
        for right in range(count):
            if left == right:
                continue
            if dominates(values[left], values[right]):
                dominated[left].append(right)
            elif dominates(values[right], values[left]):
                domination_count[left] += 1
        if domination_count[left] == 0:
            fronts[0].append(left)
    position = 0
    while position < len(fronts) and fronts[position]:
        next_front: list[int] = []
        for left in fronts[position]:
            for right in dominated[left]:
                domination_count[right] -= 1
                if domination_count[right] == 0:
                    next_front.append(right)
        if next_front:
            fronts.append(next_front)
        position += 1
    return [np.asarray(front, dtype=np.int64) for front in fronts if front]


def crowding_distance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    count, objectives = values.shape
    distance = np.zeros(count, dtype=np.float64)
    if count <= 2:
        distance[:] = np.inf
        return distance
    for objective in range(objectives):
        ordering = np.argsort(values[:, objective])
        distance[ordering[0]] = np.inf
        distance[ordering[-1]] = np.inf
        span = values[ordering[-1], objective] - values[ordering[0], objective]
        if span <= 0.0:
            continue
        for position in range(1, count - 1):
            lower = values[ordering[position - 1], objective]
            upper = values[ordering[position + 1], objective]
            distance[ordering[position]] += (upper - lower) / span
    return distance


def select_nsga2(values: NDArray[np.float64], count: int) -> NDArray[np.int64]:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    chosen: list[int] = []
    for front in fast_non_dominated_sort(values):
        remaining = count - len(chosen)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            chosen.extend(front.tolist())
            continue
        distances = crowding_distance(values[front])
        order = np.argsort(-distances, kind="stable")
        chosen.extend(front[order[:remaining]].tolist())
    return np.asarray(chosen, dtype=np.int64)


def normalize_objectives(values: NDArray[np.float64]) -> NDArray[np.float64]:
    minima = np.min(values, axis=0)
    maxima = np.max(values, axis=0)
    span = np.where(maxima > minima, maxima - minima, 1.0)
    return (values - minima) / span


def hypervolume_monte_carlo(
    values: NDArray[np.float64],
    reference: NDArray[np.float64] | None = None,
    samples: int = 100000,
    seed: int = 42,
) -> float:
    if values.ndim != 2 or values.shape[0] == 0:
        return 0.0
    ref = np.zeros(values.shape[1], dtype=np.float64) if reference is None else reference
    upper = np.max(values, axis=0)
    span = upper - ref
    if np.any(span <= 0.0):
        return 0.0
    rng = np.random.default_rng(seed)
    points = rng.uniform(ref, upper, size=(samples, values.shape[1]))
    dominated = np.zeros(samples, dtype=bool)
    for value in values:
        dominated |= np.all(value >= points, axis=1)
    return float(np.mean(dominated) * np.prod(span))


@dataclass
class ParetoArchive:
    evaluations: list[Evaluation]
    maximum_size: int = 1000

    def update(self, additions: list[Evaluation]) -> list[Evaluation]:
        feasible = [item for item in self.evaluations + additions if item.feasible]
        if not feasible:
            self.evaluations = []
            return []
        matrix = np.asarray([item.objective_vector for item in feasible], dtype=np.float64)
        indices = non_dominated_indices(matrix)
        front = [feasible[int(index)] for index in indices]
        if len(front) > self.maximum_size:
            front_matrix = np.asarray([item.objective_vector for item in front])
            selected = select_nsga2(front_matrix, self.maximum_size)
            front = [front[int(index)] for index in selected]
        self.evaluations = front
        return list(front)

    def hypervolume(self, seed: int = 42) -> float:
        if not self.evaluations:
            return 0.0
        matrix = np.asarray([item.objective_vector for item in self.evaluations])
        return hypervolume_monte_carlo(matrix, seed=seed)
