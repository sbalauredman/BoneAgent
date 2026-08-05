from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DataSplit:
    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]

    def validate(self, sample_count: int) -> None:
        combined = np.concatenate((self.train, self.validation, self.test))
        if combined.size != sample_count:
            raise ValueError("split does not cover every sample")
        if np.unique(combined).size != sample_count:
            raise ValueError("split indices overlap")
        if np.any(combined < 0) or np.any(combined >= sample_count):
            raise ValueError("split index is out of range")


def random_split(
    sample_count: int,
    seed: int,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> DataSplit:
    if sample_count < 3:
        raise ValueError("at least three samples are required")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(sample_count)
    train_end = int(np.floor(fractions[0] * sample_count))
    validation_end = train_end + int(np.floor(fractions[1] * sample_count))
    split = DataSplit(
        train=np.sort(order[:train_end]),
        validation=np.sort(order[train_end:validation_end]),
        test=np.sort(order[validation_end:]),
    )
    split.validate(sample_count)
    return split


def elements_out_split(
    element_sets: Sequence[frozenset[int]],
    seed: int,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> DataSplit:
    sample_count = len(element_sets)
    if sample_count < 3:
        raise ValueError("at least three samples are required")
    frequencies: dict[int, int] = {}
    for elements in element_sets:
        for element in elements:
            frequencies[element] = frequencies.get(element, 0) + 1
    rare_order = sorted(frequencies, key=lambda element: (frequencies[element], element))
    target_test = max(1, int(round(test_fraction * sample_count)))
    test_indices: set[int] = set()
    held_elements: set[int] = set()
    for element in rare_order:
        matching = {index for index, values in enumerate(element_sets) if element in values}
        if not matching or len(test_indices | matching) > max(target_test * 2, target_test + 1):
            continue
        test_indices |= matching
        held_elements.add(element)
        if len(test_indices) >= target_test:
            break
    if not test_indices:
        raise ValueError("no element combination can be held out")
    remainder = np.asarray(
        [index for index in range(sample_count) if index not in test_indices],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(remainder)
    validation_count = max(1, int(round(validation_fraction * sample_count)))
    validation = np.sort(remainder[:validation_count])
    train = np.sort(remainder[validation_count:])
    test = np.asarray(sorted(test_indices), dtype=np.int64)
    split = DataSplit(train=train, validation=validation, test=test)
    split.validate(sample_count)
    train_elements = set().union(*(element_sets[index] for index in train))
    if not any(element not in train_elements for element in held_elements):
        raise RuntimeError("elements-out condition was not achieved")
    return split


def grouped_split(
    groups: Sequence[Hashable],
    seed: int,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> DataSplit:
    unique = list(dict.fromkeys(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_end = int(np.floor(fractions[0] * len(unique)))
    validation_end = train_end + int(np.floor(fractions[1] * len(unique)))
    train_groups = set(unique[:train_end])
    validation_groups = set(unique[train_end:validation_end])
    test_groups = set(unique[validation_end:])
    split = DataSplit(
        train=np.asarray(
            [index for index, group in enumerate(groups) if group in train_groups],
            dtype=np.int64,
        ),
        validation=np.asarray(
            [index for index, group in enumerate(groups) if group in validation_groups],
            dtype=np.int64,
        ),
        test=np.asarray(
            [index for index, group in enumerate(groups) if group in test_groups],
            dtype=np.int64,
        ),
    )
    split.validate(len(groups))
    return split
