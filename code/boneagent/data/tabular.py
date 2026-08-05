from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True)
class TabularRecord:
    identifier: str
    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    target_mask: NDArray[np.float32]
    element_set: frozenset[int]
    source: str


class MaterialDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, records: Sequence[TabularRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        return (
            torch.from_numpy(record.features),
            torch.from_numpy(record.targets),
            torch.from_numpy(record.target_mask),
        )


def parse_float(value: str | None) -> tuple[float, float]:
    if value is None or not value.strip() or value.strip().lower() in {"na", "nan", "null"}:
        return 0.0, 0.0
    return float(value), 1.0


def read_tabular_records(
    path: str | Path,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    element_column: str = "atomic_numbers",
    identifier_column: str = "id",
    source_column: str = "source",
) -> list[TabularRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row_number, row in enumerate(reader, start=2):
            try:
                features = np.asarray([float(row[column]) for column in feature_columns])
                parsed_targets = [parse_float(row.get(column)) for column in target_columns]
                targets = np.asarray([item[0] for item in parsed_targets], dtype=np.float32)
                mask = np.asarray([item[1] for item in parsed_targets], dtype=np.float32)
                elements = frozenset(int(value) for value in row[element_column].split(";"))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid record at row {row_number}") from error
            records.append(
                TabularRecord(
                    identifier=row.get(identifier_column, str(row_number)),
                    features=features.astype(np.float32),
                    targets=targets,
                    target_mask=mask,
                    element_set=elements,
                    source=row.get(source_column, "unknown"),
                )
            )
    return records


def feature_statistics(records: Sequence[TabularRecord]) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("records must not be empty")
    matrix = np.stack([record.features for record in records])
    mean = matrix.mean(axis=0)
    standard_deviation = matrix.std(axis=0)
    standard_deviation = np.where(standard_deviation > 1e-8, standard_deviation, 1.0)
    return mean, standard_deviation


def normalized_records(
    records: Sequence[TabularRecord], mean: np.ndarray, standard_deviation: np.ndarray
) -> list[TabularRecord]:
    return [
        TabularRecord(
            identifier=record.identifier,
            features=((record.features - mean) / standard_deviation).astype(np.float32),
            targets=record.targets,
            target_mask=record.target_mask,
            element_set=record.element_set,
            source=record.source,
        )
        for record in records
    ]


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(paths: Sequence[str | Path]) -> list[dict[str, object]]:
    entries = []
    for raw_path in sorted(Path(path) for path in paths):
        stat = raw_path.stat()
        entries.append(
            {
                "name": raw_path.name,
                "size": stat.st_size,
                "sha256": file_sha256(raw_path),
            }
        )
    return entries


def write_manifest(paths: Sequence[str | Path], destination: str | Path) -> None:
    target = Path(destination)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest(paths), indent=2), encoding="utf-8")
    temporary.replace(target)


def batches(records: Sequence[TabularRecord], batch_size: int) -> Iterator[Sequence[TabularRecord]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]
