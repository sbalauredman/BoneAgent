from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MaceSettings:
    hidden_channels: int = 128
    message_layers: int = 3
    max_ell: int = 2
    correlation: int = 3
    radial_functions: int = 8
    cutoff_angstrom: float = 5.0
    pretrain_epochs: int = 500
    finetune_epochs: int = 200
    pretrain_learning_rate: float = 5e-4
    finetune_learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    gradient_clip: float = 10.0
    early_stopping_patience: int = 20


@dataclass(frozen=True)
class CascadeSettings:
    candidate_count: int = 10000
    fem_count: int = 100
    dft_count: int = 10
    maximum_cycles: int = 10
    minimum_cycles: int = 2
    convergence_fraction: float = 0.01
    convergence_cycles: int = 2
    clinical_interval: int = 5
    cost_ml: float = 1.0
    cost_fem: float = 3e4
    cost_dft: float = 1.4e6


@dataclass(frozen=True)
class ScoreSettings:
    mechanics_weight: float = 0.3
    biology_weight: float = 0.3
    degradation_weight: float = 0.2
    clinical_weight: float = 0.2
    strength_min_mpa: float = 2.0
    strength_max_mpa: float = 180.0
    modulus_min_gpa: float = 0.1
    modulus_max_gpa: float = 20.0
    ingrowth_start_weeks: float = 12.0
    ingrowth_end_weeks: float = 24.0


@dataclass(frozen=True)
class InfrastructureSettings:
    gpu_model: str = "NVIDIA A100"
    gpu_count: int = 1
    gpu_memory_gb: int = 80
    coordinator_cpu_cores: int = 128
    coordinator_memory_gb: int = 512
    dft_nodes: int = 4
    dft_cores_per_node: int = 64
    campaign_hours: float = 72.0


@dataclass(frozen=True)
class CampaignSettings:
    seed: int = 42
    output_directory: str = "records"
    mace: MaceSettings = field(default_factory=MaceSettings)
    cascade: CascadeSettings = field(default_factory=CascadeSettings)
    score: ScoreSettings = field(default_factory=ScoreSettings)
    infrastructure: InfrastructureSettings = field(default_factory=InfrastructureSettings)


def _construct_settings(values: dict[str, Any]) -> CampaignSettings:
    return CampaignSettings(
        seed=int(values.get("seed", 42)),
        output_directory=str(values.get("output_directory", "records")),
        mace=MaceSettings(**values.get("mace", {})),
        cascade=CascadeSettings(**values.get("cascade", {})),
        score=ScoreSettings(**values.get("score", {})),
        infrastructure=InfrastructureSettings(**values.get("infrastructure", {})),
    )


def load_settings(path: str | Path) -> CampaignSettings:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    return _construct_settings(values)
