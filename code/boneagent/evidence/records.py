from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum


class EvidenceGrade(str, Enum):
    RANDOMIZED_TRIAL = "randomized_trial"
    CONTROLLED_TRIAL = "controlled_trial"
    COHORT = "cohort"
    CASE_SERIES = "case_series"
    IN_VIVO = "in_vivo"
    IN_VITRO = "in_vitro"
    COMPUTATIONAL = "computational"


class EndpointKind(str, Enum):
    IMPLANT_SURVIVAL = "implant_survival"
    BONE_INGROWTH = "bone_ingrowth"
    REVISION = "revision"
    ADVERSE_EVENT = "adverse_event"
    CELL_VIABILITY = "cell_viability"
    ALP_ACTIVITY = "alp_activity"
    COMPRESSIVE_STRENGTH = "compressive_strength"
    DEGRADATION = "degradation"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    publication_date: date | None
    source_name: str
    url: str
    license_name: str
    open_access: bool


@dataclass(frozen=True)
class MaterialMention:
    source_id: str
    text: str
    start: int
    end: int
    normalized_formula: str | None
    calcium_phosphorus_ratio: float | None
    hydroxyapatite_fraction: float | None
    beta_tcp_fraction: float | None
    dopants: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class GeometryMention:
    source_id: str
    porosity: float | None
    pore_diameter_micrometer: float | None
    strut_diameter_micrometer: float | None
    interconnectivity: float | None
    confidence: float


@dataclass(frozen=True)
class PopulationRecord:
    source_id: str
    sample_size: int
    mean_age: float | None
    female_fraction: float | None
    defect_site: str | None
    defect_size_cm3: float | None
    country: str | None
    institution: str | None


@dataclass(frozen=True)
class EndpointRecord:
    source_id: str
    endpoint: EndpointKind
    value: float
    unit: str
    followup_months: float | None
    uncertainty: float | None
    sample_size: int | None
    arm_name: str | None


@dataclass(frozen=True)
class EvidenceLink:
    source_id: str
    material_index: int
    endpoint_index: int
    sentence_distance: int
    relation_probability: float
    evidence_grade: EvidenceGrade


@dataclass(frozen=True)
class HarmonizedStudy:
    source: SourceRecord
    materials: Sequence[MaterialMention]
    geometries: Sequence[GeometryMention]
    populations: Sequence[PopulationRecord]
    endpoints: Sequence[EndpointRecord]
    links: Sequence[EvidenceLink]


def validate_offsets(text: str, mention: MaterialMention) -> bool:
    if mention.start < 0 or mention.end > len(text) or mention.start >= mention.end:
        return False
    return text[mention.start : mention.end] == mention.text


def strongest_endpoint(study: HarmonizedStudy, endpoint: EndpointKind) -> EndpointRecord | None:
    matching = [item for item in study.endpoints if item.endpoint == endpoint]
    if not matching:
        return None
    return max(
        matching,
        key=lambda item: (
            item.sample_size or 0,
            item.followup_months or 0.0,
            -(item.uncertainty or 0.0),
        ),
    )
