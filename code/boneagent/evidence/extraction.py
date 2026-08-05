from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from boneagent.evidence.normalization import canonical_phase, parse_numeric_expression
from boneagent.evidence.records import EndpointKind


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Entity:
    label: str
    text: str
    start: int
    end: int
    normalized_value: str | float | None
    confidence: float


@dataclass(frozen=True)
class Relation:
    subject_index: int
    object_index: int
    relation: str
    probability: float
    sentence_distance: int


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
FORMULA_PATTERN = re.compile(
    r"(?:Ca(?:\d+(?:\.\d+)?)?)?(?:\(PO4\)|PO4)(?:\d+(?:\.\d+)?)?(?:OH|H2O)?",
    re.IGNORECASE,
)
RATIO_PATTERN = re.compile(
    r"Ca\s*/\s*P\s*(?:ratio)?\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE
)
PERCENT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
PRESSURE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(MPa|GPa|kPa)", re.IGNORECASE)
LENGTH_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(nm|[µμu]m|mm)", re.IGNORECASE)
FOLLOWUP_PATTERN = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(day|week|month|year)s?\s*(?:follow[- ]?up)?",
    re.IGNORECASE,
)


def sentences(text: str) -> list[Sentence]:
    results = []
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        value = text[start:end].strip()
        leading = len(text[start:end]) - len(text[start:end].lstrip())
        if value:
            results.append(Sentence(value, start + leading, end))
        start = match.end()
    value = text[start:].strip()
    leading = len(text[start:]) - len(text[start:].lstrip())
    if value:
        results.append(Sentence(value, start + leading, len(text)))
    return results


def _entities_from_pattern(
    text: str,
    pattern: re.Pattern[str],
    label: str,
    value_group: int | None = None,
) -> list[Entity]:
    entities = []
    for match in pattern.finditer(text):
        normalized: str | float | None = None
        if value_group is not None:
            parsed = parse_numeric_expression(match.group(value_group))
            normalized = parsed[0] if parsed else None
        entities.append(Entity(label, match.group(0), match.start(), match.end(), normalized, 1.0))
    return entities


def dictionary_entities(text: str) -> list[Entity]:
    entities = []
    phase_terms = (
        "hydroxyapatite",
        "HAp",
        "β-tricalcium phosphate",
        "beta-tricalcium phosphate",
        "β-TCP",
        "octacalcium phosphate",
        "brushite",
        "biphasic calcium phosphate",
    )
    for term in phase_terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            entities.append(
                Entity(
                    "material_phase",
                    match.group(0),
                    match.start(),
                    match.end(),
                    canonical_phase(match.group(0)),
                    1.0,
                )
            )
    entities.extend(_entities_from_pattern(text, FORMULA_PATTERN, "composition_formula"))
    entities.extend(_entities_from_pattern(text, RATIO_PATTERN, "calcium_phosphorus_ratio", 1))
    entities.extend(_entities_from_pattern(text, PRESSURE_PATTERN, "mechanical_value", 1))
    entities.extend(_entities_from_pattern(text, LENGTH_PATTERN, "geometry_value", 1))
    entities.extend(_entities_from_pattern(text, FOLLOWUP_PATTERN, "followup", 1))
    entities.sort(key=lambda item: (item.start, item.end, item.label))
    return remove_nested_duplicates(entities)


def remove_nested_duplicates(entities: list[Entity]) -> list[Entity]:
    retained = []
    for entity in entities:
        duplicate = False
        for existing in retained:
            same_span = entity.start == existing.start and entity.end == existing.end
            same_label = entity.label == existing.label
            if same_span and same_label:
                duplicate = True
                break
        if not duplicate:
            retained.append(entity)
    return retained


def endpoint_label(text: str) -> EndpointKind | None:
    lowered = text.lower()
    terms = {
        EndpointKind.IMPLANT_SURVIVAL: ("implant survival", "survival rate"),
        EndpointKind.BONE_INGROWTH: ("bone ingrowth", "new bone formation"),
        EndpointKind.REVISION: ("revision surgery", "revision rate", "reoperation"),
        EndpointKind.ADVERSE_EVENT: ("adverse event", "complication rate"),
        EndpointKind.CELL_VIABILITY: ("cell viability", "live osteoblast"),
        EndpointKind.ALP_ACTIVITY: ("alkaline phosphatase", "alp activity"),
        EndpointKind.COMPRESSIVE_STRENGTH: ("compressive strength",),
        EndpointKind.DEGRADATION: ("degradation rate", "resorption", "dissolution"),
    }
    for endpoint, aliases in terms.items():
        if any(alias in lowered for alias in aliases):
            return endpoint
    return None


def sentence_index(position: int, values: list[Sentence]) -> int:
    for index, sentence in enumerate(values):
        if sentence.start <= position < sentence.end:
            return index
    return max(len(values) - 1, 0)


def candidate_relations(
    text: str,
    entities: list[Entity],
    maximum_sentence_distance: int = 1,
) -> list[Relation]:
    sentence_values = sentences(text)
    material_labels = {"material_phase", "composition_formula", "calcium_phosphorus_ratio"}
    value_labels = {"mechanical_value", "geometry_value", "followup"}
    results = []
    for subject_index, subject in enumerate(entities):
        if subject.label not in material_labels:
            continue
        subject_sentence = sentence_index(subject.start, sentence_values)
        for object_index, object_value in enumerate(entities):
            if object_value.label not in value_labels:
                continue
            object_sentence = sentence_index(object_value.start, sentence_values)
            distance = abs(subject_sentence - object_sentence)
            if distance > maximum_sentence_distance:
                continue
            character_distance = abs(subject.end - object_value.start)
            probability = float(np.exp(-distance) * np.exp(-character_distance / 500.0))
            results.append(
                Relation(
                    subject_index,
                    object_index,
                    "associated_with",
                    probability,
                    distance,
                )
            )
    return results


def filter_relations(relations: Iterable[Relation], threshold: float = 0.5) -> list[Relation]:
    return [relation for relation in relations if relation.probability >= threshold]
