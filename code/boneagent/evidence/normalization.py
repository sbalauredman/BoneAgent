from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitDefinition:
    canonical: str
    factor: float
    offset: float = 0.0

    def convert(self, value: float) -> float:
        return value * self.factor + self.offset


PRESSURE_UNITS = {
    "pa": UnitDefinition("MPa", 1e-6),
    "kpa": UnitDefinition("MPa", 1e-3),
    "mpa": UnitDefinition("MPa", 1.0),
    "gpa": UnitDefinition("MPa", 1e3),
    "n/mm2": UnitDefinition("MPa", 1.0),
}

LENGTH_UNITS = {
    "nm": UnitDefinition("um", 1e-3),
    "µm": UnitDefinition("um", 1.0),
    "μm": UnitDefinition("um", 1.0),
    "um": UnitDefinition("um", 1.0),
    "mm": UnitDefinition("um", 1e3),
    "cm": UnitDefinition("um", 1e4),
}

TIME_UNITS = {
    "hour": UnitDefinition("week", 1.0 / 168.0),
    "hours": UnitDefinition("week", 1.0 / 168.0),
    "day": UnitDefinition("week", 1.0 / 7.0),
    "days": UnitDefinition("week", 1.0 / 7.0),
    "week": UnitDefinition("week", 1.0),
    "weeks": UnitDefinition("week", 1.0),
    "month": UnitDefinition("week", 365.25 / 84.0),
    "months": UnitDefinition("week", 365.25 / 84.0),
    "year": UnitDefinition("week", 365.25 / 7.0),
    "years": UnitDefinition("week", 365.25 / 7.0),
}


def normalized_unit(unit: str) -> str:
    return unit.strip().lower().replace("²", "2").replace(" ", "")


def convert_value(value: float, unit: str, definitions: dict[str, UnitDefinition]) -> float:
    key = normalized_unit(unit)
    if key not in definitions:
        raise ValueError(f"unsupported unit: {unit}")
    return definitions[key].convert(value)


def pressure_mpa(value: float, unit: str) -> float:
    return convert_value(value, unit, PRESSURE_UNITS)


def length_micrometer(value: float, unit: str) -> float:
    return convert_value(value, unit, LENGTH_UNITS)


def time_weeks(value: float, unit: str) -> float:
    return convert_value(value, unit, TIME_UNITS)


def fraction(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"%", "percent", "percentage"}:
        return value / 100.0
    if normalized in {"fraction", "ratio", ""}:
        return value
    raise ValueError(f"unsupported fraction unit: {unit}")


def calcium_phosphorus_ratio(formula: str) -> float | None:
    compact = re.sub(r"\s+", "", formula)
    calcium_match = re.search(r"Ca(?:\(?([0-9.]+)\)?)?", compact, re.IGNORECASE)
    phosphorus_match = re.search(r"P(?:\(?([0-9.]+)\)?)?", compact, re.IGNORECASE)
    if calcium_match is None or phosphorus_match is None:
        return None
    calcium = float(calcium_match.group(1) or 1.0)
    phosphorus = float(phosphorus_match.group(1) or 1.0)
    if phosphorus == 0.0:
        return None
    return calcium / phosphorus


def canonical_phase(text: str) -> str | None:
    lowered = text.lower()
    aliases = {
        "hydroxyapatite": "HAp",
        "hydroxylapatite": "HAp",
        "hap": "HAp",
        "β-tricalcium phosphate": "beta-TCP",
        "beta-tricalcium phosphate": "beta-TCP",
        "β-tcp": "beta-TCP",
        "beta-tcp": "beta-TCP",
        "octacalcium phosphate": "OCP",
        "brushite": "DCPD",
        "dicalcium phosphate dihydrate": "DCPD",
        "biphasic calcium phosphate": "BCP",
    }
    for alias, canonical in aliases.items():
        if alias in lowered:
            return canonical
    return None


def parse_numeric_expression(text: str) -> tuple[float, float | None] | None:
    cleaned = text.strip().replace(",", "")
    mean_sd = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s*(?:±|\+/-)\s*([-+]?[0-9]*\.?[0-9]+)", cleaned)
    if mean_sd:
        return float(mean_sd.group(1)), float(mean_sd.group(2))
    interval = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s*[-–]\s*([-+]?[0-9]*\.?[0-9]+)", cleaned)
    if interval:
        lower = float(interval.group(1))
        upper = float(interval.group(2))
        return 0.5 * (lower + upper), 0.5 * abs(upper - lower)
    single = re.search(r"[-+]?[0-9]*\.?[0-9]+", cleaned)
    if single:
        return float(single.group(0)), None
    return None


def winsorize(values: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    low_value, high_value = np.quantile(values, [lower, upper])
    return np.clip(values, low_value, high_value)


def normalize_with(values: list[float], converter: Callable[[float], float]) -> list[float]:
    return [converter(value) for value in values]
