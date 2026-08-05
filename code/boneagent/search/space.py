from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from boneagent.domain import Candidate, Composition, Processing, Scaffold


@dataclass(frozen=True)
class SpaceBounds:
    ratio: tuple[float, float] = (1.0, 2.0)
    hydroxyapatite: tuple[float, float] = (0.0, 1.0)
    dopant_fraction: tuple[float, float] = (0.0, 0.2)
    porosity: tuple[float, float] = (0.3, 0.8)
    pore_diameter: tuple[float, float] = (100.0, 600.0)
    strut_diameter: tuple[float, float] = (50.0, 300.0)
    interconnectivity: tuple[float, float] = (0.0, 1.0)
    sintering_temperature: tuple[float, float] = (500.0, 1600.0)
    oxygen_fraction: tuple[float, float] = (0.0, 1.0)
    holding_hours: tuple[float, float] = (0.1, 24.0)


class CandidateSpace:
    def __init__(
        self,
        seed: int,
        bounds: SpaceBounds | None = None,
        dopant_atomic_numbers: tuple[int, ...] = (0, 12, 14, 25, 30, 38),
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._bounds = bounds or SpaceBounds()
        self._dopants = dopant_atomic_numbers

    def _uniform(self, limits: tuple[float, float], count: int) -> NDArray[np.float64]:
        return self._rng.uniform(limits[0], limits[1], count)

    def sample(self, count: int, cycle: int = 0) -> list[Candidate]:
        ratio = self._uniform(self._bounds.ratio, count)
        hap = self._uniform(self._bounds.hydroxyapatite, count)
        dopant = self._rng.choice(self._dopants, count)
        dopant_fraction = self._uniform(self._bounds.dopant_fraction, count)
        porosity = self._uniform(self._bounds.porosity, count)
        pore = self._uniform(self._bounds.pore_diameter, count)
        strut = self._uniform(self._bounds.strut_diameter, count)
        interconnectivity = self._uniform(self._bounds.interconnectivity, count)
        temperature = self._uniform(self._bounds.sintering_temperature, count)
        oxygen = self._uniform(self._bounds.oxygen_fraction, count)
        holding = self._uniform(self._bounds.holding_hours, count)
        candidates = []
        for index in range(count):
            composition = Composition(
                calcium_phosphorus_ratio=float(ratio[index]),
                hydroxyapatite_fraction=float(hap[index]),
                beta_tcp_fraction=float(1.0 - hap[index]),
                dopant_atomic_number=int(dopant[index]),
                dopant_fraction=float(dopant_fraction[index]),
            )
            scaffold = Scaffold(
                porosity=float(porosity[index]),
                pore_diameter_micrometer=float(pore[index]),
                strut_diameter_micrometer=float(strut[index]),
                interconnectivity=float(interconnectivity[index]),
            )
            processing = Processing(
                sintering_temperature_celsius=float(temperature[index]),
                oxygen_fraction=float(oxygen[index]),
                holding_hours=float(holding[index]),
            )
            candidates.append(
                Candidate(
                    identifier=f"c{cycle:03d}-{index:07d}",
                    composition=composition,
                    scaffold=scaffold,
                    processing=processing,
                )
            )
        return candidates

    def perturb(
        self,
        parents: list[Candidate],
        count: int,
        cycle: int,
        scale: float = 0.05,
    ) -> list[Candidate]:
        if not parents:
            return self.sample(count, cycle)
        results = []
        for index in range(count):
            parent = parents[index % len(parents)]
            raw = parent.vector().copy()
            noise = self._rng.normal(0.0, scale, raw.shape)
            raw = raw + noise * np.maximum(np.abs(raw), 1.0)
            raw[0] = np.clip(raw[0], *self._bounds.ratio)
            raw[1] = np.clip(raw[1], *self._bounds.hydroxyapatite)
            raw[2] = 1.0 - raw[1]
            raw[3] = min(self._dopants, key=lambda value: abs(value - raw[3]))
            raw[4] = np.clip(raw[4], *self._bounds.dopant_fraction)
            raw[5] = np.clip(raw[5], *self._bounds.porosity)
            raw[6] = np.clip(raw[6], *self._bounds.pore_diameter)
            raw[7] = np.clip(raw[7], *self._bounds.strut_diameter)
            raw[8] = np.clip(raw[8], *self._bounds.interconnectivity)
            raw[9] = np.clip(raw[9], *self._bounds.sintering_temperature)
            raw[10] = np.clip(raw[10], *self._bounds.oxygen_fraction)
            raw[11] = np.clip(raw[11], *self._bounds.holding_hours)
            results.append(
                Candidate(
                    identifier=f"c{cycle:03d}-{index:07d}",
                    composition=Composition(
                        calcium_phosphorus_ratio=float(raw[0]),
                        hydroxyapatite_fraction=float(raw[1]),
                        beta_tcp_fraction=float(raw[2]),
                        dopant_atomic_number=int(raw[3]),
                        dopant_fraction=float(raw[4]),
                    ),
                    scaffold=Scaffold(
                        porosity=float(raw[5]),
                        pore_diameter_micrometer=float(raw[6]),
                        strut_diameter_micrometer=float(raw[7]),
                        interconnectivity=float(raw[8]),
                    ),
                    processing=Processing(
                        sintering_temperature_celsius=float(raw[9]),
                        oxygen_fraction=float(raw[10]),
                        holding_hours=float(raw[11]),
                    ),
                )
            )
        return results
