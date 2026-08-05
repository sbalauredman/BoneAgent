from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DFTSettings:
    functional: str = "PBE"
    plane_wave_cutoff_ev: float = 520.0
    electronic_tolerance_ev: float = 1e-6
    force_tolerance_ev_angstrom: float = 0.01
    calcium_hubbard_u_ev: float = 3.5
    vacuum_angstrom: float = 15.0
    kpoint_density_angstrom: float = 30.0
    vasp_version: str = "6.4.1"


@dataclass(frozen=True)
class AtomicStructure:
    symbols: tuple[str, ...]
    positions: NDArray[np.float64]
    cell: NDArray[np.float64]
    periodic: tuple[bool, bool, bool]

    def validate(self) -> None:
        if self.positions.shape != (len(self.symbols), 3):
            raise ValueError("positions must have one three-vector per symbol")
        if self.cell.shape != (3, 3):
            raise ValueError("cell must be a 3 by 3 matrix")
        if abs(np.linalg.det(self.cell)) < 1e-10:
            raise ValueError("cell must have nonzero volume")


@dataclass(frozen=True)
class DFTTask:
    identifier: str
    structure: AtomicStructure
    settings: DFTSettings
    calculation: str
    surface_normal: int | None = None


@dataclass(frozen=True)
class DFTResult:
    identifier: str
    converged: bool
    total_energy_ev: float
    formation_energy_ev_atom: float
    maximum_force_ev_angstrom: float
    surface_energy_j_m2: float | None
    dissolution_barrier_ev: float | None
    wall_seconds: float


def automatic_kpoint_mesh(
    cell: NDArray[np.float64], density_angstrom: float
) -> tuple[int, int, int]:
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    lengths = np.linalg.norm(reciprocal, axis=1)
    mesh = np.maximum(np.ceil(density_angstrom * lengths / (2.0 * np.pi)), 1).astype(int)
    return int(mesh[0]), int(mesh[1]), int(mesh[2])


def incar(settings: DFTSettings, symbols: Sequence[str], calculation: str) -> str:
    lines = [
        "SYSTEM = BoneAgent CaP",
        f"ENCUT = {settings.plane_wave_cutoff_ev:.1f}",
        f"EDIFF = {settings.electronic_tolerance_ev:.8g}",
        f"EDIFFG = {-settings.force_tolerance_ev_angstrom:.8g}",
        "GGA = PE",
        "PREC = Accurate",
        "ALGO = Normal",
        "LASPH = .TRUE.",
        "LREAL = Auto",
        "ISMEAR = 0",
        "SIGMA = 0.05",
        "LWAVE = .FALSE.",
        "LCHARG = .TRUE.",
    ]
    if "Ca" in symbols:
        unique = list(dict.fromkeys(symbols))
        ldaul = ["2" if symbol == "Ca" else "-1" for symbol in unique]
        ldauu = [
            f"{settings.calcium_hubbard_u_ev:.2f}" if symbol == "Ca" else "0.0" for symbol in unique
        ]
        ldauj = ["0.0" for _ in unique]
        lines.extend(
            [
                "LDAU = .TRUE.",
                "LDAUTYPE = 2",
                f"LDAUL = {' '.join(ldaul)}",
                f"LDAUU = {' '.join(ldauu)}",
                f"LDAUJ = {' '.join(ldauj)}",
            ]
        )
    if calculation == "relaxation":
        lines.extend(["IBRION = 2", "NSW = 300", "ISIF = 3"])
    elif calculation == "surface":
        lines.extend(["IBRION = 2", "NSW = 300", "ISIF = 2", "LDIPOL = .TRUE."])
    elif calculation == "static":
        lines.extend(["IBRION = -1", "NSW = 0", "ISIF = 2"])
    else:
        raise ValueError(f"unsupported calculation: {calculation}")
    return "\n".join(lines) + "\n"


def poscar(structure: AtomicStructure) -> str:
    structure.validate()
    unique = list(dict.fromkeys(structure.symbols))
    counts = [structure.symbols.count(symbol) for symbol in unique]
    lines = ["BoneAgent structure", "1.0"]
    lines.extend(" ".join(f"{value:.12f}" for value in row) for row in structure.cell)
    lines.append(" ".join(unique))
    lines.append(" ".join(str(count) for count in counts))
    lines.append("Cartesian")
    for symbol in unique:
        for index, atom_symbol in enumerate(structure.symbols):
            if atom_symbol == symbol:
                lines.append(" ".join(f"{value:.12f}" for value in structure.positions[index]))
    return "\n".join(lines) + "\n"


def kpoints(structure: AtomicStructure, settings: DFTSettings) -> str:
    mesh = automatic_kpoint_mesh(structure.cell, settings.kpoint_density_angstrom)
    return "Automatic mesh\n0\nGamma\n" + " ".join(str(value) for value in mesh) + "\n0 0 0\n"


def write_task(task: DFTTask, directory: str | Path) -> Path:
    target = Path(directory) / task.identifier
    target.mkdir(parents=True, exist_ok=False)
    (target / "INCAR").write_text(
        incar(task.settings, task.structure.symbols, task.calculation), encoding="utf-8"
    )
    (target / "POSCAR").write_text(poscar(task.structure), encoding="utf-8")
    (target / "KPOINTS").write_text(kpoints(task.structure, task.settings), encoding="utf-8")
    metadata = {
        "identifier": task.identifier,
        "calculation": task.calculation,
        "surface_normal": task.surface_normal,
        "settings": asdict(task.settings),
    }
    (target / "task.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def surface_energy(
    slab_energy_ev: float,
    bulk_energy_ev_atom: float,
    atom_count: int,
    surface_area_angstrom2: float,
    surface_count: int = 2,
) -> float:
    excess_ev = slab_energy_ev - atom_count * bulk_energy_ev_atom
    ev_angstrom2 = excess_ev / (surface_count * surface_area_angstrom2)
    return float(ev_angstrom2 * 16.02176634)


def formation_energy(
    total_energy_ev: float,
    symbols: Sequence[str],
    reference_energies: dict[str, float],
) -> float:
    reference = 0.0
    for symbol in symbols:
        if symbol not in reference_energies:
            raise KeyError(f"missing elemental reference for {symbol}")
        reference += reference_energies[symbol]
    return (total_energy_ev - reference) / len(symbols)
