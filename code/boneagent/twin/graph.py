from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CrystalBatch:
    atomic_numbers: Tensor
    positions: Tensor
    edge_index: Tensor
    shifts: Tensor
    batch_index: Tensor
    cell: Tensor

    def to(self, device: torch.device) -> CrystalBatch:
        return CrystalBatch(
            atomic_numbers=self.atomic_numbers.to(device),
            positions=self.positions.to(device),
            edge_index=self.edge_index.to(device),
            shifts=self.shifts.to(device),
            batch_index=self.batch_index.to(device),
            cell=self.cell.to(device),
        )


def cosine_cutoff(distance: Tensor, cutoff: float) -> Tensor:
    scaled = torch.clamp(distance / cutoff, min=0.0, max=1.0)
    values = 0.5 * (torch.cos(torch.pi * scaled) + 1.0)
    return torch.where(distance < cutoff, values, torch.zeros_like(values))


class BesselBasis(torch.nn.Module):
    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        self.count = count
        self.cutoff = cutoff
        frequencies = torch.arange(1, count + 1, dtype=torch.float32) * torch.pi
        self.register_buffer("frequencies", frequencies)

    def forward(self, distance: Tensor) -> Tensor:
        safe = torch.clamp(distance, min=1e-8)
        scaled = safe.unsqueeze(-1) / self.cutoff
        basis = torch.sin(scaled * self.frequencies) / safe.unsqueeze(-1)
        return basis * cosine_cutoff(safe, self.cutoff).unsqueeze(-1)


class SphericalFeatures(torch.nn.Module):
    def __init__(self, maximum_order: int = 2) -> None:
        super().__init__()
        if maximum_order != 2:
            raise ValueError("this basis supports maximum order two")
        self.maximum_order = maximum_order

    def forward(self, vectors: Tensor) -> Tensor:
        distance = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        unit = vectors / torch.clamp(distance, min=1e-8)
        x = unit[:, 0]
        y = unit[:, 1]
        z = unit[:, 2]
        constant = torch.ones_like(x)
        first = torch.stack((x, y, z), dim=-1)
        second = torch.stack(
            (
                x * y,
                y * z,
                0.5 * (3.0 * z.square() - 1.0),
                x * z,
                0.5 * (x.square() - y.square()),
            ),
            dim=-1,
        )
        return torch.cat((constant.unsqueeze(-1), first, second), dim=-1)


def segment_sum(values: Tensor, index: Tensor, count: int) -> Tensor:
    output_shape = (count,) + tuple(values.shape[1:])
    output = values.new_zeros(output_shape)
    output.index_add_(0, index, values)
    return output


class EquivariantInteraction(torch.nn.Module):
    def __init__(self, hidden_channels: int, radial_count: int, correlation: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.correlation = correlation
        input_channels = radial_count + 9
        self.edge_network = torch.nn.Sequential(
            torch.nn.Linear(input_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.Sigmoid(),
        )
        self.scalar_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.vector_gate = torch.nn.Linear(hidden_channels, hidden_channels)
        self.normalization = torch.nn.LayerNorm(hidden_channels)

    def forward(
        self,
        scalar: Tensor,
        vector: Tensor,
        edge_index: Tensor,
        radial: Tensor,
        spherical: Tensor,
        direction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        edge_features = torch.cat((radial, spherical), dim=-1)
        edge_weight = self.edge_network(edge_features)
        scalar_message = scalar[sender] * edge_weight
        aggregated_scalar = segment_sum(scalar_message, receiver, scalar.shape[0])
        vector_message = scalar_message.unsqueeze(-1) * direction.unsqueeze(1)
        aggregated_vector = segment_sum(vector_message, receiver, scalar.shape[0])
        vector_norm = torch.linalg.vector_norm(aggregated_vector, dim=-1)
        mixed = torch.cat((scalar, aggregated_scalar + vector_norm), dim=-1)
        update = self.scalar_network(mixed)
        gate = torch.sigmoid(self.vector_gate(update)).unsqueeze(-1)
        next_scalar = self.normalization(scalar + update)
        next_vector = vector + gate * aggregated_vector
        for _ in range(self.correlation - 1):
            norm = torch.linalg.vector_norm(next_vector, dim=-1)
            next_scalar = self.normalization(next_scalar + norm * update)
        return next_scalar, next_vector


class CrystalEquivariantNetwork(torch.nn.Module):
    def __init__(
        self,
        maximum_atomic_number: int = 118,
        hidden_channels: int = 128,
        layers: int = 3,
        radial_count: int = 8,
        cutoff: float = 5.0,
        correlation: int = 3,
        output_channels: int = 4,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.embedding = torch.nn.Embedding(maximum_atomic_number + 1, hidden_channels)
        self.radial = BesselBasis(radial_count, cutoff)
        self.spherical = SphericalFeatures(2)
        self.interactions = torch.nn.ModuleList(
            [
                EquivariantInteraction(hidden_channels, radial_count, correlation)
                for _ in range(layers)
            ]
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, output_channels),
        )

    def forward(self, batch: CrystalBatch) -> Tensor:
        scalar = self.embedding(batch.atomic_numbers)
        vector = scalar.new_zeros((scalar.shape[0], scalar.shape[1], 3))
        sender = batch.edge_index[0]
        receiver = batch.edge_index[1]
        displacement = batch.positions[receiver] - batch.positions[sender] + batch.shifts
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        direction = displacement / torch.clamp(distance.unsqueeze(-1), min=1e-8)
        radial = self.radial(distance)
        spherical = self.spherical(displacement)
        for interaction in self.interactions:
            scalar, vector = interaction(
                scalar,
                vector,
                batch.edge_index,
                radial,
                spherical,
                direction,
            )
        atom_outputs = self.readout(scalar)
        graph_count = int(batch.batch_index.max().item()) + 1
        graph_outputs = segment_sum(atom_outputs, batch.batch_index, graph_count)
        atom_counts = torch.bincount(batch.batch_index, minlength=graph_count).clamp(min=1)
        return graph_outputs / atom_counts.unsqueeze(-1)
