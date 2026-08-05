from __future__ import annotations

import torch
from torch import Tensor


class ResidualBlock(torch.nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.first = torch.nn.Linear(width, width * 2)
        self.second = torch.nn.Linear(width * 2, width)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.norm(values)
        hidden = torch.nn.functional.silu(self.first(hidden))
        hidden = self.dropout(self.second(hidden))
        return values + hidden


class ScaffoldPropertyNetwork(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        width: int = 256,
        depth: int = 6,
        output_channels: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_normalizer = torch.nn.BatchNorm1d(input_channels)
        self.input_projection = torch.nn.Linear(input_channels, width)
        self.blocks = torch.nn.ModuleList([ResidualBlock(width, dropout) for _ in range(depth)])
        self.mean_head = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, output_channels),
        )
        self.log_variance_head = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, output_channels),
        )

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.input_normalizer(values)
        hidden = self.input_projection(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(min=-12.0, max=8.0)
        return mean, log_variance


class DeepEnsemble(torch.nn.Module):
    def __init__(self, members: list[ScaffoldPropertyNetwork]) -> None:
        super().__init__()
        if not members:
            raise ValueError("ensemble requires at least one member")
        self.members = torch.nn.ModuleList(members)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        predictions = [member(values) for member in self.members]
        means = torch.stack([item[0] for item in predictions], dim=0)
        variances = torch.stack([item[1].exp() for item in predictions], dim=0)
        ensemble_mean = means.mean(dim=0)
        aleatoric = variances.mean(dim=0)
        epistemic = means.var(dim=0, unbiased=False)
        return ensemble_mean, aleatoric + epistemic


class MultiTaskLoss(torch.nn.Module):
    def __init__(self, task_count: int) -> None:
        super().__init__()
        self.log_scales = torch.nn.Parameter(torch.zeros(task_count))

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        available: Tensor | None = None,
    ) -> Tensor:
        losses = (prediction - target).square()
        if available is not None:
            losses = losses * available
            denominator = available.sum(dim=0).clamp(min=1.0)
        else:
            denominator = torch.full_like(losses.sum(dim=0), prediction.shape[0])
        per_task = losses.sum(dim=0) / denominator
        precision = torch.exp(-self.log_scales)
        return torch.sum(precision * per_task + self.log_scales)


def gaussian_negative_log_likelihood(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
    available: Tensor | None = None,
) -> Tensor:
    loss = 0.5 * (log_variance + (target - mean).square() * torch.exp(-log_variance))
    if available is not None:
        loss = loss * available
        return loss.sum() / available.sum().clamp(min=1.0)
    return loss.mean()
