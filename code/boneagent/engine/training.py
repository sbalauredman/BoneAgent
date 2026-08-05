from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from boneagent.twin.tabular import gaussian_negative_log_likelihood

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerState:
    epoch: int
    global_step: int
    best_validation: float
    epochs_without_improvement: int
    seed: int


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    training_loss: float
    validation_loss: float
    learning_rate: float
    improved: bool


class CosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        minimum_ratio: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.minimum_ratio = minimum_ratio
        self.base_rates = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_count = 0

    def _ratio(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine

    def step(self) -> None:
        ratio = self._ratio(self.step_count)
        for base_rate, group in zip(self.base_rates, self.optimizer.param_groups, strict=True):
            group["lr"] = base_rate * ratio
        self.step_count += 1

    def state_dict(self) -> dict[str, object]:
        return {"step_count": self.step_count, "base_rates": self.base_rates}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.step_count = int(state["step_count"])
        self.base_rates = [float(value) for value in state["base_rates"]]


class PropertyTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineScheduler,
        device: torch.device,
        gradient_clip: float,
        precision: str = "float32",
        accumulation_steps: int = 1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        self.precision = precision
        self.accumulation_steps = accumulation_steps
        enabled = precision == "float16" and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast_dtype(self) -> torch.dtype:
        if self.precision == "bfloat16":
            return torch.bfloat16
        if self.precision == "float16":
            return torch.float16
        return torch.float32

    def _loss(self, batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        features, targets, available = (value.to(self.device) for value in batch)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype(),
            enabled=self.precision != "float32",
        ):
            mean, log_variance = self.model(features)
            return gaussian_negative_log_likelihood(mean, log_variance, targets, available)

    def train_epoch(self, loader: Iterable[tuple[Tensor, Tensor, Tensor]]) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        batch_count = 0
        for batch_count, batch in enumerate(loader, start=1):
            loss = self._loss(batch) / self.accumulation_steps
            self.scaler.scale(loss).backward()
            if batch_count % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            loss_sum += float(loss.detach()) * self.accumulation_steps
        if batch_count == 0:
            raise ValueError("training loader is empty")
        if batch_count % self.accumulation_steps:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
        return loss_sum / batch_count

    @torch.no_grad()
    def validate(self, loader: Iterable[tuple[Tensor, Tensor, Tensor]]) -> float:
        self.model.eval()
        loss_sum = 0.0
        batch_count = 0
        for batch in loader:
            batch_count += 1
            loss_sum += float(self._loss(batch))
        if batch_count == 0:
            raise ValueError("validation loader is empty")
        return loss_sum / batch_count


def atomic_checkpoint(
    destination: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineScheduler,
    state: TrainerState,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "trainer": asdict(state),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(
    source: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineScheduler,
    device: torch.device,
) -> TrainerState:
    payload = torch.load(Path(source), map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return TrainerState(**payload["trainer"])
