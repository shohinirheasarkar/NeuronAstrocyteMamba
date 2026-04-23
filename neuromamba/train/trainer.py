"""Trainer orchestration for NeuroMamba."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..models.neuromamba import NeuroMamba
from .engine import train_one_epoch, validate_one_epoch


class Trainer:
    """High-level training loop with optimizer, scheduler, and checkpointing."""

    def __init__(
        self,
        model: NeuroMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        self.scheduler = self._build_warmup_cosine_scheduler()
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _build_warmup_cosine_scheduler(self) -> Optional[LambdaLR]:
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = int(self.cfg.max_epochs) * steps_per_epoch
        warmup_steps = int(self.cfg.warmup_epochs) * steps_per_epoch

        if total_steps <= 0:
            return None

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            if total_steps <= warmup_steps:
                return 1.0

            progress = (step - warmup_steps) / float(total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def fit(self) -> None:
        """Run full training loop across all configured epochs."""
        max_epochs = int(self.cfg.max_epochs)
        for epoch in range(max_epochs):
            train_metrics = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                cfg=self.cfg,
                epoch=epoch,
                device=self.device,
            )
            val_metrics = validate_one_epoch(
                model=self.model,
                loader=self.val_loader,
                cfg=self.cfg,
                epoch=epoch,
                device=self.device,
            )

            print(
                f"[epoch {epoch}] "
                f"train: {' '.join(f'{k}={v:.4f}' for k, v in train_metrics.items())} | "
                f"val: {' '.join(f'{k}={v:.4f}' for k, v in val_metrics.items())}"
            )

            if (epoch + 1) % int(self.cfg.save_every_n_epochs) == 0:
                ckpt_path = self.checkpoint_dir / f"epoch_{epoch + 1}.pt"
                self.save_checkpoint(epoch=epoch, path=str(ckpt_path))

    def save_checkpoint(self, epoch: int, path: str) -> None:
        """Save model/optimizer/scheduler state."""
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "cfg": self.cfg,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return stored epoch."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return int(checkpoint["epoch"])
