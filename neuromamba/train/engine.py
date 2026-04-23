"""Training and validation epoch loops for NeuroMamba."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from ..losses.total_loss import compute_total_loss
from ..models.neuromamba import NeuroMamba


def _mean_loss_dict(loss_sums: Dict[str, float], n_steps: int) -> Dict[str, float]:
    if n_steps == 0:
        return {}
    return {k: v / float(n_steps) for k, v in loss_sums.items()}


def train_one_epoch(
    model: NeuroMamba,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    cfg: DictConfig,
    epoch: int,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch and return mean loss components."""
    model.train()
    detach_c = epoch < cfg.detach_C_warmup_epochs
    loss_sums: Dict[str, float] = defaultdict(float)

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        outputs = model(x, detach_C=detach_c)
        l_total, loss_dict = compute_total_loss(outputs, {"x": x}, model, cfg)

        optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for key, value in loss_dict.items():
            loss_sums[key] += float(value)

        if step % int(cfg.log_every_n_steps) == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch={epoch} step={step} "
                + " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                + f" lr={lr:.6e}"
            )

    return _mean_loss_dict(loss_sums, len(loader))


def validate_one_epoch(
    model: NeuroMamba,
    loader: DataLoader,
    cfg: DictConfig,
    epoch: int,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation epoch and return mean loss components."""
    del epoch
    model.eval()
    loss_sums: Dict[str, float] = defaultdict(float)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            outputs = model(x, detach_C=False)
            _, loss_dict = compute_total_loss(outputs, {"x": x}, model, cfg)
            for key, value in loss_dict.items():
                loss_sums[key] += float(value)

    return _mean_loss_dict(loss_sums, len(loader))
