"""Composite NeuroMamba training objective."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from omegaconf import DictConfig
from torch import nn

from .decorrelation import decorrelation_loss
from .predictive import latent_predictive_loss
from .reconstruction import reconstruction_loss
from .sparsity import sparsity_loss
from .timescale import timescale_separation_loss
from .variance import variance_floor_loss


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    cfg: DictConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute weighted sum of all NeuroMamba losses and scalar logs."""
    x = batch["x"]

    l_recon = reconstruction_loss(x, outputs["x_recon"])
    l_pred = latent_predictive_loss(outputs["h_fast"], outputs["h_fast_pred"])
    l_decor = decorrelation_loss(outputs["h_fast"], outputs["h_slow"])
    l_var = variance_floor_loss(outputs["h_fast"], cfg.variance_gamma) + variance_floor_loss(
        outputs["h_slow"], cfg.variance_gamma
    )
    l_sparse = sparsity_loss(outputs["connectivity"])
    l_timescale = timescale_separation_loss(
        model.dual_ssm.fast_A_logs,
        model.dual_ssm.slow_A_logs,
        cfg.timescale_margin,
    )

    l_total = (
        cfg.lambda_recon * l_recon
        + cfg.lambda_pred * l_pred
        + cfg.lambda_decor * l_decor
        + cfg.lambda_var * l_var
        + cfg.lambda_sparse * l_sparse
        + cfg.lambda_timescale * l_timescale
    )

    loss_dict = {
        "total": l_total.item(),
        "recon": l_recon.item(),
        "pred": l_pred.item(),
        "decor": l_decor.item(),
        "var": l_var.item(),
        "sparse": l_sparse.item(),
        "timescale": l_timescale.item(),
    }
    return l_total, loss_dict
