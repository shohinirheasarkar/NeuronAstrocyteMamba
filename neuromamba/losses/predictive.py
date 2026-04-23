"""Latent predictive consistency loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_predictive_loss(
    h_fast: torch.Tensor, h_fast_pred: torch.Tensor
) -> torch.Tensor:
    """Predict next-step fast latent from previous-step features."""
    target = h_fast[:, :, 1:, :]
    return F.mse_loss(h_fast_pred, target)
