"""Reconstruction loss for trace autoencoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """Mean-squared reconstruction error in fluorescence space."""
    return F.mse_loss(x_recon, x)
