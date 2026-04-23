"""Variance floor regularization loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def variance_floor_loss(h: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Penalize latent dimensions whose std falls below target floor gamma."""
    bsz, n_neurons, t_steps, d_latent = h.shape
    h_flat = h.reshape(bsz * n_neurons * t_steps, d_latent)
    std = h_flat.std(dim=0)
    return F.relu(gamma - std).mean()
