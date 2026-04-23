"""Training-time monitoring metrics for NeuroMamba."""

from __future__ import annotations

from typing import Dict

import torch


def connectivity_density(c: torch.Tensor) -> float:
    """Fraction of nonzero off-diagonal entries in connectivity matrix."""
    if c.ndim != 3 or c.shape[-1] != c.shape[-2]:
        raise ValueError("Expected C with shape [B, N, N].")

    n_neurons = c.shape[-1]
    off_diag_mask = ~torch.eye(n_neurons, dtype=torch.bool, device=c.device)
    off_diag = c[:, off_diag_mask]
    return (off_diag > 0).float().mean().item()


def latent_std_stats(h: torch.Tensor) -> Dict[str, float]:
    """Return mean/min/max latent std across feature dimensions."""
    if h.ndim != 4:
        raise ValueError("Expected h with shape [B, N, T, D].")

    bsz, n_neurons, t_steps, d_latent = h.shape
    h_flat = h.reshape(bsz * n_neurons * t_steps, d_latent)
    std = h_flat.std(dim=0)
    return {
        "mean_std": std.mean().item(),
        "min_std": std.min().item(),
        "max_std": std.max().item(),
    }


def reconstruction_mse(x: torch.Tensor, x_recon: torch.Tensor) -> float:
    """Scalar reconstruction MSE."""
    return torch.mean((x_recon - x) ** 2).item()
