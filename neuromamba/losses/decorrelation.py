"""Cross-stream decorrelation loss."""

from __future__ import annotations

import torch


def decorrelation_loss(h_fast: torch.Tensor, h_slow: torch.Tensor) -> torch.Tensor:
    """Penalize cross-covariance energy between fast and slow latent streams."""
    bsz, n_neurons, t_steps, d_fast = h_fast.shape
    hf = h_fast.reshape(bsz * n_neurons * t_steps, d_fast)
    hs = h_slow.reshape(bsz * n_neurons * t_steps, h_slow.shape[-1])

    hf = hf - hf.mean(dim=0, keepdim=True)
    hs = hs - hs.mean(dim=0, keepdim=True)

    n = hf.shape[0]
    cross_cov = (hf.T @ hs) / (n - 1)
    return cross_cov.pow(2).sum()
