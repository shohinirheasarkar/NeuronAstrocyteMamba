"""Sparsity regularization on inferred connectivity."""

from __future__ import annotations

import torch


def sparsity_loss(c: torch.Tensor) -> torch.Tensor:
    """L1 penalty encouraging sparse directed connectivity."""
    return c.abs().mean()
