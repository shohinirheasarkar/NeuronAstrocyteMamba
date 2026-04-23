"""Slow-to-fast gating module for dual-timescale latents."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class SlowFastGate(nn.Module):
    """Modulate fast stream activity using a slow-stream sigmoid gate."""

    def __init__(self, d_fast: int, d_slow: int) -> None:
        super().__init__()
        if d_fast <= 0 or d_slow <= 0:
            raise ValueError("d_fast and d_slow must be positive.")
        self.d_fast = d_fast
        self.d_slow = d_slow
        self.gate_proj = nn.Linear(d_slow, d_fast, bias=True)

    def forward(
        self, h_fast: torch.Tensor, h_slow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply slow-stream gain control to the fast stream.

        Args:
            h_fast: Tensor of shape [B, N, T, d_fast].
            h_slow: Tensor of shape [B, N, T, d_slow].
        Returns:
            Tuple of:
              - modulated fast stream [B, N, T, d_fast]
              - gate tensor [B, N, T, d_fast]
        """
        if h_fast.ndim != 4 or h_fast.shape[-1] != self.d_fast:
            raise ValueError(
                f"Expected h_fast shape [B, N, T, {self.d_fast}], got {tuple(h_fast.shape)}."
            )
        if h_slow.ndim != 4 or h_slow.shape[-1] != self.d_slow:
            raise ValueError(
                f"Expected h_slow shape [B, N, T, {self.d_slow}], got {tuple(h_slow.shape)}."
            )
        if h_fast.shape[:3] != h_slow.shape[:3]:
            raise ValueError("h_fast and h_slow must match in [B, N, T].")

        gate = torch.sigmoid(self.gate_proj(h_slow))
        h_fast_mod = h_fast * gate
        return h_fast_mod, gate
