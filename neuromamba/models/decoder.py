"""Reconstruction decoder from fused fast/slow latent streams."""

from __future__ import annotations

import torch
from torch import nn


class ReconstructionDecoder(nn.Module):
    """Decode fused dual-timescale latents into fluorescence reconstructions."""

    def __init__(
        self, d_fast: int, d_slow: int, hidden_dim: int, output_dim: int = 1
    ) -> None:
        super().__init__()
        if d_fast <= 0 or d_slow <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("d_fast, d_slow, hidden_dim, and output_dim must be positive.")

        self.d_fast = d_fast
        self.d_slow = d_slow
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fusion_proj = nn.Linear(d_fast + d_slow, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.SiLU()

    def forward(self, h_fast_mod: torch.Tensor, h_slow: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct scalar fluorescence traces.

        Args:
            h_fast_mod: Tensor of shape [B, N, T, d_fast].
            h_slow: Tensor of shape [B, N, T, d_slow].
        Returns:
            Reconstructed traces of shape [B, N, T].
        """
        if h_fast_mod.ndim != 4 or h_fast_mod.shape[-1] != self.d_fast:
            raise ValueError(
                f"Expected h_fast_mod shape [B, N, T, {self.d_fast}], "
                f"got {tuple(h_fast_mod.shape)}."
            )
        if h_slow.ndim != 4 or h_slow.shape[-1] != self.d_slow:
            raise ValueError(
                f"Expected h_slow shape [B, N, T, {self.d_slow}], got {tuple(h_slow.shape)}."
            )
        if h_fast_mod.shape[:3] != h_slow.shape[:3]:
            raise ValueError("h_fast_mod and h_slow must match in [B, N, T].")

        fused = torch.cat([h_fast_mod, h_slow], dim=-1)
        hidden = self.activation(self.fusion_proj(fused))
        out = self.output_proj(hidden)
        return out.squeeze(-1)
