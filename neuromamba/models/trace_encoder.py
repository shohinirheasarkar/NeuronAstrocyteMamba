"""Shared-weight per-neuron temporal trace encoder."""

from __future__ import annotations

import torch
from torch import nn

from .mamba_block import MambaBlock


class TraceEncoder(nn.Module):
    """Encode raw traces [B, N, T] into latent sequences [B, N, T, d_model]."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_layers: int,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_state <= 0 or n_layers <= 0:
            raise ValueError("d_model, d_state, and n_layers must be positive.")

        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model, bias=True)
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode neuron traces with shared weights across neurons.

        Args:
            x: Raw fluorescence traces with shape [B, N, T].
        Returns:
            Latent sequence with shape [B, N, T, d_model].
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, N, T], got {tuple(x.shape)}.")

        bsz, n_neurons, t_steps = x.shape

        # Shared-neuron design: merge B and N so all neurons use same parameters.
        h = x.reshape(bsz * n_neurons, t_steps, 1)
        h = self.input_proj(h)
        for layer in self.layers:
            h = layer(h)
        return h.reshape(bsz, n_neurons, t_steps, self.d_model)
