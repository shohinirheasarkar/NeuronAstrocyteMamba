"""Dual-timescale SSM stacks for fast and slow latent streams."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .mamba_block import MambaBlock


class FastSSM(nn.Module):
    """Fast-timescale stack with stronger decay initialization."""

    def __init__(
        self, d_fast: int, d_state: int, n_layers: int, **mamba_kwargs
    ) -> None:
        super().__init__()
        if d_fast <= 0 or d_state <= 0 or n_layers <= 0:
            raise ValueError("d_fast, d_state, and n_layers must be positive.")

        self.d_fast = d_fast
        self.layers = nn.ModuleList(
            [MambaBlock(d_model=d_fast, d_state=d_state, **mamba_kwargs) for _ in range(n_layers)]
        )
        self._init_fast_timescale()

    def _init_fast_timescale(self) -> None:
        for layer in self.layers:
            with torch.no_grad():
                # Large |A| (after exponentiation) => fast decay / short memory.
                vals = torch.empty_like(layer.A_log).uniform_(1.0, 10.0)
                layer.A_log.copy_(torch.log(vals))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process shape [B, N, T, d_fast] by neuron-independent scanning."""
        if x.ndim != 4 or x.shape[-1] != self.d_fast:
            raise ValueError(
                f"Expected input shape [B, N, T, {self.d_fast}], got {tuple(x.shape)}."
            )
        bsz, n_neurons, t_steps, d_fast = x.shape
        h = x.reshape(bsz * n_neurons, t_steps, d_fast)
        for layer in self.layers:
            h = layer(h)
        return h.reshape(bsz, n_neurons, t_steps, d_fast)


class SlowSSM(nn.Module):
    """Slow-timescale stack with weaker decay initialization."""

    def __init__(
        self, d_slow: int, d_state: int, n_layers: int, **mamba_kwargs
    ) -> None:
        super().__init__()
        if d_slow <= 0 or d_state <= 0 or n_layers <= 0:
            raise ValueError("d_slow, d_state, and n_layers must be positive.")

        self.d_slow = d_slow
        self.layers = nn.ModuleList(
            [MambaBlock(d_model=d_slow, d_state=d_state, **mamba_kwargs) for _ in range(n_layers)]
        )
        self._init_slow_timescale()

    def _init_slow_timescale(self) -> None:
        for layer in self.layers:
            with torch.no_grad():
                # Small |A| (after exponentiation) => slow decay / long memory.
                vals = torch.empty_like(layer.A_log).uniform_(0.001, 0.1)
                layer.A_log.copy_(torch.log(vals))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process shape [B, N, T, d_slow] by neuron-independent scanning."""
        if x.ndim != 4 or x.shape[-1] != self.d_slow:
            raise ValueError(
                f"Expected input shape [B, N, T, {self.d_slow}], got {tuple(x.shape)}."
            )
        bsz, n_neurons, t_steps, d_slow = x.shape
        h = x.reshape(bsz * n_neurons, t_steps, d_slow)
        for layer in self.layers:
            h = layer(h)
        return h.reshape(bsz, n_neurons, t_steps, d_slow)


class DualTimescaleSSM(nn.Module):
    """Split latent input into fast/slow streams and process each with its own SSM stack."""

    def __init__(
        self,
        d_model: int,
        d_fast: int,
        d_slow: int,
        d_state: int,
        n_layers: int,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_fast <= 0 or d_slow <= 0:
            raise ValueError("d_model, d_fast, and d_slow must be positive.")
        if d_fast + d_slow != d_model:
            raise ValueError("d_fast + d_slow must equal d_model.")

        self.d_model = d_model
        self.d_fast = d_fast
        self.d_slow = d_slow

        self.fast_ssm = FastSSM(
            d_fast=d_fast, d_state=d_state, n_layers=n_layers, **mamba_kwargs
        )
        self.slow_ssm = SlowSSM(
            d_slow=d_slow, d_state=d_state, n_layers=n_layers, **mamba_kwargs
        )

    @property
    def fast_A_logs(self) -> List[torch.Tensor]:
        """A_log tensors from all fast-stream layers."""
        return [layer.A_log for layer in self.fast_ssm.layers]

    @property
    def slow_A_logs(self) -> List[torch.Tensor]:
        """A_log tensors from all slow-stream layers."""
        return [layer.A_log for layer in self.slow_ssm.layers]

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split [B, N, T, d_model] into fast/slow streams and process both."""
        if latent.ndim != 4 or latent.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected input shape [B, N, T, {self.d_model}], got {tuple(latent.shape)}."
            )

        x_fast, x_slow = torch.split(latent, [self.d_fast, self.d_slow], dim=-1)
        h_fast = self.fast_ssm(x_fast)
        h_slow = self.slow_ssm(x_slow)
        return h_fast, h_slow
