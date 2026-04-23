"""Top-level NeuroMamba model wiring encoder, dual SSM, gating, connectivity, and decoder."""

from __future__ import annotations

from typing import Any, Dict

import torch
from omegaconf import DictConfig
from torch import nn

from .connectivity_head import DirectedConnectivityHead
from .decoder import ReconstructionDecoder
from .dual_timescale_ssm import DualTimescaleSSM
from .gating import SlowFastGate
from .trace_encoder import TraceEncoder


def _cfg_get(cfg: DictConfig, key: str) -> Any:
    """Read key from cfg.model first, then cfg root."""
    if hasattr(cfg, "model") and key in cfg.model:
        return cfg.model[key]
    if key in cfg:
        return cfg[key]
    raise KeyError(f"Missing required config key: '{key}'.")


class NeuroMamba(nn.Module):
    """End-to-end model for directed connectivity and trace reconstruction."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d_model = int(_cfg_get(cfg, "d_model"))
        d_fast = int(_cfg_get(cfg, "d_fast"))
        d_slow = int(_cfg_get(cfg, "d_slow"))
        d_state = int(_cfg_get(cfg, "d_state"))
        d_conv = int(_cfg_get(cfg, "d_conv"))
        expand = int(_cfg_get(cfg, "expand"))
        n_layers = int(_cfg_get(cfg, "n_layers"))
        lag_window = int(_cfg_get(cfg, "lag_window"))
        entmax_alpha = float(_cfg_get(cfg, "entmax_alpha"))
        decoder_hidden = int(_cfg_get(cfg, "decoder_hidden"))

        self.encoder = TraceEncoder(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            d_conv=d_conv,
            expand=expand,
        )
        self.dual_ssm = DualTimescaleSSM(
            d_model=d_model,
            d_fast=d_fast,
            d_slow=d_slow,
            d_state=d_state,
            n_layers=n_layers,
            d_conv=d_conv,
            expand=expand,
        )
        self.gate = SlowFastGate(d_fast=d_fast, d_slow=d_slow)
        self.conn_head = DirectedConnectivityHead(
            d_fast=d_fast, lag_window=lag_window, entmax_alpha=entmax_alpha
        )
        self.decoder = ReconstructionDecoder(
            d_fast=d_fast, d_slow=d_slow, hidden_dim=decoder_hidden, output_dim=1
        )
        self.predictor = nn.Sequential(
            nn.Linear(d_fast * 2, d_fast * 2),
            nn.SiLU(),
            nn.Linear(d_fast * 2, d_fast),
        )

    def forward(self, x: torch.Tensor, detach_C: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Raw fluorescence input [B, N, T].
            detach_C: Whether to detach connectivity during warmup.
        Returns:
            Dict of reconstruction, latent states, connectivity, and predictions.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, N, T], got {tuple(x.shape)}.")

        latent = self.encoder(x)  # [B, N, T, d_model]
        h_fast, h_slow = self.dual_ssm(latent)  # [B, N, T, d_fast], [B, N, T, d_slow]
        h_fast_mod, gate = self.gate(h_fast, h_slow)  # [B, N, T, d_fast], [B, N, T, d_fast]
        c = self.conn_head(h_fast_mod, detach_C=detach_C)  # [B, N, N]
        x_recon = self.decoder(h_fast_mod, h_slow)  # [B, N, T]

        past_latents = h_fast_mod[:, :, :-1, :]  # [B, N, T-1, d_fast]
        neighbor_ctx = torch.einsum("bij,bjtd->bitd", c, past_latents)  # [B, N, T-1, d_fast]
        pred_input = torch.cat([past_latents, neighbor_ctx], dim=-1)  # [B, N, T-1, 2*d_fast]
        h_fast_pred = self.predictor(pred_input)  # [B, N, T-1, d_fast]

        return {
            "x_recon": x_recon,
            "h_fast": h_fast,
            "h_slow": h_slow,
            "h_fast_mod": h_fast_mod,
            "connectivity": c,
            "h_fast_pred": h_fast_pred,
            "gate": gate,
        }
