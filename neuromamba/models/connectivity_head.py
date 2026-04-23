"""Directed connectivity inference from fast latent states."""

from __future__ import annotations

from typing import Optional

import torch
from entmax import entmax15
from torch import nn


class DirectedConnectivityHead(nn.Module):
    """Infer directed source->target connectivity using past-vs-present asymmetry."""

    def __init__(self, d_fast: int, lag_window: int, entmax_alpha: float = 1.5) -> None:
        super().__init__()
        if d_fast <= 0:
            raise ValueError("d_fast must be positive.")
        if lag_window <= 0:
            raise ValueError("lag_window must be positive.")

        self.d_fast = d_fast
        self.lag_window = lag_window
        self.entmax_alpha = entmax_alpha

        self.q_proj = nn.Linear(d_fast, d_fast, bias=False)
        self.k_proj = nn.Linear(d_fast, d_fast, bias=False)
        self.scale = d_fast**-0.5

    def _apply_entmax(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply alpha-entmax over source dimension."""
        if abs(self.entmax_alpha - 1.5) > 1e-8:
            raise ValueError(
                "This implementation currently supports entmax_alpha=1.5 only "
                "(uses entmax15)."
            )
        c = entmax15(scores, dim=-1)
        # Exclude self-edges by construction for explicit structural sparsity.
        n_neurons = c.shape[-1]
        diag_mask = torch.eye(n_neurons, dtype=torch.bool, device=c.device).unsqueeze(0)
        return c.masked_fill(diag_mask, 0.0)

    def _compute_scores_at_t(self, h_fast_mod: torch.Tensor, t: int) -> torch.Tensor:
        """
        Compute raw target-source scores at a specific timestep t.

        Args:
            h_fast_mod: [B, N, T, d_fast]
            t: Present timestep index used for query states.
        Returns:
            Scores of shape [B, N_target, N_source].
        """
        q_states = h_fast_mod[:, :, t, :]  # [B, N, d_fast]
        q = self.q_proj(q_states)  # [B, N, d_fast]

        past_window = h_fast_mod[:, :, t - self.lag_window : t, :]  # [B, N, lag, d_fast]
        source_summary = past_window.mean(dim=2)  # [B, N, d_fast]
        k = self.k_proj(source_summary)  # [B, N, d_fast]

        scores = (q[:, :, None, :] * k[:, None, :, :]).sum(dim=-1) * self.scale
        return scores

    def forward(
        self, h_fast_mod: torch.Tensor, current_epoch: int = 0, detach_C: bool = False
    ) -> torch.Tensor:
        """
        Fast single-slice connectivity estimate for training.

        Args:
            h_fast_mod: [B, N, T, d_fast]
            current_epoch: Included for compatibility with training loops.
            detach_C: If True, return detached C (warmup use case).
        Returns:
            Connectivity matrix [B, N, N].
        """
        del current_epoch

        if h_fast_mod.ndim != 4 or h_fast_mod.shape[-1] != self.d_fast:
            raise ValueError(
                f"Expected h_fast_mod shape [B, N, T, {self.d_fast}], "
                f"got {tuple(h_fast_mod.shape)}."
            )

        t_steps = h_fast_mod.shape[2]
        if t_steps < self.lag_window + 1:
            raise ValueError(
                f"Need at least lag_window+1={self.lag_window + 1} timesteps, got {t_steps}."
            )

        # Present = last timestep; source keys summarize the preceding lag window.
        scores = self._compute_scores_at_t(h_fast_mod, t=t_steps - 1)
        c = self._apply_entmax(scores)
        return c.detach() if detach_C else c

    def compute_temporal_connectivity(self, h_fast_mod: torch.Tensor) -> torch.Tensor:
        """
        Full temporal connectivity estimate for inference.

        For each t in [lag_window, T-1], compute e_t then average scores over t,
        and apply entmax once at the end: C = entmax(mean_t e_t).
        """
        if h_fast_mod.ndim != 4 or h_fast_mod.shape[-1] != self.d_fast:
            raise ValueError(
                f"Expected h_fast_mod shape [B, N, T, {self.d_fast}], "
                f"got {tuple(h_fast_mod.shape)}."
            )

        t_steps = h_fast_mod.shape[2]
        if t_steps < self.lag_window + 1:
            raise ValueError(
                f"Need at least lag_window+1={self.lag_window + 1} timesteps, got {t_steps}."
            )

        score_sum: Optional[torch.Tensor] = None
        n_terms = 0
        for t in range(self.lag_window, t_steps):
            scores_t = self._compute_scores_at_t(h_fast_mod, t=t)
            score_sum = scores_t if score_sum is None else score_sum + scores_t
            n_terms += 1

        mean_scores = score_sum / float(n_terms)
        return self._apply_entmax(mean_scores)
