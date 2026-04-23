"""Pure-PyTorch Mamba-style selective SSM block."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F


class MambaBlock(nn.Module):
    """Mamba-1 style block with causal depthwise conv and selective scan."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError("d_model, d_state, d_conv, and expand must be positive.")
        if dt_min <= 0.0 or dt_max <= 0.0 or dt_min >= dt_max:
            raise ValueError("Require 0 < dt_min < dt_max.")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        elif isinstance(dt_rank, int) and dt_rank > 0:
            self.dt_rank = dt_rank
        else:
            raise ValueError("dt_rank must be a positive int or 'auto'.")

        factory_kwargs = {"device": device}
        self.in_proj = nn.Linear(
            d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )

        # dt (per channel), plus per-token selective B, C, and A modulation gates.
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + (2 * d_state) + self.d_inner,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

        # A is strictly negative for stable continuous-time dynamics.
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32, device=device))
            .unsqueeze(0)
            .repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        # Initialize dt bias so softplus(dt_bias) starts in [dt_min, dt_max].
        dt = torch.empty(self.d_inner, device=device).uniform_(dt_min, dt_max)
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse softplus: x = log(exp(y) - 1)
        inv_softplus = torch.log(torch.expm1(dt))
        with torch.no_grad():
            self.dt_proj.weight.zero_()
            self.dt_proj.bias.copy_(inv_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor with shape [B, L, d_model].
        Returns:
            Tensor with shape [B, L, d_model].
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected input shape [B, L, {self.d_model}], got {tuple(x.shape)}."
            )

        bsz, seq_len, _ = x.shape
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_branch, z_branch = torch.chunk(xz, chunks=2, dim=-1)

        # Causal depthwise conv over sequence.
        u = x_branch.transpose(1, 2)  # [B, d_inner, L]
        u = self.conv1d(u)[..., :seq_len]
        u = u.transpose(1, 2)  # [B, L, d_inner]
        u = F.silu(u)

        proj = self.x_proj(u)
        dt_raw, b_raw, c_raw, a_gate_raw = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state, self.d_inner], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))  # [B, L, d_inner]
        b_t = b_raw.unsqueeze(2)  # [B, L, 1, d_state]
        c_t = c_raw.unsqueeze(2)  # [B, L, 1, d_state]

        # Base diagonal A from exposed log-param; selective modulation is input-dependent.
        a_base = -torch.exp(self.A_log)  # [d_inner, d_state]
        a_gate = torch.sigmoid(a_gate_raw).unsqueeze(-1)  # [B, L, d_inner, 1]
        a_t = a_base.unsqueeze(0).unsqueeze(0) * (0.5 + a_gate)  # [B, L, d_inner, d_state]

        state = x.new_zeros(bsz, self.d_inner, self.d_state)
        y_steps = []
        for t in range(seq_len):
            dt_t = dt[:, t].unsqueeze(-1)  # [B, d_inner, 1]
            u_t = u[:, t].unsqueeze(-1)  # [B, d_inner, 1]
            b_cur = b_t[:, t]  # [B, 1, d_state]
            c_cur = c_t[:, t]  # [B, 1, d_state]
            a_cur = a_t[:, t]  # [B, d_inner, d_state]

            decay = torch.exp(dt_t * a_cur)
            state = decay * state + dt_t * b_cur * u_t
            y_t = torch.sum(state * c_cur, dim=-1) + self.D.unsqueeze(0) * u[:, t]
            y_steps.append(y_t)

        y = torch.stack(y_steps, dim=1)  # [B, L, d_inner]
        y = y * F.silu(z_branch)
        return self.out_proj(y)
