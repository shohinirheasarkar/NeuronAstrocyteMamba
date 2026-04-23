"""Timescale separation loss between fast and slow SSM streams."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def timescale_separation_loss(
    fast_A_logs: List[torch.Tensor],
    slow_A_logs: List[torch.Tensor],
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Encourage fast stream to have larger decay magnitude than slow stream.

    With A = -exp(A_log), |A| = exp(A_log).
    """
    fast_decay = torch.stack([a.exp().mean() for a in fast_A_logs]).mean()
    slow_decay = torch.stack([a.exp().mean() for a in slow_A_logs]).mean()
    return F.relu(margin - (fast_decay - slow_decay))
