"""Unit tests for NeuroMamba loss functions."""

from __future__ import annotations

import torch

from losses.decorrelation import decorrelation_loss
from losses.predictive import latent_predictive_loss
from losses.reconstruction import reconstruction_loss
from losses.sparsity import sparsity_loss
from losses.timescale import timescale_separation_loss
from losses.variance import variance_floor_loss


def _is_scalar_tensor(x: torch.Tensor) -> bool:
    return isinstance(x, torch.Tensor) and x.ndim == 0


def test_losses_run_and_return_scalar_tensors():
    torch.manual_seed(0)
    bsz, n, t, d_fast, d_slow = 2, 8, 16, 6, 4
    x = torch.randn(bsz, n, t)
    x_recon = torch.randn(bsz, n, t)
    h_fast = torch.randn(bsz, n, t, d_fast)
    h_fast_pred = torch.randn(bsz, n, t - 1, d_fast)
    h_slow = torch.randn(bsz, n, t, d_slow)
    c = torch.randn(bsz, n, n)
    fast_a_logs = [torch.randn(12, 16), torch.randn(12, 16)]
    slow_a_logs = [torch.randn(12, 16), torch.randn(12, 16)]

    losses = [
        reconstruction_loss(x, x_recon),
        latent_predictive_loss(h_fast, h_fast_pred),
        decorrelation_loss(h_fast, h_slow),
        variance_floor_loss(h_fast, gamma=1.0),
        sparsity_loss(c),
        timescale_separation_loss(fast_a_logs, slow_a_logs, margin=0.1),
    ]
    assert all(_is_scalar_tensor(loss) for loss in losses)


def test_reconstruction_loss_zero_for_perfect_reconstruction():
    x = torch.randn(2, 4, 8)
    loss = reconstruction_loss(x, x.clone())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_latent_predictive_loss_zero_for_perfect_prediction():
    h_fast = torch.randn(2, 3, 10, 5)
    h_fast_pred = h_fast[:, :, 1:, :].clone()
    loss = latent_predictive_loss(h_fast, h_fast_pred)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_decorrelation_loss_lower_for_orthogonal_streams():
    torch.manual_seed(0)
    bsz, n, t, d = 2, 4, 10, 6
    hf = torch.randn(bsz, n, t, d)
    hs_corr = hf + 0.01 * torch.randn_like(hf)
    hs_orth = torch.randn(bsz, n, t, d)
    hs_orth = hs_orth - hs_orth.mean(dim=(0, 1, 2), keepdim=True)
    hs_orth = hs_orth - (hs_orth * hf).mean() * hf / (hf.pow(2).mean() + 1e-8)

    l_corr = decorrelation_loss(hf, hs_corr)
    l_orth = decorrelation_loss(hf, hs_orth)
    assert l_orth < l_corr


def test_variance_floor_loss_zero_when_std_above_gamma():
    torch.manual_seed(0)
    h = torch.randn(4, 5, 6, 8) * 3.0
    loss = variance_floor_loss(h, gamma=1.0)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_sparsity_loss_zero_for_zero_connectivity():
    c = torch.zeros(2, 5, 5)
    loss = sparsity_loss(c)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_timescale_separation_loss_zero_when_fast_exceeds_slow_by_margin():
    fast_a_logs = [torch.log(torch.full((10, 16), 2.0)), torch.log(torch.full((10, 16), 3.0))]
    slow_a_logs = [torch.log(torch.full((10, 16), 0.1)), torch.log(torch.full((10, 16), 0.2))]
    loss = timescale_separation_loss(fast_a_logs, slow_a_logs, margin=0.1)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)
