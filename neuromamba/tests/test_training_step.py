"""Training-step sanity test."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from losses.total_loss import compute_total_loss
from models.neuromamba import NeuroMamba


def _train_cfg():
    return OmegaConf.create(
        {
            "d_model": 32,
            "d_fast": 16,
            "d_slow": 16,
            "d_state": 8,
            "d_conv": 3,
            "expand": 2,
            "n_layers": 1,
            "lag_window": 5,
            "entmax_alpha": 1.5,
            "decoder_hidden": 32,
            "lambda_recon": 1.0,
            "lambda_pred": 1.0,
            "lambda_decor": 0.1,
            "lambda_var": 0.1,
            "lambda_sparse": 0.05,
            "lambda_timescale": 0.1,
            "variance_gamma": 1.0,
            "timescale_margin": 0.1,
        }
    )


def test_single_batch_overfit_and_no_nan_gradients():
    torch.manual_seed(0)
    cfg = _train_cfg()
    model = NeuroMamba(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    x = torch.randn(2, 10, 32)
    batch = {"x": x}

    losses = []
    for _ in range(5):
        outputs = model(x, detach_C=False)
        l_total, _ = compute_total_loss(outputs, batch, model, cfg)
        optimizer.zero_grad(set_to_none=True)
        l_total.backward()

        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()

        optimizer.step()
        losses.append(l_total.item())

    assert losses[-1] < losses[0]
