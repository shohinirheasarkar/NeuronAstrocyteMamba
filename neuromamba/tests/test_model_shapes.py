"""Shape and basic output property tests for NeuroMamba."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from models.neuromamba import NeuroMamba


def _default_cfg():
    return OmegaConf.create(
        {
            "d_model": 128,
            "d_fast": 64,
            "d_slow": 64,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "n_layers": 2,
            "lag_window": 10,
            "entmax_alpha": 1.5,
            "decoder_hidden": 128,
        }
    )


def test_neuromamba_output_shapes_and_connectivity_properties():
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = NeuroMamba(cfg)

    x = torch.randn(2, 10, 64)
    outputs = model(x, detach_C=False)

    assert outputs["x_recon"].shape == (2, 10, 64)
    assert outputs["h_fast"].shape == (2, 10, 64, cfg.d_fast)
    assert outputs["h_slow"].shape == (2, 10, 64, cfg.d_slow)
    assert outputs["h_fast_mod"].shape == (2, 10, 64, cfg.d_fast)
    assert outputs["connectivity"].shape == (2, 10, 10)
    assert outputs["h_fast_pred"].shape == (2, 10, 63, cfg.d_fast)
    assert outputs["gate"].shape == (2, 10, 64, cfg.d_fast)

    c = outputs["connectivity"]
    assert torch.all(c >= 0.0)
    assert torch.all(c <= 1.0)

    row_sums = c.sum(dim=-1)
    assert torch.all(row_sums <= 1.0 + 1e-5)

    # Entmax should produce some exact zeros for sparse connectivity.
    assert (c == 0).any()
