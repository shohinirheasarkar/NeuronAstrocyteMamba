"""Preprocessing utilities for calcium trace arrays."""

from __future__ import annotations

import numpy as np


def zscore_traces(x: np.ndarray) -> np.ndarray:
    """Z-score each neuron's trace across time."""
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    # Return zeros for zero-variance neurons.
    safe_std = np.where(std > 0.0, std, 1.0)
    z = (x - mean) / safe_std
    return np.where(std > 0.0, z, 0.0).astype(np.float32)


def dff_normalize(x: np.ndarray, percentile: float = 10.0) -> np.ndarray:
    """Compute per-neuron delta-F over F normalization."""
    x = np.asarray(x, dtype=np.float32)
    f0 = np.percentile(x, percentile, axis=1, keepdims=True)
    return ((x - f0) / (f0 + 1e-6)).astype(np.float32)


def clip_outliers(x: np.ndarray, n_std: float = 5.0) -> np.ndarray:
    """Clip per-neuron values to mean +/- n_std * std."""
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    lower = mean - n_std * std
    upper = mean + n_std * std
    return np.clip(x, lower, upper).astype(np.float32)


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along the time axis per neuron."""
    x = np.asarray(x, dtype=np.float32).copy()
    n_neurons, t_steps = x.shape
    t_idx = np.arange(t_steps)

    for n in range(n_neurons):
        row = x[n]
        nan_mask = np.isnan(row)
        if not np.any(nan_mask):
            continue
        valid_mask = ~nan_mask
        if not np.any(valid_mask):
            # If all values are NaN, default to zeros.
            x[n] = 0.0
            continue
        x[n, nan_mask] = np.interp(t_idx[nan_mask], t_idx[valid_mask], row[valid_mask])

    return x.astype(np.float32)
