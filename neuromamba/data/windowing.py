"""Windowing and train/val/test splitting utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def make_sliding_windows(x: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Create sliding windows from traces of shape [N, T]."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Expected x with shape [N, T].")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")

    n_neurons, t_steps = x.shape
    if window_size > t_steps:
        raise ValueError("window_size cannot exceed number of timesteps.")

    starts = np.arange(0, t_steps - window_size + 1, stride)
    windows = np.stack([x[:, s : s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32)


def split_train_val_test(
    windows: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    by_session: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windows into train/val/test.

    If by_session=True, keeps split contiguous by time-order (session-like split)
    so no future windows leak into train from validation/test segments.
    If by_session=False, performs a random split at window level.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError("Expected windows with shape [num_windows, N, T].")
    if not (0.0 <= val_fraction < 1.0) or not (0.0 <= test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0.")

    num_windows = windows.shape[0]
    n_test = int(np.floor(num_windows * test_fraction))
    n_val = int(np.floor(num_windows * val_fraction))
    n_train = num_windows - n_val - n_test

    if n_train <= 0:
        raise ValueError("Split results in empty training set.")

    if by_session:
        train_end = n_train
        val_end = n_train + n_val

        train = windows[:train_end]
        val = windows[train_end:val_end]
        test = windows[val_end:]
        return train, val, test

    perm = np.random.permutation(num_windows)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train = windows[train_idx]
    val = windows[val_idx]
    test = windows[test_idx]
    return train, val, test
