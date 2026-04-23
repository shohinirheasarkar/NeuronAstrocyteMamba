"""Data loading helpers and PyTorch DataLoader builders."""

from __future__ import annotations

from typing import Tuple

import h5py
import numpy as np
from torch.utils.data import DataLoader

from .dataset import CalciumTraceDataset


def build_dataloaders(
    train_data: CalciumTraceDataset,
    val_data: CalciumTraceDataset,
    test_data: CalciumTraceDataset,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with standard shuffle behavior."""
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def load_traces_from_npy(filepath: str) -> np.ndarray:
    """Load calcium traces array [N, T] from a .npy file."""
    traces = np.load(filepath)
    traces = np.asarray(traces, dtype=np.float32)
    if traces.ndim != 2:
        raise ValueError("Expected traces with shape [N, T].")
    return traces


def load_traces_from_h5(filepath: str, key: str = "traces") -> np.ndarray:
    """Load calcium traces array [N, T] from an HDF5 file."""
    with h5py.File(filepath, "r") as handle:
        if key not in handle:
            raise KeyError(f"Key '{key}' not found in HDF5 file.")
        traces = np.asarray(handle[key], dtype=np.float32)
    if traces.ndim != 2:
        raise ValueError("Expected traces with shape [N, T].")
    return traces
