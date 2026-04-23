"""Dataset and windowing tests."""

from __future__ import annotations

import numpy as np

from data.dataset import CalciumTraceDataset
from data.windowing import make_sliding_windows, split_train_val_test


def test_calcium_trace_dataset_shapes():
    windows = np.random.randn(12, 7, 20).astype(np.float32)
    ds = CalciumTraceDataset(windows=windows, session_ids=None, neuron_ids=None, transform=None)

    assert len(ds) == 12
    item = ds[3]
    assert set(item.keys()) == {"x", "session_id", "window_idx"}
    assert tuple(item["x"].shape) == (7, 20)
    assert isinstance(item["session_id"], str)
    assert item["window_idx"] == 3


def test_make_sliding_windows_count():
    x = np.random.randn(5, 100).astype(np.float32)
    window_size = 20
    stride = 10
    windows = make_sliding_windows(x, window_size=window_size, stride=stride)

    expected = ((100 - window_size) // stride) + 1
    assert windows.shape == (expected, 5, window_size)


def test_split_sizes_are_approximately_correct():
    windows = np.random.randn(101, 6, 16).astype(np.float32)
    val_fraction = 0.15
    test_fraction = 0.10
    train, val, test = split_train_val_test(
        windows, val_fraction=val_fraction, test_fraction=test_fraction, by_session=True
    )

    assert len(train) + len(val) + len(test) == len(windows)
    assert abs(len(val) - int(np.floor(len(windows) * val_fraction))) <= 1
    assert abs(len(test) - int(np.floor(len(windows) * test_fraction))) <= 1


def test_no_temporal_leakage_in_contiguous_split():
    # Encode time index directly in values so window starts are recoverable.
    x = np.arange(0, 60, dtype=np.float32).reshape(1, 60)
    windows = make_sliding_windows(x, window_size=10, stride=5)
    train, val, test = split_train_val_test(
        windows, val_fraction=0.2, test_fraction=0.2, by_session=True
    )

    def starts(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0:
            return np.array([], dtype=np.float32)
        return arr[:, 0, 0]

    train_starts = starts(train)
    val_starts = starts(val)
    test_starts = starts(test)

    if len(val_starts) > 0:
        assert val_starts.min() > train_starts.max()
    if len(test_starts) > 0:
        ref = val_starts.max() if len(val_starts) > 0 else train_starts.max()
        assert test_starts.min() > ref
