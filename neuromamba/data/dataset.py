"""Dataset definitions for calcium trace windows."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CalciumTraceDataset(Dataset):
    """PyTorch dataset over calcium windows of shape [num_windows, N, T]."""

    def __init__(
        self,
        windows: np.ndarray,
        session_ids: Optional[List[str]] = None,
        neuron_ids: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 3:
            raise ValueError("Expected windows with shape [num_windows, N, T].")

        self.windows = windows
        self.transform = transform
        self.neuron_ids = neuron_ids

        num_windows = windows.shape[0]
        if session_ids is None:
            self.session_ids = [f"session_{i}" for i in range(num_windows)]
        else:
            if len(session_ids) != num_windows:
                raise ValueError("session_ids length must match number of windows.")
            self.session_ids = [str(sid) for sid in session_ids]

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.windows[idx]
        if self.transform is not None:
            x = self.transform(x)

        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        return {
            "x": x_tensor,
            "session_id": self.session_ids[idx],
            "window_idx": idx,
        }
