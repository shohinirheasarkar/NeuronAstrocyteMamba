"""Evaluation metrics for inferred directed connectivity."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _off_diagonal_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def auroc_auprc(C_pred: np.ndarray, C_true: np.ndarray) -> Dict[str, float]:
    """
    Compute AUROC and AUPRC on off-diagonal connectivity entries only.
    """
    c_pred = np.asarray(C_pred)
    c_true = np.asarray(C_true)
    if c_pred.shape != c_true.shape or c_pred.ndim != 2 or c_pred.shape[0] != c_pred.shape[1]:
        raise ValueError("Expected C_pred and C_true as same-shape [N, N] matrices.")

    mask = _off_diagonal_mask(c_pred.shape[0])
    y_score = c_pred[mask].reshape(-1)
    y_true = c_true[mask].reshape(-1)

    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def aggregate_connectivity_over_windows(
    C_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate window-level connectivity with element-wise mean and std.
    """
    if len(C_list) == 0:
        raise ValueError("C_list cannot be empty.")

    stack = np.stack([np.asarray(c) for c in C_list], axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def connectivity_stability(C_list: List[np.ndarray]) -> float:
    """
    Mean pairwise correlation across off-diagonal entries over windows.
    """
    if len(C_list) < 2:
        return 1.0

    stack = np.stack([np.asarray(c) for c in C_list], axis=0)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise ValueError("Expected list of [N, N] matrices.")

    mask = _off_diagonal_mask(stack.shape[1])
    flat = stack[:, mask]  # [W, N*(N-1)]

    corrs = []
    for i, j in combinations(range(flat.shape[0]), 2):
        a = flat[i]
        b = flat[j]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        corrs.append(corr)

    return float(np.mean(corrs)) if corrs else 0.0
