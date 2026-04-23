"""Visualization utilities for NeuroMamba outputs and diagnostics."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _maybe_save(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_connectivity_heatmap(
    C: np.ndarray,
    title: str = "Directed Connectivity",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot directed connectivity matrix as a heatmap."""
    c = np.asarray(C)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("Expected C with shape [N, N].")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(c, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Source neuron (j)")
    ax.set_ylabel("Target neuron (i)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Connectivity weight")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_reconstruction(
    x: np.ndarray,
    x_recon: np.ndarray,
    n_neurons: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay original and reconstructed traces for example neurons."""
    x_arr = np.asarray(x)
    x_rec_arr = np.asarray(x_recon)
    if x_arr.shape != x_rec_arr.shape or x_arr.ndim != 2:
        raise ValueError("Expected x and x_recon with matching shape [N, T].")

    n_total = x_arr.shape[0]
    n_plot = max(1, min(int(n_neurons), n_total))
    idxs = np.arange(n_plot)

    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.2 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        ax.plot(x_arr[idx], label="original", linewidth=1.5)
        ax.plot(x_rec_arr[idx], label="recon", linewidth=1.2, alpha=0.85)
        ax.set_ylabel(f"n={idx}")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Reconstruction Overlay", y=1.02)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_latent_streams(
    h_fast: np.ndarray,
    h_slow: np.ndarray,
    neuron_idx: int = 0,
    n_dims: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot first latent dimensions of fast and slow streams side by side."""
    hf = np.asarray(h_fast)
    hs = np.asarray(h_slow)

    if hf.ndim == 3:
        if neuron_idx < 0 or neuron_idx >= hf.shape[0]:
            raise ValueError("neuron_idx out of bounds for h_fast.")
        hf_plot = hf[neuron_idx]
    elif hf.ndim == 2:
        hf_plot = hf
    else:
        raise ValueError("h_fast must have shape [N, T, d] or [T, d].")

    if hs.ndim == 3:
        if neuron_idx < 0 or neuron_idx >= hs.shape[0]:
            raise ValueError("neuron_idx out of bounds for h_slow.")
        hs_plot = hs[neuron_idx]
    elif hs.ndim == 2:
        hs_plot = hs
    else:
        raise ValueError("h_slow must have shape [N, T, d] or [T, d].")

    if hf_plot.ndim != 2 or hs_plot.ndim != 2 or hf_plot.shape[0] != hs_plot.shape[0]:
        raise ValueError("Selected fast/slow arrays must be [T, d] with matching T.")

    d_plot = max(1, min(int(n_dims), hf_plot.shape[1], hs_plot.shape[1]))
    fig, axes = plt.subplots(d_plot, 2, figsize=(12, 2.2 * d_plot), sharex=True)
    if d_plot == 1:
        axes = np.asarray([axes])

    for d in range(d_plot):
        axes[d, 0].plot(hf_plot[:, d], color="tab:blue", linewidth=1.3)
        axes[d, 0].set_ylabel(f"dim {d}")
        axes[d, 0].grid(alpha=0.2)
        axes[d, 1].plot(hs_plot[:, d], color="tab:orange", linewidth=1.3)
        axes[d, 1].grid(alpha=0.2)

    axes[0, 0].set_title("Fast stream")
    axes[0, 1].set_title("Slow stream")
    axes[-1, 0].set_xlabel("Timestep")
    axes[-1, 1].set_xlabel("Timestep")
    fig.suptitle(f"Latent Streams (neuron {neuron_idx})", y=1.02)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_gate_values(
    gate: np.ndarray, neuron_idx: int = 0, save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize gate tensor as a time-by-channel heatmap for one neuron."""
    g = np.asarray(gate)
    if g.ndim != 3:
        raise ValueError("Expected gate with shape [N, T, d_fast].")
    if neuron_idx < 0 or neuron_idx >= g.shape[0]:
        raise ValueError("neuron_idx out of bounds for gate.")

    g_neuron = g[neuron_idx]  # [T, d_fast]
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(g_neuron.T, aspect="auto", cmap="magma", origin="lower")
    ax.set_title(f"Gate Values (neuron {neuron_idx})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fast latent dim")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gate value")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_loss_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot train/val loss curves per component."""
    components = ["recon", "pred", "decor", "var", "sparse", "timescale"]
    n_comp = len(components)
    n_cols = 2
    n_rows = int(np.ceil(n_comp / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.2 * n_rows), sharex=False)
    axes = np.asarray(axes).reshape(-1)

    for i, comp in enumerate(components):
        ax = axes[i]
        tr = train_losses.get(comp, [])
        va = val_losses.get(comp, [])
        if len(tr) > 0:
            ax.plot(tr, label="train", linewidth=1.5)
        if len(va) > 0:
            ax.plot(va, label="val", linewidth=1.5)
        ax.set_title(comp)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    for j in range(n_comp, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Loss Curves", y=1.02)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
