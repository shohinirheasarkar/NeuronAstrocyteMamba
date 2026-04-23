"""Run trained NeuroMamba inference and save directed connectivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from neuromamba.data.loaders import load_traces_from_h5, load_traces_from_npy
from neuromamba.data.preprocessing import zscore_traces
from neuromamba.data.windowing import make_sliding_windows
from neuromamba.eval.connectivity_metrics import (
    aggregate_connectivity_over_windows,
    connectivity_stability,
)
from neuromamba.eval.visualization import plot_connectivity_heatmap
from neuromamba.models.neuromamba import NeuroMamba


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer directed connectivity using NeuroMamba.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Input trace file (.npy/.h5).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--window_size", type=int, required=True, help="Sliding window size.")
    parser.add_argument("--stride", type=int, required=True, help="Sliding window stride.")
    return parser


def _load_cfg_from_checkpoint(raw_cfg: object) -> DictConfig:
    if isinstance(raw_cfg, DictConfig):
        return raw_cfg
    if isinstance(raw_cfg, dict):
        return OmegaConf.create(raw_cfg)
    raise ValueError("Checkpoint does not contain a valid cfg object.")


def _load_traces(data_path: str) -> np.ndarray:
    suffix = Path(data_path).suffix.lower()
    if suffix == ".npy":
        return load_traces_from_npy(data_path)
    if suffix in {".h5", ".hdf5"}:
        return load_traces_from_h5(data_path, key="traces")
    raise ValueError("Unsupported file format. Expected .npy, .h5, or .hdf5.")


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = _load_cfg_from_checkpoint(checkpoint["cfg"])

    device_str = str(getattr(cfg, "device", "cpu"))
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model = NeuroMamba(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    traces = _load_traces(args.data_path)  # [N, T]
    traces = zscore_traces(traces)
    windows = make_sliding_windows(
        traces, window_size=int(args.window_size), stride=int(args.stride)
    )  # [W, N, T_ctx]

    c_list = []
    with torch.no_grad():
        for window in windows:
            x_win = torch.from_numpy(window).unsqueeze(0).to(device)  # [1, N, T_ctx]
            outputs = model(x_win, detach_C=True)
            c = outputs["connectivity"].squeeze(0).detach().cpu().numpy()  # [N, N]
            c_list.append(c)

    aggregated_C, aggregated_std = aggregate_connectivity_over_windows(c_list)
    np.save(output_dir / "connectivity.npy", aggregated_C)
    np.save(output_dir / "connectivity_std.npy", aggregated_std)

    fig = plot_connectivity_heatmap(
        aggregated_C, title="Directed Connectivity", save_path=str(output_dir / "connectivity.png")
    )
    # Explicit close for long-running scripts.
    import matplotlib.pyplot as plt

    plt.close(fig)

    n_neurons, t_steps = traces.shape
    off_diag_mask = ~np.eye(n_neurons, dtype=bool)
    density = float((aggregated_C[off_diag_mask] > 0).mean())
    stability = connectivity_stability(c_list)

    print(f"N neurons: {n_neurons}")
    print(f"T timesteps: {t_steps}")
    print(f"Num windows: {len(c_list)}")
    print(f"Connectivity density (off-diagonal > 0): {density:.6f}")
    print(f"Connectivity stability: {stability:.6f}")
    print(f"Saved connectivity to: {output_dir / 'connectivity.npy'}")
    print(f"Saved std map to: {output_dir / 'connectivity_std.npy'}")
    print(f"Saved heatmap to: {output_dir / 'connectivity.png'}")


if __name__ == "__main__":
    main()
