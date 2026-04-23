# NeuroMamba

NeuroMamba is an end-to-end deep learning model for inferring **directed functional connectivity** from calcium fluorescence recordings. Given neuronal traces with shape `[N, T]`, it encodes each neuron with shared temporal dynamics, separates latent activity into fast and slow timescales, uses slow-to-fast gating, and computes asymmetric source-to-target connectivity from source past states and target present states. The model jointly reconstructs input signals and learns sparse, directed connectivity through a multi-term objective.

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart: Train With Default Config

From the `neuromamba/` directory:

```bash
python -m scripts.train
```

If your training script expects explicit config paths, use:

```bash
python -m scripts.train --config-name default
```

## Inference on a New Recording

Use the inference script to generate a connectivity matrix from a trained checkpoint:

```bash
python -m scripts.infer_connectivity \
  --checkpoint checkpoints/epoch_100.pt \
  --data_path /path/to/traces.npy \
  --output_dir outputs/inference_run \
  --window_size 128 \
  --stride 64
```

Outputs saved to `output_dir`:
- `connectivity.npy` (aggregated mean connectivity `[N, N]`)
- `connectivity_std.npy` (window-wise variability `[N, N]`)
- `connectivity.png` (heatmap visualization)

## Config Files and Key Parameters

Configuration lives in:
- `configs/model.yaml`
- `configs/train.yaml`
- `configs/default.yaml` (composes model + train settings)

Important parameters to tune:
- **Model capacity:** `d_model`, `d_fast`, `d_slow`, `n_layers`, `d_state`
- **Temporal context:** `lag_window`, `window_size`, `window_stride`
- **Sparsity behavior:** `entmax_alpha`, `lambda_sparse`
- **Optimization:** `learning_rate`, `weight_decay`, `max_epochs`, `warmup_epochs`, `grad_clip`
- **Auxiliary constraints:** `lambda_decor`, `lambda_var`, `lambda_timescale`, `variance_gamma`, `timescale_margin`
- **Connectivity warmup:** `detach_C_warmup_epochs`

## Expected Input Format

- Primary input is a NumPy array with shape `[N, T]`:
  - `N` = number of neurons
  - `T` = number of timesteps
- Supported file formats in inference:
  - `.npy`
  - `.h5` / `.hdf5` (dataset key: `traces`)

## Expected Output Format

- Predicted connectivity matrix with shape `[N, N]`
  - Entry `(i, j)` is the directed influence score from source neuron `j` to target neuron `i`
  - Matrix is generally asymmetric (`C[i, j] != C[j, i]`)
  - Entmax sparsification produces exact zeros for weak/nonexistent edges

## Run Unit Tests

From `neuromamba/`:

```bash
pytest tests/
```

