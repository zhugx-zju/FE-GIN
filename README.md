# FE-GIN: Finite-Element-Guided Inversion Network for Full-Field Modulus Reconstruction

This repository collects the code used to study full-field Young's modulus reconstruction for functionally graded materials (FGMs) from displacement measurements. The project combines:

- IGFE-based synthetic data generation in MATLAB
- physics-guided U-Net training and evaluation in PyTorch
- an adjoint/L-BFGS-B inverse solver baseline with Tikhonov regularization
- comparison utilities for pure U-Net inference, cold-start ASM inversion, and UNet-warm-start ASM inversion

The codebase follows the FE-GIN workflow described in the manuscript *Finite-element-guided inversion network for full-field reconstruction of graded modulus fields in continuous high-throughput mechanical characterization*.

## What This Repository Covers

- generate displacement-modulus pairs for BIL, EXP, and GRF modulus fields
- preprocess MATLAB data into tensors and FE residual metadata for neural training
- train U-Net models with data, physics-only, and hybrid losses
- benchmark against a classical adjoint-state inverse solver
- compare reconstruction quality, noise robustness, and warm-start behavior on shared fixed test sets

## Repository Layout

```text
.
|-- asm_log/
|   |-- fgm_asm/                  # Adjoint/FE forward-inverse solver package
|   |-- forward_job.py            # Forward simulation entry point
|   |-- inverse_main.py           # Cold-start inverse solver entry point
|   |-- inverse_l_curve.py        # L-curve gamma selection + rerun
|   `-- requirements.txt
|-- asm_unet_compare/
|   |-- pipeline/                 # U-Net / ASM / warm-start comparison utilities
|   |-- run_sample_unet.py
|   |-- run_sample_asm.py
|   |-- run_sample_warm_start.py
|   |-- plot_sample.py
|   |-- plot_sample_asm.py
|   `-- plot_true_modulus.py
|-- data/
|   |-- fixed_test_sets/          # Shared fixed test sets used by batch evaluation
|   `-- data_*/                   # Generated or processed training datasets
|-- data_generation/
|   |-- stenglib-master/          # Third-party MATLAB helper library
|   `-- uniform_pressure_load/    # IGFE forward solver and batch generation scripts
|-- igfe_unet/
|   |-- architectures/            # U-Net, FE residual operators, custom losses
|   |-- configs/                  # Dataset-specific experiment configs
|   |-- model/                    # Training and testing logic
|   |-- postprocess/              # CSV/figure/statistics generation
|   |-- script/                   # Main training, testing, and comparison entry points
|   `-- utils/
|-- trained_models_mix/           # Saved checkpoints and histories
`-- comparison_sample_GN/         # Example comparison outputs
```

## Physical and Data Setup

- problem type: 2D plane-stress elasticity
- specimen size: `9 mm x 9 mm`
- Poisson ratio: `nu = 0.3`
- loading: right-edge force / traction case (`force_load` is the main path used here)
- field resolution used by the learning pipeline: `40 x 40` nodal field
- modulus families: `BIL` (bilinear), `EXP` (exponential), `GRF` (Gaussian-random-field-based), and `MIX` (mixed subsets)

The processed fixed test sets currently stored in `data/fixed_test_sets/force_load` follow the tensor layout:

- displacement input: `[N, 2, 40, 40]`
- modulus output: `[N, 40, 40]`

## Requirements

### MATLAB

`data_generation/` depends on MATLAB plus the bundled `stenglib-master` helper library.

- recommended: MATLAB R2020a or newer
- third-party helper: `data_generation/stenglib-master/`

### Python

The deep-learning, adjoint baseline, and comparison scripts require a standard scientific Python stack plus PyTorch.

Minimum packages used across modules:

- `torch`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

Quick install example:

```bash
pip install -r requirements.txt
```

The root `requirements.txt` covers the common Python dependencies used across the PyTorch and ASM utilities. The file `asm_log/requirements.txt` remains as a minimal baseline-only dependency list.

## End-to-End Workflow

### 1. Generate MATLAB data

Use the scripts in `data_generation/uniform_pressure_load`.

For analytical fields:

```matlab
cd data_generation/uniform_pressure_load
% set dis_type = 'bil' or 'exp' inside batch.m
batch
```

For GRF fields:

```matlab
cd data_generation/uniform_pressure_load
batch_grf
```

These scripts write datasets into:

- `data/data_bil/force_load`
- `data/data_exp/force_load`
- `data/data_grf/force_load`

### 2. Preprocess data for PyTorch

Set the dataset type in `igfe_unet/script/config.json`:

```json
{
  "config_type": "mix"
}
```

Supported values map to `igfe_unet/configs/config_*.py`, for example:

- `bil`
- `exp`
- `grf`
- `mix`
- `layer`

Then run:

```bash
cd igfe_unet/script
python data_process.py
```

This step prepares the tensors and FE residual metadata used by the custom loss functions, including files such as `dof.npy`, `force_ele.npy`, and `force.npy`.

### 3. Train and test U-Net models

Single experiment:

```bash
cd igfe_unet/script
python train_model.py
python test_model.py
```

The default `mix` configuration in `igfe_unet/configs/config_mix.py` currently uses:

- `filters_list = [2, 32, 64, 128]`
- `method = 'LocMixloss'`
- `gamma = 100000000`
- mix ratio `exp:bil:grf = 0.60:0.10:0.30`
- test noise levels `[0, 2, 4, 6, 8, 10]`

Supported loss names in the training code:

- `MSE`
- `LocResloss`
- `GloResloss`
- `LocMixloss`
- `GloMixloss`

In the manuscript terminology these correspond to:

- `MSE-M`
- `LE-M`
- `GE-M`
- `LM-M`
- `GM-M`

### 4. Run batch U-Net evaluation and postprocessing

Useful entry points under `igfe_unet/script`:

- `test_all_models_noise.py`: batch test on fixed mix test sets
- `val_all_models_noise.py`: validation-style batch evaluation
- `compare_robustness.py`: compare MSE / local residual / global residual models
- `compare_gamma.py`: gamma sweep analysis for hybrid losses
- `plot_history.py`: summarize training curves
- `train_mse_ratio.py`, `test_all_models_noise_ratio.py`, `compare_robustness_ratio.py`: dataset-ratio study
- `train_architectures_mse.py`, `test_all_models_noise_arch.py`, `compare_architectures.py`: architecture study

Examples:

```bash
cd igfe_unet/script
python test_all_models_noise.py
python compare_robustness.py
python compare_gamma.py
```

### 5. Run the ASM / adjoint baseline

The classical baseline lives in `asm_log`.

Forward solve:

```bash
cd asm_log
python forward_job.py
```

Cold-start inverse solve:

```bash
cd asm_log
python inverse_main.py
```

L-curve gamma selection followed by a rerun:

```bash
cd asm_log
python inverse_l_curve.py
```

The baseline solver uses SciPy L-BFGS-B in log-modulus space with Tikhonov regularization. Its geometry, mesh, gamma range, bounds, and stopping tolerances are configured in `asm_log/config.py`.

### 6. Compare U-Net, ASM, and warm-start ASM

The comparison pipeline is under `asm_unet_compare`.

Typical manual workflow:

```bash
cd asm_unet_compare
python run_sample_unet.py
python run_sample_asm.py
python run_sample_warm_start.py
python plot_sample.py
python plot_sample_asm.py
python plot_true_modulus.py
```

What each script does:

- `run_sample_unet.py`: save U-Net predictions for selected datasets, sample indices, and noise levels
- `run_sample_asm.py`: run cold-start ASM on selected cases
- `run_sample_warm_start.py`: reuse a saved U-Net prediction as `E_init` for ASM and optionally reuse the cold-start optimal gamma
- `plot_sample.py`: compare U-Net and ASM reconstructions
- `plot_sample_asm.py`: inspect ASM diagnostics such as iteration or L-curve behavior
- `run_sample_batch.py`: batch-style ASM runs for the current config

Important: the comparison pipeline expects the shared fixed test set under `data/fixed_test_sets/force_load`.

## Output Conventions

- MATLAB data generation writes to `data/data_{type}/force_load`
- processed fixed test sets live in `data/fixed_test_sets/force_load`
- U-Net checkpoints and histories are stored under `trained_models_{config_type}/{load_type}/{exp_id}`
- U-Net comparison statistics and figures are written by the `igfe_unet/postprocess` scripts
- ASM forward/inverse outputs are saved into run-specific result folders with per-noise subdirectories
- `asm_unet_compare` writes figures and summaries to configurable output folders such as `comparison_sample` or `comparison_sample_GN`

## Notes on Third-Party Code

`data_generation/stenglib-master` is third-party MATLAB code by Stefan Engblom. Its bundled README states that redistribution is allowed with attribution. If you publish this repository, keep the attribution and the original license statement intact.

## Notes for a Public GitHub Release

This repository currently contains generated artifacts in addition to source code. Before making it public, it is recommended to:

- add a root `.gitignore`
- add a root `LICENSE`
- decide whether large files such as `.mat`, `.pt`, `.pkl`, and generated `.png` files should remain in the main repo, move to Git LFS, or be released separately
- clean any absolute local paths that may still appear in metadata files

The documentation has been centralized in this root README so that the project has a single entry point for future GitHub users.

## Citation

If you use this repository in academic work, please cite the accompanying FE-GIN manuscript:

- Gengxuan Zhu, Ronghao Bao, Weiqiu Chen
- *Finite-element-guided inversion network for full-field reconstruction of graded modulus fields in continuous high-throughput mechanical characterization*

If the paper is published later, add the final journal citation or BibTeX entry here.
