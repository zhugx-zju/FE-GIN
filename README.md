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
|-- data/                         # Local-only datasets, fixed test sets, and processed tensors
|-- trained_models_mix/           # Local-only checkpoints and histories
`-- results/                      # Local-only statistics, figures, and ASM/U-Net outputs
```

The current `force_load` asset layout is organized by experiment purpose:

```text
trained_models_mix/force_load/
|-- std/                          # Standard physical-loss models
|-- gamma/                        # Physical-term coefficient sweeps
|-- arch/                         # Architecture sweep models
|-- ratio/                        # Dataset-ratio sweep models
`-- final_model/                  # Three selected models for final sample comparisons
    |-- MSE_UNet_GN_arch_32-64-128/
    |-- LocMix_UNet_GN_arch_32-64-128_gamma_100000/
    `-- GloMix_UNet_GN_arch_32-64-128_gamma_10000/
```

The models under `final_model/` are copied-in, user-selected checkpoints. They
are used by the selected-sample U-Net, ASM, warm-start, and comparison
workflows; they are not an additional training experiment group.

The corresponding grouped analysis and comparison outputs are stored below:

```text
results/force_load/
|-- standard_models/GN/
|-- gamma_sweep/GN/{val,test}/
|-- architecture_sweep/GN/{val,test}/
|-- loss_ratio_sweep/GN/{val,test}/
`-- asm_unet_comparison/GN/
    |-- sample_<index>/<dataset>/comparison/
    `-- fixed_gamma_<value>/GN/sample_<index>/<dataset>/comparison/
```

`GN` means the model uses Group Normalization. The analysis directories mirror
the checkpoint experiment groups, while ASM/U-Net sample outputs are grouped
under `asm_unet_comparison`.

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

## External Assets

This GitHub repository is source-only. The dataset and pretrained model assets are released separately on Hugging Face:

- dataset repository: `zhugx/fe_gin_data`
- model repository: `zhugx/fe_gin_models`

Recommended local layout after download:

```text
FE-GIN/
|-- data/                 # contents from the Hugging Face dataset repo
`-- trained_models_mix/   # downloaded model package or extracted model directory
```

Current asset usage:

- the dataset repository should be placed at `data/`
- the model repository currently distributes the trained model package separately from the code repo
- if the model repository is provided as a compressed archive, extract it so that `trained_models_mix/force_load/...` exists locally

Minimal setup after cloning:

```bash
git clone https://github.com/zhugx-zju/FE-GIN.git
cd FE-GIN
pip install -r requirements.txt
```

Then download:

- FE-GIN dataset assets from the Hugging Face dataset repo and place them under `data/`
- FE-GIN model assets from the Hugging Face model repo and place/extract them under `trained_models_mix/`

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

For a complete local evaluation setup, create or obtain the fixed test sets
under `data/fixed_test_sets/force_load` before running batch test scripts. The
repository already contains the fixed test-set layout used by the current
comparison workflow.

### 3. Train U-Net model groups

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

For the grouped studies, use the scripts in this order:

1. `train_model.py` for standard physical-loss experiments and individually configured runs.
2. `train_architectures_mse.py` for the architecture group.
3. `train_mse_ratio.py` for the dataset-ratio group.

The architecture group changes the network architecture while using the
current prepared mix dataset. The ratio group explicitly rebuilds the mix
dataset for each requested ratio.

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

The usual evaluation order is:

```bash
cd igfe_unet/script
python val_all_models_noise.py
python val_all_models_noise_arch.py
python val_all_models_noise_ratio.py
python test_all_models_noise.py
python test_all_models_noise_arch.py
python test_all_models_noise_ratio.py
python compare_robustness.py
python compare_gamma.py
python compare_architectures.py
python compare_robustness_ratio.py
```

The three `test_all_models_noise*.py` scripts use the shared fixed test sets.
The `val_all_models_noise*.py` scripts evaluate the current prepared dataset
split and should be run according to the dataset-generation order used for the
corresponding experiment group.

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

For the current selected `final_model/` checkpoints, the fixed-gamma smoke
comparison can be run with:

```bash
cd asm_unet_compare
python run_sample_unet.py
python run_fixed_gamma_comparison.py
```

`run_fixed_gamma_comparison.py` runs cold-start ASM with several fixed gamma
values and compares each result with the saved MSE, LocMixloss, and GloMixloss
predictions. Each gamma is written to its own directory so ASM results cannot
overwrite one another.

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
- `asm_unet_compare` writes figures and summaries under `results/force_load/asm_unet_comparison/GN/`
- fixed-gamma ASM comparison outputs are written under
  `results/force_load/asm_unet_comparison/fixed_gamma_<value>/GN/`

## Data and Model Assets

This GitHub repository is intended to be source-only. The following directories are treated as local assets and are ignored by git:

- `data/`
- `trained_models*/`
- `comparison_*`
- `history/`

To reproduce the full workflow after cloning, you need to separately obtain or regenerate:

- raw/generated MATLAB datasets
- processed fixed test sets
- trained PyTorch checkpoints
- comparison figures and baseline outputs

For the current public release workflow:

- source code lives in this GitHub repository
- datasets live in the Hugging Face dataset repository `zhugx/fe_gin_data`
- pretrained models live in the Hugging Face model repository `zhugx/fe_gin_models`

## Notes on Third-Party Code

`data_generation/stenglib-master` is third-party MATLAB code by Stefan Engblom. Its bundled README states that redistribution is allowed with attribution. If you publish this repository, keep the attribution and the original license statement intact.

## Notes for a Public GitHub Release

The repository has been prepared as a source-only public codebase:

- generated datasets are excluded from git
- trained checkpoints and comparison outputs are excluded from git
- a root `.gitignore`, `LICENSE`, `requirements.txt`, and `CITATION.cff` are included
- known absolute local paths in fixed-test-set metadata were removed from tracked files

The documentation has been centralized in this root README so that the project has a single entry point for future GitHub users.

## Citation

If you use this repository in academic work, please cite the accompanying FE-GIN manuscript:

- Gengxuan Zhu, Ronghao Bao, Weiqiu Chen
- *Finite-element-guided inversion network for full-field reconstruction of graded modulus fields in continuous high-throughput mechanical characterization*

If the paper is published later, add the final journal citation or BibTeX entry here.
