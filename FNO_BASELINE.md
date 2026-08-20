# FNO baseline

This branch adds two Fourier Neural Operator baselines while keeping the
repository's existing data, model, and script layout:

- `FNO-MSE`: the repository-local implementation in `architectures/fno.py`.
- `FNO-neuraloperator-MSE`: the optional `neuraloperator` implementation in
  `fno/neuraloperator.py`.

The model accepts displacement tensors with shape `[N, 2, 40, 40]` and returns
modulus fields with shape `[N, 40, 40]`. The coordinate grid is generated
inside the model and is not written into the dataset. No input or target
normalization is added, matching the current U-Net loader.

## Environment

No third-party FNO package is required. `igfe_unet/architectures/fno.py`
implements the spectral convolution with `torch.fft`; the FNO training and
testing functions are organized under `igfe_unet/fno/`. Install PyTorch first,
using the command matching the remote server.

For a CUDA server, check the driver/CUDA combination first:

```bash
nvidia-smi
```

Then use the matching official PyTorch wheel. Examples:

```bash
# CUDA 12.4 server: install a cu124 PyTorch wheel in the active conda env.
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

# CPU-only alternative
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
```

After PyTorch is installed, install the remaining repository dependencies:

```bash
python -m pip install numpy scipy matplotlib pandas
```

The custom FNO needs no package beyond the repository's normal dependencies.
For the optional package-backed comparison, install it in the same activated
environment after PyTorch:

```bash
python -m pip install -r requirements_fno_neuralop.txt
```

For a clean CUDA 12.4 conda environment on the remote server, the complete
installation sequence is:

```bash
conda create -n fno_neuralop python=3.10 -y
conda activate fno_neuralop
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -r requirements_fno_neuralop.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import neuralop; print(neuralop.__file__)"
```

The NVIDIA driver must support CUDA 12.4. The locally installed CUDA toolkit is
not used by the standard PyTorch wheel; `nvidia-smi` is the authoritative
driver check. If the server's driver cannot run the cu124 wheel, install the
newest PyTorch CUDA wheel supported by that driver and keep the rest of the
commands unchanged.

This installs the package that provides the `neuralop` Python module. Verify
the environment before running the project:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The current Windows development environment uses:

```powershell
C:\Users\zhu_g\.conda\envs\torch\python.exe
```

On a remote Linux server, the equivalent is normally `python` from the
activated conda or virtual environment.

## Data layout

FNO uses the same `data_path` and split layout as U-Net. The optional
`FNO_DATA_PATH` environment variable can override that path for a remote run.
The preferred layout is:

```text
data/data_mix/force_load/
|-- train/input.mat
|-- train/output.mat
|-- val/input.mat
|-- val/output.mat
`-- dataset_manifest.json
```

The fixed public test sets should remain separate:

```text
data/fixed_test_sets/force_load/data_mix/input.mat
data/fixed_test_sets/force_load/data_mix/output.mat
```

The evaluator derives the BIL, EXP, and GRF fixed-test paths from the MIX path.
Do not train from `fixed_test_sets`.

## Run order

Run the following steps from the repository root.

### 1. Select the dataset and validate imports

```bash
export FNO_DATA_PATH=/absolute/path/to/data/data_mix/force_load
python -m py_compile \
  igfe_unet/architectures/fno.py \
  igfe_unet/configs/config_fno.py \
  igfe_unet/configs/config_fno_neuralop.py \
  igfe_unet/fno/config.py \
  igfe_unet/fno/common.py \
  igfe_unet/fno/neuraloperator.py \
  igfe_unet/fno/train.py \
  igfe_unet/fno/test.py \
  igfe_unet/script/train_fno.py \
  igfe_unet/script/test_fno.py \
  igfe_unet/script/train_fno_neuralop.py \
  igfe_unet/script/test_fno_neuralop.py
```

On PowerShell, use `$env:FNO_DATA_PATH = 'D:\path\to\data_mix\force_load'`.

### 2. Train the custom FNO baseline

The default data path is `data/data_mix/force_load`; `FNO_DATA_PATH` overrides
it without changing `config_mix.py`:

```bash
python igfe_unet/script/train_fno.py
```

The defaults are stored in `igfe_unet/configs/config_fno.py`: `width=32`,
`modes1=16`, `modes2=16`, `n_layers=4`, `n_epochs=1500`, and `seed=42`. The
network settings follow a common literature-style FNO configuration. The
optimizer, scheduler, early stopping, and batch size remain fixed to the
repository settings. Edit the FNO-only structural parameters in that file for
a controlled comparison; each changed width/mode/layer configuration gets a
separate model directory. The custom and NeuralOperator backends load the same
structural defaults.

The default model has about 4.20 million trainable real-scalar parameters. This
is a conventional FNO capacity reference rather than a parameter-count-matched
U-Net baseline. A parameter-matched setting can be created by editing the FNO
structural parameters in the configuration. Complex spectral weights are
counted as two real scalar values for comparison with U-Net.

Outputs use the same checkpoint and history format as U-Net and are written
under:

```text
trained_models_fno/force_load/std/FNO_custom_w32_m16x16_l4/
trained_models_fno/force_load/std/FNO_neuralop_w32_m16x16_l4/
```

### 3. Evaluate the selected checkpoint

After training, run the test script directly. It resolves the checkpoint using
the same configuration and run directory as U-Net:

```bash
python igfe_unet/script/test_fno.py
```

The evaluator uses the shared fixed test-set convention and writes U-Net-style
`all_samples_*` or `selected_samples_*` folders with `L1` text files and
`npz`/`mat` prediction files. The dataset types, noise levels, sample index,
and optional sample limit are configured in `igfe_unet/configs/config_fno.py`.

### 4. Train and evaluate FNO-neuraloperator

The optional implementation uses the same defaults, data protocol, checkpoint
format, and test output format. Install `requirements_fno_neuralop.txt` before
starting this step. Its checkpoint is isolated from the custom baseline by the
backend in the run identifier:

```text
trained_models_fno/force_load/std/FNO_neuralop_w32_m16x16_l4/
```

Train it:

```bash
python igfe_unet/script/train_fno_neuralop.py
```

Evaluate the checkpoint written by that run:

```bash
python igfe_unet/script/test_fno_neuralop.py
```

The default run id is `FNO_neuralop_w32_m16x16_l4`. Programmatic callers can
pass a specific checkpoint to `fno.test.test_fno` when a historical run must
be evaluated.

### 5. Inspect outputs

Use the saved `L1` files and prediction arrays for the method comparison, in
the same way as U-Net. Only after both implementations are validated should
their results be merged into `asm_unet_compare`.

### 6. Select and visualize the baseline

After an architecture sweep has produced validation outputs, select the FNO
architecture from `all_samples_val/all_L1_val_mix.txt` only:

```bash
python igfe_unet/script/select_fno_baseline.py
```

The script records the validation score and trainable parameter count and
copies the winner into the `final_model` group without replacing an existing
destination. To compare representative fields with the selected MSE-U-Net,
LocMix-U-Net, and GloMix-U-Net models on the same fixed-test samples and
noise levels:

```bash
python igfe_unet/script/compare_fno_unet_fields.py
```

The comparison reuses the existing ASM/U-Net mesh contour and colourbar
conventions and saves prediction/error PNG/PDF panels, per-noise `.npz`
fields, FNO-compatible result/config files, and a metrics CSV into the
existing `results/force_load/asm_unet_comparison/GN/sample_<index>/<dataset>/`
hierarchy without replacing the existing ASM/U-Net figures.

### 7. Reproduce the Appendix tables and FNO-only figures

After the architecture candidates have been validated and the selected final
model has been tested, export the exact statistics used in Tables F1 and F2:

```bash
python igfe_unet/script/compare_fno_architectures.py
```

This produces a validation-only architecture table, a selected-model test
table covering BIL/EXP/GRF/MIX at all six noise levels, and a ready-to-copy
Markdown file:

```text
results_fno/force_load/architecture_sweep/all_experiments.csv
results_fno/force_load/architecture_sweep/selected_fno_test_results.csv
results_fno/force_load/architecture_sweep/fno_appendix_tables.md
```

Generate the Appendix sample predictions and relative-error grids directly
from the selected FNO checkpoint with:

```bash
python igfe_unet/script/generate_fno_appendix_figures.py
```

This command performs FNO inference for BIL sample 600, EXP sample 200, and
GRF sample 410 at noise levels 0%, 2%, 4%, 6%, 8%, and 10%. It does not require
pre-existing `.npz` fields or any U-Net/ASM checkpoint. The final PNG and PDF
figures are written to
`results/force_load/asm_unet_comparison/GN/fno_sample_grid/`.
