# FNO baseline

This branch adds two Fourier Neural Operator baselines while keeping the
repository's existing data, model, and script layout:

- `FNO-MSE`: the repository-local implementation in `architectures/fno.py`.
- `FNO-neuraloperator-MSE`: the optional `neuraloperator` implementation in
  `architectures/fno_neuralop.py`.

The model accepts displacement tensors with shape `[N, 2, 40, 40]` and returns
modulus fields with shape `[N, 40, 40]`. The coordinate grid is generated
inside the model and is not written into the dataset. No input or target
normalization is added, matching the current U-Net loader.

## Environment

No third-party FNO package is required. `igfe_unet/architectures/fno.py`
implements the spectral convolution with `torch.fft`. Install PyTorch first,
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

Set `FNO_DATA_PATH` to the same dataset used by U-Net. The preferred layout is:

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
  igfe_unet/architectures/fno_neuralop.py \
  igfe_unet/model/fno_train.py \
  igfe_unet/model/fno_test.py \
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
python igfe_unet/script/train_fno.py \
  --device cuda \
  --epochs 1500
```

The default run id is `fno_mse_w21_m8x8_l4_s42`. For a validation sweep, vary
`--width`, `--modes1`, `--modes2`, `--layers`, and `--seed`; each configuration
gets a separate model directory.

The default model has about 454,189 trainable real-scalar parameters, compared
with about 472,545 for the current U-Net `[2, 32, 64, 128]`. Its raw PyTorch
tensor `numel` is smaller because the spectral weights are stored as complex
tensors; each complex value represents two real scalars. The reported count
uses the real-scalar convention for comparison with U-Net.

Outputs are written under:

```text
results/force_load/fno_custom/
```

### 3. Evaluate the selected checkpoint

After training, use the checkpoint and its saved configuration:

```bash
python igfe_unet/script/test_fno.py \
  --checkpoint results/force_load/fno_custom/models/<run_id>/model.pt \
  --config results/force_load/fno_custom/configs/<run_id>.json \
  --device cuda \
  --dataset-types mix,bil,exp,grf \
  --noise-levels 0,2,4,6,8,10
```

The evaluator uses the shared fixed test-set convention and writes
`metrics/summary.csv`, `metrics/per_sample_fno_mse.csv`, representative NPZ
fields, and PNG panels. Dataset types and noise levels can be selected with
`--dataset-types` and `--noise-levels`. To limit a diagnostic evaluation, add
`--max-samples 4`; the default evaluates every available test sample.

### 4. Train and evaluate FNO-neuraloperator

The optional implementation uses the same defaults and data protocol. Install
`requirements_fno_neuralop.txt` before starting this step. Its outputs are
isolated from the custom baseline:

```text
results/force_load/fno_neuralop/
```

Train it:

```bash
python igfe_unet/script/train_fno_neuralop.py \
  --device cuda \
  --epochs 1500 \
  --seed 42
```

Evaluate the checkpoint written by that run:

```bash
python igfe_unet/script/test_fno_neuralop.py \
  --checkpoint results/force_load/fno_neuralop/models/<run_id>/model.pt \
  --config results/force_load/fno_neuralop/configs/<run_id>.json \
  --device cuda \
  --dataset-types mix,bil,exp,grf \
  --noise-levels 0,2,4,6,8,10
```

The default run id is `fno_neuralop_mse_w21_m8x8_l4_s42`. Compare its
`metrics/summary.csv` with the custom FNO summary using the same dataset and
noise-level rows.

### 5. Inspect outputs

Use `metrics/summary.csv` for the method comparison and
`metrics/per_sample_fno_mse.csv` for distributions and robustness plots.
Use the files in `figures/` for the representative prediction/error panels.
Only after both implementations are validated should their summaries be
merged into `asm_unet_compare`.
