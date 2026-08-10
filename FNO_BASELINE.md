# FNO baseline

This branch adds a PyTorch-only Fourier Neural Operator baseline while keeping
the repository's existing data and script layout.

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
# CUDA 12.1 example
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU-only example
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

After PyTorch is installed, install the remaining repository dependencies:

```bash
python -m pip install numpy scipy matplotlib pandas
```

Do not install `neuraloperator` unless a later branch explicitly changes the
implementation to use it. Verify the environment before running the project:

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
  igfe_unet/model/fno_train.py \
  igfe_unet/model/fno_test.py \
  igfe_unet/script/train_fno.py \
  igfe_unet/script/test_fno.py
python igfe_unet/script/fno_smoke_test.py
```

On PowerShell, use `$env:FNO_DATA_PATH = 'D:\path\to\data_mix\force_load'`.

### 2. Run a short data/training smoke test

Once `train/` and `val/` exist, run two epochs in a separate output root:

```bash
python igfe_unet/script/train_fno.py \
  --device cuda \
  --epochs 2 \
  --batch-size 4 \
  --output-root /tmp/fno_baseline_smoke
```

Check that `models/*/model.pt`, `configs/*.json`, and `logs/*/history.csv`
were created. This is only a pipeline check, not a reported experiment.

### 3. Train the baseline

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

Early stopping monitors validation MAE, requires an improvement larger than
`1e-5`, and stops after 25 consecutive non-improving epochs. The log prints
the current best value and the stale counter.

Outputs are written under:

```text
results/fno_baseline/
```

### 4. Evaluate the selected checkpoint

After training, use the checkpoint and its saved configuration:

```bash
python igfe_unet/script/test_fno.py \
  --checkpoint results/fno_baseline/models/<run_id>/model.pt \
  --config results/fno_baseline/configs/<run_id>.json \
  --device cuda \
  --dataset-types mix,bil,exp,grf \
  --noise-levels 0,2,4,6,8,10
```

The evaluator uses the shared fixed test-set convention and writes
`metrics/summary.csv`, `metrics/per_sample_fno_mse.csv`, representative NPZ
fields, and PNG panels. Dataset types and noise levels can be selected with
`--dataset-types` and `--noise-levels`. For a local smoke run, add
`--max-samples 4`; the default evaluates every available test sample.

### 5. Inspect outputs

Use `metrics/summary.csv` for the method comparison and
`metrics/per_sample_fno_mse.csv` for distributions and robustness plots.
Use the files in `figures/` for the representative prediction/error panels.
Only after this baseline is validated should its summary be merged into
`asm_unet_compare`.

## Standalone smoke test

This check does not require training data:

```bash
python igfe_unet/script/fno_smoke_test.py
```
