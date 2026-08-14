# GRF correlation-length generalization test

This workflow performs an independent, test-only experiment for Reviewer 8,
Comment 6. It evaluates the three unchanged `final_model` checkpoints; it does
not retrain the networks or modify `config_mix.py`.

## Test design

- Generated GRF correlation lengths: `25, 20, 15, 10, 8, 5 mm`.
- `25 mm` is the original reference distribution.
- `20, 15, 10, 8 mm` form the main generalization test. In particular,
  `8 mm` is shorter than the `9 mm` specimen dimension.
- `5 mm` is an extreme boundary test and is stored under `supplementary/`; it
  is intentionally excluded from the main summary table and curve.
- A continuous steep sigmoid case is also evaluated in the complete metrics.
- All GRF scales use paired latent samples and identical `E_max` ordering, so
  the comparison isolates the correlation-length effect.
- The default noise protocol is `0, 2, 4, 6, 8, 10%`, consistent with the
  manuscript robustness experiment.

The RBF covariance and `tanh` mapping reproduce `GRF_Generate.m`. The grid is
`40 x 40` nodes (`39 x 39` elements), which matches the selected U-Net tensor
size. The older MATLAB batch script produces `41 x 41` nodes and should not be
used directly with these checkpoints.

## Code organization

The three run scripts only load the configuration and call package functions;
they do not define or invoke a `main()` function:

```text
grf_generalization/
|-- config.py
|-- run_generate_cases.py
|-- run_preview_samples.py
|-- run_compare_cases.py
`-- pipeline/
    |-- common.py
    |-- data_generation.py
    |-- evaluation.py
    `-- visualization.py
```

When running from this Git worktree while ignored model assets remain in the
primary checkout, set `FE_GIN_ASSET_ROOT` to the primary checkout before model
evaluation.

## 1. Generate the independent test set

```bash
python grf_generalization/run_generate_cases.py
```

The test-only dataset and manifest are written under
`data/generalization_test_sets/force_load/`.

## 2. Preview and select samples

```bash
python grf_generalization/run_preview_samples.py
```

The main catalogs are saved as:

```text
results/grf_ood/sample_previews/
    true_modulus_sample_catalog_grf_l{20,15,10,8}.(png|pdf)
    scale_fields/true_modulus_grf_sample_<index>.(png|pdf)
```

The `l=5 mm` catalog is separated as:

```text
results/grf_ood/supplementary/sample_previews/
    true_modulus_sample_catalog_grf_l5.(png|pdf)
```

Every sample panel has its own colorbar. After inspecting the catalogs, edit
the corresponding `sample_index` values in `grf_generalization/config.py`.
The indices may be selected independently; using the same index preserves the
paired latent sample for a direct cross-scale comparison.

## 3. Evaluate the unchanged models

```bash
python grf_generalization/run_compare_cases.py
```

The default checkpoints are loaded from
`trained_models_mix/force_load/final_model/`. Outputs are organized as:

```text
results/grf_ood/
|-- manifests/
|-- metrics/
|   |-- per_sample_all.csv
|   |-- ood_summary.csv
|   `-- grf_noise_statistics_table.csv
|-- figures/
|   |-- grf_correlation_length.(png|pdf)
|   `-- grf_noise_statistics_table.(png|pdf)
|-- cases/grf_l{20,15,10,8}/sample_<index>/
`-- supplementary/
    |-- metrics/grf_l5_stress_summary.csv
    `-- cases/grf_l5/sample_<index>/
```

`ood_summary.csv` remains the complete audit table and includes every generated
condition, including `l=5 mm` and the steep-gradient field. The manuscript-style
table and correlation-length curve contain only `l=25, 20, 15, 10, 8 mm`.

## Interpretation boundary

The main experiment examines shorter correlation lengths within the intended
class of smooth continuous modulus fields. The `l=5 mm` result is retained as
an explicit supplementary stress test to show where the unchanged model begins
to fail; it should be described as a limitation rather than as evidence of
generalization to arbitrary high-frequency media or discontinuous interfaces.
