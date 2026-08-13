# GRF correlation-length and continuous-gradient generalization test

This workflow performs a small, test-only experiment for Reviewer 8, Comment 6. It keeps the three selected `final_model` checkpoints unchanged and evaluates them on independently generated fields.

## Scope

- GRF correlation lengths: `25, 20, 15, 10 mm`;
- `25 mm` is the original in-distribution reference;
- `20, 15, 10 mm` are progressively more demanding test-only cases;
- the steep case is a continuous sigmoid transition rather than a discontinuous interface;
- default evaluation uses clean displacement fields; optional noise levels may be supplied explicitly;
- no retraining and no modification of `config_mix.py` are involved.

The RBF covariance and `tanh` mapping reproduce `GRF_Generate.m`. The generated grid is deliberately `40 x 40` nodes (`39 x 39` elements), because that is the tensor size used by the selected models. The older MATLAB batch script currently uses `40 x 40` elements and therefore produces `41 x 41` nodes; it should not be used directly with these checkpoints.

The four GRF conditions form a paired test: they use the same independently generated latent normal samples and the same `E_max` ordering, while only the covariance length changes. This isolates the correlation-length effect. The continuous steep-gradient cases use a separate seed.

## 1. Generate independent test cases

From the repository root:

```bash
python igfe_unet/script/generate_generalization_test_set.py
```

For a quick smoke test:

```bash
python igfe_unet/script/generate_generalization_test_set.py --samples 1 --steep-samples 1
```

The dataset is written below `data/generalization_test_sets/force_load/`. Its manifest explicitly marks every condition as test-only and records the mesh, seeds, correlation lengths, forward-problem parameters, sample metadata, and file hashes.

## 2. Evaluate the unchanged final models

```bash
python igfe_unet/script/evaluate_generalization.py --device cpu
```

The default final-model directory is:

```text
trained_models_mix/force_load/final_model/
```

Optional deterministic noise evaluation can be requested without changing the clean-test default:

```bash
python igfe_unet/script/evaluate_generalization.py --device cpu --noise-levels 0 2 4 6 8 10
```

Results are saved under:

```text
results/revision/grf_ood/
├── manifests/
├── metrics/per_sample_all.csv
├── metrics/ood_summary.csv
├── figures/grf_correlation_length.(png|pdf)
└── figures/ood_examples.(png|pdf)
```

`ood_summary.csv` reports relative L1, MAE and RMSE. The column `relative_l1_ratio_to_l25` is the error ratio relative to the `l=25 mm` reference under the same model and noise level. Values above one indicate degradation relative to the original GRF condition.

## Interpretation boundary

The shorter correlation lengths are boundary tests within the paper's intended class of smooth continuously graded fields. The experiment does not claim applicability to arbitrary high-frequency random media or discontinuous interfaces. A degradation at `l=10 mm` should be reported as the boundary of the current training distribution, not concealed by retraining.
