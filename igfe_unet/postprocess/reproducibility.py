"""Metrics and statistical summaries for multi-seed experiments."""

import csv
import itertools
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.train import Training
from utils.utils_test import generate_noise_data, load_test_data


METRICS = ("relative_l1", "mae", "rmse")
DEFAULT_MODEL_ORDER = ("MSE-M", "LM-M", "GM-M")
DEFAULT_EVAL_TYPE_ORDER = ("bil", "exp", "grf", "mix")


def relative_l1(target, prediction):
    """Compute the full-field relative L1 error for one sample."""
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    denominator = np.sum(np.abs(target))
    if np.isclose(denominator, 0.0):
        return 0.0
    return float(np.sum(np.abs(target - prediction)) / denominator)


def build_noisy_test_inputs(inputs, noise_level):
    """Return deterministic per-sample noisy inputs using the paper protocol."""
    noise_level = float(noise_level)
    if np.isclose(noise_level, 0.0):
        return inputs
    return torch.stack(
        [
            generate_noise_data(inputs[index], noise_level, seed=index)
            for index in range(int(inputs.shape[0]))
        ],
        dim=0,
    )


def load_eval_type_mapping(manifest_path, test_count=None):
    """Map ordered MIX sample IDs to EXP/BIL/GRF labels from the manifest."""
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    mix_info = manifest["datasets"]["data_mix"]
    counts = mix_info["per_case_counts"]
    case_order = manifest.get("case_order") or mix_info.get("source_cases")
    labels = []
    for case_name in case_order:
        labels.extend([str(case_name).lower()] * int(counts[case_name]))
    expected = int(mix_info["test_count"])
    if len(labels) != expected:
        raise ValueError(
            f"Manifest case counts total {len(labels)}, expected {expected} samples"
        )
    if test_count is not None and len(labels) != int(test_count):
        raise ValueError(
            f"Manifest describes {len(labels)} samples, but test data has {test_count}"
        )
    return labels


def evaluate_checkpoint(
        cfg, checkpoint_path, batch_size, noise_level=0.0,
        inputs=None, targets=None, eval_type_by_sample=None):
    """Evaluate a checkpoint on one deterministic noisy fixed test set."""
    if inputs is None or targets is None:
        inputs, targets = load_test_data(cfg)
        inputs = build_noisy_test_inputs(inputs, noise_level)
    if int(inputs.shape[0]) != int(targets.shape[0]):
        raise ValueError("Input and target sample counts differ")
    if eval_type_by_sample is not None and len(eval_type_by_sample) != int(inputs.shape[0]):
        raise ValueError("Evaluation-type mapping length does not match test data")
    network = Training(cfg).net
    network.load_state_dict(
        torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
    )
    network = network.to(cfg.device)
    network.eval()

    rows = []
    total = int(inputs.shape[0])
    with torch.no_grad():
        for start in range(0, total, int(batch_size)):
            end = min(start + int(batch_size), total)
            outputs = network(inputs[start:end].to(cfg.device))
            batch_targets = targets[start:end].to(cfg.device)
            if outputs.ndim == 4 and outputs.shape[1] == 1:
                outputs = outputs[:, 0]
            if batch_targets.ndim == 4 and batch_targets.shape[1] == 1:
                batch_targets = batch_targets[:, 0]

            for offset in range(end - start):
                target = batch_targets[offset].detach().cpu().numpy().squeeze()
                prediction = outputs[offset].detach().cpu().numpy().squeeze()
                error = target - prediction
                rows.append(
                    {
                        "sample_id": start + offset,
                        "noise_level": float(noise_level),
                        "eval_type": (
                            eval_type_by_sample[start + offset]
                            if eval_type_by_sample is not None else "mix"
                        ),
                        "relative_l1": relative_l1(target, prediction),
                        "mae": float(np.mean(np.abs(error))),
                        "rmse": float(np.sqrt(np.mean(error ** 2))),
                    }
                )
    return rows


def write_noise_per_sample_metrics(path, results):
    """Write all method/seed/noise sample errors to one long-format CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "seed", "noise_level", "sample_id", "eval_type", *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (model, seed, noise_level), rows in results.items():
            for row in rows:
                writer.writerow(
                    {
                        "model": model,
                        "seed": int(seed),
                        "noise_level": float(noise_level),
                        **{
                            key: row[key]
                            for key in ["sample_id", "eval_type", *METRICS]
                        },
                    }
                )


def combine_noise_per_sample_metrics(path, run_files):
    """Combine completed per-run CSVs without keeping all sample rows in memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "seed", "noise_level", "sample_id", "eval_type", *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        writer.writeheader()
        for run_file in run_files:
            with Path(run_file).open(newline="", encoding="utf-8") as input_handle:
                for row in csv.DictReader(input_handle):
                    writer.writerow({field: row[field] for field in fields})


def read_noise_per_sample_metrics(path):
    """Load one per-run noise CSV and restore numeric field types."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "noise_level": float(row["noise_level"]),
                    "eval_type": row["eval_type"],
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )
    return rows


def summarize_seed_noise_results(results, eval_type_order=DEFAULT_EVAL_TYPE_ORDER):
    """Return one test-set mean per method/seed/noise/evaluation subset."""
    summary = []
    for (model, seed, noise_level), rows in results.items():
        for eval_type in eval_type_order:
            selected = rows if eval_type == "mix" else [
                row for row in rows if row["eval_type"] == eval_type
            ]
            if not selected:
                continue
            record = {
                "model": model,
                "seed": int(seed),
                "noise_level": float(noise_level),
                "eval_type": eval_type,
                "sample_count": len(selected),
            }
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_sample_std"] = sample_std(values)
            summary.append(record)
    return summary


def write_seed_noise_summary(path, rows):
    """Write the per-seed test-subset means used for cross-seed statistics."""
    fields = ["model", "seed", "noise_level", "eval_type", "sample_count"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_sample_std"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_cross_seed_noise_summary(
        path, seed_rows, model_order=DEFAULT_MODEL_ORDER,
        eval_type_order=DEFAULT_EVAL_TYPE_ORDER):
    """Write screenshot-style relative-L1 mean/std columns across seeds."""
    fields = ["noise_level", "method"]
    for eval_type in eval_type_order:
        fields.extend([f"{eval_type}_mean", f"{eval_type}_std"])
    noise_levels = sorted({float(row["noise_level"]) for row in seed_rows})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for noise_level in noise_levels:
            for model in model_order:
                record = {
                    "noise_level": noise_level,
                    "method": model,
                }
                for eval_type in eval_type_order:
                    values = np.asarray(
                        [
                            row["relative_l1_mean"]
                            for row in seed_rows
                            if row["model"] == model
                            and np.isclose(float(row["noise_level"]), noise_level)
                            and row["eval_type"] == eval_type
                        ],
                        dtype=float,
                    )
                    if not values.size:
                        record[f"{eval_type}_mean"] = ""
                        record[f"{eval_type}_std"] = ""
                        continue
                    record[f"{eval_type}_mean"] = float(np.mean(values))
                    record[f"{eval_type}_std"] = sample_std(values)
                writer.writerow(record)


def _exact_paired_permutation_pvalue(deltas):
    """Two-sided exact sign-flip test for a small vector of paired deltas."""
    deltas = np.asarray(deltas, dtype=float)
    if not deltas.size or np.allclose(deltas, 0.0):
        return 1.0
    observed = abs(float(np.mean(deltas)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(deltas.size)):
        statistic = abs(float(np.mean(deltas * np.asarray(signs))))
        exceed += statistic >= observed - 1e-15
        total += 1
    return float(exceed / total)


def _holm_adjust(p_values):
    """Return Holm-adjusted p-values in their original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = int(p_values.size)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def write_paired_noise_comparison(path, seed_rows):
    """Compare LM-M/GM-M with MSE-M using paired seed-level means."""
    comparisons = (("LM-M", "MSE-M"), ("GM-M", "MSE-M"))
    noise_levels = sorted({float(row["noise_level"]) for row in seed_rows})
    eval_types = [
        eval_type for eval_type in DEFAULT_EVAL_TYPE_ORDER
        if any(row["eval_type"] == eval_type for row in seed_rows)
    ]
    index = {
        (row["model"], int(row["seed"]), float(row["noise_level"]), row["eval_type"]):
        float(row["relative_l1_mean"])
        for row in seed_rows
    }
    output = []
    for noise_level in noise_levels:
        for eval_type in eval_types:
            for candidate, baseline in comparisons:
                candidate_seeds = {
                    seed for model, seed, noise, subset in index
                    if model == candidate and np.isclose(noise, noise_level)
                    and subset == eval_type
                }
                baseline_seeds = {
                    seed for model, seed, noise, subset in index
                    if model == baseline and np.isclose(noise, noise_level)
                    and subset == eval_type
                }
                seeds = sorted(candidate_seeds & baseline_seeds)
                if not seeds:
                    continue
                deltas = np.asarray(
                    [
                        index[(candidate, seed, noise_level, eval_type)]
                        - index[(baseline, seed, noise_level, eval_type)]
                        for seed in seeds
                    ],
                    dtype=float,
                )
                low, high = mean_ci95(deltas)
                output.append(
                    {
                        "noise_level": noise_level,
                        "eval_type": eval_type,
                        "comparison": f"{candidate} minus {baseline}",
                        "n_seeds": len(seeds),
                        "delta_mean": float(np.mean(deltas)),
                        "delta_std": sample_std(deltas),
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value": _exact_paired_permutation_pvalue(deltas),
                        "win_count": int(np.sum(deltas < 0.0)),
                    }
                )
    for noise_level in noise_levels:
        for eval_type in eval_types:
            group = [
                row for row in output
                if np.isclose(row["noise_level"], noise_level)
                and row["eval_type"] == eval_type
            ]
            adjusted = _holm_adjust([row["p_value"] for row in group])
            for row, p_adjusted in zip(group, adjusted):
                row["p_value_holm"] = float(p_adjusted)

    fields = [
        "noise_level", "eval_type", "comparison", "n_seeds",
        "delta_mean", "delta_std", "ci95_low", "ci95_high",
        "p_value", "p_value_holm", "win_count",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def plot_multiseed_noise_robustness(
        path_png, path_pdf, seed_rows, model_order=DEFAULT_MODEL_ORDER,
        eval_type_order=DEFAULT_EVAL_TYPE_ORDER):
    """Plot mean relative-L1 curves with 95% CIs across initialization seeds."""
    colors = {"MSE-M": "#e41a1c", "LM-M": "#984ea3", "GM-M": "#ff7f00"}
    markers = {"MSE-M": "o", "LM-M": "s", "GM-M": "^"}
    titles = {"bil": "BIL Dataset", "exp": "EXP Dataset",
              "grf": "GRF Dataset", "mix": "MIX Dataset"}
    fig, axes = plt.subplots(1, len(eval_type_order), figsize=(14.5, 3.6), squeeze=False)
    axes = axes[0]
    for panel_index, (axis, eval_type) in enumerate(zip(axes, eval_type_order)):
        for model in model_order:
            model_rows = [
                row for row in seed_rows
                if row["model"] == model and row["eval_type"] == eval_type
            ]
            noise_levels = sorted({float(row["noise_level"]) for row in model_rows})
            means = []
            lows = []
            highs = []
            for noise_level in noise_levels:
                values = np.asarray(
                    [
                        row["relative_l1_mean"] * 100.0
                        for row in model_rows
                        if np.isclose(float(row["noise_level"]), noise_level)
                    ],
                    dtype=float,
                )
                mean = float(np.mean(values))
                low, high = mean_ci95(values)
                means.append(mean)
                lows.append(low)
                highs.append(high)
            if not noise_levels:
                continue
            axis.plot(
                noise_levels, means, color=colors[model], marker=markers[model],
                linewidth=1.5, markersize=4.5, label=model,
            )
            axis.fill_between(
                noise_levels, lows, highs, color=colors[model], alpha=0.13,
                linewidth=0.0,
            )
        axis.set_title(titles.get(eval_type, eval_type.upper()))
        axis.set_xlabel("Noise level (%)")
        axis.set_xticks(sorted({float(row["noise_level"]) for row in seed_rows}))
        axis.tick_params(direction="in", labelsize=10)
        axis.grid(False)
        axis.text(-0.16, 1.05, f"({chr(97 + panel_index)})", transform=axis.transAxes)
        if panel_index == 0:
            axis.set_ylabel(r"Relative $L_1$ error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path_png, dpi=600, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def write_per_sample_metrics(path, model, seed, rows):
    """Save relative L1, MAE and RMSE for every test sample."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "sample_id", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": model,
                    "seed": int(seed),
                    **{key: row[key] for key in ["sample_id", *METRICS]},
                }
            )


def sample_std(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def mean_ci95(values):
    """Return a two-sided 95% t interval for independent seed means."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean
    try:
        from scipy.stats import t

        critical = float(t.ppf(0.975, values.size - 1))
    except Exception:
        critical = 1.96
    half_width = critical * sample_std(values) / np.sqrt(values.size)
    return mean - half_width, mean + half_width


def write_seed_summary(path, results):
    """Write per-seed mean/std over the fixed test samples."""
    fields = [
        "model", "seed", "test_count",
        "relative_l1_mean", "relative_l1_std",
        "mae_mean", "mae_std",
        "rmse_mean", "rmse_std",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (model, seed), rows in results.items():
            arrays = {
                metric: np.array([row[metric] for row in rows], dtype=float)
                for metric in METRICS
            }
            writer.writerow(
                {
                    "model": model,
                    "seed": seed,
                    "test_count": len(rows),
                    **{
                        f"{metric}_mean": float(np.mean(values))
                        for metric, values in arrays.items()
                    },
                    **{
                        f"{metric}_std": sample_std(values)
                        for metric, values in arrays.items()
                    },
                }
            )


def write_cross_seed_summary(path, results):
    """Write mean/std/95% CI across per-seed test-set means."""
    fields = ["model", "n_seeds"]
    for metric in METRICS:
        fields.extend(
            [
                f"{metric}_mean", f"{metric}_std",
                f"{metric}_ci95_low", f"{metric}_ci95_high",
            ]
        )

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        models = list(OrderedDict((model, None) for model, _ in results).keys())
        for model in models:
            seeds = sorted(seed for candidate, seed in results if candidate == model)
            row = {"model": model, "n_seeds": len(seeds)}
            for metric in METRICS:
                values = np.array(
                    [np.mean([item[metric] for item in results[(model, seed)]]) for seed in seeds],
                    dtype=float,
                )
                low, high = mean_ci95(values)
                row.update(
                    {
                        f"{metric}_mean": float(np.mean(values)),
                        f"{metric}_std": sample_std(values),
                        f"{metric}_ci95_low": low,
                        f"{metric}_ci95_high": high,
                    }
                )
            writer.writerow(row)


def write_paired_comparison(path, results):
    """Compare LM-M and GM-M using the same sample IDs for each seed."""
    lm_seeds = {seed for model, seed in results if model == "LM-M"}
    gm_seeds = {seed for model, seed in results if model == "GM-M"}
    common_seeds = sorted(lm_seeds & gm_seeds)
    if not common_seeds:
        return

    fields = [
        "metric", "n_seeds", "n_samples_per_seed", "delta_definition",
        "delta_mean_lm_minus_gm", "delta_std_across_seed_means",
        "ci95_low", "ci95_high", "lm_win_rate_sample_pooled",
        "lm_win_rate_seed_means",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            seed_deltas = []
            pooled_wins = []
            for seed in common_seeds:
                lm = {
                    row["sample_id"]: row[metric]
                    for row in results[("LM-M", seed)]
                }
                gm = {
                    row["sample_id"]: row[metric]
                    for row in results[("GM-M", seed)]
                }
                if set(lm) != set(gm):
                    raise ValueError(f"LM-M and GM-M sample IDs differ for seed {seed}")
                sample_ids = sorted(lm)
                lm_values = np.array([lm[index] for index in sample_ids], dtype=float)
                gm_values = np.array([gm[index] for index in sample_ids], dtype=float)
                delta = lm_values - gm_values
                seed_deltas.append(float(np.mean(delta)))
                pooled_wins.extend((lm_values < gm_values).tolist())

            seed_deltas = np.array(seed_deltas, dtype=float)
            low, high = mean_ci95(seed_deltas)
            writer.writerow(
                {
                    "metric": metric,
                    "n_seeds": len(seed_deltas),
                    "n_samples_per_seed": len(results[("LM-M", common_seeds[0])]),
                    "delta_definition": "LM-M minus GM-M; negative favors LM-M",
                    "delta_mean_lm_minus_gm": float(np.mean(seed_deltas)),
                    "delta_std_across_seed_means": sample_std(seed_deltas),
                    "ci95_low": low,
                    "ci95_high": high,
                    "lm_win_rate_sample_pooled": float(np.mean(pooled_wins)),
                    "lm_win_rate_seed_means": float(np.mean(seed_deltas < 0)),
                }
            )


def plot_seed_reproducibility(path_png, path_pdf, results, model_order=DEFAULT_MODEL_ORDER):
    """Plot per-seed test means with within-seed sample standard deviations."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.5), squeeze=False)
    axes = axes[0]
    colors = {"MSE-M": "#4C78A8", "LM-M": "#F58518", "GM-M": "#54A24B"}
    available_seeds = sorted({seed for _, seed in results})

    for axis, metric in zip(axes, METRICS):
        for model in model_order:
            model_seeds = sorted(seed for candidate, seed in results if candidate == model)
            if not model_seeds:
                continue
            means = [
                np.mean([row[metric] for row in results[(model, seed)]])
                for seed in model_seeds
            ]
            stds = [
                sample_std([row[metric] for row in results[(model, seed)]])
                for seed in model_seeds
            ]
            positions = [available_seeds.index(seed) for seed in model_seeds]
            axis.errorbar(
                positions, means, yerr=stds, marker="o", capsize=3,
                label=model, color=colors.get(model),
            )
        axis.set_title(metric.replace("_", " ").upper())
        axis.set_xlabel("seed")
        axis.set_xticks(range(len(available_seeds)))
        axis.set_xticklabels(available_seeds, rotation=45, ha="right")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("test-set metric")
    axes[0].legend(frameon=False)
    fig.suptitle("Multi-seed reproducibility on the fixed public test set")
    fig.tight_layout()
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "METRICS",
    "relative_l1",
    "build_noisy_test_inputs",
    "load_eval_type_mapping",
    "evaluate_checkpoint",
    "write_per_sample_metrics",
    "write_noise_per_sample_metrics",
    "combine_noise_per_sample_metrics",
    "read_noise_per_sample_metrics",
    "sample_std",
    "mean_ci95",
    "write_seed_summary",
    "write_cross_seed_summary",
    "write_paired_comparison",
    "plot_seed_reproducibility",
    "summarize_seed_noise_results",
    "write_seed_noise_summary",
    "write_cross_seed_noise_summary",
    "write_paired_noise_comparison",
    "plot_multiseed_noise_robustness",
]
