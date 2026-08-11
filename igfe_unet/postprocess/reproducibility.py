"""Metrics and statistical summaries for multi-seed experiments."""

import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.train import Training
from utils.utils_test import load_test_data


METRICS = ("relative_l1", "mae", "rmse")
DEFAULT_MODEL_ORDER = ("MSE-M", "LM-M", "GM-M")


def relative_l1(target, prediction):
    """Compute the full-field relative L1 error for one sample."""
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    denominator = np.sum(np.abs(target))
    if np.isclose(denominator, 0.0):
        return 0.0
    return float(np.sum(np.abs(target - prediction)) / denominator)


def evaluate_checkpoint(cfg, checkpoint_path, batch_size):
    """Evaluate a checkpoint on the ordered fixed test set."""
    inputs, targets = load_test_data(cfg)
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
                        "relative_l1": relative_l1(target, prediction),
                        "mae": float(np.mean(np.abs(error))),
                        "rmse": float(np.sqrt(np.mean(error ** 2))),
                    }
                )
    return rows


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
    "evaluate_checkpoint",
    "write_per_sample_metrics",
    "sample_std",
    "mean_ci95",
    "write_seed_summary",
    "write_cross_seed_summary",
    "write_paired_comparison",
    "plot_seed_reproducibility",
]
