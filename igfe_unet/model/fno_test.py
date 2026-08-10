"""Testing manager for the isolated FNO baseline."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from architectures.fno import build_fno_model
from utils.utils_test import generate_noise_data, load_test_data


def dataset_path(base_path, dataset_type):
    base = Path(base_path)
    dataset_type = dataset_type.lower()
    if dataset_type == "mix":
        return str(base)
    return str(base.parent / f"data_{dataset_type}" / base.name)


def metric_row(target, prediction, dataset_type, noise_level, sample_index):
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    error = target - prediction
    denominator = np.sum(np.abs(target))
    relative_l1 = 0.0 if np.isclose(denominator, 0.0) else np.sum(np.abs(error)) / denominator
    return {
        "dataset": dataset_type,
        "noise_level": noise_level,
        "sample_index": sample_index,
        "relative_l1": float(relative_l1),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def save_prediction_panel(path, target, prediction):
    path.parent.mkdir(parents=True, exist_ok=True)
    error = np.abs(target - prediction)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (target, prediction, error),
        ("True modulus", "FNO-MSE prediction", "Absolute error"),
    ):
        plot = axis.imshow(image, origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(path, dpi=200)
    plt.close(figure)


class FNOTester:
    def __init__(self, cfg, checkpoint, output_root, model_builder=build_fno_model):
        self.cfg = cfg
        self.output_root = Path(output_root)
        self.checkpoint = Path(checkpoint).resolve()
        state = torch.load(self.checkpoint, map_location=cfg.device, weights_only=True)
        self.net = model_builder(cfg).to(cfg.device)
        self.net.load_state_dict(state)
        self.net.eval()

    def evaluate_dataset(self, dataset_type, noise_levels, batch_size, sample_index, max_samples=None):
        eval_cfg = type(self.cfg)(**vars(self.cfg))
        eval_cfg.data_path = dataset_path(self.cfg.data_path, dataset_type)
        inputs, targets = load_test_data(eval_cfg)
        if max_samples is not None:
            max_samples = min(int(max_samples), inputs.shape[0])
            inputs = inputs[:max_samples]
            targets = targets[:max_samples]
        print(f"Evaluating {dataset_type}: {inputs.shape[0]} samples")
        targets_np = targets.cpu().numpy()
        rows = []
        for noise_level in noise_levels:
            predictions = []
            with torch.no_grad():
                for start in range(0, inputs.shape[0], batch_size):
                    end = min(start + batch_size, inputs.shape[0])
                    batch_inputs = []
                    for index in range(start, end):
                        sample = inputs[index]
                        if noise_level > 0:
                            sample = generate_noise_data(sample, noise_level, seed=index)
                        batch_inputs.append(sample)
                    batch = torch.stack(batch_inputs).to(self.cfg.device)
                    predictions.append(self.net(batch).cpu().numpy())
            predictions = np.concatenate(predictions, axis=0)
            rows.extend(
                metric_row(target, prediction, dataset_type, noise_level, index)
                for index, (target, prediction) in enumerate(zip(targets_np, predictions))
            )

            if noise_level == 0:
                selected = min(max(sample_index, 0), len(predictions) - 1)
                save_prediction_panel(
                    self.output_root / "figures" / f"{self.cfg.model_tag}_{dataset_type}_noise_0.png",
                    targets_np[selected],
                    predictions[selected],
                )
                np.savez(
                    self.output_root / "figures" / f"{self.cfg.model_tag}_{dataset_type}_sample_{selected}.npz",
                    target=targets_np[selected],
                    prediction=predictions[selected],
                    error=np.abs(targets_np[selected] - predictions[selected]),
                )
        return rows

    def evaluate(self, dataset_types, noise_levels, batch_size=32, sample_index=0, max_samples=None):
        rows = []
        for dataset_type in dataset_types:
            try:
                rows.extend(
                    self.evaluate_dataset(
                        dataset_type,
                        noise_levels,
                        batch_size,
                        sample_index,
                        max_samples=max_samples,
                    )
                )
            except FileNotFoundError as error:
                print(f"Skipping {dataset_type}: {error}")
        if not rows:
            raise RuntimeError("No test datasets were available")

        per_sample_path = self.output_root / "metrics" / f"per_sample_{self.cfg.model_tag}.csv"
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        grouped = {}
        for row in rows:
            grouped.setdefault((row["dataset"], row["noise_level"]), []).append(row)
        summary_rows = []
        for (dataset_type, noise_level), group in sorted(grouped.items()):
            summary_rows.append({
                "method": getattr(self.cfg, "method_label", self.cfg.model_tag),
                "dataset": dataset_type,
                "noise_level": noise_level,
                "n_samples": len(group),
                "relative_l1_mean": float(np.mean([r["relative_l1"] for r in group])),
                "relative_l1_std": float(np.std([r["relative_l1"] for r in group])),
                "mae_mean": float(np.mean([r["mae"] for r in group])),
                "rmse_mean": float(np.mean([r["rmse"] for r in group])),
            })
        summary_path = self.output_root / "metrics" / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved per-sample metrics: {per_sample_path}")
        print(f"Saved summary metrics: {summary_path}")
