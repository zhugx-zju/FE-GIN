"""Evaluate unchanged final U-Net checkpoints on independent OOD cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import runpy
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "igfe_unet"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from architectures.unet import UNet
from generalization.metrics import apply_relative_noise, field_metrics


MODEL_LABELS = {
    "MSE_UNet_GN_arch_32-64-128": "MSE-M",
    "LocMix_UNet_GN_arch_32-64-128_gamma_100000": "LM-M",
    "GloMix_UNet_GN_arch_32-64-128_gamma_10000": "GM-M",
}
PER_SAMPLE_FIELDS = [
    "model",
    "condition_id",
    "field_type",
    "distribution_status",
    "correlation_length_mm",
    "transition_width_10_90_mm",
    "noise_level_percent",
    "sample_id",
    "relative_l1",
    "mae",
    "rmse",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_condition(data_root: Path, condition: dict) -> tuple[np.ndarray, np.ndarray]:
    directory = data_root / condition["directory"]
    inputs = np.asarray(loadmat(directory / "input.mat")["U"], dtype=np.float32)
    targets = np.asarray(loadmat(directory / "output.mat")["E"], dtype=np.float32)
    if inputs.ndim != 4 or inputs.shape[1] != 2:
        raise ValueError(f"Invalid input shape for {condition['condition_id']}: {inputs.shape}")
    if targets.shape != (inputs.shape[0], inputs.shape[2], inputs.shape[3]):
        raise ValueError(
            f"Target shape {targets.shape} does not match input shape {inputs.shape}"
        )
    return inputs, targets


def _load_model(model_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    config_path = model_dir / "config.py"
    checkpoint_path = model_dir / "model.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing config.py or model.pt under {model_dir}")
    config = runpy.run_path(str(config_path))
    filters = list(config.get("filters_list", [2, 32, 64, 128]))
    kernel_size = int(config.get("kernel_size", 3))
    use_normalization = bool(config.get("use_batch_norm", False))
    model = UNet(filters, kernel_size, use_batch_norm=use_normalization)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "filters_list": filters,
        "kernel_size": kernel_size,
        "use_batch_norm": use_normalization,
        "training_method": config.get("method"),
        "gamma": config.get("gamma"),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["condition_id"], row["noise_level_percent"])].append(row)

    summaries = []
    for (model, condition_id, noise), selected in grouped.items():
        first = selected[0]
        record = {
            "model": model,
            "condition_id": condition_id,
            "field_type": first["field_type"],
            "distribution_status": first["distribution_status"],
            "correlation_length_mm": first["correlation_length_mm"],
            "transition_width_10_90_mm": first["transition_width_10_90_mm"],
            "noise_level_percent": noise,
            "sample_count": len(selected),
        }
        for metric in ("relative_l1", "mae", "rmse"):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(record)

    id_errors = {
        (row["model"], row["noise_level_percent"]): row["relative_l1_mean"]
        for row in summaries
        if row["condition_id"] == "grf_l25"
    }
    for row in summaries:
        baseline = id_errors.get((row["model"], row["noise_level_percent"]))
        row["relative_l1_ratio_to_l25"] = (
            float(row["relative_l1_mean"] / baseline)
            if baseline is not None and not np.isclose(baseline, 0.0)
            else np.nan
        )
    return sorted(
        summaries,
        key=lambda row: (float(row["noise_level_percent"]), row["condition_id"], row["model"]),
    )


def _plot_correlation_lengths(output_dir: Path, summaries: list[dict]) -> None:
    clean = [
        row for row in summaries
        if row["field_type"] == "grf" and np.isclose(row["noise_level_percent"], 0.0)
    ]
    if not clean:
        return
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    colors = {"MSE-M": "#d62728", "LM-M": "#9467bd", "GM-M": "#ff7f0e"}
    markers = {"MSE-M": "o", "LM-M": "s", "GM-M": "^"}
    for model in ("MSE-M", "LM-M", "GM-M"):
        selected = sorted(
            [row for row in clean if row["model"] == model],
            key=lambda row: float(row["correlation_length_mm"]),
        )
        if not selected:
            continue
        axis.errorbar(
            [row["correlation_length_mm"] for row in selected],
            [100.0 * row["relative_l1_mean"] for row in selected],
            yerr=[100.0 * row["relative_l1_std"] for row in selected],
            label=model,
            color=colors[model],
            marker=markers[model],
            capsize=3,
        )
    axis.set_xlabel("GRF correlation length, $l$ (mm)")
    axis.set_ylabel(r"Relative $L_1$ error (%)")
    axis.set_xticks(sorted({row["correlation_length_mm"] for row in clean}))
    axis.legend(frameon=False)
    axis.tick_params(direction="in")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / "figures" / f"grf_correlation_length.{suffix}", dpi=300)
    plt.close(figure)


def _plot_examples(
    output_dir: Path,
    example_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> None:
    conditions = ["grf_l25", "grf_l10", "steep_sigmoid"]
    models = [model for model in ("MSE-M", "LM-M", "GM-M") if (model, conditions[0]) in example_data]
    if not models:
        return
    figure, axes = plt.subplots(
        len(conditions),
        len(models) + 2,
        figsize=(3.1 * (len(models) + 2), 8.2),
        constrained_layout=True,
    )
    for row_index, condition_id in enumerate(conditions):
        target = example_data[(models[0], condition_id)][0]
        predictions = [example_data[(model, condition_id)][1] for model in models]
        field_min = min([target.min(), *[prediction.min() for prediction in predictions]])
        field_max = max([target.max(), *[prediction.max() for prediction in predictions]])
        image = axes[row_index, 0].imshow(target, origin="lower", cmap="viridis", vmin=field_min, vmax=field_max)
        axes[row_index, 0].set_title("Ground truth")
        for column_index, (model, prediction) in enumerate(zip(models, predictions), start=1):
            axes[row_index, column_index].imshow(
                prediction, origin="lower", cmap="viridis", vmin=field_min, vmax=field_max
            )
            axes[row_index, column_index].set_title(model)
        best_prediction = predictions[0]
        error = np.abs(best_prediction - target)
        error_image = axes[row_index, -1].imshow(error, origin="lower", cmap="magma")
        axes[row_index, -1].set_title(f"|Error| ({models[0]})")
        axes[row_index, 0].set_ylabel(condition_id.replace("_", " "))
        figure.colorbar(image, ax=axes[row_index, :-1], fraction=0.015, pad=0.015)
        figure.colorbar(error_image, ax=axes[row_index, -1], fraction=0.05, pad=0.03)
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / "figures" / f"ood_examples.{suffix}", dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of final models on unseen fields.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "generalization_test_sets" / "force_load",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "trained_models_mix" / "force_load" / "final_model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "revision" / "grf_ood",
    )
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sample-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_dir.resolve()
    model_root = args.model_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("training_or_validation_use_permitted") is not False:
        raise ValueError("Dataset manifest does not explicitly mark the data as test-only")
    geometry = manifest["geometry"]
    if (geometry["nodes_x"], geometry["nodes_y"]) != (40, 40):
        raise ValueError("The selected final models require 40 x 40 nodal fields")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "manifests").mkdir(exist_ok=True)
    shutil.copy2(manifest_path, output_dir / "manifests" / "dataset_manifest.json")

    loaded_conditions = {}
    for condition in manifest["conditions"]:
        inputs, targets = _load_condition(data_root, condition)
        if args.sample_limit is not None:
            inputs = inputs[: args.sample_limit]
            targets = targets[: args.sample_limit]
        loaded_conditions[condition["condition_id"]] = (condition, inputs, targets)

    rows = []
    model_manifest = {}
    examples = {}
    for directory_name, model_label in MODEL_LABELS.items():
        model, model_info = _load_model(model_root / directory_name, device)
        model_manifest[model_label] = model_info
        for condition_id, (condition, inputs, targets) in loaded_conditions.items():
            for noise_level in args.noise_levels:
                predictions = []
                for start in range(0, len(inputs), args.batch_size):
                    batch = np.stack(
                        [
                            apply_relative_noise(inputs[index], noise_level, seed=86060000 + index)
                            for index in range(start, min(start + args.batch_size, len(inputs)))
                        ]
                    )
                    with torch.no_grad():
                        prediction = model(torch.from_numpy(batch).float().to(device))[:, 0]
                    predictions.append(prediction.detach().cpu().numpy())
                predictions_array = np.concatenate(predictions, axis=0)
                if np.isclose(noise_level, 0.0):
                    examples[(model_label, condition_id)] = (targets[0], predictions_array[0])
                for sample_id, (target, prediction) in enumerate(zip(targets, predictions_array)):
                    rows.append(
                        {
                            "model": model_label,
                            "condition_id": condition_id,
                            "field_type": condition["field_type"],
                            "distribution_status": condition["distribution_status"],
                            "correlation_length_mm": condition.get("correlation_length_mm", ""),
                            "transition_width_10_90_mm": condition.get(
                                "transition_width_10_90_mm", ""
                            ),
                            "noise_level_percent": float(noise_level),
                            "sample_id": sample_id,
                            **field_metrics(target, prediction),
                        }
                    )

    _write_csv(output_dir / "metrics" / "per_sample_all.csv", PER_SAMPLE_FIELDS, rows)
    summaries = _summarize(rows)
    summary_fields = [
        "model",
        "condition_id",
        "field_type",
        "distribution_status",
        "correlation_length_mm",
        "transition_width_10_90_mm",
        "noise_level_percent",
        "sample_count",
        "relative_l1_mean",
        "relative_l1_std",
        "relative_l1_ratio_to_l25",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
    ]
    _write_csv(output_dir / "metrics" / "ood_summary.csv", summary_fields, summaries)
    _plot_correlation_lengths(output_dir, summaries)
    _plot_examples(output_dir, examples)

    evaluation_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "device": str(device),
        "noise_levels_percent": [float(value) for value in args.noise_levels],
        "sample_limit": args.sample_limit,
        "models": model_manifest,
    }
    with (output_dir / "manifests" / "evaluation_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(evaluation_manifest, handle, indent=2)
    print(f"Saved zero-shot generalization results to: {output_dir}")


if __name__ == "__main__":
    main()
