"""Export the FNO statistics reported in Appendix Tables F1 and F2.

Unlike the previous generic experiment summary, this script keeps model
selection and final testing separate:

* ``all_experiments.csv`` contains clean MIX *validation* statistics for every
  architecture candidate (Table F1); and
* ``selected_fno_test_results.csv`` contains BIL/EXP/GRF/MIX *test* statistics
  for the validation-selected FNO at all reported noise levels (Table F2).

The script only reads saved per-sample relative-L1 files. Run validation,
selection, and final-model testing before invoking it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
UNET_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = UNET_ROOT.parent
if str(UNET_ROOT) not in sys.path:
    sys.path.insert(0, str(UNET_ROOT))
from postprocess.common import find_all_experiments


DEFAULT_LOAD_TYPE = "force_load"
DEFAULT_BACKEND = "custom"
DEFAULT_NOISE_LEVELS = (0, 2, 4, 6, 8, 10)
DEFAULT_EVAL_TYPES = ("bil", "exp", "grf", "mix")
STRUCTURE_LABELS = ("Structure I", "Structure II", "Structure III", "Structure IV")


def _read_config(path: Path) -> dict:
    namespace = {}
    with path.open("r", encoding="utf-8") as handle:
        exec(compile(handle.read(), str(path), "exec"), {}, namespace)
    return {key: value for key, value in namespace.items() if not key.startswith("__")}


def _sample_stats(path: Path) -> tuple[float, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    values = np.asarray(np.loadtxt(path), dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Result file is empty or contains non-finite values: {path}")
    return float(np.mean(values)), float(np.std(values)), int(values.size)


def _parameter_count_from_config(config: dict) -> int:
    """Count real trainable scalars without importing PyTorch."""
    if config.get("parameter_count") is not None:
        return int(config["parameter_count"])
    input_channels = int(config.get("input_channels", 2))
    output_channels = int(config.get("output_channels", 1))
    width = int(config["width"])
    modes1 = int(config["modes1"])
    modes2 = int(config["modes2"])
    n_layers = int(config["n_layers"])
    lifted_channels = input_channels + (2 if bool(config.get("use_coordinates", True)) else 0)
    lifting = width * lifted_channels + width
    spectral_per_layer = 4 * width * width * modes1 * modes2
    pointwise_per_layer = width * width + width
    group_norm_per_layer = 2 * width
    projection = (width * width + width) + (width * output_channels + output_channels)
    return int(
        lifting
        + n_layers * (spectral_per_layer + pointwise_per_layer + group_norm_per_layer)
        + projection
    )


def discover_candidates(
    project_root: Path,
    load_type: str,
    backend: str,
) -> list[dict]:
    candidates = []
    for config_type, found_load_type, exp_id, raw_path in find_all_experiments(
        base_dir=project_root,
        experiment_group="arch",
    ):
        exp_path = Path(raw_path)
        if found_load_type != load_type or config_type not in {"fno", "fno_neuralop"}:
            continue
        config_path = exp_path / "config.py"
        validation_path = exp_path / "all_samples_val" / "all_L1_val_mix.txt"
        if not (exp_path / "model.pt").is_file() or not config_path.is_file():
            continue
        config = _read_config(config_path)
        if str(config.get("model_type", "")).lower() != "fno":
            continue
        if str(config.get("fno_backend", "custom")).lower() != backend.lower():
            continue
        if str(config.get("method", "MSE")) != "MSE":
            continue
        try:
            mean, std, count = _sample_stats(validation_path)
            parameter_count = _parameter_count_from_config(config)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            print(f"Skipping {exp_id}: {exc}")
            continue
        candidates.append({
            "config_type": config_type,
            "load_type": found_load_type,
            "exp_id": exp_id,
            "exp_path": str(exp_path.resolve()),
            "validation_relative_l1_mean": mean,
            "validation_relative_l1_std": std,
            "validation_sample_count": count,
            "parameter_count": parameter_count,
            "training_time": config.get("time"),
            "config": config,
        })
    return sorted(
        candidates,
        key=lambda item: (item["validation_relative_l1_mean"], item["exp_id"]),
    )


def _architecture_key(candidate: dict) -> tuple[int, int, int, int, str]:
    config = candidate["config"]
    return (
        int(config["width"]),
        int(config["modes1"]),
        int(config["modes2"]),
        int(config["n_layers"]),
        candidate["exp_id"],
    )


def build_architecture_rows(candidates: list[dict]) -> list[dict]:
    ordered = sorted(candidates, key=_architecture_key)
    if len(ordered) > len(STRUCTURE_LABELS):
        raise ValueError(
            f"Appendix Table F1 supports at most {len(STRUCTURE_LABELS)} structures; "
            f"found {len(ordered)}."
        )

    rows = []
    for label, candidate in zip(STRUCTURE_LABELS, ordered):
        config = candidate["config"]
        modes1 = int(config["modes1"])
        modes2 = int(config["modes2"])
        training_time = candidate.get("training_time")
        if training_time is None:
            raise ValueError(f"Training time is missing from {candidate['exp_id']}/config.py")
        rows.append({
            "model": label,
            "width": int(config["width"]),
            "retained_fourier_modes": f"{modes1} x {modes2}",
            "fourier_layers": int(config["n_layers"]),
            "trainable_parameters": int(candidate["parameter_count"]),
            "mix_validation_mean": float(candidate["validation_relative_l1_mean"]),
            "mix_validation_std": float(candidate["validation_relative_l1_std"]),
            "validation_sample_count": int(candidate["validation_sample_count"]),
            "training_time_s": float(training_time),
            "exp_id": candidate["exp_id"],
        })
    return rows


def _test_l1_path(exp_path: Path, eval_type: str, noise_level: float) -> Path:
    noise_suffix = "" if np.isclose(float(noise_level), 0.0) else f"_noise_{int(noise_level)}"
    return exp_path / "all_samples_test" / f"all_L1_test_{eval_type}{noise_suffix}.txt"


def resolve_selected_test_path(project_root: Path, load_type: str, selected: dict) -> Path:
    final_path = (
        project_root
        / "trained_models_fno"
        / load_type
        / "final_model"
        / selected["exp_id"]
    )
    if not (final_path / "model.pt").is_file():
        raise FileNotFoundError(
            f"Selected final-model checkpoint not found: {final_path}. "
            "Run select_fno_baseline.py and test_fno_final_models.py first."
        )
    return final_path


def build_selected_test_rows(
    exp_path: Path,
    noise_levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS,
    eval_types: tuple[str, ...] = DEFAULT_EVAL_TYPES,
) -> list[dict]:
    rows = []
    expected_count = None
    for noise_level in noise_levels:
        row = {"noise_level_pct": float(noise_level)}
        for eval_type in eval_types:
            mean, std, count = _sample_stats(_test_l1_path(exp_path, eval_type, noise_level))
            if expected_count is None:
                expected_count = count
            elif count != expected_count:
                raise ValueError(
                    f"Inconsistent test subset sizes: expected {expected_count}, "
                    f"found {count} for {eval_type} at {noise_level}% noise."
                )
            row[f"{eval_type}_mean"] = mean
            row[f"{eval_type}_std"] = std
        row["test_sample_count"] = expected_count
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def _markdown_table(headers: list[str], alignments: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(alignments) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_appendix_markdown(
    path: Path,
    architecture_rows: list[dict],
    test_rows: list[dict],
    selected: dict,
) -> None:
    table_f1 = _markdown_table(
        [
            "Model", "Width", "Retained Fourier modes", "Fourier layers",
            "Trainable parameters", "MIX validation mean", "MIX validation std",
            "Training time (s)",
        ],
        [":---", "---:", ":---:", "---:", "---:", "---:", "---:", "---:"],
        [
            [
                row["model"], str(row["width"]),
                row["retained_fourier_modes"].replace(" x ", " × "),
                str(row["fourier_layers"]), f"{row['trainable_parameters']:,}",
                f"{row['mix_validation_mean']:.6f}",
                f"{row['mix_validation_std']:.6f}", f"{row['training_time_s']:,.0f}",
            ]
            for row in architecture_rows
        ],
    )
    table_f2 = _markdown_table(
        [
            "Noise level (%)", "BIL mean", "BIL std", "EXP mean", "EXP std",
            "GRF mean", "GRF std", "MIX mean", "MIX std",
        ],
        ["---:"] * 9,
        [
            [
                f"{row['noise_level_pct']:g}",
                f"{row['bil_mean']:.4f}", f"{row['bil_std']:.4f}",
                f"{row['exp_mean']:.4f}", f"{row['exp_std']:.4f}",
                f"{row['grf_mean']:.4f}", f"{row['grf_std']:.4f}",
                f"{row['mix_mean']:.4f}", f"{row['mix_std']:.4f}",
            ]
            for row in test_rows
        ],
    )
    text = (
        "# FNO appendix tables\n\n"
        "## Table F1\n\n"
        "Architectural specifications, trainable parameters, validation error "
        "statistics, and training time of the MSE-trained FNO models.\n\n"
        f"{table_f1}\n\n"
        "## Table F2\n\n"
        "Mean and standard deviation of the relative L1 errors obtained by the "
        f"validation-selected FNO (`{selected['exp_id']}`) on the fixed test sets.\n\n"
        f"{table_f2}\n"
    )
    path.write_text(text, encoding="utf-8")
    print(f"Saved: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--load-type", default=DEFAULT_LOAD_TYPE)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results_fno/<load-type>/architecture_sweep under project-root.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = project_root / "results_fno" / args.load_type / "architecture_sweep"
    elif not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()

    candidates = discover_candidates(project_root, args.load_type, args.backend)
    if not candidates:
        raise SystemExit(
            "No validated FNO architecture candidates were found. "
            "Run train_fno_architectures.py and val_fno_architectures.py first."
        )

    architecture_rows = build_architecture_rows(candidates)
    selected = candidates[0]
    selected_test_path = resolve_selected_test_path(project_root, args.load_type, selected)
    test_rows = build_selected_test_rows(selected_test_path)

    _write_csv(output_dir / "all_experiments.csv", architecture_rows)
    _write_csv(output_dir / "selected_fno_test_results.csv", test_rows)
    write_appendix_markdown(
        output_dir / "fno_appendix_tables.md",
        architecture_rows,
        test_rows,
        selected,
    )
    print(
        "Validation-selected FNO: "
        f"{selected['exp_id']} "
        f"(mean={selected['validation_relative_l1_mean']:.8f}, "
        f"std={selected['validation_relative_l1_std']:.8f})"
    )
    return architecture_rows, test_rows


if __name__ == "__main__":
    main()
