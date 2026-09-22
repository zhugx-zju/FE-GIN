"""Evaluate the saved multi-seed checkpoints on deterministic noisy test data.

Edit the constants in CUSTOMIZABLE PARAMETERS when needed, then run this file
directly. Completed model/seed/noise combinations are reused on restart.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
IGFE_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = IGFE_ROOT.parent
if str(IGFE_ROOT) not in sys.path:
    sys.path.insert(0, str(IGFE_ROOT))

from postprocess.reproducibility import (
    build_noisy_test_inputs,
    combine_noise_per_sample_metrics,
    evaluate_checkpoint,
    load_eval_type_mapping,
    plot_multiseed_noise_robustness,
    read_noise_per_sample_metrics,
    summarize_seed_noise_results,
    write_cross_seed_noise_summary,
    write_noise_per_sample_metrics,
    write_paired_noise_comparison,
    write_seed_noise_summary,
)
from utils.utils_process import Config
from utils.utils_test import load_test_data


# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
RESULT_ROOT = PROJECT_ROOT / "results" / "reproducibility"
FIXED_TEST_DIR = PROJECT_ROOT / "data" / "fixed_test_sets" / "force_load" / "data_mix"
NOISE_LEVELS = [0, 2, 4, 6, 8, 10]
EVAL_BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REUSE_COMPLETED = True
# ============================================================================

METHODS = OrderedDict(
    [
        ("MSE-M", {"method": "MSE", "gamma": None}),
        ("LM-M", {"method": "LocMixloss", "gamma": 100000.0}),
        ("GM-M", {"method": "GloMixloss", "gamma": 10000.0}),
    ]
)
SEEDS = [7, 17, 27, 37, 42, 47, 123, 2024, 2025, 3407]


def _load_run_config(model, seed):
    config_path = RESULT_ROOT / "configs" / f"{model}_seed_{seed}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    cfg = Config("mix")
    cfg.config_type = "mix"
    cfg.dataset_type = "mix"
    cfg.device = DEVICE
    cfg.fixed_test_data_dir_override = str(FIXED_TEST_DIR)
    cfg.method = saved["internal_method"]
    cfg.filters_list = list(saved["filters_list"])
    cfg.kernel_size = int(saved["kernel_size"])
    cfg.use_batch_norm = bool(saved["use_batch_norm"])
    cfg.seed = int(seed)
    if saved.get("gamma") is not None:
        cfg.gamma = float(saved["gamma"])
    return cfg


def _run_file(model, seed, noise_level):
    noise_tag = f"{float(noise_level):g}".replace(".", "p")
    return (
        RESULT_ROOT / "metrics" / "noise_runs"
        / f"per_sample_{model}_seed_{seed}_noise_{noise_tag}.csv"
    )


def _validate_inputs():
    required_test_files = [FIXED_TEST_DIR / "input.mat", FIXED_TEST_DIR / "output.mat"]
    missing = [str(path) for path in required_test_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Fixed test files are missing:\n  - " + "\n  - ".join(missing))
    missing_checkpoints = [
        str(RESULT_ROOT / "models" / model / f"seed_{seed}" / "model.pt")
        for model in METHODS for seed in SEEDS
        if not (RESULT_ROOT / "models" / model / f"seed_{seed}" / "model.pt").is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Reproducibility checkpoints are missing:\n  - "
            + "\n  - ".join(missing_checkpoints)
        )


def run():
    _validate_inputs()
    metrics_dir = RESULT_ROOT / "metrics"
    figures_dir = RESULT_ROOT / "figures"
    (metrics_dir / "noise_runs").mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reference_cfg = _load_run_config(next(iter(METHODS)), SEEDS[0])
    reference_cfg.device = "cpu"
    clean_inputs, targets = load_test_data(reference_cfg)
    clean_inputs = clean_inputs.cpu()
    targets = targets.cpu()
    manifest_path = RESULT_ROOT / "configs" / "fixed_test_manifest.json"
    eval_type_by_sample = load_eval_type_mapping(
        manifest_path, test_count=int(clean_inputs.shape[0])
    )

    print("=" * 80)
    print("Multi-seed noisy-test reproducibility evaluation")
    print(f"Device: {DEVICE}")
    print(f"Noise levels: {NOISE_LEVELS}")
    print(f"Test samples: {len(eval_type_by_sample)}")
    print("Statistics: mean/std across 10 initialization-seed test-set means")
    print("=" * 80)

    run_files = []
    seed_rows = []
    for noise_level in NOISE_LEVELS:
        print(f"\nPreparing deterministic {noise_level:g}% noisy test set ...")
        noisy_inputs = build_noisy_test_inputs(clean_inputs, noise_level)
        for model in METHODS:
            for seed in SEEDS:
                run_file = _run_file(model, seed, noise_level)
                run_files.append(run_file)
                if REUSE_COMPLETED and run_file.is_file():
                    rows = read_noise_per_sample_metrics(run_file)
                    cache_is_complete = (
                        len(rows) == len(eval_type_by_sample)
                        and [row["sample_id"] for row in rows]
                        == list(range(len(eval_type_by_sample)))
                        and all(
                            row["eval_type"] == eval_type_by_sample[row["sample_id"]]
                            and np.isclose(row["noise_level"], float(noise_level))
                            for row in rows
                        )
                    )
                    if cache_is_complete:
                        print(f"Reuse: {model}, seed={seed}, noise={noise_level:g}%")
                    else:
                        print(f"Recompute incomplete cache: {run_file}")
                else:
                    rows = None
                if not (REUSE_COMPLETED and run_file.is_file() and cache_is_complete):
                    cfg = _load_run_config(model, seed)
                    checkpoint = RESULT_ROOT / "models" / model / f"seed_{seed}" / "model.pt"
                    print(f"Evaluate: {model}, seed={seed}, noise={noise_level:g}%")
                    rows = evaluate_checkpoint(
                        cfg,
                        checkpoint,
                        EVAL_BATCH_SIZE,
                        noise_level=noise_level,
                        inputs=noisy_inputs,
                        targets=targets,
                        eval_type_by_sample=eval_type_by_sample,
                    )
                    write_noise_per_sample_metrics(
                        run_file, {(model, seed, float(noise_level)): rows}
                    )
                seed_rows.extend(
                    summarize_seed_noise_results(
                        {(model, seed, float(noise_level)): rows}
                    )
                )

    combine_noise_per_sample_metrics(
        metrics_dir / "per_sample_noise.csv", run_files
    )
    write_seed_noise_summary(metrics_dir / "seed_noise_summary.csv", seed_rows)
    write_cross_seed_noise_summary(
        metrics_dir / "cross_seed_noise_summary.csv", seed_rows
    )
    write_paired_noise_comparison(
        metrics_dir / "paired_noise_comparison.csv", seed_rows
    )
    plot_multiseed_noise_robustness(
        figures_dir / "multiseed_noise_robustness.png",
        figures_dir / "multiseed_noise_robustness.pdf",
        seed_rows,
    )
    print("\nOutputs written to:", RESULT_ROOT)


if __name__ == "__main__":
    run()
