"""Select the FNO baseline using validation relative-L1 only.

The selector intentionally reads exactly one score file per candidate:
``all_samples_val/all_L1_val_mix.txt``.  Test outputs are not loaded while
choosing the architecture.  The selected checkpoint is copied into the
``final_model`` group without overwriting an existing destination by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
UNET_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = UNET_ROOT.parent
if str(UNET_ROOT) not in sys.path:
    sys.path.insert(0, str(UNET_ROOT))

from architectures.fno import FNO2d
from fno.common import count_parameters, count_tensor_parameters
from postprocess.common import find_all_experiments


DEFAULT_GROUP = "arch"
DEFAULT_LOAD_TYPE = "force_load"
DEFAULT_BACKEND = "custom"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "trained_models_fno"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_fno"


def _read_config(config_path: Path) -> dict:
    """Read a saved config.py without loading any result files."""
    namespace = {}
    with config_path.open("r", encoding="utf-8") as handle:
        exec(compile(handle.read(), str(config_path), "exec"), {}, namespace)
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__")
    }


def _validation_file(exp_path: Path, eval_type: str = "mix") -> Path:
    if str(eval_type).lower() != "mix":
        raise ValueError("FNO baseline selection is defined on the MIX validation split.")
    return exp_path / "all_samples_val" / "all_L1_val_mix.txt"


def _validation_score(exp_path: Path) -> tuple[float, int]:
    """Return mean validation relative-L1 and sample count."""
    score_path = _validation_file(exp_path)
    if not score_path.is_file():
        raise FileNotFoundError(f"Validation score file not found: {score_path}")
    values = np.asarray(np.loadtxt(score_path), dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Validation score file is empty or non-finite: {score_path}")
    return float(np.mean(values)), int(values.size)


def _parameter_counts(config: dict) -> tuple[int, int]:
    """Use saved counts when present, otherwise count the configured FNO."""
    saved_real = config.get("parameter_count")
    saved_tensor = config.get("parameter_tensor_count")
    if saved_real is not None and saved_tensor is not None:
        return int(saved_real), int(saved_tensor)

    model = FNO2d(
        input_channels=int(config.get("input_channels", 2)),
        output_channels=int(config.get("output_channels", 1)),
        width=int(config["width"]),
        modes1=int(config["modes1"]),
        modes2=int(config["modes2"]),
        n_layers=int(config["n_layers"]),
        use_coordinates=bool(config.get("use_coordinates", True)),
    )
    return count_parameters(model), count_tensor_parameters(model)


def discover_candidates(
    group: str = DEFAULT_GROUP,
    load_type: str = DEFAULT_LOAD_TYPE,
    backend: str = DEFAULT_BACKEND,
) -> list[dict]:
    """Discover FNO architecture candidates and score them on validation only."""
    candidates = []
    for config_type, found_load_type, exp_id, raw_path in find_all_experiments(
        experiment_group=group,
    ):
        exp_path = Path(raw_path)
        if found_load_type != load_type or config_type not in {"fno", "fno_neuralop"}:
            continue
        if not (exp_path / "model.pt").is_file():
            continue

        # This is deliberately a direct config read.  load_experiment_results
        # also scans test outputs and is therefore inappropriate for selection.
        config = _read_config(exp_path / "config.py")
        if str(config.get("model_type", "")).lower() != "fno":
            continue
        if str(config.get("fno_backend", "custom")).lower() != backend.lower():
            continue
        if str(config.get("method", "MSE")) != "MSE":
            continue

        try:
            score, sample_count = _validation_score(exp_path)
            parameter_count, parameter_tensor_count = _parameter_counts(config)
        except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            print(f"Skipping {exp_id}: {exc}")
            continue

        candidates.append({
            "config_type": config_type,
            "load_type": found_load_type,
            "experiment_group": group,
            "exp_id": exp_id,
            "exp_path": str(exp_path.resolve()),
            "validation_file": str(_validation_file(exp_path).resolve()),
            "validation_relative_l1_mean": score,
            "validation_sample_count": sample_count,
            "parameter_count": parameter_count,
            "parameter_tensor_count": parameter_tensor_count,
            "config": config,
        })

    return sorted(
        candidates,
        key=lambda item: (item["validation_relative_l1_mean"], item["exp_id"]),
    )


def _copy_selected_model(
    selected: dict,
    output_root: Path,
    load_type: str,
    overwrite: bool = False,
) -> Path:
    source = Path(selected["exp_path"])
    destination = output_root / load_type / "final_model" / selected["exp_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}. "
                "Use --overwrite only when replacing it is intentional."
            )
        shutil.rmtree(destination)

    shutil.copytree(source, destination)
    metadata = {
        "selection_rule": "minimum mean validation relative L1 on MIX",
        "selection_split": "val",
        "selection_score_file": selected["validation_file"],
        "validation_relative_l1_mean": selected["validation_relative_l1_mean"],
        "validation_sample_count": selected["validation_sample_count"],
        "parameter_count": selected["parameter_count"],
        "parameter_tensor_count": selected["parameter_tensor_count"],
        "source_experiment": selected["exp_path"],
        "source_experiment_group": selected["experiment_group"],
    }
    (destination / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Keep the metadata visible to the existing post-processing readers too.
    with (destination / "config.py").open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(f"parameter_count = {selected['parameter_count']!r}\n")
        handle.write(f"parameter_tensor_count = {selected['parameter_tensor_count']!r}\n")
        handle.write("selection_split = 'val'\n")
        handle.write(
            f"validation_relative_l1_mean = {selected['validation_relative_l1_mean']!r}\n"
        )
        handle.write("selection_rule = 'minimum mean validation relative L1 on MIX'\n")
    return destination


def _write_selection_table(candidates: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exp_id",
        "validation_relative_l1_mean",
        "validation_sample_count",
        "parameter_count",
        "parameter_tensor_count",
        "validation_file",
        "exp_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({key: candidate[key] for key in fieldnames})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--load-type", default=DEFAULT_LOAD_TYPE)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="FNO checkpoint root used for the final_model copy.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory for the validation-only candidate table.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the winner without copying it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final_model destination.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    candidates = discover_candidates(args.group, args.load_type, args.backend)
    if not candidates:
        raise SystemExit(
            "No FNO-MSE candidates with all_samples_val/all_L1_val_mix.txt were found."
        )

    table_path = args.results_root / args.load_type / "architecture_sweep" / "validation_selection.csv"
    _write_selection_table(candidates, table_path)
    selected = candidates[0]
    print("Validation-only FNO selection:")
    for candidate in candidates:
        print(
            f"  {candidate['exp_id']}: "
            f"val_relative_L1={candidate['validation_relative_l1_mean']:.8f}, "
            f"params={candidate['parameter_count']}"
        )
    print(f"Saved validation selection table: {table_path.resolve()}")

    if args.dry_run:
        print(f"Selected (dry run): {selected['exp_id']}")
        return selected

    destination = _copy_selected_model(
        selected,
        output_root=args.output_root,
        load_type=args.load_type,
        overwrite=args.overwrite,
    )
    print(f"Selected FNO copied to: {destination.resolve()}")
    return selected


if __name__ == "__main__":
    main()
