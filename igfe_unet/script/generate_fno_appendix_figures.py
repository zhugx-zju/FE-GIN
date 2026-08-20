"""Run FNO-only inference and generate Appendix Figs. F1 and F2 in one step.

The default columns are BIL sample 600, EXP sample 200, and GRF sample 410.
Rows use 0%, 2%, 4%, 6%, 8%, and 10% input noise.  Noise is generated with
the sample index as the random seed, matching the full quantitative evaluator.

This script does not require U-Net or ASM checkpoints and does not depend on
previously saved FNO ``.npz`` files.  It loads the validation-selected FNO,
runs inference, saves the per-noise fields, and then calls the shared grid
plotter to create PNG and PDF outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
UNET_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = UNET_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(UNET_ROOT) not in sys.path:
    sys.path.insert(0, str(UNET_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from asm_unet_compare.pipeline.common import (
    build_noisy_input_by_noise,
    predict_unet_panel,
    resolve_runtime_device,
)
from compare_fno_unet_fields import (
    _augment_metrics,
    _fno_file_stem,
    _load_fno_model,
    _read_config,
    _save_fno_panel_outputs,
    _write_summary,
)
from plot_fno_sample_grid import main as plot_grid_main
from utils.utils_process import Config
from utils.utils_test import load_test_data


DEFAULT_CASES = ("bil:600", "exp:200", "grf:410")
DEFAULT_NOISE_LEVELS = (0, 2, 4, 6, 8, 10)


@dataclass(frozen=True)
class SampleCase:
    dataset: str
    sample_index: int

    @property
    def token(self) -> str:
        return f"{self.dataset}:{self.sample_index}"


def _parse_case(value: str) -> SampleCase:
    try:
        dataset, raw_index = value.split(":", maxsplit=1)
        dataset = dataset.strip().lower()
        sample_index = int(raw_index)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid case '{value}'. Expected DATASET:SAMPLE_INDEX."
        ) from exc
    if dataset not in {"bil", "exp", "grf", "mix"}:
        raise argparse.ArgumentTypeError(f"Unsupported dataset '{dataset}'.")
    if sample_index < 0:
        raise argparse.ArgumentTypeError("Sample index must be non-negative.")
    return SampleCase(dataset, sample_index)


def _selection_score(exp_path: Path, config: dict) -> float:
    metadata_path = exp_path / "selection_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value = metadata.get("validation_relative_l1_mean")
    else:
        value = config.get("validation_relative_l1_mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def resolve_fno_checkpoint(
    project_root: Path,
    load_type: str,
    explicit_path: Path | None,
) -> tuple[Path, dict]:
    if explicit_path is not None:
        exp_path = explicit_path if explicit_path.is_absolute() else project_root / explicit_path
        exp_path = exp_path.resolve()
        if not (exp_path / "model.pt").is_file():
            raise FileNotFoundError(f"FNO checkpoint not found: {exp_path}")
        return exp_path, _read_config(exp_path / "config.py")

    final_root = project_root / "trained_models_fno" / load_type / "final_model"
    candidates = []
    if final_root.is_dir():
        for exp_path in final_root.iterdir():
            config_path = exp_path / "config.py"
            if not (exp_path / "model.pt").is_file() or not config_path.is_file():
                continue
            config = _read_config(config_path)
            if str(config.get("model_type", "")).lower() != "fno":
                continue
            if str(config.get("fno_backend", "custom")).lower() != "custom":
                continue
            candidates.append((_selection_score(exp_path, config), exp_path.name, exp_path, config))
    if not candidates:
        raise FileNotFoundError(
            f"No validation-selected custom FNO found under {final_root}. "
            "Run select_fno_baseline.py first or pass --fno-exp-path."
        )
    if len(candidates) > 1 and all(not np.isfinite(item[0]) for item in candidates):
        raise RuntimeError(
            "Multiple FNO final models lack validation-selection metadata; "
            "pass --fno-exp-path explicitly."
        )
    _, _, exp_path, config = min(candidates, key=lambda item: (item[0], item[1]))
    return exp_path, config


def load_fixed_sample(
    data_root: Path,
    load_type: str,
    case: SampleCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = Config("fno")
    cfg.device = "cpu"
    cfg.load_type = load_type
    cfg.data_path = str((data_root / f"data_{case.dataset}" / load_type).resolve())
    inputs, targets = load_test_data(cfg)
    if case.sample_index >= int(inputs.shape[0]):
        raise IndexError(
            f"{case.dataset.upper()} sample {case.sample_index} is out of range; "
            f"the fixed test set contains {inputs.shape[0]} samples."
        )
    return inputs[case.sample_index], targets[case.sample_index]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--load-type", default="force_load")
    parser.add_argument("--fno-exp-path", type=Path, default=None)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=_parse_case,
        default=[_parse_case(value) for value in DEFAULT_CASES],
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=list(DEFAULT_NOISE_LEVELS),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to results/force_load/asm_unet_comparison/GN under project-root.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--error-vmax", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    data_root = (args.data_root or (project_root / "data")).resolve()
    output_root = args.output_root
    if output_root is None:
        output_root = (
            project_root
            / "results"
            / args.load_type
            / "asm_unet_comparison"
            / "GN"
        )
    elif not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()

    cases = list(dict.fromkeys(args.cases))
    noise_levels = list(dict.fromkeys(float(value) for value in args.noise_levels))
    if not cases or not noise_levels:
        raise ValueError("At least one sample case and one noise level are required.")
    if args.dpi <= 0 or args.error_vmax <= 0:
        raise ValueError("--dpi and --error-vmax must be positive.")

    exp_path, config = resolve_fno_checkpoint(
        project_root=project_root,
        load_type=args.load_type,
        explicit_path=args.fno_exp_path,
    )
    model_info = _load_fno_model(exp_path, config)
    device = resolve_runtime_device(args.device)
    model_info["net"] = model_info["net"].to(device)
    model_info["device"] = device
    model_info["label"] = "FNO"
    file_stem = _fno_file_stem(config, model_info["exp_id"])
    print(f"Selected FNO: {model_info['exp_id']}")
    print(f"Inference device: {device}")

    summary_rows = []
    for case in cases:
        input_sample, target_sample = load_fixed_sample(data_root, args.load_type, case)
        noisy_by_noise = build_noisy_input_by_noise(
            input_sample=input_sample,
            noise_levels=noise_levels,
            sample_seed=case.sample_index,
        )
        panel = predict_unet_panel(
            net=model_info["net"],
            device=device,
            input_sample=input_sample,
            target_sample=target_sample,
            noise_levels=noise_levels,
            sample_seed=case.sample_index,
            noisy_input_by_noise=noisy_by_noise,
        )
        case_dir = output_root / f"sample_{case.sample_index}" / case.dataset
        case_cfg = {
            "dataset": case.dataset,
            "sample_index": case.sample_index,
            "resolved_sample_index": case.sample_index,
            "noise_levels": list(noise_levels),
        }
        _save_fno_panel_outputs(
            case_dir=case_dir,
            panel_data=panel,
            case_cfg=case_cfg,
            model_info=model_info,
            file_stem=file_stem,
        )
        metadata = {
            **case_cfg,
            "noise_seed": case.sample_index,
            "fno_file_stem": file_stem,
            "fno_exp_id": model_info["exp_id"],
            "fno_exp_path": model_info["exp_path"],
        }
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        summary_rows.extend(
            _augment_metrics(panel, model_info, case.dataset, case.sample_index)
        )
        print(f"Saved per-noise fields: {case_dir}")

    _write_summary(summary_rows, output_root / "fno_appendix_sample_results.csv")
    figure_dir = output_root / "fno_sample_grid"
    plot_args = [
        "--comparison-root", str(output_root),
        "--cases", *[case.token for case in cases],
        "--noise-levels", *[f"{value:g}" for value in noise_levels],
        "--fno-file-stem", file_stem,
        "--output-dir", str(figure_dir),
        "--dpi", str(args.dpi),
        "--error-vmax", str(args.error_vmax),
    ]
    figure_paths = plot_grid_main(plot_args)
    print("FNO appendix figure generation completed.")
    return figure_paths


if __name__ == "__main__":
    main()
