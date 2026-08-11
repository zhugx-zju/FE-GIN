"""Compare FNO and the selected U-Net models on identical field panels.

The script follows the existing ASM/U-Net sample workflow:

* load the same fixed-test sample for each MIX/BIL/EXP/GRF case;
* generate the same deterministic 0--10% noisy displacement inputs;
* evaluate MSE-U-Net, LocMix-U-Net, GloMix-U-Net and FNO on those inputs;
* draw prediction and relative-error contour panels with shared mesh and
  colour limits; and
* save per-noise fields plus a CSV containing L1, MAE, RMSE, time and counts.

ASM is intentionally left in its existing comparison workflow because its
iterative solve time and noise handling have different semantics from a
single forward pass.  It can still be compared using the existing ASM/U-Net
scripts with the same saved displacement inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
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

from architectures.fno import FNO2d
from fno.common import count_parameters, count_tensor_parameters
from postprocess.common import find_all_experiments
from utils.utils_process import Config

from asm_unet_compare.pipeline.asm_unet_compare import (
    _plot_error_figure,
    _plot_prediction_figure,
)
from asm_unet_compare.pipeline.common import (
    build_asm_context,
    build_noisy_input_by_noise,
    compute_error_fields,
    format_decimal_token,
    load_sample,
    load_unet_from_experiment,
    predict_unet_panel,
)


DEFAULT_DATASETS = ("mix", "bil", "exp", "grf")
DEFAULT_NOISE_LEVELS = (0, 2, 4, 6, 8, 10)
DEFAULT_UNET_METHODS = ("MSE", "LocMixloss", "GloMixloss")
DEFAULT_COMPARISON_ROOT = (
    PROJECT_ROOT / "results" / "force_load" / "asm_unet_comparison"
)
FNO_SUMMARY_FILENAME = "fno_unet_summary_GN.csv"


def _read_config(config_path: Path) -> dict:
    namespace = {}
    with config_path.open("r", encoding="utf-8") as handle:
        exec(compile(handle.read(), str(config_path), "exec"), {}, namespace)
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__")
    }


def _history_validation_score(exp_path: Path) -> float:
    history_path = exp_path / "history_mae.txt"
    if not history_path.is_file():
        return float("inf")
    try:
        history = np.asarray(np.loadtxt(history_path), dtype=float)
        if history.ndim == 1:
            history = history.reshape(1, -1)
        if history.shape[1] < 2:
            return float("inf")
        values = history[:, 1]
        values = values[np.isfinite(values)]
        return float(np.min(values)) if values.size else float("inf")
    except (OSError, ValueError):
        return float("inf")


def _resolve_unet_models(
    group: str,
    load_type: str,
    methods: list[str],
    architecture: str,
    use_batch_norm: bool,
) -> dict[str, dict]:
    """Resolve selected U-Net checkpoints using saved validation histories."""
    experiments = find_all_experiments(experiment_group=group)
    resolved = {}
    for method in methods:
        candidates = []
        for config_type, found_load_type, exp_id, raw_path in experiments:
            if config_type != "mix" or found_load_type != load_type:
                continue
            exp_path = Path(raw_path)
            if not (exp_path / "model.pt").is_file():
                continue
            config = _read_config(exp_path / "config.py")
            if config.get("method") != method:
                continue
            if bool(config.get("use_batch_norm", False)) != bool(use_batch_norm):
                continue
            if str(config.get("filters_list")) != architecture:
                continue
            candidates.append((
                _history_validation_score(exp_path),
                exp_id,
                exp_path,
                config,
            ))

        if not candidates:
            raise FileNotFoundError(
                f"No U-Net checkpoint found for method={method}, group={group}, "
                f"architecture={architecture}, use_batch_norm={use_batch_norm}."
            )
        _, exp_id, exp_path, config = min(candidates, key=lambda item: (item[0], item[1]))
        net, device = load_unet_from_experiment(
            config_type="mix",
            load_type=load_type,
            exp_path=str(exp_path),
            exp_cfg=config,
        )
        resolved[method] = {
            "net": net,
            "device": device,
            "exp_id": exp_id,
            "exp_path": str(exp_path.resolve()),
            "config": config,
            "parameter_count": count_parameters(net),
            "parameter_tensor_count": count_tensor_parameters(net),
            "training_time": config.get("time"),
        }
        print(f"[U-Net] {method}: {exp_id}")
    return resolved


def _resolve_fno_experiment(
    group: str,
    load_type: str,
    explicit_path: str | None = None,
) -> tuple[Path, dict]:
    if explicit_path:
        exp_path = Path(explicit_path)
        if not exp_path.is_absolute():
            exp_path = PROJECT_ROOT / exp_path
        exp_path = exp_path.resolve()
        if not (exp_path / "model.pt").is_file():
            raise FileNotFoundError(f"FNO checkpoint not found: {exp_path}")
        return exp_path, _read_config(exp_path / "config.py")

    candidates = []
    for config_type, found_load_type, exp_id, raw_path in find_all_experiments(
        experiment_group=group,
    ):
        if config_type not in {"fno", "fno_neuralop"} or found_load_type != load_type:
            continue
        exp_path = Path(raw_path)
        if not (exp_path / "model.pt").is_file():
            continue
        config = _read_config(exp_path / "config.py")
        if str(config.get("model_type", "")).lower() != "fno":
            continue
        if str(config.get("fno_backend", "custom")).lower() != "custom":
            continue
        if str(config.get("method", "MSE")) != "MSE":
            continue
        selection_score = config.get("validation_relative_l1_mean", float("inf"))
        try:
            selection_score = float(selection_score)
        except (TypeError, ValueError):
            selection_score = float("inf")
        candidates.append((selection_score, exp_id, exp_path, config))

    if not candidates:
        raise FileNotFoundError(
            "No selected custom FNO found. Run select_fno_baseline.py first "
            "or pass --fno-exp-path explicitly."
        )
    if len(candidates) > 1 and all(not np.isfinite(item[0]) for item in candidates):
        raise RuntimeError(
            "Multiple final-model FNO checkpoints were found without validation "
            "selection metadata; pass --fno-exp-path explicitly."
        )
    _, _, exp_path, config = min(candidates, key=lambda item: (item[0], item[1]))
    return exp_path, config


def _load_fno_model(exp_path: Path, config: dict) -> dict:
    runtime = Config("fno")
    runtime.config_type = "fno"
    runtime.device = "cpu"
    for key in (
        "input_channels", "output_channels", "width", "modes1", "modes2",
        "n_layers", "use_coordinates",
    ):
        if key in config:
            setattr(runtime, key, config[key])
    net = FNO2d(
        input_channels=int(runtime.input_channels),
        output_channels=int(runtime.output_channels),
        width=int(runtime.width),
        modes1=int(runtime.modes1),
        modes2=int(runtime.modes2),
        n_layers=int(runtime.n_layers),
        use_coordinates=bool(runtime.use_coordinates),
    )
    state = torch.load(exp_path / "model.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return {
        "net": net,
        "device": "cpu",
        "exp_id": exp_path.name,
        "exp_path": str(exp_path.resolve()),
        "config": config,
        "parameter_count": count_parameters(net),
        "parameter_tensor_count": count_tensor_parameters(net),
        "training_time": config.get("time"),
    }


def _fno_file_stem(config: dict, exp_id: str) -> str:
    """Use the same lowercase per-method stem convention as UNet inference."""
    backend = str(config.get("fno_backend", "custom")).lower()
    width = int(config.get("width", 0))
    modes1 = int(config.get("modes1", 0))
    modes2 = int(config.get("modes2", 0))
    layers = int(config.get("n_layers", 0))
    if width and modes1 and modes2 and layers:
        return f"fno_{backend}_w{width}_m{modes1}x{modes2}_l{layers}"
    return str(exp_id).lower()


def _write_fno_case_config(
    noise_dir: Path,
    case_cfg: dict,
    row: dict,
    file_stem: str,
    model_info: dict,
) -> None:
    config_path = noise_dir / f"{file_stem}_config.py"
    ordered_items = [
        ("dataset", case_cfg["dataset"]),
        ("sample_index", int(case_cfg["sample_index"])),
        ("resolved_sample_index", int(case_cfg["resolved_sample_index"])),
        ("noise_levels", list(case_cfg["noise_levels"])),
        ("noise_level", float(row["noise"])),
        ("model_type", "fno"),
        ("fno_backend", model_info["config"].get("fno_backend", "custom")),
        ("fno_exp_id", model_info["exp_id"]),
        ("fno_exp_path", model_info["exp_path"]),
        ("parameter_count", int(model_info["parameter_count"])),
        ("parameter_tensor_count", int(model_info["parameter_tensor_count"])),
        ("elapsed_time", float(row.get("elapsed_time", 0.0))),
        ("relative_l1", float(row["l1"])),
        ("mae", float(row["mae"])),
        ("rmse", float(row["rmse"])),
    ]
    with config_path.open("w", encoding="utf-8") as handle:
        for key, value in ordered_items:
            handle.write(f"{key} = {value!r}\n")


def _save_fno_panel_outputs(
    case_dir: Path,
    panel_data: list[dict],
    case_cfg: dict,
    model_info: dict,
    file_stem: str,
) -> None:
    """Save FNO fields into the existing sample/noise result layout."""
    for row in panel_data:
        noise = float(row["noise"])
        noise_tag = format_decimal_token(noise)
        noise_dir = case_dir / f"noise_{noise_tag or '0'}"
        noise_dir.mkdir(parents=True, exist_ok=True)
        target = np.asarray(row["target"], dtype=float)
        pred = np.asarray(row["pred"], dtype=float)
        delta = target - pred
        serializable_row = dict(row)
        serializable_row.update({
            "target": target,
            "pred": pred,
            "rel_err": np.asarray(row["rel_err"]),
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
        })
        np.savez(
            noise_dir / f"{file_stem}.npz",
            target=np.asarray(row["target"]),
            pred=np.asarray(row["pred"]),
            rel_err=np.asarray(row["rel_err"]),
            noise=np.asarray(noise),
            l1=np.asarray(row["l1"]),
            mae_pct=np.asarray(row.get("mae_pct", np.nan)),
            elapsed_time=np.asarray(row.get("elapsed_time", np.nan)),
        )
        with (noise_dir / f"{file_stem}_results.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "data_type": case_cfg["dataset"],
                    "sample_index": int(case_cfg["resolved_sample_index"]),
                    "noise_levels": [noise],
                    "method": "FNO",
                    "fno_backend": model_info["config"].get("fno_backend", "custom"),
                    "exp_id": model_info["exp_id"],
                    "exp_path": model_info["exp_path"],
                    "file_stem": file_stem,
                    "panel_data": [serializable_row],
                    "case_config": dict(case_cfg),
                },
                handle,
            )
        _write_fno_case_config(
            noise_dir=noise_dir,
            case_cfg=case_cfg,
            row=serializable_row,
            file_stem=file_stem,
            model_info=model_info,
        )


def _parse_sample_index(path: Path) -> int | None:
    match = re.fullmatch(r"sample_(\d+)", path.name)
    return None if match is None else int(match.group(1))


def _parse_noise_tag(name: str) -> float | None:
    if not name.startswith("noise_"):
        return None
    try:
        return float(name[len("noise_"):].replace("p", "."))
    except ValueError:
        return None


def _read_existing_summary_noise_levels(
    comparison_root: Path,
    dataset: str,
    sample_index: int,
) -> set[float]:
    noise_levels = set()
    for filename in ("comparison_summary_GN.csv", "unet_summary_GN.csv"):
        summary_path = comparison_root / filename
        if not summary_path.is_file():
            continue
        try:
            with summary_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if (
                        str(row.get("data_type", "")).lower() == dataset
                        and int(float(row.get("sample_index", -1))) == sample_index
                    ):
                        noise_levels.add(float(row["noise"]))
        except (OSError, TypeError, ValueError, KeyError):
            continue
    return noise_levels


def _case_noise_levels(
    comparison_root: Path,
    dataset: str,
    sample_index: int,
    override: list[float] | None,
) -> list[float]:
    if override is not None:
        return sorted({float(value) for value in override})

    case_dir = comparison_root / f"sample_{sample_index}" / dataset
    noise_levels = set()
    if case_dir.is_dir():
        noise_levels = {
            parsed
            for child in case_dir.iterdir()
            for parsed in [_parse_noise_tag(child.name)]
            if parsed is not None
        }
    noise_levels.update(
        _read_existing_summary_noise_levels(comparison_root, dataset, sample_index)
    )
    return sorted(noise_levels or {float(value) for value in DEFAULT_NOISE_LEVELS})


def discover_existing_cases(comparison_root: Path) -> list[tuple[int, str]]:
    """Discover sample/data cases under the existing ASM/U-Net GN root."""
    cases = []
    if not comparison_root.is_dir():
        return cases
    for sample_dir in sorted(comparison_root.glob("sample_*")):
        sample_index = _parse_sample_index(sample_dir)
        if sample_index is None:
            continue
        for dataset_dir in sorted(sample_dir.iterdir()):
            if dataset_dir.is_dir() and dataset_dir.name in {"mix", "bil", "exp", "grf"}:
                cases.append((sample_index, dataset_dir.name))
    return cases


def _load_saved_noisy_inputs(
    case_dir: Path,
    noise_levels: list[float],
) -> dict[float, torch.Tensor]:
    """Prefer measured displacement files already saved by ASM/U-Net."""
    saved = {}
    for noise in noise_levels:
        noise_dir = case_dir / f"noise_{format_decimal_token(noise)}"
        npz_path = noise_dir / "measured_displacement.npz"
        if not npz_path.is_file():
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                input_array = np.asarray(data["input"], dtype=np.float32)
            saved[float(noise)] = torch.from_numpy(input_array)
        except (OSError, KeyError, ValueError):
            continue
    return saved


def _resolve_comparison_root(path: Path) -> Path:
    root = path if path.is_absolute() else PROJECT_ROOT / path
    root = root.resolve()
    gn_root = root / "GN"
    return gn_root if gn_root.is_dir() else root


def _augment_metrics(panel_data: list[dict], method_info: dict, dataset: str, sample_index: int) -> list[dict]:
    rows = []
    for row in panel_data:
        target = np.asarray(row["target"], dtype=float)
        pred = np.asarray(row["pred"], dtype=float)
        delta = target - pred
        rows.append({
            "dataset": dataset,
            "sample_index": int(sample_index),
            "method": method_info["label"],
            "exp_id": method_info["exp_id"],
            "noise": float(row["noise"]),
            "relative_l1": float(row["l1"]),
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "relative_mae_pct": float(row.get("mae_pct", np.nan)),
            "inference_time_s": float(row.get("elapsed_time", np.nan)),
            "parameter_count": int(method_info["parameter_count"]),
            "parameter_tensor_count": int(method_info["parameter_tensor_count"]),
            "training_time_s": method_info["training_time"],
            "exp_path": method_info["exp_path"],
        })
    return rows


def _json_safe_model_info(info: dict) -> dict:
    return {
        key: value
        for key, value in info.items()
        if key not in {"net", "device"}
    }


def _write_summary(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "sample_index", "method", "exp_id", "noise",
        "relative_l1", "mae", "rmse", "relative_mae_pct",
        "inference_time_s", "parameter_count", "parameter_tensor_count",
        "training_time_s", "exp_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=DEFAULT_COMPARISON_ROOT,
        help="Existing asm_unet_comparison root; its GN subdirectory is used when present.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset filter. Without it, discover cases already present under comparison-root.",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=None,
        help="Override existing case noise levels; defaults to saved levels or 0,2,...,10.",
    )
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--unet-group", default="final_model")
    parser.add_argument("--fno-group", default="final_model")
    parser.add_argument("--load-type", default="force_load")
    parser.add_argument("--unet-architecture", default="[2, 32, 64, 128]")
    parser.add_argument("--unet-use-batch-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fno-exp-path", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional separate output root. By default write into comparison-root/GN.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-contour", action="store_true", help="Use image panels instead of FE mesh contours.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    comparison_root = _resolve_comparison_root(args.comparison_root)
    discovered_cases = discover_existing_cases(comparison_root)

    if args.sample_index is not None:
        requested_datasets = (
            [str(item).lower() for item in args.datasets]
            if args.datasets is not None
            else list(DEFAULT_DATASETS)
        )
        cases = [(int(args.sample_index), dataset) for dataset in requested_datasets]
    else:
        cases = discovered_cases
        if args.datasets is not None:
            requested_datasets = {str(item).lower() for item in args.datasets}
            cases = [case for case in cases if case[1] in requested_datasets]

    for _, dataset in cases:
        if dataset not in {"mix", "bil", "exp", "grf"}:
            raise SystemExit(f"Unsupported dataset: {dataset}")
    if not cases:
        raise SystemExit(
            f"No sample/data cases found under {comparison_root}. "
            "Pass --sample-index and --datasets explicitly if needed."
        )

    unet_models = _resolve_unet_models(
        group=args.unet_group,
        load_type=args.load_type,
        methods=[str(item) for item in DEFAULT_UNET_METHODS],
        architecture=args.unet_architecture,
        use_batch_norm=args.unet_use_batch_norm,
    )
    fno_path, fno_config = _resolve_fno_experiment(
        group=args.fno_group,
        load_type=args.load_type,
        explicit_path=args.fno_exp_path,
    )
    fno_model = _load_fno_model(fno_path, fno_config)
    fno_model["label"] = "FNO"
    print(f"[FNO] {fno_model['exp_id']}")

    model_infos = {}
    for method, info in unet_models.items():
        info["label"] = method
        model_infos[method] = info
    model_infos["FNO"] = fno_model

    if args.output_dir is None:
        output_root = comparison_root
    else:
        output_root = args.output_dir
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    methods_order = ["MSE", "LocMixloss", "GloMixloss", "FNO"]
    fno_file_stem = _fno_file_stem(fno_config, fno_model["exp_id"])

    asm_ctx = None
    if not args.no_contour:
        asm_ctx = build_asm_context(
            project_root=str(PROJECT_ROOT),
            nodesx=40,
            nodesy=40,
            asm_gamma=None,
            asm_max_iter=1,
            asm_ftol=1e-12,
            asm_gtol=1e-8,
        )

    for sample_index, dataset in cases:
        noise_levels = _case_noise_levels(
            comparison_root=comparison_root,
            dataset=dataset,
            sample_index=sample_index,
            override=args.noise_levels,
        )
        input_sample, target_sample, resolved_index = load_sample(
            config_type="mix",
            load_type=args.load_type,
            data_type=dataset,
            sample_index=sample_index,
        )
        noisy_by_noise = build_noisy_input_by_noise(
            input_sample=input_sample,
            noise_levels=noise_levels,
            sample_seed=resolved_index,
        )
        source_case_dir = comparison_root / f"sample_{resolved_index}" / dataset
        noisy_by_noise.update(
            _load_saved_noisy_inputs(source_case_dir, noise_levels)
        )
        case_dir = output_root / f"sample_{resolved_index}" / dataset
        case_dir.mkdir(parents=True, exist_ok=True)
        method_to_panel_data = {}

        for method in methods_order:
            info = model_infos[method]
            panel = predict_unet_panel(
                net=info["net"],
                device=info["device"],
                input_sample=input_sample,
                target_sample=target_sample,
                noise_levels=noise_levels,
                sample_seed=resolved_index,
                noisy_input_by_noise=noisy_by_noise,
            )
            method_to_panel_data[method] = panel
            if method == "FNO":
                fno_case_cfg = {
                    "dataset": dataset,
                    "sample_index": int(sample_index),
                    "resolved_sample_index": int(resolved_index),
                    "noise_levels": list(noise_levels),
                }
                _save_fno_panel_outputs(
                    case_dir=case_dir,
                    panel_data=panel,
                    case_cfg=fno_case_cfg,
                    model_info=info,
                    file_stem=fno_file_stem,
                )
            summary_rows.extend(
                _augment_metrics(panel, info, dataset, resolved_index)
            )

        metadata = {
            "dataset": dataset,
            "sample_index": int(resolved_index),
            "noise_levels": noise_levels,
            "input_shape": list(input_sample.shape),
            "target_shape": list(target_sample.shape),
            "noise_seed": int(resolved_index),
            "comparison_root": str(comparison_root),
            "source_case_dir": str(source_case_dir),
            "fno_file_stem": fno_file_stem,
            "models": {
                method: _json_safe_model_info(info)
                for method, info in model_infos.items()
            },
        }
        (case_dir / "case_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

        if asm_ctx is not None:
            comparison_dir = case_dir / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            _plot_prediction_figure(
                methods_order=methods_order,
                noise_order=noise_levels,
                method_to_panel_data=method_to_panel_data,
                save_path=str(
                    comparison_dir
                    / f"fno_prediction_sample_{resolved_index}_{dataset}_GN.png"
                ),
                title=f"{dataset.upper()} sample {resolved_index} - prediction",
                mesh_info=asm_ctx["mesh"],
                contour_fn=asm_ctx["create_smooth_contour"],
                dpi=args.dpi,
            )
            _plot_error_figure(
                methods_order=methods_order,
                noise_order=noise_levels,
                method_to_panel_data=method_to_panel_data,
                save_path=str(
                    comparison_dir
                    / f"fno_error_sample_{resolved_index}_{dataset}_GN.png"
                ),
                title=f"{dataset.upper()} sample {resolved_index} - relative error",
                mesh_info=asm_ctx["mesh"],
                contour_fn=asm_ctx["create_smooth_contour"],
                dpi=args.dpi,
            )
        print(f"Saved FNO/U-Net field comparison: {case_dir}")

    summary_path = output_root / FNO_SUMMARY_FILENAME
    _write_summary(summary_rows, summary_path)
    print(f"Saved FNO/U-Net field-comparison metrics: {summary_path}")
    return summary_rows


if __name__ == "__main__":
    main()
