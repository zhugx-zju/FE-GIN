"""Plot saved FNO fields as noise-level by sample-case grids.

This script performs no model inference.  It reads the per-noise ``.npz``
files written by ``compare_fno_unet_fields.py`` and creates two figures:

* FNO-predicted modulus fields, with noise levels in rows and sample cases in
  columns; and
* the corresponding pointwise relative-error fields in the same layout.

By default the columns are EXP sample 200, GRF sample 410, and BIL sample 600,
and the rows are 0%, 2%, 4%, 6%, 8%, and 10% input noise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


CURRENT_DIR = Path(__file__).resolve().parent
UNET_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = UNET_ROOT.parent
RELATIVE_ERROR_COLORBAR_MAX_PCT = 10.0


DEFAULT_COMPARISON_ROOT = (
    PROJECT_ROOT / "results" / "force_load" / "asm_unet_comparison"
)
DEFAULT_CASES = ("exp:200", "grf:410", "bil:600")
DEFAULT_NOISE_LEVELS = (0, 2, 4, 6, 8, 10)
DEFAULT_OUTPUT_DIRNAME = "fno_sample_grid"


def format_decimal_token(value: float, precision: int = 10) -> str:
    """Format numeric values like the existing per-noise result folders."""
    return (
        f"{float(value):.{precision}f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )


@dataclass(frozen=True)
class SampleCase:
    dataset: str
    sample_index: int

    @property
    def title(self) -> str:
        return f"{self.dataset.upper()} sample {self.sample_index}"


def _parse_case(value: str) -> SampleCase:
    try:
        dataset, raw_index = value.split(":", maxsplit=1)
        sample_index = int(raw_index)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid case {value!r}; expected DATASET:SAMPLE_INDEX."
        ) from exc
    dataset = dataset.strip().lower()
    if dataset not in {"mix", "bil", "exp", "grf"}:
        raise argparse.ArgumentTypeError(f"Unsupported dataset: {dataset}")
    if sample_index < 0:
        raise argparse.ArgumentTypeError("Sample index must be non-negative.")
    return SampleCase(dataset=dataset, sample_index=sample_index)


def _resolve_comparison_root(path: Path) -> Path:
    root = path if path.is_absolute() else PROJECT_ROOT / path
    root = root.resolve()
    gn_root = root / "GN"
    return gn_root if gn_root.is_dir() else root


def _read_case_metadata(case_dir: Path) -> dict:
    metadata_path = case_dir / "case_metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_fno_stem(case_dir: Path, noise_levels: list[float], explicit: str | None) -> str:
    if explicit:
        return explicit

    metadata = _read_case_metadata(case_dir)
    saved_stem = metadata.get("fno_file_stem")
    if saved_stem:
        return str(saved_stem)

    candidates = set()
    for noise in noise_levels:
        noise_tag = format_decimal_token(noise) or "0"
        noise_dir = case_dir / f"noise_{noise_tag}"
        candidates.update(path.stem for path in noise_dir.glob("fno*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not infer one FNO file stem under {case_dir}; found "
            f"{sorted(candidates)}. Pass --fno-file-stem explicitly."
        )
    return candidates.pop()


def _load_case_fields(
    comparison_root: Path,
    case: SampleCase,
    noise_levels: list[float],
    explicit_stem: str | None,
) -> dict[float, dict[str, np.ndarray | float]]:
    case_dir = comparison_root / f"sample_{case.sample_index}" / case.dataset
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Sample case directory not found: {case_dir}")
    file_stem = _resolve_fno_stem(case_dir, noise_levels, explicit_stem)

    rows = {}
    reference_target = None
    for noise in noise_levels:
        noise_tag = format_decimal_token(noise) or "0"
        npz_path = case_dir / f"noise_{noise_tag}" / f"{file_stem}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"FNO result not found: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            missing = {"target", "pred", "rel_err", "l1"}.difference(data.files)
            if missing:
                raise KeyError(f"{npz_path} is missing fields: {sorted(missing)}")
            target = np.asarray(data["target"], dtype=float).squeeze()
            pred = np.asarray(data["pred"], dtype=float).squeeze()
            rel_err = np.asarray(data["rel_err"], dtype=float).squeeze()
            l1 = float(np.asarray(data["l1"]).item())

        if target.ndim != 2 or pred.shape != target.shape or rel_err.shape != target.shape:
            raise ValueError(
                f"Incompatible field shapes in {npz_path}: target={target.shape}, "
                f"pred={pred.shape}, rel_err={rel_err.shape}."
            )
        if not all(np.all(np.isfinite(field)) for field in (target, pred, rel_err)):
            raise ValueError(f"Non-finite field value found in {npz_path}")
        if reference_target is None:
            reference_target = target
        elif not np.allclose(target, reference_target, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Target field changes across noise levels for {case.title}.")

        rows[float(noise)] = {
            "target": target,
            "pred": pred,
            "rel_err": rel_err,
            "l1": l1,
        }
    return rows


def _draw_field(ax, values, cmap, vmin, vmax):
    image = ax.imshow(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _style_colorbar(colorbar, label: str) -> None:
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.set_label(label, fontsize=15)
    colorbar.ax.tick_params(
        direction="in", which="both", labelsize=13, length=4.0, width=0.8
    )
    colorbar.outline.set_linewidth(0.8)


def _save_figure(fig, png_path: Path, dpi: int) -> tuple[Path, Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    return png_path, pdf_path


def _plot_grid(
    cases: list[SampleCase],
    noise_levels: list[float],
    case_fields: dict[SampleCase, dict[float, dict]],
    field_key: str,
    output_path: Path,
    dpi: int,
    error_vmax: float,
) -> tuple[Path, Path]:
    n_rows = len(noise_levels)
    n_cols = len(cases)
    is_error = field_key == "rel_err"

    if is_error:
        vmin, vmax, cmap = 0.0, float(error_vmax), "Blues"
    else:
        targets = [case_fields[case][noise_levels[0]]["target"] for case in cases]
        vmin = float(min(np.min(target) for target in targets))
        vmax = float(max(np.max(target) for target in targets))
        cmap = "viridis"

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.75 * n_cols + 0.9, 2.62 * n_rows + 0.45),
        squeeze=False,
    )
    image_ref = None
    for row_idx, noise in enumerate(noise_levels):
        for col_idx, case in enumerate(cases):
            ax = axes[row_idx, col_idx]
            row = case_fields[case][float(noise)]
            image_ref = _draw_field(
                ax=ax,
                values=row[field_key],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if row_idx == 0:
                ax.set_title(case.title, fontsize=15, pad=3)
            if col_idx == 0:
                ax.text(
                    -0.13,
                    0.50,
                    f"{noise:g}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=14,
                )
            if is_error:
                ax.text(
                    0.00,
                    0.97,
                    f"$L_1$={row['l1']:.2e}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="white",
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.35,
                        "edgecolor": "none",
                        "boxstyle": "square,pad=0.0",
                    },
                )

    fig.subplots_adjust(
        left=0.115,
        right=0.875,
        bottom=0.025,
        top=0.955,
        wspace=0.055,
        hspace=0.085,
    )
    fig.text(
        0.025,
        0.50,
        "Input noise level",
        rotation=90,
        ha="center",
        va="center",
        fontsize=15,
    )
    cax = fig.add_axes([0.900, 0.14, 0.020, 0.74])
    colorbar = fig.colorbar(image_ref, cax=cax)
    _style_colorbar(
        colorbar,
        "Relative error (%)" if is_error else "Modulus (MPa)",
    )
    return _save_figure(fig, output_path, dpi)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=DEFAULT_COMPARISON_ROOT,
        help="asm_unet_comparison root; its GN directory is used when present.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        type=_parse_case,
        default=[_parse_case(value) for value in DEFAULT_CASES],
        metavar="DATASET:SAMPLE_INDEX",
        help="Column order, e.g. exp:200 grf:410 bil:600.",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=list(DEFAULT_NOISE_LEVELS),
        help="Row order; defaults to 0 2 4 6 8 10.",
    )
    parser.add_argument(
        "--fno-file-stem",
        default=None,
        help="Optional saved FNO npz stem; otherwise infer it per case.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to comparison-root/fno_sample_grid.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--error-vmax",
        type=float,
        default=RELATIVE_ERROR_COLORBAR_MAX_PCT,
        help="Upper limit of the shared relative-error color scale in percent.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    comparison_root = _resolve_comparison_root(args.comparison_root)
    cases = list(dict.fromkeys(args.cases))
    noise_levels = list(dict.fromkeys(float(value) for value in args.noise_levels))
    if not cases:
        raise SystemExit("At least one sample case is required.")
    if not noise_levels:
        raise SystemExit("At least one noise level is required.")
    if args.error_vmax <= 0:
        raise SystemExit("--error-vmax must be positive.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = comparison_root / DEFAULT_OUTPUT_DIRNAME
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    case_fields = {
        case: _load_case_fields(
            comparison_root=comparison_root,
            case=case,
            noise_levels=noise_levels,
            explicit_stem=args.fno_file_stem,
        )
        for case in cases
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    prediction_paths = _plot_grid(
        cases=cases,
        noise_levels=noise_levels,
        case_fields=case_fields,
        field_key="pred",
        output_path=output_dir / "fno_prediction_noise_by_sample_GN.png",
        dpi=args.dpi,
        error_vmax=args.error_vmax,
    )
    error_paths = _plot_grid(
        cases=cases,
        noise_levels=noise_levels,
        case_fields=case_fields,
        field_key="rel_err",
        output_path=output_dir / "fno_error_noise_by_sample_GN.png",
        dpi=args.dpi,
        error_vmax=args.error_vmax,
    )
    return prediction_paths + error_paths


if __name__ == "__main__":
    main()
