"""Generate independent GRF/steep-gradient cases for zero-shot evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "igfe_unet"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from generalization.data import DEFAULT_CORRELATION_LENGTHS, build_generalization_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test-only GRF correlation-length and continuous steep-gradient cases."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "generalization_test_sets" / "force_load",
    )
    parser.add_argument("--samples", type=int, default=20, help="Samples per GRF condition.")
    parser.add_argument("--steep-samples", type=int, default=None)
    parser.add_argument(
        "--correlation-lengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_CORRELATION_LENGTHS),
        metavar="MM",
    )
    parser.add_argument("--seed", type=int, default=8606)
    parser.add_argument(
        "--transition-width",
        type=float,
        default=0.75,
        help="10-90%% transition width of the continuous sigmoid field in mm.",
    )
    parser.add_argument("--nodes-x", type=int, default=40)
    parser.add_argument("--nodes-y", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_generalization_dataset(
        output_root=args.output_dir,
        sample_count=args.samples,
        steep_count=args.steep_samples,
        correlation_lengths=tuple(args.correlation_lengths),
        seed=args.seed,
        transition_width_mm=args.transition_width,
        nodes_x=args.nodes_x,
        nodes_y=args.nodes_y,
    )
    print(f"Generated test-only generalization dataset: {manifest}")


if __name__ == "__main__":
    main()
