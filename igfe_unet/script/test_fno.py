"""Evaluate an FNO checkpoint on the shared fixed test sets and noise levels."""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from configs import config_fno
from model.fno_test import FNOTester
from utils.utils_fno import (
    checkpoint_config_path,
    module_config,
    output_root,
    read_json,
    resolve_checkpoint,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path; defaults to the newest checkpoint in the FNO output directory.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--noise-levels", default="0,2,4,6,8,10")
    parser.add_argument("--dataset-types", default="mix,bil,exp,grf")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(args, checkpoint):
    values = module_config(config_fno)
    config_path = Path(args.config).expanduser() if args.config else checkpoint_config_path(checkpoint)
    if config_path.is_file():
        values.update(read_json(config_path))
    elif args.config:
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if args.data_path:
        values["data_path"] = args.data_path
    if args.device:
        values["device"] = args.device
    values["device"] = resolve_device(values["device"])
    return SimpleNamespace(**values)


def run_testing(args=None):
    args = parse_args() if args is None else args
    search_root = args.output_root or config_fno.output_dir
    try:
        checkpoint = resolve_checkpoint(args.checkpoint, search_root)
        cfg = load_config(args, checkpoint)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    root = output_root(args.output_root or cfg.output_dir)
    noise_levels = [float(value) for value in args.noise_levels.split(",") if value.strip()]
    dataset_types = [value.strip().lower() for value in args.dataset_types.split(",") if value.strip()]
    tester = FNOTester(cfg, checkpoint, root)
    tester.evaluate(
        dataset_types,
        noise_levels,
        batch_size=args.batch_size,
        sample_index=args.sample_index,
        max_samples=args.max_samples,
    )


run_testing()
