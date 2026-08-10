"""Evaluate an FNO checkpoint on the shared fixed test sets and noise levels."""

import argparse
import os
import sys
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from configs import config_fno
from model.fno_test import FNOTester
from utils.utils_fno import module_config, output_root, read_json, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
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


def load_config(args):
    values = module_config(config_fno)
    if args.config:
        values.update(read_json(args.config))
    if args.data_path:
        values["data_path"] = args.data_path
    if args.device:
        values["device"] = args.device
    values["device"] = resolve_device(values["device"])
    return SimpleNamespace(**values)


def main():
    args = parse_args()
    cfg = load_config(args)
    root = output_root(args.output_root)
    noise_levels = [float(value) for value in args.noise_levels.split(",") if value.strip()]
    dataset_types = [value.strip().lower() for value in args.dataset_types.split(",") if value.strip()]
    tester = FNOTester(cfg, args.checkpoint, root)
    tester.evaluate(
        dataset_types,
        noise_levels,
        batch_size=args.batch_size,
        sample_index=args.sample_index,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
