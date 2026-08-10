"""Train the isolated FNO-MSE baseline using the repository data protocol."""

import argparse
import os
import sys
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from configs import config_fno
from model.fno_train import FNOTrainer
from utils.utils_fno import module_config, output_root, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--modes1", type=int, default=None)
    parser.add_argument("--modes2", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def build_config(args):
    overrides = {
        key: value for key, value in {
            "data_path": args.data_path,
            "device": args.device,
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "width": args.width,
            "modes1": args.modes1,
            "modes2": args.modes2,
            "n_layers": args.layers,
        }.items() if value is not None
    }
    values = module_config(config_fno, overrides)
    values["device"] = resolve_device(values["device"])
    return SimpleNamespace(**values)


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)
    root = output_root(args.output_root)
    trainer = FNOTrainer(cfg, root)
    trainer.run_training()


if __name__ == "__main__":
    main()
