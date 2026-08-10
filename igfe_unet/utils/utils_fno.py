"""Small utilities shared by the FNO training and evaluation scripts."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root(custom_root=None):
    root = (
        Path(custom_root)
        if custom_root
        else project_root() / "results" / "force_load" / "fno_custom"
    )
    for name in ("configs", "models", "logs", "metrics", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root.resolve()


def resolve_checkpoint(checkpoint=None, search_root=None):
    """Return an explicit checkpoint or the newest checkpoint in a result root."""
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    root = Path(search_root) if search_root else output_root()
    candidates = [path for path in (root / "models").glob("*/model.pt") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {root}. Train the model first or pass --checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def checkpoint_config_path(checkpoint):
    """Infer the saved config path from ``models/<run_id>/model.pt``."""
    checkpoint = Path(checkpoint).resolve()
    run_name = checkpoint.parent.name
    return checkpoint.parent.parent.parent / "configs" / f"{run_name}.json"


def run_id(cfg):
    return (
        f"{getattr(cfg, 'model_tag', 'fno_mse')}_w{int(cfg.width)}"
        f"_m{int(cfg.modes1)}x{int(cfg.modes2)}"
        f"_l{int(cfg.n_layers)}_s{int(cfg.seed)}"
    )


def count_parameters(model):
    """Count trainable real scalar values, including both parts of complex weights."""
    return int(sum(
        (2 if parameter.is_complex() else 1) * parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ))


def count_tensor_parameters(model):
    """Return raw tensor element count for compatibility/debugging."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def module_config(module, overrides=None):
    values = {
        key: value
        for key, value in vars(module).items()
        if not key.startswith("_") and not callable(value)
    }
    if overrides:
        values.update(overrides)
    return values


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True, default=str)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(requested):
    requested = str(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return "cpu"
    return requested
