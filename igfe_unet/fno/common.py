"""Shared filesystem, reproducibility, and checkpoint helpers for FNO."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root(custom_root=None):
    root = Path(custom_root) if custom_root else project_root() / "results" / "force_load" / "fno_custom"
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
            f"No checkpoint found under {root}. Train the model first or pass a checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def checkpoint_config_path(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    return checkpoint.parent.parent.parent / "configs" / f"{checkpoint.parent.name}.json"


def run_id(cfg):
    return (
        f"{getattr(cfg, 'model_tag', 'fno_mse')}_w{int(cfg.width)}"
        f"_m{int(cfg.modes1)}x{int(cfg.modes2)}"
        f"_l{int(cfg.n_layers)}_s{int(cfg.seed)}"
    )


def count_parameters(model):
    """Count trainable real scalar values, including complex weight parts."""
    return int(sum(
        (2 if parameter.is_complex() else 1) * parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ))


def count_tensor_parameters(model):
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


__all__ = [
    "checkpoint_config_path",
    "count_parameters",
    "count_tensor_parameters",
    "output_root",
    "read_json",
    "resolve_checkpoint",
    "resolve_device",
    "run_id",
    "set_seed",
    "write_json",
]
