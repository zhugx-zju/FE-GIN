"""Randomness controls shared by reproducibility experiments.

The training scripts historically relied on the process-global PyTorch random
state.  That is enough for a single run, but it makes it easy for a new
DataLoader or worker to consume randomness unexpectedly.  The helpers here
keep the global state deterministic and provide independent, seedable loader
generators.
"""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch for one experiment.

    ``warn_only=True`` is used for deterministic algorithms so that a run does
    not fail merely because an optional backend has no deterministic
    implementation.  The selected seed is returned as an int for convenient
    recording in experiment metadata.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)

    return seed


def make_dataloader_generator(seed, offset=0):
    """Return a private CPU generator for a DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def seed_worker(worker_id):
    """Seed NumPy/Python RNGs when DataLoader workers are enabled."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def json_default(value):
    """Serialize common NumPy/path values used in experiment metadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path, payload):
    """Write a UTF-8 JSON metadata file, creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def sha256_file(path):
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_project_path(path, project_root):
    """Resolve a CLI path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


__all__ = [
    "seed_everything",
    "make_dataloader_generator",
    "seed_worker",
    "json_default",
    "write_json",
    "sha256_file",
    "resolve_project_path",
]
