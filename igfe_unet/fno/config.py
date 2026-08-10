"""Compatibility accessors for the shared FNO configuration modules."""

import os

from utils.utils_process import Config


def get_config(backend="custom", overrides=None):
    """Return a U-Net-compatible config with FNO-specific model settings."""
    backend = str(backend).strip().lower()
    if backend not in {"custom", "neuralop"}:
        raise ValueError("backend must be 'custom' or 'neuralop'")

    config_type = "fno" if backend == "custom" else "fno_neuralop"
    cfg = Config(config_type)
    if overrides:
        for key, value in overrides.items():
            setattr(cfg, key, value)

    # Preserve the old environment override while keeping the default path
    # identical to the U-Net mix configuration.
    data_override = os.environ.get("FNO_DATA_PATH")
    if data_override:
        cfg.data_path = os.path.abspath(data_override)
    return cfg


__all__ = ["get_config"]
