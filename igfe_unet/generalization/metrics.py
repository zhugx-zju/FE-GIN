"""Metrics shared by the generalization evaluation scripts."""

from __future__ import annotations

import numpy as np


def apply_relative_noise(values: np.ndarray, noise_level_percent: float, seed: int) -> np.ndarray:
    """Apply deterministic noise with the relative L2 norm used in the paper."""
    values = np.asarray(values)
    level = float(noise_level_percent)
    if np.isclose(level, 0.0):
        return values.copy()
    rng = np.random.default_rng(int(seed))
    standard_noise = rng.standard_normal(values.shape)
    noise_norm = np.linalg.norm(standard_noise)
    signal_norm = np.linalg.norm(values)
    if np.isclose(noise_norm, 0.0) or np.isclose(signal_norm, 0.0):
        return values.copy()
    scaled_noise = (level / 100.0) * signal_norm * standard_noise / noise_norm
    return values + scaled_noise


def field_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute full-field relative L1, MAE and RMSE."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, prediction {prediction.shape}")
    error = prediction - target
    denominator = np.sum(np.abs(target))
    relative_l1 = np.nan if np.isclose(denominator, 0.0) else np.sum(np.abs(error)) / denominator
    return {
        "relative_l1": float(relative_l1),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }
