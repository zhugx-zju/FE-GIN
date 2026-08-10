"""Compatibility entry point for FNO evaluation.

The actual data loading and result serialization are shared with U-Net in
``model.test``.
"""

from pathlib import Path

from model.test import Testing
from utils.utils_process import construct_paths


def test_fno(
    cfg,
    model_builder=None,
    checkpoint=None,
    config_path=None,
    output_root_override=None,
    data_path=None,
    device=None,
    batch_size=None,
    noise_levels=None,
    dataset_types=None,
    sample_index=None,
    max_samples=None,
):
    """Evaluate FNO using the same files and output format as U-Net."""
    if model_builder is not None and "neuraloperator" in getattr(model_builder, "__module__", ""):
        cfg.fno_backend = "neuralop"
    if output_root_override is not None:
        cfg.model_root_override = output_root_override
    if data_path:
        cfg.data_path = data_path
    if device:
        cfg.device = device
    if noise_levels is not None:
        cfg.noise_levels = list(noise_levels)
    if dataset_types is not None:
        cfg.eval_types = list(dataset_types)
    if sample_index is not None:
        cfg.sample_index = int(sample_index)
    if max_samples is not None:
        configured_num = getattr(cfg, "num", "all")
        cfg.num = (
            int(max_samples)
            if configured_num == "all"
            else min(int(max_samples), int(configured_num))
        )

    experiment_path = None
    if checkpoint:
        experiment_path = str(Path(checkpoint).expanduser().resolve().parent)
    else:
        checkpoint_path, _, experiment_path = construct_paths(cfg)
        if not Path(checkpoint_path).is_file():
            raise SystemExit(
                f"No checkpoint found at {checkpoint_path}. Train the model first "
                "or pass an explicit checkpoint."
            )

    tester = Testing(cfg, experiment_path=experiment_path)
    tester.compute_and_save_predictions()
    return tester


__all__ = ["test_fno"]
