"""Compatibility entry point for FNO training.

The actual training lifecycle is shared with U-Net in ``model.train``.
"""

from model.train import Training
from .common import set_seed


def train_fno(cfg, model_builder=None, output_root_override=None):
    """Train FNO through the repository's standard trainer."""
    if model_builder is not None and "neuraloperator" in getattr(model_builder, "__module__", ""):
        # Retain compatibility with the old NeuralOperator injection API.
        cfg.fno_backend = "neuralop"
    if output_root_override is not None:
        cfg.model_root_override = output_root_override
    set_seed(getattr(cfg, "seed", 42))
    manager = Training(cfg)
    trainer = manager.select_trainer(cfg.method)
    return trainer.run_training()


__all__ = ["train_fno"]
