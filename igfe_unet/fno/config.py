"""Editable defaults for the custom and NeuralOperator FNO experiments."""

import os
from types import SimpleNamespace


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.environ.get(
    "FNO_DATA_PATH",
    os.path.join(project_root, "data", "data_mix", "force_load"),
)

# Shared optimization and tensor-shape settings. Edit these values for a run.
lr_start = 3e-4
n_epochs = 1500
batch_size = 32
train_rto = 0.70
valid_rto = 0.20
patience_lr = 10
patience_stop = 25
weight_decay = 0.0
seed = 42
device = "cuda"

input_channels = 2
output_channels = 1
width = 21
modes1 = 8
modes2 = 8
n_layers = 4
use_coordinates = True

nodesx = 40
nodesy = 40
num = "all"
noise_levels = [0, 2, 4, 6, 8, 10]
dataset_types = ["mix", "bil", "exp", "grf"]
sample_index = 0

custom_output_dir = os.path.join(project_root, "results", "force_load", "fno_custom")
neuralop_output_dir = os.path.join(project_root, "results", "force_load", "fno_neuralop")


def get_config(backend="custom", overrides=None):
    """Build a config object for one FNO backend."""
    backend = str(backend).strip().lower()
    if backend not in {"custom", "neuralop"}:
        raise ValueError("backend must be 'custom' or 'neuralop'")

    values = {
        name: value
        for name, value in globals().items()
        if not name.startswith("_")
        and name not in {"os", "SimpleNamespace"}
        and not callable(value)
        and name not in {"custom_output_dir", "neuralop_output_dir"}
    }
    values.update(
        {
            "model_tag": "fno_mse" if backend == "custom" else "fno_neuralop_mse",
            "method_label": "FNO-MSE" if backend == "custom" else "FNO-neuraloperator-MSE",
            "output_dir": custom_output_dir if backend == "custom" else neuralop_output_dir,
            "backend": backend,
        }
    )
    if overrides:
        values.update(overrides)
    return SimpleNamespace(**values)


__all__ = ["get_config"]
