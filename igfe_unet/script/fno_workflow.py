"""Shared discovery/configuration helpers for FNO experiment scripts."""

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from postprocess.common import find_all_experiments, load_experiment_results
from utils.utils_process import Config


def fno_noise_levels(cfg):
    return list(getattr(cfg, "noise_levels", [0, 2, 4, 6, 8, 10]))


def is_fno_config(config, backend=None):
    if not isinstance(config, dict):
        return False
    if str(config.get("model_type", "")).lower() != "fno":
        return False
    return backend is None or str(config.get("fno_backend", "custom")).lower() == backend


def select_fno_experiments(group, load_type="force_load", backend="custom"):
    """Find FNO checkpoints and return their saved configs."""
    selected = []
    for exp_info in find_all_experiments(experiment_group=group):
        config_type, found_load_type, exp_id, exp_path = exp_info
        if config_type not in {"fno", "fno_neuralop"} or found_load_type != load_type:
            continue
        results = load_experiment_results(exp_path)
        config = results.get("config") or {}
        if is_fno_config(config, backend=backend) and os.path.isfile(os.path.join(exp_path, "model.pt")):
            selected.append((exp_info, config))
    return sorted(selected, key=lambda item: item[0][2])


def config_from_saved_fno(saved_config, group, load_type="force_load"):
    """Rebuild a runtime Config while retaining the saved FNO architecture."""
    config_type = "fno_neuralop" if str(saved_config.get("fno_backend", "custom")).lower() == "neuralop" else "fno"
    cfg = Config(config_type)
    cfg.config_type = config_type
    cfg.dataset_type = "mix"
    cfg.load_type = load_type
    cfg.experiment_group = group
    for key in (
        "fno_backend", "input_channels", "output_channels", "width", "modes1",
        "modes2", "n_layers", "use_coordinates", "method", "num",
        "eval_types", "save_format", "nodesx", "nodesy", "noise_levels",
    ):
        if key in saved_config:
            setattr(cfg, key, saved_config[key])
    cfg.noise_level = 0
    cfg.eval_split = "test"
    return cfg


__all__ = [
    "config_from_saved_fno",
    "fno_noise_levels",
    "is_fno_config",
    "select_fno_experiments",
]
