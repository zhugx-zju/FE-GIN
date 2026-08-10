"""Train the custom FNO architecture sweep using the U-Net trainer."""

import os
import random
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.train import Training
from utils.utils_process import Config


TARGET_BACKEND = "custom"
TARGET_LOAD_TYPE = "force_load"
MODEL_INIT_SEED = 42

# Keep this list small enough for a practical reviewer ablation. Every run is
# independently checkpointed under trained_models_fno/.../arch/.
TARGET_ARCHITECTURES = [
    {"width": 21, "modes1": 8, "modes2": 8, "n_layers": 4},
    {"width": 32, "modes1": 12, "modes2": 12, "n_layers": 4},
    {"width": 32, "modes1": 16, "modes2": 16, "n_layers": 4},
    {"width": 64, "modes1": 16, "modes2": 16, "n_layers": 4},
]


def reset_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


print("=" * 80)
print("Batch Training: FNO architecture sweep with MSE")
print("=" * 80)

for architecture in TARGET_ARCHITECTURES:
    cfg = Config("fno")
    cfg.config_type = "fno"
    cfg.fno_backend = TARGET_BACKEND
    cfg.load_type = TARGET_LOAD_TYPE
    cfg.experiment_group = "arch"
    cfg.method = "MSE"
    cfg.preload = False
    cfg.exp_id_override = None
    for key, value in architecture.items():
        setattr(cfg, key, value)

    print(f"\nArchitecture: {architecture}")
    reset_global_seed(MODEL_INIT_SEED)
    Training(cfg).run_train()

print("\nFNO architecture training completed.")
