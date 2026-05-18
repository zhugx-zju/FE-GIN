import sys
import os
import random
import numpy as np
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from model.train import Training
from utils.utils_process import Config, config_from_json


MODEL_INIT_SEED = 42


def reset_global_seed(seed):
    """Set a unified random seed at training entry."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# %% Specify the dataset -------------------------
config_type = config_from_json()
# %% Load the configuration file -------------------------
cfg = Config(config_type)
cfg.config_type = config_type
reset_global_seed(MODEL_INIT_SEED)
print(f"Global seed: {MODEL_INIT_SEED}")
print(f"Using device: {cfg.device}")
trainer = Training(cfg)
trainer.run_train()
# %%
