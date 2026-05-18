import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from utils.utils_process import Config, config_from_json
from model.test import Testing
# %% Specify the dataset -------------------------
config_type = config_from_json()
# %% Load the configuration file -------------------------
cfg = Config(config_type)
cfg.config_type = config_type
print(f"Using device: {cfg.device}")
testor = Testing(cfg)
testor.compute_and_save_predictions()
# %%
