import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from model.train import Training
from utils.utils_process import Config, config_from_json, resolve_experiment_group
from utils.reproducibility import seed_everything


MODEL_INIT_SEED = int(os.environ.get('SEED', '42'))


def reset_global_seed(seed):
    """Set a unified random seed at training entry."""
    return seed_everything(seed)

# %% Specify the dataset -------------------------
config_type = config_from_json()
# %% Load the configuration file -------------------------
cfg = Config(config_type)
cfg.config_type = config_type
cfg.seed = MODEL_INIT_SEED
cfg.experiment_group = resolve_experiment_group(
    cfg,
    override=os.environ.get('EXPERIMENT_GROUP') or None,
)
reset_global_seed(MODEL_INIT_SEED)
print(f"Global seed: {MODEL_INIT_SEED}")
print(f"Using device: {cfg.device}")
print(f"Experiment group: {cfg.experiment_group}")
trainer = Training(cfg)
trainer.run_train()
# %%
