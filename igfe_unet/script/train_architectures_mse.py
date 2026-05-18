import os
import random
import sys

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from model.train import Training
from utils.utils_process import Config, generate_arch_id, get_filepath, network_variant_name

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'
TARGET_LOAD_TYPE = 'force_load'
TARGET_METHOD = 'MSE'
MODEL_INIT_SEED = 42
# TARGET_ARCHITECTURES = [
#     [2, 16, 32],
#     [2, 16, 32, 64],
#     [2, 16, 32, 64, 128],
#     [2, 32, 64],
#     [2, 32, 64, 128],
#     [2, 32, 64, 128, 256],
#     [2, 64, 128],
#     [2, 64, 128, 256],
#     [2, 32, 96, 192],
#     [2, 32, 128, 192],
#     [2, 32, 128, 256],
# ]
TARGET_ARCHITECTURES = [
    [2, 32, 64],
    [2, 32, 64, 128],
    [2, 32, 64, 128, 256],
    [2, 32, 96, 192],
    [2, 32, 128, 256],
]
# ============================================================================


def reset_global_seed(seed):
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


print('=' * 80)
print('Batch Training: Architecture sweep with MSE')
print('=' * 80)
print(f'Config type: {TARGET_CONFIG_TYPE}')
print(f'Load type: {TARGET_LOAD_TYPE}')
print(f'Method: {TARGET_METHOD}')
print(f'Model init seed: {MODEL_INIT_SEED}')
print(f'Target architecture count: {len(TARGET_ARCHITECTURES)}')

for filters_list in TARGET_ARCHITECTURES:
    cfg = Config(TARGET_CONFIG_TYPE)
    cfg.config_type = TARGET_CONFIG_TYPE
    cfg.load_type = TARGET_LOAD_TYPE
    cfg.method = TARGET_METHOD
    cfg.filters_list = list(filters_list)
    cfg.preload = False
    cfg.exp_id_override = None

    method_short = get_filepath(cfg.method)
    arch_id = generate_arch_id(cfg.filters_list)
    exp_id = f'{method_short}_{network_variant_name(cfg)}_arch_{arch_id}'

    print('\n' + '-' * 80)
    print(f'Architecture: {cfg.filters_list}')
    print(f'Experiment id: {exp_id}')
    print(f'- Reset model seed: {MODEL_INIT_SEED}')
    reset_global_seed(MODEL_INIT_SEED)

    print('- Train model...')
    trainer = Training(cfg)
    trainer.run_train()

print('\n' + '=' * 80)
print('Batch architecture training completed.')
print('=' * 80)
