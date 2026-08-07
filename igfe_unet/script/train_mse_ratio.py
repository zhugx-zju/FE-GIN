import os
import random
import sys

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from data_process import create_mix_dataset
from model.train import Training
from utils.utils_process import (
    Config,
    generate_arch_id,
    generate_mix_ratio_tag,
    get_filepath,
    network_variant_name,
)

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'
TARGET_METHOD = 'MSE'
TARGET_LOAD_TYPE = 'force_load'
TARGET_RATIOS = [
    (0.33, 0.33, 0.34),
    (0.10, 0.80, 0.10),
    (0.10, 0.70, 0.20),
    (0.10, 0.60, 0.30),
    (0.10, 0.50, 0.40),
    (0.10, 0.40, 0.50),
    (0.10, 0.30, 0.60),
    (0.10, 0.20, 0.70),
    (0.10, 0.10, 0.80),
]
DATA_SEED = 42
MODEL_INIT_SEED = 42
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
print('Batch Training: MSE with different mix ratios')
print('=' * 80)
print(f'Config type: {TARGET_CONFIG_TYPE}')
print(f'Method: {TARGET_METHOD}')
print(f'Load type: {TARGET_LOAD_TYPE}')
print(f'Data seed: {DATA_SEED}')
print(f'Model init seed: {MODEL_INIT_SEED}')
print(f'Ratio sets: {TARGET_RATIOS}')

for bil_ratio, exp_ratio, grf_ratio in TARGET_RATIOS:
    ratio_sum = bil_ratio + exp_ratio + grf_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f'Invalid ratio set {(bil_ratio, exp_ratio, grf_ratio)}; sum={ratio_sum}'
        )

    cfg = Config(TARGET_CONFIG_TYPE)
    cfg.config_type = TARGET_CONFIG_TYPE
    cfg.load_type = TARGET_LOAD_TYPE
    cfg.experiment_group = 'ratio'
    cfg.method = TARGET_METHOD
    cfg.preload = False
    cfg.bil_ratio = float(bil_ratio)
    cfg.exp_ratio = float(exp_ratio)
    cfg.grf_ratio = float(grf_ratio)

    ratio_tag = generate_mix_ratio_tag(cfg)
    method_short = get_filepath(cfg.method)
    arch_id = generate_arch_id(cfg.filters_list)
    cfg.exp_id_override = f'{method_short}_{network_variant_name(cfg)}_arch_{arch_id}_{ratio_tag}'

    print('\n' + '-' * 80)
    print(f'Ratio tag: {ratio_tag}')
    print(f'Experiment id: {cfg.exp_id_override}')
    print('- Build mix dataset...')
    ratio_data_dir = os.path.abspath(
        os.path.join(root_dir, '..', 'data', 'mix_ratio_datasets', ratio_tag, cfg.load_type)
    )
    ds_info = create_mix_dataset(
        seed=DATA_SEED,
        bil_ratio=cfg.bil_ratio,
        exp_ratio=cfg.exp_ratio,
        grf_ratio=cfg.grf_ratio,
        train_rto=cfg.train_rto,
        valid_rto=cfg.valid_rto,
        load_type=cfg.load_type,
        output_dir=ratio_data_dir,
        verbose=True,
    )
    cfg.data_path = ds_info['mix_dir']
    cfg.dataset_manifest = ds_info['manifest_path']
    cfg.dataset_id = ratio_tag
    cfg.variable_names = list(cfg.variable_names)
    for key in ['dataset_id', 'dataset_manifest']:
        if key not in cfg.variable_names:
            cfg.variable_names.append(key)
    print(
        f"- Dataset ready: train/val/test = "
        f"{ds_info['split_sizes']['train']}/"
        f"{ds_info['split_sizes']['val']}/"
        f"{ds_info['split_sizes']['test']}"
    )

    print(f'- Reset model seed: {MODEL_INIT_SEED}')
    reset_global_seed(MODEL_INIT_SEED)

    print('- Train model...')
    trainer = Training(cfg)
    trainer.run_train()

print('\n' + '=' * 80)
print('Batch ratio training completed.')
print('=' * 80)
