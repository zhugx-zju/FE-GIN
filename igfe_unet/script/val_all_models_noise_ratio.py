import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from model.test import Testing
from data_process import create_mix_dataset
from postprocess.common import (
    extract_mix_ratio_info,
    find_all_experiments,
    load_experiment_results,
)
from utils.utils_process import Config

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'
TARGET_LOAD_TYPE = 'force_load'
TARGET_METHOD = 'MSE'
TARGET_USE_BATCH_NORM = True
TARGET_EXPERIMENT_GROUP = 'ratio'
DATA_SEED = 4
TARGET_RATIO_TAGS = [
    'b0p33_e0p33_g0p34',
    'b0p1_e0p8_g0p1',
    'b0p1_e0p7_g0p2',
    'b0p1_e0p6_g0p3',
    'b0p1_e0p5_g0p4',
    'b0p1_e0p4_g0p5',
    'b0p1_e0p3_g0p6',
    'b0p1_e0p2_g0p7',
    'b0p1_e0p1_g0p8',
]
# ============================================================================

print('=' * 80)
print('Batch Validation: MSE ratio experiments with Multi-Noise')
print('=' * 80)
print(f'Target config type: {TARGET_CONFIG_TYPE}')
print(f'Target load type: {TARGET_LOAD_TYPE}')
print(f'Target method: {TARGET_METHOD}')
print(f'Target use_batch_norm: {TARGET_USE_BATCH_NORM}')
print(f'Data seed (for rebuilding mix set): {DATA_SEED}')
print(f'Target ratio tags: {TARGET_RATIO_TAGS}')

base_cfg = Config(TARGET_CONFIG_TYPE)
base_cfg.config_type = TARGET_CONFIG_TYPE
noise_levels = list(getattr(base_cfg, 'noise_levels', [0, 1, 3]))
eval_types = getattr(base_cfg, 'eval_types', 'all')
val_num = getattr(base_cfg, 'num', 'all')

print(f'Noise levels: {noise_levels}')
print(f'Evaluation types: {eval_types}')
print(f'Number of validation samples (num): {val_num}')
print('Evaluation split: val')
print('=' * 80)

print('\nSearching for experiments...')
experiments = find_all_experiments(experiment_group=TARGET_EXPERIMENT_GROUP)
print(f'Found {len(experiments)} total experiments')

selected = []
for exp_info in experiments:
    config_type, load_type, exp_id, exp_path = exp_info
    if config_type != TARGET_CONFIG_TYPE or load_type != TARGET_LOAD_TYPE:
        continue

    results = load_experiment_results(exp_path)
    exp_cfg = results.get('config') or {}
    if not exp_cfg:
        continue
    if exp_cfg.get('method') != TARGET_METHOD:
        continue
    exp_use_batch_norm = bool(exp_cfg.get('use_batch_norm', False))
    if exp_use_batch_norm != TARGET_USE_BATCH_NORM:
        continue
    if not os.path.exists(os.path.join(exp_path, 'model.pt')):
        continue

    ratio_info = extract_mix_ratio_info(results, exp_id=exp_id, exp_path=exp_path)
    ratio_tag = ratio_info.get('ratio_tag')
    if TARGET_RATIO_TAGS and ratio_tag not in TARGET_RATIO_TAGS:
        continue

    selected.append((exp_info, exp_cfg, ratio_tag, ratio_info))

print(f'Filtered to {len(selected)} experiments')
if not selected:
    print('No experiments found. Exit.')
    sys.exit(0)

ratio_order = {tag: i for i, tag in enumerate(TARGET_RATIO_TAGS)}
selected.sort(key=lambda x: (ratio_order.get(x[2], 10**9), x[0][2]))

for (_, load_type, exp_id, exp_path), exp_cfg, ratio_tag, ratio_info in selected:
    print('\n' + '-' * 80)
    print(f'Model: {exp_id}')
    print(f'Ratio: {ratio_tag}')
    print(f'Path: {exp_path}')

    cfg = Config(TARGET_CONFIG_TYPE)
    cfg.config_type = TARGET_CONFIG_TYPE
    cfg.load_type = load_type
    cfg.num = val_num
    cfg.eval_types = eval_types
    cfg.save_format = getattr(base_cfg, 'save_format', 'both')
    cfg.device = getattr(base_cfg, 'device', cfg.device)
    cfg.eval_split = 'val'

    if 'method' in exp_cfg:
        cfg.method = exp_cfg['method']
    if 'filters_list' in exp_cfg:
        cfg.filters_list = exp_cfg['filters_list']
    if 'kernel_size' in exp_cfg:
        cfg.kernel_size = exp_cfg['kernel_size']
    if 'use_batch_norm' in exp_cfg:
        cfg.use_batch_norm = exp_cfg['use_batch_norm']
    if 'gamma' in exp_cfg:
        cfg.gamma = exp_cfg['gamma']
    if 'bil_ratio' in exp_cfg:
        cfg.bil_ratio = exp_cfg['bil_ratio']
    if 'exp_ratio' in exp_cfg:
        cfg.exp_ratio = exp_cfg['exp_ratio']
    if 'grf_ratio' in exp_cfg:
        cfg.grf_ratio = exp_cfg['grf_ratio']

    bil_ratio = ratio_info.get('bil_ratio')
    exp_ratio = ratio_info.get('exp_ratio')
    grf_ratio = ratio_info.get('grf_ratio')
    if bil_ratio is not None and exp_ratio is not None and grf_ratio is not None:
        print(
            f"Rebuild mix dataset for ratio: "
            f"bil={bil_ratio:.4f}, exp={exp_ratio:.4f}, grf={grf_ratio:.4f}"
        )
        create_mix_dataset(
            seed=DATA_SEED,
            bil_ratio=bil_ratio,
            exp_ratio=exp_ratio,
            grf_ratio=grf_ratio,
            train_rto=cfg.train_rto,
            valid_rto=cfg.valid_rto,
            load_type=cfg.load_type,
            verbose=False,
        )

    cfg.noise_level = 0
    tester = Testing(cfg, experiment_path=exp_path)

    for noise in noise_levels:
        tester.noise_level = noise if noise != 0 else None
        print(f'\nRunning validation prediction with noise_level={noise} ...')
        tester.compute_and_save_predictions()

print('\n' + '=' * 80)
print('Batch ratio validation completed.')
print('=' * 80)

