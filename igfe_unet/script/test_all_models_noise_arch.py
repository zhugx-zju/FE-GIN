import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from model.test import Testing
from postprocess.common import (
    extract_mix_ratio_info,
    find_all_experiments,
    load_experiment_results,
)
from utils.utils_process import Config
from utils.utils_test import fixed_test_set_exists, get_fixed_test_data_dir

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'
TARGET_LOAD_TYPE = 'force_load'
TARGET_METHOD = 'MSE'
TARGET_USE_BATCH_NORM = True
EXCLUDE_RATIO_EXPERIMENTS = True
# ============================================================================

print("=" * 80)
print("Batch Testing: Architecture experiments with Multi-Noise")
print("=" * 80)
print(f"Target config type: {TARGET_CONFIG_TYPE}")
print(f"Target load type: {TARGET_LOAD_TYPE}")
print(f"Target method: {TARGET_METHOD}")
print(f"Target use_batch_norm: {TARGET_USE_BATCH_NORM}")
print(f"Exclude ratio experiments: {EXCLUDE_RATIO_EXPERIMENTS}")

base_cfg = Config(TARGET_CONFIG_TYPE)
base_cfg.config_type = TARGET_CONFIG_TYPE
noise_levels = list(getattr(base_cfg, 'noise_levels', [0, 1, 3]))
eval_types = getattr(base_cfg, 'eval_types', 'all')
test_num = getattr(base_cfg, 'num', 'all')

print(f"Noise levels: {noise_levels}")
print(f"Evaluation types: {eval_types}")
print(f"Number of test samples (num): {test_num}")
print("Evaluation split: test")
if fixed_test_set_exists(base_cfg):
    print(f"Shared fixed test set: {get_fixed_test_data_dir(base_cfg)}")
else:
    print("Shared fixed test set: not found, fallback to sliced cfg.data_path test split")
print("=" * 80)

print("\nSearching for experiments...")
experiments = find_all_experiments()
print(f"Found {len(experiments)} total experiments")

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

    if EXCLUDE_RATIO_EXPERIMENTS:
        ratio_info = extract_mix_ratio_info(results, exp_id=exp_id, exp_path=exp_path)
        if ratio_info.get('ratio_tag') is not None:
            continue

    selected.append((exp_info, exp_cfg))

print(f"Filtered to {len(selected)} experiments")
if not selected:
    print("No experiments found. Exit.")
    sys.exit(0)

selected.sort(key=lambda x: x[0][2])

for (_, load_type, exp_id, exp_path), exp_cfg in selected:
    print("\n" + "-" * 80)
    print(f"Model: {exp_id}")
    print(f"Path: {exp_path}")

    cfg = Config(TARGET_CONFIG_TYPE)
    cfg.config_type = TARGET_CONFIG_TYPE
    cfg.load_type = load_type
    cfg.num = test_num
    cfg.eval_types = eval_types
    cfg.save_format = getattr(base_cfg, 'save_format', 'both')
    cfg.device = getattr(base_cfg, 'device', cfg.device)
    cfg.eval_split = 'test'

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

    cfg.noise_level = 0
    tester = Testing(cfg, experiment_path=exp_path)

    for noise in noise_levels:
        tester.noise_level = noise if noise != 0 else None
        print(f"\nRunning prediction with noise_level={noise} ...")
        tester.compute_and_save_predictions()

print("\n" + "=" * 80)
print("Batch architecture testing completed.")
print("=" * 80)

