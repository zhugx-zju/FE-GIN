import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.utils_process import Config
from postprocess.common import find_all_experiments, load_experiment_results
from model.test import Testing

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'
TARGET_LOAD_TYPE = 'force_load'
TARGET_USE_BATCH_NORM = True
# ============================================================================

print("=" * 80)
print("Batch Validation: All trained_models_mix with Multi-Noise")
print("=" * 80)
print(f"Target config type: {TARGET_CONFIG_TYPE}")
print(f"Target load type: {TARGET_LOAD_TYPE}")
print(f"Target use_batch_norm: {TARGET_USE_BATCH_NORM}")

base_cfg = Config(TARGET_CONFIG_TYPE)
base_cfg.config_type = TARGET_CONFIG_TYPE
noise_levels = list(getattr(base_cfg, 'noise_levels', [0, 1, 3]))
eval_types = getattr(base_cfg, 'eval_types', 'all')
val_num = getattr(base_cfg, 'num', 'all')

print(f"Noise levels: {noise_levels}")
print(f"Evaluation types: {eval_types}")
print(f"Number of validation samples (num): {val_num}")
print("Evaluation split: val")
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
    exp_use_batch_norm = bool(exp_cfg.get('use_batch_norm', False))
    if exp_use_batch_norm != TARGET_USE_BATCH_NORM:
        continue
    if not os.path.exists(os.path.join(exp_path, 'model.pt')):
        continue
    selected.append((exp_info, exp_cfg))

print(f"Filtered to {len(selected)} experiments")
if not selected:
    print("No experiments found. Exit.")
    sys.exit(0)

selected.sort(key=lambda x: x[0][2])

for (config_type, load_type, exp_id, exp_path), exp_cfg in selected:
    print("\n" + "-" * 80)
    print(f"Model: {exp_id}")
    print(f"Path: {exp_path}")

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

    cfg.noise_level = 0
    tester = Testing(cfg, experiment_path=exp_path)

    for noise in noise_levels:
        tester.noise_level = noise if noise != 0 else None
        print(f"\nRunning validation prediction with noise_level={noise} ...")
        tester.compute_and_save_predictions()

print("\n" + "=" * 80)
print("Batch validation completed.")
print("=" * 80)

