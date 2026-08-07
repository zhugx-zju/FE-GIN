import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from postprocess.common import (
    find_all_experiments,
    load_experiment_results,
    resolve_analysis_output_dir,
)
from postprocess.common import extract_mix_ratio_info
from postprocess.common import _gn_suffix
from postprocess.history import plot_history_row

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'
TARGET_USE_BATCH_NORM = True
TARGET_ARCHITECTURE = '[2, 32, 64, 128]'
TARGET_LOAD_TYPE = 'force_load'
TARGET_METHODS = ['MSE', 'LocResloss', 'GloResloss']
TARGET_EXPERIMENT_GROUP = 'std'
EXCLUDE_RATIO_EXPERIMENTS = True
# ============================================================================

print("=" * 80)
print("One-row Loss History Plot")
print("=" * 80)
print("Configuration:")
print(f"  Dataset: {TARGET_CONFIG_TYPE}")
print(f"  Use Batch Norm: {TARGET_USE_BATCH_NORM}")
print(f"  Architecture: {TARGET_ARCHITECTURE}")
print(f"  Load Type: {TARGET_LOAD_TYPE}")
print(f"  Methods: {TARGET_METHODS}")
print(f"  Exclude ratio experiments: {EXCLUDE_RATIO_EXPERIMENTS}")
print("=" * 80)

print("\nSearching for experiments...")
experiments = find_all_experiments(experiment_group=TARGET_EXPERIMENT_GROUP)
print(f"Found {len(experiments)} total experiments")

experiments_data = {}
for exp_info in experiments:
    config_type, load_type, exp_id, exp_path = exp_info
    if config_type != TARGET_CONFIG_TYPE or load_type != TARGET_LOAD_TYPE:
        continue

    results = load_experiment_results(exp_path)
    config = results.get('config') or {}
    if not config:
        continue
    method = config.get('method', '')
    use_batch_norm = bool(config.get('use_batch_norm', False))
    architecture = str(config.get('filters_list', ''))

    if method not in TARGET_METHODS:
        continue
    if use_batch_norm != TARGET_USE_BATCH_NORM:
        continue
    if architecture != TARGET_ARCHITECTURE:
        continue
    if EXCLUDE_RATIO_EXPERIMENTS:
        ratio_info = extract_mix_ratio_info(results, exp_id=exp_id, exp_path=exp_path)
        if ratio_info.get('ratio_tag') is not None:
            continue
    print(f"  Loading: {config_type}/{load_type}/{exp_id}")
    experiments_data[exp_info] = results

print(f"\nFiltered to {len(experiments_data)} experiments matching criteria")

if not experiments_data:
    print("\nERROR: No experiments found matching criteria.")
    sys.exit(1)

print("\nGenerating one-row history figure...")
fig_file = plot_history_row(
    experiments_data,
    target_methods=TARGET_METHODS,
    output_dir=resolve_analysis_output_dir(
        TARGET_EXPERIMENT_GROUP,
        load_type=TARGET_LOAD_TYPE,
        use_batch_norm=TARGET_USE_BATCH_NORM,
    ),
    filename=f'loss_history_row{_gn_suffix(TARGET_USE_BATCH_NORM)}.png',
    dpi=600
)

if fig_file is None:
    sys.exit(1)

print("\nDone.")
