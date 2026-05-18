import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from postprocess.common import find_all_experiments, load_experiment_results
from postprocess.common import extract_mix_ratio_info
from postprocess.common import _gn_suffix
from postprocess.robustness import (
    plot_ecdf_curve,
    plot_noise_mean_std_from_csv,
    save_noise_statistics,
)

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'  # Only compare on mix dataset
TARGET_USE_BATCH_NORM = True
TARGET_ARCHITECTURE = '[2, 32, 64, 128]'  # Target network architecture
TARGET_LOAD_TYPE = 'force_load'  # Load type to compare
TARGET_METHODS = ['MSE', 'LocResloss', 'GloResloss']  # Robustness methods only
EXCLUDE_RATIO_EXPERIMENTS = True  # Exclude ratio-sweep experiments from robustness comparison
# ============================================================================
OUTPUT_DIR = os.path.join(root_dir, '..', f'comparison_results{_gn_suffix(TARGET_USE_BATCH_NORM)}')

print("=" * 80)
print("Robustness Comparison")
print("=" * 80)
print("Configuration:")
print(f"  Dataset: {TARGET_CONFIG_TYPE}")
print(f"  Use Batch Norm: {TARGET_USE_BATCH_NORM}")
print(f"  Architecture: {TARGET_ARCHITECTURE}")
print(f"  Load Type: {TARGET_LOAD_TYPE}")
print(f"  Methods: {', '.join(TARGET_METHODS)}")
print(f"  Exclude ratio experiments: {EXCLUDE_RATIO_EXPERIMENTS}")
print("=" * 80)

print("\nSearching for experiments...")
experiments = find_all_experiments()
print(f"Found {len(experiments)} total experiments")

print("\nLoading and filtering experiment results...")
experiments_data = {}
for exp_info in experiments:
    config_type, load_type, exp_id, exp_path = exp_info

    # Filter: only mix dataset with target load type
    if config_type != TARGET_CONFIG_TYPE or load_type != TARGET_LOAD_TYPE:
        continue

    results = load_experiment_results(exp_path)

    # Filter: only target architecture and selected methods
    config = results.get('config') or {}
    if not config:
        continue
    use_batch_norm = bool(config.get('use_batch_norm', False))
    architecture = str(config.get('filters_list', ''))
    method = config.get('method', '')
    if use_batch_norm != TARGET_USE_BATCH_NORM:
        continue
    if architecture != TARGET_ARCHITECTURE:
        continue
    if method not in TARGET_METHODS:
        continue
    if EXCLUDE_RATIO_EXPERIMENTS:
        ratio_info = extract_mix_ratio_info(results, exp_id=exp_id, exp_path=exp_path)
        if ratio_info.get('ratio_tag') is not None:
            continue

    print(f"  Loading: {config_type}/{load_type}/{exp_id}")
    experiments_data[exp_info] = results

print(f"\nFiltered to {len(experiments_data)} experiments matching criteria")

if not experiments_data:
    print("\nERROR: No experiments found matching the criteria!")
    print("Please check:")
    print(f"  1. Dataset type: {TARGET_CONFIG_TYPE}")
    print(f"  2. Architecture: {TARGET_ARCHITECTURE}")
    print(f"  3. Load type: {TARGET_LOAD_TYPE}")
    print(f"  4. Methods: {', '.join(TARGET_METHODS)}")
    sys.exit(1)

print("\nSaving noise-level statistics...")
noise_stats_csv = save_noise_statistics(
    experiments_data,
    output_dir=OUTPUT_DIR,
    filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
)

print("\nGenerating ECDF plots...")
plot_ecdf_curve(experiments_data, output_dir=OUTPUT_DIR, filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM))

print("\nGenerating mean+/-std plot from noise statistics...")
plot_noise_mean_std_from_csv(
    csv_file=noise_stats_csv,
    output_dir=OUTPUT_DIR,
    target_methods=TARGET_METHODS,
    filename=f'mean_std_combined{_gn_suffix(TARGET_USE_BATCH_NORM)}.png',
)

print("\n" + "=" * 80)
print("Robustness comparison complete!")
print("=" * 80)
