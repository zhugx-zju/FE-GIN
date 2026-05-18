import os
import sys
from fnmatch import fnmatch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from postprocess.common import (
    _gn_suffix,
    extract_mix_ratio_info,
    find_all_experiments,
    load_experiment_results,
)
from postprocess.robustness import (
    plot_ecdf_curve,
    plot_ratio_mix_heatmap_from_csv,
    plot_ratio_representative_trends_from_csv,
    save_noise_statistics,
)

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'
TARGET_LOAD_TYPE = 'force_load'
TARGET_METHOD = 'MSE'
TARGET_USE_BATCH_NORM = True
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
OUTPUT_DIR = os.path.join(root_dir, '..', f'ratio_robustness_comparison{_gn_suffix(TARGET_USE_BATCH_NORM)}')
# ============================================================================

print("=" * 80)
print("MSE Ratio Robustness Comparison")
print("=" * 80)
print("Configuration:")
print(f"  Dataset: {TARGET_CONFIG_TYPE}")
print(f"  Load Type: {TARGET_LOAD_TYPE}")
print(f"  Method: {TARGET_METHOD}")
print(f"  Use Batch Norm: {TARGET_USE_BATCH_NORM}")
print(f"  Ratio tags: {TARGET_RATIO_TAGS}")
print(f"  Output dir: {OUTPUT_DIR}")
print("=" * 80)

print("\nSearching for experiments...")
experiments = find_all_experiments()
print(f"Found {len(experiments)} total experiments")

print("\nLoading and filtering experiment results...")
experiments_data = {}
for exp_info in experiments:
    config_type, load_type, exp_id, exp_path = exp_info

    if config_type != TARGET_CONFIG_TYPE or load_type != TARGET_LOAD_TYPE:
        continue

    results = load_experiment_results(exp_path)
    config = results.get('config') or {}
    method = config.get('method', '')
    use_batch_norm = bool(config.get('use_batch_norm', False))
    if method != TARGET_METHOD:
        continue
    if use_batch_norm != TARGET_USE_BATCH_NORM:
        continue

    ratio_info = extract_mix_ratio_info(results, exp_id=exp_id, exp_path=exp_path)
    ratio_tag = ratio_info.get('ratio_tag')
    if TARGET_RATIO_TAGS and ratio_tag not in TARGET_RATIO_TAGS:
        continue

    print(f"  Loading: {exp_id} (ratio={ratio_tag})")
    experiments_data[exp_info] = results

print(f"\nFiltered to {len(experiments_data)} experiments matching criteria")

if not experiments_data:
    print("\nERROR: No experiments found matching the criteria!")
    print("Please check:")
    print(f"  1. Dataset type: {TARGET_CONFIG_TYPE}")
    print(f"  2. Load type: {TARGET_LOAD_TYPE}")
    print(f"  3. Method: {TARGET_METHOD}")
    print(f"  4. Ratio tags: {TARGET_RATIO_TAGS}")
    sys.exit(1)

print("\nSaving noise-level statistics...")
csv_file = save_noise_statistics(
    experiments_data,
    output_dir=OUTPUT_DIR,
    filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
    label_mode='ratio',
)

print("\nGenerating ECDF plot...")
plot_ecdf_curve(
    experiments_data,
    output_dir=OUTPUT_DIR,
    filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
    label_mode='ratio',
)

print("\nGenerating ratio heatmap and representative trend plots...")
plot_ratio_mix_heatmap_from_csv(
    csv_file,
    output_dir=OUTPUT_DIR,
    filename=f'mix_mean_heatmap{_gn_suffix(TARGET_USE_BATCH_NORM)}.png',
)
plot_ratio_representative_trends_from_csv(
    csv_file,
    output_dir=OUTPUT_DIR,
    filename=f'ratio_representative_trends{_gn_suffix(TARGET_USE_BATCH_NORM)}.png',
)

print("\n" + "=" * 80)
print("MSE ratio robustness comparison complete!")
print("=" * 80)
