import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from postprocess.common import _gn_suffix, find_all_experiments, load_experiment_results
from postprocess.common import extract_mix_ratio_info
from postprocess.gamma import (
    save_gamma_statistics,
    grouped_data_from_gamma_statistics_csv,
    plot_gamma_error_heatmaps,
    plot_gamma_ecdf_panels,
    plot_gamma_mean_std_bars,
    plot_gamma_history_2x2,
)

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_CONFIG_TYPE = 'mix'
TARGET_USE_BATCH_NORM = True
TARGET_ARCHITECTURE = '[2, 32, 64, 128]'
TARGET_LOAD_TYPE = 'force_load'
TARGET_METHODS = ['MSE', 'LocResloss', 'GloResloss', 'LocMixloss', 'GloMixloss']
TARGET_EVAL_TYPES = ['bil', 'exp', 'grf', 'mix']
EXCLUDE_RATIO_EXPERIMENTS = True
# Set to None to auto-detect all available noise levels.
TARGET_NOISE_LEVELS = [0, 2, 4, 6, 8, 10]
TARGET_HEATMAP_NOISE_LEVELS = TARGET_NOISE_LEVELS
TARGET_DATA_SPLIT = os.environ.get('TARGET_DATA_SPLIT', 'test')  # options: 'val', 'test'
TARGET_DATA_SPLIT = TARGET_DATA_SPLIT.strip().lower()
if TARGET_DATA_SPLIT not in ['val', 'test']:
    raise ValueError(f"Invalid TARGET_DATA_SPLIT: {TARGET_DATA_SPLIT}. Use 'val' or 'test'.")
OUTPUT_DIR = os.path.abspath(os.path.join(root_dir, '..', f'gamma_comparison{_gn_suffix(TARGET_USE_BATCH_NORM)}', TARGET_DATA_SPLIT))
ENABLE_DETAILED_CURVES = True  # set False to skip ECDF/curve/history figures
# ============================================================================


def _fmt_noise_levels(levels):
    if levels is None:
        return 'auto'
    return str(levels)


print('=' * 90)
print('Gamma Comparison')
print('=' * 90)
print('Configuration:')
print(f"  Dataset: {TARGET_CONFIG_TYPE}")
print(f"  Use Batch Norm: {TARGET_USE_BATCH_NORM}")
print(f"  Architecture: {TARGET_ARCHITECTURE}")
print(f"  Load Type: {TARGET_LOAD_TYPE}")
print(f"  Methods: {', '.join(TARGET_METHODS)}")
print(f"  Eval types: {', '.join(TARGET_EVAL_TYPES)}")
print(f"  Exclude ratio experiments: {EXCLUDE_RATIO_EXPERIMENTS}")
print(f"  Noise levels: {_fmt_noise_levels(TARGET_NOISE_LEVELS)}")
print(f"  Heatmap noise levels: {_fmt_noise_levels(TARGET_HEATMAP_NOISE_LEVELS)}")
print(f"  Data split: {TARGET_DATA_SPLIT}")
print(f"  Output dir: {OUTPUT_DIR}")
print('=' * 90)

print('\nSearching for experiments...')
experiments = find_all_experiments()
print(f"Found {len(experiments)} total experiments")

print('\nLoading and filtering experiment results...')
experiments_data = {}
for exp_info in experiments:
    config_type, load_type, exp_id, exp_path = exp_info

    if config_type != TARGET_CONFIG_TYPE or load_type != TARGET_LOAD_TYPE:
        continue

    results = load_experiment_results(exp_path)
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
    print('\nNo experiments found matching the criteria.')
    csv_fallback = os.path.join(OUTPUT_DIR, f'gamma_statistics{_gn_suffix(TARGET_USE_BATCH_NORM)}.csv')
    if not os.path.exists(csv_fallback):
        print(f"Fallback gamma_statistics.csv not found at: {csv_fallback}")
        print('Please check:')
        print(f"  1. Dataset type: {TARGET_CONFIG_TYPE}")
        print(f"  2. Architecture: {TARGET_ARCHITECTURE}")
        print(f"  3. Load type: {TARGET_LOAD_TYPE}")
        print(f"  4. Methods: {', '.join(TARGET_METHODS)}")
        print(f"  5. Output dir: {OUTPUT_DIR}")
        sys.exit(1)

    print(f"Using existing statistics CSV: {csv_fallback}")
    grouped_data = grouped_data_from_gamma_statistics_csv(
        csv_fallback,
        eval_types=TARGET_EVAL_TYPES,
        noise_levels=TARGET_NOISE_LEVELS,
    )
    if not grouped_data:
        print('ERROR: Fallback CSV has no usable rows after filters.')
        sys.exit(1)

    print('\nGenerating gamma overview heatmaps from CSV...')
    plot_gamma_error_heatmaps(
        experiments_data={},
        output_dir=OUTPUT_DIR,
        eval_types=TARGET_EVAL_TYPES,
        noise_levels=TARGET_HEATMAP_NOISE_LEVELS,
        data_split=TARGET_DATA_SPLIT,
        grouped_data=grouped_data,
        filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
    )
    if ENABLE_DETAILED_CURVES:
        print('Detailed curves require raw sample files. Skipping in CSV fallback mode.')

    print('\n' + '=' * 90)
    print('Gamma comparison complete! (CSV fallback mode)')
    print('=' * 90)
    sys.exit(0)

print('\nSaving gamma statistics...')
save_gamma_statistics(
    experiments_data,
    output_dir=OUTPUT_DIR,
    eval_types=TARGET_EVAL_TYPES,
    noise_levels=TARGET_NOISE_LEVELS,
    data_split=TARGET_DATA_SPLIT,
    filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
)

print('\nGenerating gamma overview heatmaps...')
plot_gamma_error_heatmaps(
    experiments_data,
    output_dir=OUTPUT_DIR,
    eval_types=TARGET_EVAL_TYPES,
    noise_levels=TARGET_HEATMAP_NOISE_LEVELS,
    data_split=TARGET_DATA_SPLIT,
    filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
)

if ENABLE_DETAILED_CURVES:
    print('\nGenerating gamma ECDF panels...')
    plot_gamma_ecdf_panels(
        experiments_data,
        output_dir=OUTPUT_DIR,
        eval_types=TARGET_EVAL_TYPES,
        noise_levels=TARGET_NOISE_LEVELS,
        data_split=TARGET_DATA_SPLIT,
        filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
    )

    print('\nGenerating gamma performance curves (mean+/-std)...')
    plot_gamma_mean_std_bars(
        experiments_data,
        output_dir=OUTPUT_DIR,
        eval_types=TARGET_EVAL_TYPES,
        noise_levels=TARGET_NOISE_LEVELS,
        data_split=TARGET_DATA_SPLIT,
        filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
    )

    print('\nGenerating gamma history 2x2 plot...')
    plot_gamma_history_2x2(
        experiments_data,
        output_dir=OUTPUT_DIR,
        filename_suffix=_gn_suffix(TARGET_USE_BATCH_NORM),
    )

print('\n' + '=' * 90)
print('Gamma comparison complete!')
print('=' * 90)

