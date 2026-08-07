import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from postprocess.common import (
    _gn_suffix,
    find_all_experiments,
    load_experiment_results,
    generate_comparison_dataframe,
    generate_paraset_table,
    save_results,
    resolve_analysis_output_dir,
)
from pathlib import Path

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_METHOD = 'MSE'  # Only compare MSE loss
TARGET_USE_BATCH_NORM = True
TARGET_CONFIG_TYPE = 'mix'  # Dataset to compare on
TARGET_LOAD_TYPE = 'force_load'  # Load type to compare
TARGET_EXPERIMENT_GROUP = 'arch'
# ============================================================================
OUTPUT_DIR = resolve_analysis_output_dir(
    TARGET_EXPERIMENT_GROUP,
    load_type=TARGET_LOAD_TYPE,
    use_batch_norm=TARGET_USE_BATCH_NORM,
)
TXT_FILE = OUTPUT_DIR / f"tableC1_architectures{_gn_suffix(TARGET_USE_BATCH_NORM)}.txt"

print("="*80)
print("Table 1: Architecture Comparison")
print("="*80)
print(f"Configuration:")
print(f"  Loss Function: {TARGET_METHOD}")
print(f"  Use Batch Norm: {TARGET_USE_BATCH_NORM}")
print(f"  Dataset: {TARGET_CONFIG_TYPE}")
print(f"  Load Type: {TARGET_LOAD_TYPE}")
print("="*80)

print("\nSearching for experiments...")
experiments = find_all_experiments(experiment_group=TARGET_EXPERIMENT_GROUP)
print(f"Found {len(experiments)} total experiments")

print("\nLoading and filtering experiment results...")
experiments_data = {}
for exp_info in experiments:
    config_type, load_type, exp_id, exp_path = exp_info

    # Filter: only target dataset and load type
    if config_type != TARGET_CONFIG_TYPE or load_type != TARGET_LOAD_TYPE:
        continue

    results = load_experiment_results(exp_path)

    # Filter: only MSE method
    config = results.get('config') or {}
    use_batch_norm = bool(config.get('use_batch_norm', False))
    method = config.get('method', '')
    if use_batch_norm != TARGET_USE_BATCH_NORM:
        continue
    if method != TARGET_METHOD:
        continue

    print(f"  Loading: {config_type}/{load_type}/{exp_id}")
    experiments_data[exp_info] = results

print(f"\nFiltered to {len(experiments_data)} experiments matching criteria")

if not experiments_data:
    print("\nERROR: No experiments found matching the criteria!")
    print("Please check:")
    print(f"  1. Loss function: {TARGET_METHOD}")
    print(f"  2. Dataset type: {TARGET_CONFIG_TYPE}")
    print(f"  3. Load type: {TARGET_LOAD_TYPE}")
    sys.exit(1)

print("\nGenerating comparison dataframe...")
df = generate_comparison_dataframe(experiments_data)

print("\nSaving results...")
csv_file = save_results(df, output_dir=OUTPUT_DIR)
print(f"Saved comparison table to: {csv_file}")

print("\nGenerating Table 1: Architecture comparison...")
paraset_table = generate_paraset_table(df)

if paraset_table:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(TXT_FILE, 'w') as f:
        f.write(paraset_table)
    print(f"Saved: {TXT_FILE}")
    print("\nTable 1 Preview:")
    print(paraset_table)
else:
    print("No data available for architecture comparison table")

print("\n" + "="*80)
print("Architecture comparison complete!")
print("="*80)
