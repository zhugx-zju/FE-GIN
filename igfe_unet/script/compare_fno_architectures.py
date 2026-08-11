"""Compare FNO architecture candidates using the shared U-Net tables."""

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from postprocess.common import (
    find_all_experiments,
    generate_comparison_dataframe,
    generate_paraset_table,
    load_experiment_results,
    resolve_analysis_output_dir,
    save_results,
)


TARGET_BACKEND = "custom"
TARGET_LOAD_TYPE = "force_load"
TARGET_EXPERIMENT_GROUP = "arch"
TARGET_METHOD = "MSE"
TARGET_RESULTS_ROOT = "results_fno"

OUTPUT_DIR = resolve_analysis_output_dir(
    TARGET_EXPERIMENT_GROUP,
    load_type=TARGET_LOAD_TYPE,
    root_name=TARGET_RESULTS_ROOT,
)
TXT_FILE = OUTPUT_DIR / f"tableC1_fno_architectures_{TARGET_BACKEND}.txt"

experiments_data = {}
for exp_info in find_all_experiments(experiment_group=TARGET_EXPERIMENT_GROUP):
    config_type, load_type, exp_id, exp_path = exp_info
    if config_type not in {"fno", "fno_neuralop"} or load_type != TARGET_LOAD_TYPE:
        continue
    if not os.path.isfile(os.path.join(exp_path, "model.pt")):
        continue
    results = load_experiment_results(exp_path)
    config = results.get("config") or {}
    if str(config.get("model_type", "")).lower() != "fno":
        continue
    if str(config.get("fno_backend", "custom")).lower() != TARGET_BACKEND:
        continue
    if config.get("method") != TARGET_METHOD:
        continue
    experiments_data[exp_info] = results

if not experiments_data:
    raise SystemExit(
        "No tested FNO architecture experiments found. "
        "Run test_fno_architectures.py first."
    )

df = generate_comparison_dataframe(experiments_data)
csv_file = save_results(df, output_dir=OUTPUT_DIR)
table = generate_paraset_table(df)
if table:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TXT_FILE.write_text(table, encoding="utf-8")
    print(table)
print(f"Saved FNO architecture comparison CSV: {csv_file}")
print(f"Saved FNO architecture comparison table: {TXT_FILE}")
