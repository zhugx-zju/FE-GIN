"""Generate a shared U-Net/FNO metric table from final-model outputs."""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from postprocess.common import (
    find_all_experiments,
    load_experiment_results,
    resolve_analysis_output_dir,
    save_unified_metrics_table,
)


TARGET_LOAD_TYPE = "force_load"
TARGET_EXPERIMENT_GROUP = "final_model"
TARGET_NOISE_LEVELS = [0, 2, 4, 6, 8, 10]
TARGET_EVAL_TYPES = ["mix", "bil", "exp", "grf"]

experiments_data = {}
for exp_info in find_all_experiments(experiment_group=TARGET_EXPERIMENT_GROUP):
    config_type, load_type, exp_id, exp_path = exp_info
    if load_type != TARGET_LOAD_TYPE:
        continue
    results = load_experiment_results(exp_path)
    config = results.get("config") or {}
    dataset_type = str(config.get("dataset_type", "mix" if config_type == "mix" else config_type)).lower()
    if dataset_type != "mix":
        continue
    if str(config.get("model_type", "unet")).lower() not in {"unet", "fno"}:
        continue
    experiments_data[exp_info] = results

if not experiments_data:
    raise SystemExit(
        "No final_model U-Net/FNO experiments found. Test selected models first."
    )

output_dir = resolve_analysis_output_dir(
    TARGET_EXPERIMENT_GROUP,
    load_type=TARGET_LOAD_TYPE,
    root_name="results",
)
save_unified_metrics_table(
    experiments_data,
    output_dir=output_dir,
    noise_levels=TARGET_NOISE_LEVELS,
    eval_types=TARGET_EVAL_TYPES,
)
