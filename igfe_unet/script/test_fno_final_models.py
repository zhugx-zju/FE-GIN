"""Test manually selected FNO checkpoints across the complete noise sweep."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.test import Testing
from utils.utils_process import Config
from fno_workflow import config_from_saved_fno, fno_noise_levels, select_fno_experiments


TARGET_LOAD_TYPE = "force_load"
TARGET_BACKEND = "custom"  # Set to "neuralop" only when that package is installed.
TARGET_NUM = "all"

base_cfg = Config("fno")
selected = select_fno_experiments("final_model", TARGET_LOAD_TYPE, TARGET_BACKEND)

print(f"Found {len(selected)} manually selected FNO final models")
for (config_type, load_type, exp_id, exp_path), saved_config in selected:
    cfg = config_from_saved_fno(saved_config, "final_model", load_type)
    cfg.eval_split = "test"
    cfg.num = TARGET_NUM
    print(f"\nFinal-model test: {exp_id}")
    tester = Testing(cfg, experiment_path=exp_path)
    for noise in fno_noise_levels(base_cfg):
        tester.noise_level = noise if noise else None
        print(f"  noise_level={noise}")
        tester.compute_and_save_predictions()

print("\nFNO final-model testing completed.")
