import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
unet_root = os.path.join(project_root, 'igfe_unet')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if unet_root not in sys.path:
    sys.path.insert(0, unet_root)

from config import get_config
from pipeline.asm_diagnostics import plot_asm_results_cases


# ---------------------------------------------------------------------------
# Manual ASM plotting cases
# Keep these settings aligned with the ASM runs you want to visualize.
# The script will read the corresponding ASM result file and generate:
# 1) true vs ASM prediction figures
# 2) ASM relative-error figures
# 3) iteration / L-curve diagnostics for the selected noise level
# If use_warm_start=True, it will read the method-specific warm-start result
# file based on warm_start_method.
# ---------------------------------------------------------------------------
CASES = [
    {
        'dataset': 'exp',
        'sample_index': 200,
        'noise_level': 6.0,
        'nodesx': 40,
        'nodesy': 40,
        'asm_gamma': None,
        'asm_max_iter': 1000,
        'asm_ftol': 1e-18,
        'asm_gtol': 1e-12,
        'use_warm_start': False,
        'warm_start_method': 'GloResloss',
    },
]

cfg = get_config()

print("=" * 80)
print("Stage 2: Plot ASM Results")
print("=" * 80)
print(f"Output directory: {cfg['output_dir']}")
print(f"UNet use_batch_norm: {cfg.get('unet_use_batch_norm', False)}")
print(f"Number of manual cases: {len(CASES)}")
for idx, case in enumerate(CASES, start=1):
    print(
        f"[{idx}] dataset={case['dataset']}, sample_index={case['sample_index']}, "
        f"noise_level={case['noise_level']}, "
        f"use_warm_start={case.get('use_warm_start', cfg.get('use_warm_start', False))}, "
        f"warm_start_method={case.get('warm_start_method', cfg.get('warm_start_method', None))}"
    )
print("=" * 80)

plot_asm_results_cases(project_root=project_root, cfg=cfg, cases=CASES)

print("=" * 80)
print("ASM result plotting complete.")
print("=" * 80)
