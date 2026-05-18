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
from pipeline.asm_inversion import run_asm_inversion_cases


# ---------------------------------------------------------------------------
# Manual ASM cases
# Edit this list directly when you want to tune one case at a time.
# Each case corresponds to one dataset + one sample + one set of noise levels.
# That way every requested noise level is run, saved, and analyzed independently.
# Each case can have its own mesh info and ASM / L-curve parameters.
# This script is intended for cold-start ASM only.
# Use run_sample_warm_start.py for the UNet-initialized warm-start workflow.
# ---------------------------------------------------------------------------
CASES = [
    {
        'dataset': 'exp',
        'sample_index': 200,
        'noise_level': 6.0,
        # 'noise_levels': [0, 2, 4, 6, 8, 10], 
        'nodesx': 40,
        'nodesy': 40,
        'asm_dof_order': 'C',
        'asm_gamma': None,
        'asm_max_iter': 1000,
        'asm_ftol': 1e-20,
        'asm_gtol': 1e-14,
        'enable_lcurve': True,
        'lcurve_points': 50,
        'lcurve_gamma_min': 5e-7,
        'lcurve_gamma_max': 1e-7,
    },
]


cfg = get_config()

print("=" * 80)
print("Stage 1: ASM Inversion (Manual Case Mode)")
print("=" * 80)
print(f"Output directory: {cfg['output_dir']}")
print(f"UNet use_batch_norm: {cfg.get('unet_use_batch_norm', False)}")
print(f"Number of manual cases: {len(CASES)}")
for idx, case in enumerate(CASES, start=1):
    noise_desc = case.get('noise_levels', case.get('noise_level'))
    print(
        f"[{idx}] dataset={case['dataset']}, sample_index={case['sample_index']}, "
        f"noise_levels={noise_desc}, nodes=({case['nodesx']}, {case['nodesy']}), "
        "mode=cold_start"
    )
print("=" * 80)

run_asm_inversion_cases(project_root=project_root, cfg=cfg, cases=CASES)

print("=" * 80)
print("ASM manual-case inversion stage complete.")
print("=" * 80)
