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
# Manual ASM warm-start cases
# Edit this list directly when you want to run one warm-start case at a time.
#
# Workflow:
# 1) run_sample_unet.py
#    Save the UNet predictions for the same dataset / sample / noise level.
# 2) run_sample_asm.py
#    Save the corresponding cold-start ASM result so gamma_opt is available.
# 3) run this script
#    Use one saved UNet prediction as E_init, and automatically reuse the
#    corresponding cold-start gamma_opt. No manual gamma setting is needed.
#    The ASM warm-start outputs are saved with a method-specific suffix, so
#    different UNet initializers will not overwrite each other.
#
# This script is intentionally manual:
# - one dataset
# - one sample
# - one noise level
# - one chosen warm_start_method
# - optional mix gamma selector when warm_start_method is LocMixloss/GloMixloss
# ---------------------------------------------------------------------------
CASES = [
    {
        'dataset': 'exp',
        'sample_index': 1000,
        'noise_level': 0.0,
        'nodesx': 40,
        'nodesy': 40,
        'asm_dof_order': 'C',
        'asm_max_iter': 600,
        'asm_ftol': 1e-12,
        'asm_gtol': 1e-8,
        'enable_lcurve': False,
        'use_warm_start': True,
        'warm_start_method': 'GloResloss',
        # For Mix warm-start methods, optionally pin gamma:
        # 'mix_gamma': 300000,
        # 'mix_gamma_by_method': {'GloMixloss': 100000, 'LocMixloss': 300000},
        # 'strict_mix_gamma': True,
        'warm_start_output_dir': 'results/force_load/asm_unet_comparison',
        'use_cold_start_gamma': True,
        'cold_start_gamma_source_dir': 'results/force_load/asm_unet_comparison',
    },
]


cfg = get_config()
# Warm-start keeps its historical GloResloss initializer from the standard
# experiment group; final-sample U-Net inference uses the dedicated group.
cfg['unet_experiment_group'] = 'std'

print("=" * 80)
print("Stage 1: ASM Warm-Start Inversion (Manual Case Mode)")
print("=" * 80)
print(f"Output directory: {cfg['output_dir']}")
print(f"UNet use_batch_norm: {cfg.get('unet_use_batch_norm', False)}")
print(f"Number of manual cases: {len(CASES)}")
print(f"Default mix gamma: {cfg.get('mix_gamma', None)}")
print(f"Default mix gamma by method: {cfg.get('mix_gamma_by_method', None)}")
print(f"Strict mix gamma match: {cfg.get('strict_mix_gamma', True)}")
for idx, case in enumerate(CASES, start=1):
    noise_desc = case.get('noise_levels', case.get('noise_level'))
    print(
        f"[{idx}] dataset={case['dataset']}, sample_index={case['sample_index']}, "
        f"noise_levels={noise_desc}, nodes=({case['nodesx']}, {case['nodesy']}), "
        f"warm_start_method={case['warm_start_method']}, "
        f"reuse_cold_start_gamma={case.get('use_cold_start_gamma', True)}, "
        f"mix_gamma={case.get('mix_gamma', cfg.get('mix_gamma', None))}, "
        f"mix_gamma_by_method={case.get('mix_gamma_by_method', cfg.get('mix_gamma_by_method', None))}"
    )
print("=" * 80)

run_asm_inversion_cases(project_root=project_root, cfg=cfg, cases=CASES)

print("=" * 80)
print("ASM manual warm-start stage complete.")
print("=" * 80)
