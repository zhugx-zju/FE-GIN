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
from pipeline.asm_unet_compare import compare_asm_unet_cases


# ---------------------------------------------------------------------------
# Manual ASM-vs-UNet comparison cases
# Keep each case aligned with the saved outputs produced by:
# 1) run_sample_unet.py
# 2) run_sample_asm.py or run_sample_warm_start.py
#
# Each case corresponds to one dataset + one sample + one noise set.
# Optional:
# 1) use `unet_methods` to restrict which saved UNet results are plotted
# 2) use `mix_gamma` / `mix_gamma_by_method` for Mix-model file selection
# 3) set `use_warm_start=True` to compare against saved warm-start ASM results
# ---------------------------------------------------------------------------
CASES = [
    {
        'dataset': 'exp',
        'sample_index': 200,
        'noise_levels': [0, 2, 4, 6, 8, 10],
        # 'unet_methods': ['MSE', 'GloResloss'],
        'unet_methods': ['MSE', 'LocMixloss', 'GloMixloss'],
        # 'mix_gamma': 300000,
        'mix_gamma_by_method': {
            'LocMixloss': 100000,
            'GloMixloss': 10000,
        },
        'use_warm_start': False,
        'warm_start_method': 'GloResloss',
    },
]


cfg = get_config()

# Optional: choose which saved Mix-model gamma outputs to load in plots.
# If None, plotting uses defaults from config.py or generic method files.
PLOT_MIX_GAMMA = None
PLOT_MIX_GAMMA_BY_METHOD = None
PLOT_STRICT_MIX_GAMMA = None

if PLOT_MIX_GAMMA is not None:
    cfg['mix_gamma'] = PLOT_MIX_GAMMA
if PLOT_MIX_GAMMA_BY_METHOD is not None:
    cfg['mix_gamma_by_method'] = PLOT_MIX_GAMMA_BY_METHOD
if PLOT_STRICT_MIX_GAMMA is not None:
    cfg['strict_mix_gamma'] = bool(PLOT_STRICT_MIX_GAMMA)

print("=" * 80)
print("Stage 3: Plot Saved ASM vs UNet Results (Manual Case Mode)")
print("=" * 80)
print(f"Output directory: {cfg['output_dir']}")
print(f"UNet use_batch_norm: {cfg.get('unet_use_batch_norm', False)}")
print(f"Number of manual cases: {len(CASES)}")
print(f"Default mix gamma: {cfg.get('mix_gamma', None)}")
print(f"Default mix gamma by method: {cfg.get('mix_gamma_by_method', None)}")
print(f"Strict mix gamma match: {cfg.get('strict_mix_gamma', True)}")
for idx, case in enumerate(CASES, start=1):
    noise_desc = case.get('noise_levels', case.get('noise_level'))
    methods_desc = case.get('unet_methods', cfg['unet_methods'])
    gamma_desc = case.get('mix_gamma', cfg.get('mix_gamma', None))
    gamma_map_desc = case.get('mix_gamma_by_method', cfg.get('mix_gamma_by_method', None))
    print(
        f"[{idx}] dataset={case['dataset']}, sample_index={case['sample_index']}, "
        f"noise_levels={noise_desc}, methods={methods_desc}, "
        f"use_warm_start={case.get('use_warm_start', cfg.get('use_warm_start', False))}, "
        f"warm_start_method={case.get('warm_start_method', cfg.get('warm_start_method', None))}, "
        f"mix_gamma={gamma_desc}, mix_gamma_by_method={gamma_map_desc}"
    )
print("=" * 80)

compare_asm_unet_cases(project_root=project_root, cfg=cfg, cases=CASES)

print("=" * 80)
print("Saved ASM vs UNet plotting complete.")
print("=" * 80)
