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
from pipeline.unet_inference import run_unet_inference_cases


# ---------------------------------------------------------------------------
# Manual UNet cases
# Edit this list directly when you want to evaluate one sample at one or
# multiple noise levels, while automatically sweeping all UNet methods listed
# in config.py.
# Each case corresponds to one dataset + one sample + one set of noise levels.
# Optional:
# 1) use `unet_methods` to restrict the methods for one case
# 2) use `exp_paths` to point each method to a specific trained model folder
# 3) use `noise_level` instead of `noise_levels` for a single-noise run
# 4) for LocMixloss/GloMixloss, use:
#    - `mix_gamma`: one gamma for all Mix methods in this case
#    - `mix_gamma_by_method`: per-method gamma override (higher priority)
#    - `strict_mix_gamma`: True means gamma mismatch/missing raises error
# ---------------------------------------------------------------------------
CASES = [
    {
        'dataset': 'grf',
        'sample_index': 900,
        'noise_levels': [0, 2, 4, 6, 8, 10],
        # 'unet_methods': ['MSE', 'GloResloss'],
        'unet_methods': ['MSE', 'LocMixloss', 'GloMixloss'],
        # 'mix_gamma': 300000,
        'mix_gamma_by_method': {
            'LocMixloss': 100000,
            'GloMixloss': 10000,
        },
        # 'strict_mix_gamma': True,
        # 'exp_paths': {
        #     'MSE': r'trained_models/your_mse_experiment',
        #     'GloResloss': r'trained_models/your_glores_experiment',
        # },
    },
]


cfg = get_config()

print("=" * 80)
print("Stage 1: UNet Inference (Manual Case Mode)")
print("=" * 80)
print(f"Output directory: {cfg['output_dir']}")
print(f"UNet use_batch_norm: {cfg.get('unet_use_batch_norm', False)}")
print(f"Number of manual cases: {len(CASES)}")
print(f"Default UNet methods: {cfg['unet_methods']}")
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
        f"mix_gamma={gamma_desc}, mix_gamma_by_method={gamma_map_desc}"
    )
print("=" * 80)

run_unet_inference_cases(project_root=project_root, cfg=cfg, cases=CASES)

print("=" * 80)
print("UNet manual-case inference and plotting stage complete.")
print("=" * 80)
