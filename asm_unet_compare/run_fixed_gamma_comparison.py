"""Run a small fixed-gamma ASM versus saved U-Net comparison sweep."""

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
from pipeline.asm_unet_compare import compare_asm_unet_cases


DATASET = 'grf'
SAMPLE_INDEX = 900
NOISE_LEVELS = [6.0]
FIXED_GAMMAS = [1e-7, 5e-7, 1e-6]
UNET_OUTPUT_DIR = 'results/force_load/asm_unet_comparison'


def _gamma_tag(gamma):
    return f'{gamma:.0e}'.replace('+', '')


for gamma in FIXED_GAMMAS:
    output_dir = f'results/force_load/asm_unet_comparison/fixed_gamma_{_gamma_tag(gamma)}'
    cfg = get_config()
    cfg.update(
        {
            'output_dir': output_dir,
            'asm_output_dir': output_dir,
            'displacement_source_dir': UNET_OUTPUT_DIR,
            'unet_output_dir': UNET_OUTPUT_DIR,
            'asm_gamma': gamma,
            'enable_lcurve': False,
            'asm_max_iter': 600,
        }
    )
    case = {
        'dataset': DATASET,
        'sample_index': SAMPLE_INDEX,
        'noise_levels': NOISE_LEVELS,
        'asm_gamma': gamma,
        'enable_lcurve': False,
        'use_warm_start': False,
    }
    print('=' * 80)
    print(f'Fixed gamma test: gamma={gamma:.6e}')
    print(f'ASM output: {output_dir}')
    print(f'UNet source: {UNET_OUTPUT_DIR}')
    run_asm_inversion_cases(project_root=project_root, cfg=cfg, cases=[case])
    compare_asm_unet_cases(project_root=project_root, cfg=cfg, cases=[
        {
            'dataset': DATASET,
            'sample_index': SAMPLE_INDEX,
            'noise_levels': NOISE_LEVELS,
            'unet_methods': ['MSE', 'LocMixloss', 'GloMixloss'],
            'mix_gamma_by_method': {
                'LocMixloss': 100000,
                'GloMixloss': 10000,
            },
            'unet_use_batch_norm': True,
            'asm_output_dir': output_dir,
            'unet_output_dir': UNET_OUTPUT_DIR,
            'use_warm_start': False,
        }
    ])

print('=' * 80)
print('Fixed-gamma ASM versus UNet comparison sweep complete.')
