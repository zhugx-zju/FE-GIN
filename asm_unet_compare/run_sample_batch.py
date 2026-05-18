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
from pipeline.asm_inversion import run_asm_inversion_batch

cfg = get_config()

print("=" * 80)
print("Stage 1: ASM Inversion Batch")
print("=" * 80)
print(f"Output directory: {cfg['output_dir']}")
print(f"UNet use_batch_norm: {cfg.get('unet_use_batch_norm', False)}")
print(f"Datasets: {cfg['datasets']}")
print(f"Sample index map: {cfg['sample_index']}")
print(f"Noise levels (%): {cfg['noise_levels']}")
print("=" * 80)

run_asm_inversion_batch(project_root=project_root, cfg=cfg)

print("=" * 80)
print("ASM batch inversion stage complete.")
print("=" * 80)
