from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grf_generalization.config import get_config
from grf_generalization.pipeline.sample_preview import generate_sample_previews


cfg = get_config(PROJECT_ROOT)

print('=' * 80)
print('Preview Paired GRF Ground-Truth Samples')
print('=' * 80)
print(f"Sample indices: {cfg['sample_preview_indices']}")
print(f"Catalog condition: {cfg['sample_catalog_condition']}")
print(f"Output directory: {cfg['output_dir'] / 'sample_previews'}")

outputs = generate_sample_previews(cfg)

print(f"Saved sample catalog: {outputs['catalog_png']}")
print(f"Saved paired sample figures: {len(outputs['paired_samples'])}")
