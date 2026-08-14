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
print('Preview GRF Ground-Truth Samples for Manual Case Selection')
print('=' * 80)
print(f"Sample indices: {cfg['sample_preview_indices']}")
print(f"Catalog conditions: {cfg['sample_catalog_conditions']}")
print(f"Output directory: {cfg['output_dir'] / 'sample_previews'}")

outputs = generate_sample_previews(cfg)

for condition_id, paths in outputs['catalogs'].items():
    print(f'Saved {condition_id} sample catalog: {paths[0]}')
print(f"Saved selected-scale sample figures: {len(outputs['scale_samples'])}")
