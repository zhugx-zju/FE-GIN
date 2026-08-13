from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grf_generalization.config import get_config
from grf_generalization.pipeline.data_generation import generate_generalization_cases


cfg = get_config(PROJECT_ROOT)

print('=' * 80)
print('Generate GRF Generalization Test Cases')
print('=' * 80)
print(f"Correlation lengths (mm): {cfg['correlation_lengths_mm']}")
print(f"Samples per condition: {cfg['sample_count']}")
print(f"Output directory: {cfg['data_dir']}")

manifest_path = generate_generalization_cases(cfg)

print(f'Generated test-only dataset manifest: {manifest_path}')
