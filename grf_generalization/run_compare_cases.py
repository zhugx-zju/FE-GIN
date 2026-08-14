from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grf_generalization.config import get_config
from grf_generalization.pipeline.evaluation import run_generalization_comparison


cfg = get_config(PROJECT_ROOT)

print('=' * 80)
print('Compare Final Models on GRF Generalization Cases')
print('=' * 80)
print(f"Cases: {[case['condition'] for case in cfg['cases']]}")
print(
    'Supplementary stress cases: '
    f"{[case['condition'] for case in cfg['supplementary_cases']]}"
)
print(f"Noise levels (%): {cfg['noise_levels']}")
print(f"Models: {list(cfg['models'].values())}")
print(f"Output directory: {cfg['output_dir']}")

outputs = run_generalization_comparison(cfg)

print(f"Saved per-sample metrics: {outputs['per_sample_csv']}")
print(f"Saved supplementary stress summary: {outputs['stress_summary_csv']}")
print(f"Saved paper-format table: {outputs['paper_table_csv']}")
print(f"Saved rendered table: {outputs['paper_table_png']}")
