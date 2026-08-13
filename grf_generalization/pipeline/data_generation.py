"""Project-style orchestration for independent generalization test data."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNET_ROOT = PROJECT_ROOT / 'igfe_unet'
for import_root in (PROJECT_ROOT, UNET_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from .generators import build_generalization_dataset


def generate_generalization_cases(cfg):
    """Generate all conditions described by a generalization config."""
    correlation_lengths = tuple(float(value) for value in cfg['correlation_lengths_mm'])
    steep_count = cfg.get('steep_sample_count', cfg['sample_count'])
    if not cfg.get('include_steep_gradient', True):
        steep_count = 0
    return build_generalization_dataset(
        output_root=cfg['data_dir'],
        sample_count=int(cfg['sample_count']),
        steep_count=int(steep_count),
        correlation_lengths=correlation_lengths,
        seed=int(cfg['seed']),
        transition_width_mm=float(cfg['steep_transition_width_mm']),
        nodes_x=int(cfg['nodes_x']),
        nodes_y=int(cfg['nodes_y']),
        include_steep_gradient=bool(cfg.get('include_steep_gradient', True)),
    )
