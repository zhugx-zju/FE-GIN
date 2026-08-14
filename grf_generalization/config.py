from pathlib import Path
import os


def get_config(project_root=None, asset_root=None):
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    asset_root = Path(
        asset_root or os.environ.get('FE_GIN_ASSET_ROOT', project_root)
    ).resolve()
    sample_count = 20
    return {
        # Test-only generated data and analysis outputs.
        'data_dir': project_root / 'data' / 'generalization_test_sets' / 'force_load',
        'output_dir': project_root / 'results' / 'grf_ood',
        # Set asset_root separately when code runs from a Git worktree whose
        # ignored data/model assets live in the primary checkout.
        'asset_root': asset_root,
        'model_root': asset_root / 'trained_models_mix' / 'force_load' / 'final_model',
        # Match the selected U-Net input/output grid: 40 x 40 nodes.
        'nodes_x': 40,
        'nodes_y': 40,
        'sample_count': sample_count,
        'seed': 8606,
        'correlation_lengths_mm': [25.0, 20.0, 15.0, 10.0],
        'include_steep_gradient': True,
        'steep_sample_count': 20,
        'steep_transition_width_mm': 0.75,
        # Same noise protocol and levels as the manuscript robustness table.
        'noise_levels': [0, 2, 4, 6, 8, 10],
        'batch_size': 16,
        'device': 'cpu',
        'dpi': 600,
        # Preview all paired GRF samples before choosing a representative case.
        'sample_preview_indices': list(range(sample_count)),
        'sample_catalog_condition': 'grf_l10',
        # Selected final checkpoints. Keys are folder names; values are labels.
        'models': {
            'MSE_UNet_GN_arch_32-64-128': 'MSE-M',
            'LocMix_UNet_GN_arch_32-64-128_gamma_100000': 'LM-M',
            'GloMix_UNet_GN_arch_32-64-128_gamma_10000': 'GM-M',
        },
        # Representative cases follow the asm_unet_compare manual-case pattern.
        # l=20/15/10 mm are the three additional reviewer-response cases.
        'cases': [
            {'condition': 'grf_l25', 'sample_index': 0},
            {'condition': 'grf_l20', 'sample_index': 0},
            {'condition': 'grf_l15', 'sample_index': 0},
            {'condition': 'grf_l10', 'sample_index': 0},
        ],
    }
