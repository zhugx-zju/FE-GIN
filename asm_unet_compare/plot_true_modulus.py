import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
unet_root = os.path.join(project_root, 'igfe_unet')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if unet_root not in sys.path:
    sys.path.insert(0, unet_root)

from config import get_config
from pipeline.common import load_asm_modules, load_sample, resolve_output_root, resolve_variant_output_dir
from igfe_unet.postprocess.common import _add_panel_labels


# ---------------------------------------------------------------------------
# Manual true-modulus plotting cases
# Edit this list directly when you want to export the ground-truth modulus
# field for one or more fixed-test samples.
# Each case corresponds to one dataset + one sample.
# Optional:
# 1) set `nodesx` / `nodesy` if you want a different plotting mesh
# 2) set `unet_config_type` / `unet_load_type` to read another fixed test set
# 3) set `title` to override each subplot title
# ---------------------------------------------------------------------------
CASES = [
    {
        'dataset': 'bil',
        'sample_index': 600,
    },
    {
        'dataset': 'exp',
        'sample_index': 200,
    },
    {
        'dataset': 'grf',
        'sample_index': 410,
    },
]


def _build_plot_mesh_context(asm_mods, nodesx, nodesy):
    asm_cfg = asm_mods['cfg_module']
    forward_cfg = asm_cfg.get_forward_config()

    nel_x = int(nodesx) - 1
    nel_y = int(nodesy) - 1
    if nel_x != int(forward_cfg.nel_x) or nel_y != int(forward_cfg.nel_y):
        print(
            "Warning: UNet mesh and asm_log config mesh mismatch. "
            f"Using UNet mesh {nel_x}x{nel_y} for plotting."
        )

    mesh = asm_mods['MeshInfo'](
        float(forward_cfg.geo_l),
        float(forward_cfg.geo_h),
        nel_x,
        nel_y,
    )
    return {
        'mesh': mesh,
        'create_smooth_contour': asm_mods['create_smooth_contour'],
    }


def plot_true_modulus_cases(project_root, cfg, cases):
    output_dir = resolve_variant_output_dir(
        cfg['output_dir'],
        use_batch_norm=cfg.get('unet_use_batch_norm', False),
    )
    output_root = resolve_output_root(project_root, output_dir)
    os.makedirs(output_root, exist_ok=True)

    asm_mods = load_asm_modules(project_root)
    mesh_ctx_cache = {}
    filename_suffix = '_GN' if bool(cfg.get('unet_use_batch_norm', False)) else ''
    panels = []

    for idx, case in enumerate(cases, start=1):
        data_type = case.get('dataset', case.get('data_type'))
        if data_type is None:
            raise KeyError(f"Case {idx} must provide 'dataset'.")

        sample_index = int(case.get('sample_index', cfg.get('sample_index', {}).get(data_type, 0)))
        config_type = case.get('unet_config_type', cfg['unet_config_type'])
        load_type = case.get('unet_load_type', cfg['unet_load_type'])
        nodesx = int(case.get('nodesx', cfg['nodesx']))
        nodesy = int(case.get('nodesy', cfg['nodesy']))

        _, target_sample, resolved_idx = load_sample(
            config_type=config_type,
            load_type=load_type,
            data_type=data_type,
            sample_index=sample_index,
        )
        target = target_sample.squeeze().detach().cpu().numpy()

        ctx_key = (nodesx, nodesy)
        if ctx_key not in mesh_ctx_cache:
            mesh_ctx_cache[ctx_key] = _build_plot_mesh_context(asm_mods, nodesx, nodesy)
        mesh_ctx = mesh_ctx_cache[ctx_key]

        panels.append({
            'case_idx': idx,
            'data_type': data_type,
            'sample_index': sample_index,
            'resolved_idx': resolved_idx,
            'target': target,
            'mesh_ctx': mesh_ctx,
            'title': case.get('title', f'{data_type.upper()} Sample'),
        })

    if not panels:
        print("No true-modulus cases to plot.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.4))
    axes = np.atleast_1d(axes)

    for panel_idx, (panel, ax) in enumerate(zip(panels, axes)):
        target = panel['target']
        mesh_ctx = panel['mesh_ctx']
        vmin = float(np.min(target))
        vmax = float(np.max(target))
        im = mesh_ctx['create_smooth_contour'](
            mesh_ctx['mesh'].plot_x,
            mesh_ctx['mesh'].plot_y,
            target,
            ax,
            levels=128,
            cmap='viridis',
        )
        im.set_clim(vmin, vmax)
        ax.axis('equal')
        ax.axis('off')

        ax.set_title(panel['title'], fontsize=20, fontweight='normal', pad=6)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        colorbar.set_label('Modulus (MPa)', fontsize=15, fontweight='normal')
        colorbar.ax.tick_params(
            direction='in', which='both', labelsize=15, length=4.0, width=0.8
        )
        colorbar.outline.set_linewidth(0.8)

    _add_panel_labels(list(axes), x=-0.10, y=1.06, fontsize=18)
    # fig.suptitle('True Modulus Fields', fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.06, top=0.87, wspace=0.30)

    case_tag = '_'.join(f"{panel['data_type']}{panel['resolved_idx']}" for panel in panels)
    save_path = os.path.join(output_root, f'true_modulus_{case_tag}{filename_suffix}.png')
    fig.savefig(save_path, dpi=cfg['dpi'], bbox_inches='tight', pad_inches=0.03)
    pdf_path = os.path.splitext(save_path)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

    print(f"Saved combined true modulus plot: {save_path}")
    print(f"Saved combined true modulus PDF: {pdf_path}")
    for panel in panels:
        print(
            f"[{panel['case_idx']}] dataset={panel['data_type']}, "
            f"requested_sample_index={panel['sample_index']}, "
            f"resolved_sample_index={panel['resolved_idx']}"
        )


cfg = get_config()
output_dir = resolve_variant_output_dir(
    cfg['output_dir'],
    use_batch_norm=cfg.get('unet_use_batch_norm', False),
)

print("=" * 80)
print("Stage X: Plot True Modulus (Manual Case Mode)")
print("=" * 80)
print(f"Output directory: {output_dir}")
print(f"UNet use_batch_norm: {cfg.get('unet_use_batch_norm', False)}")
print(f"Number of manual cases: {len(CASES)}")
for idx, case in enumerate(CASES, start=1):
    print(
        f"[{idx}] dataset={case['dataset']}, sample_index={case['sample_index']}, "
        f"nodes=({case.get('nodesx', cfg['nodesx'])}, {case.get('nodesy', cfg['nodesy'])})"
    )
print("=" * 80)

plot_true_modulus_cases(project_root=project_root, cfg=cfg, cases=CASES)

print("=" * 80)
print("True modulus plotting complete.")
print("=" * 80)
