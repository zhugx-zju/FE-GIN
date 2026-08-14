"""Preview paired GRF ground-truth fields before selecting comparison cases."""

import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .visualization import _add_panel_labels, _style_field_colorbar


DEFAULT_PREVIEW_CONDITIONS = ('grf_l20', 'grf_l15', 'grf_l10', 'grf_l5')


def resolve_preview_indices(requested_indices, sample_count):
    """Return unique indices in user order and reject accidental clamping."""
    resolved = []
    seen = set()
    for raw_index in requested_indices:
        sample_index = int(raw_index)
        if sample_index < 0 or sample_index >= int(sample_count):
            raise IndexError(
                f'Sample index {sample_index} is outside the available range '
                f'0..{int(sample_count) - 1}.'
            )
        if sample_index not in seen:
            seen.add(sample_index)
            resolved.append(sample_index)
    if not resolved:
        raise ValueError('sample_preview_indices must contain at least one index.')
    return resolved


def _load_grf_targets(data_root, condition_ids):
    data_root = Path(data_root).resolve()
    manifest_path = data_root / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'Generalization dataset manifest not found: {manifest_path}\n'
            'Run grf_generalization/run_generate_cases.py first.'
        )
    with manifest_path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('training_or_validation_use_permitted') is not False:
        raise ValueError('The generalization dataset is not explicitly marked test-only.')

    condition_lookup = {
        condition['condition_id']: condition for condition in manifest['conditions']
    }
    targets = {}
    for condition_id in condition_ids:
        if condition_id not in condition_lookup:
            raise KeyError(f'Missing GRF preview condition: {condition_id}')
        condition = condition_lookup[condition_id]
        output_path = data_root / condition['directory'] / 'output.mat'
        targets[condition_id] = np.asarray(loadmat(output_path)['E'], dtype=float)

    sample_counts = {values.shape[0] for values in targets.values()}
    if len(sample_counts) != 1:
        raise ValueError(f'Paired GRF conditions have unequal sample counts: {sample_counts}')
    return targets


def _condition_title(condition_id):
    length = condition_id.split('l')[-1]
    return rf'GRF $l={length}$ mm'


def _draw_true_field(axis, values, vmin, vmax):
    y = np.linspace(0.0, 9.0, values.shape[0])
    x = np.linspace(0.0, 9.0, values.shape[1])
    if np.isclose(vmin, vmax):
        vmax = vmin + np.finfo(float).eps
    levels = np.linspace(vmin, vmax, 129)
    image = axis.contourf(x, y, values, levels=levels, cmap='viridis')
    axis.set_aspect('equal')
    axis.axis('off')
    return image


def _save_figure_pair(figure, png_path, dpi):
    png_path = Path(png_path)
    pdf_path = png_path.with_suffix('.pdf')
    figure.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    figure.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)
    return png_path, pdf_path


def _plot_scale_sample(output_dir, targets, condition_ids, sample_index, dpi):
    scale_dir = Path(output_dir) / 'sample_previews' / 'scale_fields'
    scale_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        len(condition_ids),
        figsize=(3.8 * len(condition_ids), 3.4),
    )
    axes = np.atleast_1d(axes)
    for condition_id, axis in zip(condition_ids, axes):
        field = targets[condition_id][sample_index]
        image = _draw_true_field(axis, field, float(np.min(field)), float(np.max(field)))
        axis.set_title(
            _condition_title(condition_id),
            fontsize=20,
            fontweight='normal',
            pad=6,
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        _style_field_colorbar(colorbar, 'Modulus (MPa)')

    _add_panel_labels(axes, x=-0.10, y=1.06, fontsize=18)
    figure.subplots_adjust(
        left=0.055,
        right=0.975,
        bottom=0.06,
        top=0.87,
        wspace=0.30,
    )
    output_path = scale_dir / f'true_modulus_grf_sample_{sample_index:02d}.png'
    paths = _save_figure_pair(figure, output_path, dpi)
    plt.close(figure)
    return paths


def _plot_sample_catalog(output_dir, fields, condition_id, sample_indices, dpi):
    preview_dir = Path(output_dir) / 'sample_previews'
    preview_dir.mkdir(parents=True, exist_ok=True)
    column_count = min(5, len(sample_indices))
    row_count = int(ceil(len(sample_indices) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(3.8 * column_count, 3.4 * row_count),
        squeeze=False,
    )
    used_axes = []
    for panel_index, axis in enumerate(axes.ravel()):
        if panel_index >= len(sample_indices):
            axis.axis('off')
            continue
        sample_index = sample_indices[panel_index]
        field = fields[sample_index]
        image = _draw_true_field(
            axis,
            field,
            float(np.min(field)),
            float(np.max(field)),
        )
        axis.set_title(
            f'Sample {sample_index}',
            fontsize=14,
            fontweight='normal',
            pad=4,
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        _style_field_colorbar(colorbar, 'Modulus (MPa)')
        used_axes.append(axis)

    _add_panel_labels(used_axes, x=-0.10, y=1.06, fontsize=13)
    figure.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.045,
        top=0.935,
        wspace=0.42,
        hspace=0.28,
    )
    output_path = preview_dir / f'true_modulus_sample_catalog_{condition_id}.png'
    paths = _save_figure_pair(figure, output_path, dpi)
    plt.close(figure)
    return paths


def generate_sample_previews(cfg):
    """Write catalogs and selected-scale figures for manual sample selection."""
    condition_ids = tuple(
        str(value)
        for value in cfg.get('sample_catalog_conditions', DEFAULT_PREVIEW_CONDITIONS)
    )
    if not condition_ids:
        raise ValueError('sample_catalog_conditions must not be empty.')
    targets = _load_grf_targets(cfg['data_dir'], condition_ids)
    sample_count = next(iter(targets.values())).shape[0]
    sample_indices = resolve_preview_indices(
        cfg.get('sample_preview_indices', range(sample_count)),
        sample_count,
    )
    catalog_paths = {
        condition_id: _plot_sample_catalog(
            cfg['output_dir'],
            targets[condition_id],
            condition_id,
            sample_indices,
            int(cfg.get('dpi', 600)),
        )
        for condition_id in condition_ids
    }
    scale_paths = {
        sample_index: _plot_scale_sample(
            cfg['output_dir'],
            targets,
            condition_ids,
            sample_index,
            int(cfg.get('dpi', 600)),
        )
        for sample_index in sample_indices
    }
    return {
        'catalogs': catalog_paths,
        'scale_samples': scale_paths,
    }
