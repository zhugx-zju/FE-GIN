"""asm_unet_compare-style case figures and manuscript statistics tables."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .common import (
    CONDITION_DISPLAY,
    GRF_CONDITION_ORDER,
    METHOD_COLORS,
    METHOD_MARKERS,
    METHOD_ORDER,
)


# Keep every figure in this evaluation consistent with the manuscript plotting
# utilities in igfe_unet/postprocess/common.py.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.8

RELATIVE_ERROR_COLORBAR_MAX_PCT = 10.0


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['model'], row['condition_id'], float(row['noise_level_percent']))
        grouped[key].append(row)

    summaries = []
    for (model, condition_id, noise_level), selected in grouped.items():
        first = selected[0]
        summary = {
            'model': model,
            'condition_id': condition_id,
            'field_type': first['field_type'],
            'distribution_status': first['distribution_status'],
            'correlation_length_mm': first['correlation_length_mm'],
            'transition_width_10_90_mm': first['transition_width_10_90_mm'],
            'noise_level_percent': noise_level,
            'sample_count': len(selected),
        }
        for metric in ('relative_l1', 'mae', 'rmse'):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        summaries.append(summary)

    id_reference = {
        (row['model'], row['noise_level_percent']): row['relative_l1_mean']
        for row in summaries
        if row['condition_id'] == 'grf_l25'
    }
    for row in summaries:
        reference = id_reference.get((row['model'], row['noise_level_percent']))
        row['relative_l1_ratio_to_l25'] = (
            float(row['relative_l1_mean'] / reference)
            if reference is not None and not np.isclose(reference, 0.0)
            else np.nan
        )
    return sorted(
        summaries,
        key=lambda row: (
            float(row['noise_level_percent']),
            row['condition_id'],
            METHOD_ORDER.index(row['model']) if row['model'] in METHOD_ORDER else 99,
        ),
    )


def build_paper_table(summaries):
    """Create screenshot-style rows: noise/model x GRF condition mean/std."""
    lookup = {
        (row['model'], row['condition_id'], float(row['noise_level_percent'])): row
        for row in summaries
    }
    noise_levels = sorted({float(row['noise_level_percent']) for row in summaries})
    rows = []
    for noise_level in noise_levels:
        for model in METHOD_ORDER:
            row = {
                'noise_level_percent': int(noise_level) if noise_level.is_integer() else noise_level,
                'model': model,
            }
            for condition_id in GRF_CONDITION_ORDER:
                summary = lookup.get((model, condition_id, noise_level))
                prefix = condition_id.replace('grf_', 'grf_')
                row[f'{prefix}_mean'] = (
                    summary['relative_l1_mean'] if summary is not None else np.nan
                )
                row[f'{prefix}_std'] = (
                    summary['relative_l1_std'] if summary is not None else np.nan
                )
            rows.append(row)
    return rows


def save_paper_table_csv(path, summaries):
    rows = build_paper_table(summaries)
    fields = ['noise_level_percent', 'model']
    for condition_id in GRF_CONDITION_ORDER:
        fields.extend([f'{condition_id}_mean', f'{condition_id}_std'])
    return write_csv(path, fields, rows)


def plot_paper_table(path_png, path_pdf, summaries, dpi=600):
    rows = build_paper_table(summaries)
    column_labels = ['Noise level (%)', 'Model']
    for condition_id in GRF_CONDITION_ORDER:
        column_labels.extend(['mean', 'std'])

    cell_text = []
    previous_noise = None
    for row in rows:
        noise_value = row['noise_level_percent']
        display_noise = str(noise_value) if noise_value != previous_noise else ''
        previous_noise = noise_value
        values = [display_noise, row['model']]
        for condition_id in GRF_CONDITION_ORDER:
            values.extend(
                [
                    f"{float(row[f'{condition_id}_mean']):.4f}",
                    f"{float(row[f'{condition_id}_std']):.4f}",
                ]
            )
        cell_text.append(values)

    figure_height = 1.55 + 0.34 * len(rows)
    figure, axis = plt.subplots(figsize=(12.5, figure_height))
    axis.axis('off')
    table = axis.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc='center',
        colLoc='center',
        loc='center',
        colWidths=[0.11, 0.10] + [0.095] * 8,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)

    total_columns = len(column_labels)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor('white')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.0)
        if row_index == 0:
            cell.set_text_props(weight='bold')
            cell.visible_edges = 'BT'
            cell.set_linewidth(1.0)
        else:
            data_index = row_index - 1
            is_group_end = (data_index + 1) % len(METHOD_ORDER) == 0
            cell.visible_edges = 'B' if is_group_end else ''
            if is_group_end:
                cell.set_linewidth(0.7)
    last_row = len(rows)
    for column_index in range(total_columns):
        table[(last_row, column_index)].visible_edges = 'B'
        table[(last_row, column_index)].set_linewidth(1.4)

    # Add the upper grouped headings used by the manuscript table. Matplotlib
    # tables do not support colspan, so place one centered label over each
    # mean/std pair after the table layout has been resolved.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for pair_index, condition_id in enumerate(GRF_CONDITION_ORDER):
        first_column = 2 + 2 * pair_index
        second_column = first_column + 1
        first_box = table[(0, first_column)].get_window_extent(renderer)
        second_box = table[(0, second_column)].get_window_extent(renderer)
        center_display = (
            0.5 * (first_box.x0 + second_box.x1),
            first_box.y1 + 5.0,
        )
        center_axes = axis.transAxes.inverted().transform(center_display)
        ell = condition_id.split('l')[-1]
        axis.text(
            center_axes[0],
            center_axes[1],
            rf'GRF $l={ell}$ mm',
            transform=axis.transAxes,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    axis.set_title(
        r'Statistical results of the relative $L_1$ error for GRF correlation lengths '
        r'across models and noise levels',
        fontsize=12,
        pad=10,
    )
    figure.tight_layout()
    figure.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    figure.savefig(path_pdf, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(figure)


def _draw_field(axis, values, cmap, vmin=None, vmax=None):
    image = axis.imshow(
        values,
        origin='lower',
        extent=[0, 9, 0, 9],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def _row_tag(row_index):
    return f"({chr(ord('a') + row_index)})"


def _add_row_tags(first_axes, x=-0.16, y=1.10, fontsize=16):
    for row_index, axis in sorted(first_axes.items()):
        axis.text(
            x,
            y,
            _row_tag(row_index),
            transform=axis.transAxes,
            fontsize=fontsize,
            fontweight='normal',
            va='top',
            ha='left',
        )


def _add_panel_labels(axes, x=-0.16, y=1.08, fontsize=16):
    for panel_index, axis in enumerate(np.asarray(axes).ravel()):
        axis.text(
            x,
            y,
            _row_tag(panel_index),
            transform=axis.transAxes,
            fontsize=fontsize,
            fontweight='normal',
            va='bottom',
            ha='left',
        )


def _style_field_colorbar(colorbar, label):
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    colorbar.set_label(label, fontsize=15, fontweight='normal')
    colorbar.ax.tick_params(
        direction='in', which='both', labelsize=15, length=4.0, width=0.8
    )
    colorbar.outline.set_linewidth(0.8)


def _save_figure_pair(figure, png_path, dpi):
    png_path = Path(png_path)
    pdf_path = png_path.with_suffix('.pdf')
    for path in (png_path, pdf_path):
        save_dpi = dpi if path.suffix.lower() == '.png' else None
        figure.savefig(path, dpi=save_dpi, bbox_inches='tight', pad_inches=0.03)


def plot_case_comparison(output_dir, case, target, panel_data, noise_levels, dpi=600):
    """Save prediction/error matrices using asm_unet_compare's visual grammar."""
    condition_id = case['condition']
    sample_index = int(case['sample_index'])
    case_dir = Path(output_dir) / 'cases' / condition_id / f'sample_{sample_index}'
    case_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if method in panel_data]

    field_min = float(np.min(target))
    field_max = float(np.max(target))

    figure = plt.figure(figsize=(2.55 * len(methods) + 0.55, 2.60 * len(noise_levels) + 0.75))
    grid = figure.add_gridspec(len(noise_levels), len(methods), wspace=0.045, hspace=0.085)
    first_axes = {}
    image = None
    for row_index, noise_level in enumerate(noise_levels):
        for column_index, method in enumerate(methods):
            axis = figure.add_subplot(grid[row_index, column_index])
            image = _draw_field(
                axis,
                panel_data[method][float(noise_level)]['prediction'],
                'viridis',
                field_min,
                field_max,
            )
            if row_index == 0:
                axis.set_title(method, fontsize=16, fontweight='normal', pad=0)
            if column_index == 0:
                first_axes[row_index] = axis
    figure.subplots_adjust(left=0.070, right=0.895, bottom=0.035, top=0.935, wspace=0.045, hspace=0.085)
    _add_row_tags(first_axes, x=-0.16, y=1.10, fontsize=16)
    color_axis = figure.add_axes([0.915, 0.14, 0.016, 0.74])
    colorbar = figure.colorbar(image, cax=color_axis)
    _style_field_colorbar(colorbar, 'Modulus (MPa)')
    prediction_path = case_dir / f'prediction_{condition_id}_sample_{sample_index}.png'
    _save_figure_pair(figure, prediction_path, dpi)
    plt.close(figure)

    figure = plt.figure(figsize=(2.55 * len(methods) + 0.55, 2.60 * len(noise_levels) + 0.75))
    grid = figure.add_gridspec(len(noise_levels), len(methods), wspace=0.045, hspace=0.085)
    first_axes = {}
    image = None
    for row_index, noise_level in enumerate(noise_levels):
        for column_index, method in enumerate(methods):
            axis = figure.add_subplot(grid[row_index, column_index])
            result = panel_data[method][float(noise_level)]
            image = _draw_field(
                axis,
                result['relative_error_percent'],
                'Blues',
                0.0,
                RELATIVE_ERROR_COLORBAR_MAX_PCT,
            )
            axis.text(
                0.00,
                0.969,
                f"$L_1$={result['relative_l1']:.2e}",
                transform=axis.transAxes,
                ha='left',
                va='top',
                fontsize=11,
                color='white',
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.35,
                    'edgecolor': 'none',
                    'boxstyle': 'square,pad=0.0',
                },
            )
            if row_index == 0:
                axis.set_title(method, fontsize=16, fontweight='normal', pad=2)
            if column_index == 0:
                first_axes[row_index] = axis
    figure.subplots_adjust(left=0.070, right=0.895, bottom=0.035, top=0.935, wspace=0.045, hspace=0.085)
    _add_row_tags(first_axes, x=-0.16, y=1.10, fontsize=16)
    color_axis = figure.add_axes([0.915, 0.14, 0.016, 0.74])
    colorbar = figure.colorbar(image, cax=color_axis)
    _style_field_colorbar(colorbar, 'Relative error (%)')
    error_path = case_dir / f'error_{condition_id}_sample_{sample_index}.png'
    _save_figure_pair(figure, error_path, dpi)
    plt.close(figure)
    return prediction_path, error_path


def plot_correlation_length_curves(output_dir, summaries, dpi=600):
    figure_dir = Path(output_dir) / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    grf_rows = [row for row in summaries if row['condition_id'] in GRF_CONDITION_ORDER]
    noise_levels = sorted({float(row['noise_level_percent']) for row in grf_rows})
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=True)
    for axis, noise_level in zip(axes.ravel(), noise_levels):
        for model in METHOD_ORDER:
            selected = sorted(
                [
                    row for row in grf_rows
                    if row['model'] == model
                    and np.isclose(float(row['noise_level_percent']), noise_level)
                ],
                key=lambda row: float(row['correlation_length_mm']),
            )
            axis.errorbar(
                [row['correlation_length_mm'] for row in selected],
                [100.0 * row['relative_l1_mean'] for row in selected],
                yerr=[100.0 * row['relative_l1_std'] for row in selected],
                label=model,
                color=METHOD_COLORS[model],
                marker=METHOD_MARKERS[model],
                linewidth=1.4,
                capsize=2.5,
            )
        axis.set_title(f'Noise level {noise_level:g}%', fontsize=16, fontweight='normal')
        axis.set_xticks([10, 15, 20, 25])
        axis.tick_params(direction='in', labelsize=10, width=0.8)
    for axis in axes[-1]:
        axis.set_xlabel('GRF correlation length, $l$ (mm)')
    for axis in axes[:, 0]:
        axis.set_ylabel(r'Relative $L_1$ error (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_panel_labels(axes, x=-0.14, y=1.05, fontsize=16)
    figure.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=1.0,
        fancybox=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ('png', 'pdf'):
        figure.savefig(figure_dir / f'grf_correlation_length.{suffix}', dpi=dpi)
    plt.close(figure)
