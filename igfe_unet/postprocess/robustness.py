import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .common import (
    _resolve_output_dir,
    _method_label,
    _parse_test_stem,
    _apply_axis_style,
    _sorted_labels,
    _method_color,
    _legend_with_frame,
    _dataset_title,
    _add_panel_labels,
    _draw_grid_figure,
    _top_shared_legend_kwargs,
    extract_mix_ratio_info,
    mse_ratio_label,
)

X_AXIS_MAX_PERCENT = 10.0

_LEGEND_RATIO_COLUMNS = (
    ('0.1:0.1:0.8', '0.1:0.2:0.7', '0.1:0.3:0.6'),
    ('0.33:0.33:0.34', '0.1:0.4:0.5', '0.1:0.5:0.4'),
    ('0.1:0.6:0.3', '0.1:0.7:0.2', '0.1:0.8:0.1'),
)
_CURVE_COLORS = (
    '#0072B2', '#E69F00', '#009E73',
    '#D55E00', '#CC79A7', '#56B4E9',
    '#7F3C8D', '#11A579', '#B79F00',
)
_LINESTYLE_FAMILY = ('-', '--', '-.')
_MARKER_FAMILY = ('o', 's', '^')
_TARGET_MARKERS_PER_CURVE = 8
_CURVE_LINEWIDTH = 1.2
_CURVE_ALPHA = 0.95
_METHOD_ECDF_STYLES = {
    'MSE-M': {
        'color': '#e41a1c',
        'linestyle': '-',
        'marker': 'o',
    },
    'LE-M': {
        'color': '#377eb8',
        'linestyle': '--',
        'marker': 's',
    },
    'GE-M': {
        'color': '#4daf4a',
        'linestyle': '-.',
        'marker': '^',
    },
}


def _save_figure_pair(fig, fig_file, dpi):
    """Save raster and vector versions using the same figure state."""
    fig.savefig(fig_file, dpi=dpi, bbox_inches='tight')
    pdf_file = Path(fig_file).with_suffix('.pdf')
    fig.savefig(pdf_file, bbox_inches='tight')
    print(f"Saved PDF: {pdf_file}")


def _legend_ratio_order():
    return [ratio for col in _LEGEND_RATIO_COLUMNS for ratio in col]


_TARGET_RATIO_ORDER = tuple(_legend_ratio_order())
_TABLE_RATIO_ORDER = tuple(_TARGET_RATIO_ORDER)


def _ratio_display_from_values(bil_ratio, exp_ratio, grf_ratio):
    return f'{float(bil_ratio):.3g}:{float(exp_ratio):.3g}:{float(grf_ratio):.3g}'


def _ratio_text_from_label(label):
    text = str(label).strip()
    if 'B:E:G=' in text:
        return text.split('B:E:G=')[-1].rstrip(')').strip()
    if text in _TARGET_RATIO_ORDER:
        return text
    return None


def _legend_label(raw_label):
    ratio_text = _ratio_text_from_label(raw_label)
    return ratio_text if ratio_text is not None else str(raw_label)


def _ordered_labels(labels):
    labels = list(labels)
    ratio_to_label = {}
    for label in labels:
        ratio_text = _ratio_text_from_label(label)
        if ratio_text is not None and ratio_text not in ratio_to_label:
            ratio_to_label[ratio_text] = label

    ordered = []
    used = set()
    for ratio_text in _TARGET_RATIO_ORDER:
        label = ratio_to_label.get(ratio_text)
        if label is not None and label not in used:
            ordered.append(label)
            used.add(label)

    for label in _sorted_labels(labels):
        if label not in used:
            ordered.append(label)
            used.add(label)

    return ordered


def _build_label_styles(data):
    labels = _ordered_labels({
        label
        for noise_data in data.values()
        for eval_data in noise_data.values()
        for label in eval_data.keys()
    })

    label_styles = {}
    for idx, label in enumerate(labels):
        style_idx = idx % len(_CURVE_COLORS)
        group_idx = style_idx // 3
        row_idx = style_idx % 3
        label_styles[label] = {
            'color': _CURVE_COLORS[style_idx],
            'linestyle': _LINESTYLE_FAMILY[row_idx],
            'marker': _MARKER_FAMILY[group_idx],
            'linewidth': _CURVE_LINEWIDTH,
            'alpha': _CURVE_ALPHA,
            'zorder': 3,
        }
    return label_styles


def _shared_legend_kwargs(fontsize=11):
    return {
        'loc': 'upper left',
        'bbox_to_anchor': (0.04, 0.955, 0.94, 0.001),
        'mode': 'expand',
        'ncol': len(_TARGET_RATIO_ORDER),
        'fontsize': fontsize,
        'labelspacing': 0.45,
        'columnspacing': 0.8,
        'handlelength': 2.2,
        'handletextpad': 0.5,
        'markerscale': 1.05,
        'borderaxespad': 0.2,
        'frameon': False,
        'facecolor': 'white',
        'edgecolor': 'black',
        'framealpha': 1.0,
        'fancybox': False,
        'layout_top': 0.945,
    }


def _sparse_markevery(point_count, target_points=_TARGET_MARKERS_PER_CURVE):
    if point_count <= 1:
        return [0]
    count = max(2, min(int(target_points), int(point_count)))
    return np.unique(np.linspace(0, point_count - 1, count, dtype=int)).tolist()


def _label_for_results(results, exp_id, exp_path, label_mode):
    config = results.get('config') or {}
    if label_mode == 'ratio':
        if config.get('method') != 'MSE':
            return None
        return _legend_label(mse_ratio_label(results, exp_id=exp_id, exp_path=exp_path))
    method = config.get('method', 'Unknown')
    gamma = config.get('gamma', None)
    return _method_label(method, gamma)


def _label_color(label, label_mode, label_styles=None):
    if label_mode == 'ratio' and label_styles is not None:
        return label_styles[label]['color']
    return _method_color(label)


def _collect_histogram_data(experiments_data, label_mode='method'):
    data = {}
    for (config_type, load_type, exp_id, exp_path), results in experiments_data.items():
        if config_type != 'mix':
            continue
        label = _label_for_results(results, exp_id, exp_path, label_mode)
        if label is None:
            continue
        search_paths = [Path(exp_path) / 'all_samples_test', Path(exp_path) / 'all_samples']
        seen = set()
        for all_samples_dir in search_paths:
            if not all_samples_dir.exists():
                continue
            for test_file in all_samples_dir.glob('all_L1_test*.txt'):
                file_key = str(test_file.resolve())
                if file_key in seen:
                    continue
                seen.add(file_key)
                noise_level, eval_type = _parse_test_stem(test_file.stem)
                L1_errors = np.loadtxt(test_file)
                data.setdefault(noise_level, {}).setdefault(eval_type, {}).setdefault(label, []).append(L1_errors)

    all_arrays = [arr for nl_data in data.values()
                  for method_data in nl_data.values()
                  for arr in method_data.values()]
    global_x_max = float(np.percentile(np.concatenate(all_arrays), 99)) if all_arrays else 1.0
    return data, global_x_max


def _draw_ecdf_panel(ax, eval_type, noise_level, data, global_x_max, label_mode='method',
                     label_styles=None, show_legend=True, legend_loc='lower right'):
    if noise_level not in data or eval_type not in data[noise_level]:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        _apply_axis_style(ax)
        return
    labels = _ordered_labels(data[noise_level][eval_type]) if label_mode == 'ratio' else _sorted_labels(data[noise_level][eval_type])
    for label in labels:
        errors = np.sort(np.concatenate(data[noise_level][eval_type][label]))
        errors = errors * 100.0
        y = np.arange(1, len(errors) + 1) / len(errors)
        if label_mode == 'ratio' and label_styles is not None:
            style = label_styles[label]
            ax.plot(
                errors, y, label=label,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                marker=style['marker'],
                markersize=3.4,
                markerfacecolor='white',
                markeredgecolor=style['color'],
                markeredgewidth=0.9,
                markevery=_sparse_markevery(len(errors)),
                alpha=style['alpha'],
                zorder=style['zorder'],
            )
        else:
            style = _METHOD_ECDF_STYLES.get(label, {
                'color': _method_color(label),
                'linestyle': '-',
                'marker': None,
            })
            ax.plot(errors, y, label=label,
                    color=style['color'], linestyle=style['linestyle'],
                    linewidth=1.2, marker=style['marker'],
                    markersize=3.4, markerfacecolor='white',
                    markeredgecolor=style['color'], markeredgewidth=0.9,
                    markevery=_sparse_markevery(len(errors)))
    ax.set_xlabel(r'Relative $L_1$ Error (%)', fontsize=17)
    ax.set_ylabel('ECDF')
    ax.set_xlim(0, X_AXIS_MAX_PERCENT)
    ax.set_xticks(np.arange(0, X_AXIS_MAX_PERCENT + 1, 2))
    ax.set_ylim(0, 1)
    if show_legend:
        _legend_with_frame(ax, fontsize=10, loc=legend_loc)
    _apply_axis_style(ax)


def plot_ecdf_curve(experiments_data, output_dir=None, filename_suffix='', label_mode='method'):
    output_path = _resolve_output_dir(output_dir, 'comparison_results')
    data, global_x_max = _collect_histogram_data(experiments_data, label_mode=label_mode)
    if not data:
        print("No data available for ECDF plotting")
        return None

    label_styles = _build_label_styles(data) if label_mode == 'ratio' else None
    saved_files = []
    noise_levels = sorted(data.keys())
    all_eval_types = sorted({et for nl_data in data.values() for et in nl_data})

    if label_mode == 'ratio':
        ecdf_legend_kwargs = _shared_legend_kwargs(fontsize=12)
        draw_ecdf = lambda ax, eval_type, noise_level, d, gx, show_legend=True: _draw_ecdf_panel(
            ax, eval_type, noise_level, d, gx, label_mode=label_mode,
            label_styles=label_styles, show_legend=show_legend
        )
        use_shared_legend = True
    else:
        draw_ecdf = lambda ax, eval_type, noise_level, d, gx, show_legend=True: _draw_ecdf_panel(
            ax, eval_type, noise_level, d, gx,
            show_legend=False,
        )
        ecdf_legend_kwargs = None
        use_shared_legend = False
    fig_file = _draw_grid_figure(noise_levels, all_eval_types, data, global_x_max,
                                 draw_ecdf, '',
                                 output_path, f'ecdf_combined{filename_suffix}.png',
                                 show_noise_label=True,
                                 shared_legend=use_shared_legend,
                                 legend_kwargs=ecdf_legend_kwargs,
                                 panel_label_kwargs={
                                     'x': -0.13,
                                     'y': 1.06,
                                     'fontsize': 18,
                                 })
    print(f"Saved ECDF: {fig_file}")
    saved_files.append(str(fig_file))

    return saved_files


def _load_ratio_statistics_table(csv_file):
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Ratio statistics CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    required_cols = {
        'noise_level', 'ratio',
        'bil_mean', 'exp_mean', 'grf_mean', 'mix_mean',
    }
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        print(f"Missing required columns in ratio statistics CSV: {sorted(missing_cols)}")
        return None

    df['noise_level'] = pd.to_numeric(df['noise_level'], errors='coerce')
    for col in ['bil_mean', 'exp_mean', 'grf_mean', 'mix_mean']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ratio'] = df['ratio'].astype(str).str.strip()
    df = df.dropna(subset=['noise_level', 'ratio'])
    if df.empty:
        print("No valid rows found in ratio statistics CSV")
        return None
    return df


def plot_ratio_mix_heatmap_from_csv(csv_file, output_dir=None,
                                    filename='mix_mean_heatmap.png', dpi=600):
    output_path = _resolve_output_dir(output_dir, 'comparison_results')
    df = _load_ratio_statistics_table(csv_file)
    if df is None:
        return None

    ratios_present = list(dict.fromkeys(df['ratio'].tolist()))
    ratio_order = [ratio for ratio in _TABLE_RATIO_ORDER if ratio in ratios_present]
    ratio_order.extend(sorted([ratio for ratio in ratios_present if ratio not in ratio_order]))
    ratio_order = list(reversed(ratio_order))
    noise_levels = sorted(df['noise_level'].unique().tolist())

    pivot = (
        df.pivot_table(index='ratio', columns='noise_level', values='mix_mean', aggfunc='first')
        .reindex(index=ratio_order, columns=noise_levels)
    )
    values = pivot.to_numpy(dtype=float) * 100.0
    if values.size == 0:
        print("No valid MIX mean data available for heatmap plotting")
        return None

    fig_w = max(6.2, 0.9 * len(noise_levels) + 3.0)
    fig_h = max(6.8, 0.38 * len(ratio_order) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap('Blues')
    finite_vals = values[np.isfinite(values)]
    vmin = float(np.min(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0
    im = ax.imshow(values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    for col_idx in range(values.shape[1]):
        col_vals = values[:, col_idx]
        finite_idx = np.where(np.isfinite(col_vals))[0]
        if finite_idx.size == 0:
            continue
        best_row = finite_idx[int(np.argmin(col_vals[finite_idx]))]
        rect = plt.Rectangle(
            (col_idx - 0.5, best_row - 0.5), 1.0, 1.0,
            fill=False, edgecolor='#212529', linewidth=2.0
        )
        ax.add_patch(rect)

    midpoint = (vmin + vmax) / 2.0
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            val = values[row_idx, col_idx]
            if not np.isfinite(val):
                continue
            text_color = 'white' if val > midpoint else 'black'
            ax.text(
                col_idx, row_idx, f'{val:.2f}',
                ha='center', va='center', fontsize=8.5,
                color=text_color, fontweight='semibold'
            )

    ax.set_xticks(np.arange(len(noise_levels)))
    ax.set_xticklabels([
        f'{int(v)}' if float(v).is_integer() else f'{v:g}'
        for v in noise_levels
    ])
    ax.set_yticks(np.arange(len(ratio_order)))
    ax.set_yticklabels(ratio_order)
    ax.set_xlabel('Noise level (%)')
    ax.set_ylabel(r'Training ratio $r_{\mathrm{BIL}}:r_{\mathrm{EXP}}:r_{\mathrm{GRF}}$')
    # ax.set_title('MIX Test Mean Relative $L_1$ Error (%)')
    _apply_axis_style(ax)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'Mean relative $L_1$ error (%)')

    plt.tight_layout()
    fig_file = output_path / filename
    _save_figure_pair(fig, fig_file, dpi)
    plt.close()
    print(f"Saved MIX mean heatmap: {fig_file}")
    return str(fig_file)


def plot_ratio_representative_trends_from_csv(
        csv_file,
        output_dir=None,
        filename='ratio_representative_trends.png',
        dpi=600,
        representative_ratios=None):
    output_path = _resolve_output_dir(output_dir, 'comparison_results')
    df = _load_ratio_statistics_table(csv_file)
    if df is None:
        return None

    if representative_ratios is None:
        representative_ratios = [
            '0.33:0.33:0.34',
            '0.1:0.6:0.3',
            '0.1:0.8:0.1',
            '0.1:0.2:0.7',
        ]

    representative_ratios = [
        ratio for ratio in representative_ratios
        if ratio in set(df['ratio'].tolist())
    ]
    if not representative_ratios:
        print("No representative ratios found in ratio statistics CSV")
        return None

    noise_levels = sorted(df['noise_level'].unique().tolist())
    eval_types = ['bil', 'exp', 'grf', 'mix']
    panel_titles = {
        'bil': 'BIL Test Set',
        'exp': 'EXP Test Set',
        'grf': 'GRF Test Set',
        'mix': 'MIX Test Set',
    }
    style_map = {
        '0.33:0.33:0.34': {'color': 'k', 'linestyle': '-', 'marker': 's', 'linewidth': 1.5},
        '0.1:0.6:0.3': {'color': 'C1', 'linestyle': '-', 'marker': 'o', 'linewidth': 1.8},
        '0.1:0.8:0.1': {'color': 'C0', 'linestyle': '-', 'marker': '^', 'linewidth': 1.5},
        '0.1:0.2:0.7': {'color': 'C2', 'linestyle': '--', 'marker': 'd', 'linewidth': 1.5},
    }
    fallback_colors = ['#CC79A7', '#56B4E9', '#7F3C8D', '#11A579']
    x_min = float(min(noise_levels))
    x_max = float(max(noise_levels))
    y_cols = [f'{eval_type}_mean' for eval_type in eval_types if f'{eval_type}_mean' in df.columns]
    finite_y = (df[y_cols].to_numpy(dtype=float) * 100.0) if y_cols else np.array([])
    finite_y = finite_y[np.isfinite(finite_y)]
    y_tick_step = 2.0
    y_max = float(np.max(finite_y)) if finite_y.size else 10.0
    y_upper = 6.5
    # y_upper = max(y_tick_step, float(np.ceil(y_max / y_tick_step) * y_tick_step))
    y_ticks = np.arange(0.0, y_upper + 0.5 * y_tick_step, y_tick_step)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6), sharex=True)
    axes = axes.reshape(-1)

    handles = None
    labels = None
    for ax, eval_type in zip(axes, eval_types):
        col = f'{eval_type}_mean'
        for idx, ratio in enumerate(representative_ratios):
            sub = (
                df[df['ratio'] == ratio]
                .sort_values('noise_level')
            )
            if sub.empty or col not in sub.columns:
                continue
            style = dict(style_map.get(ratio, {}))
            if not style:
                style = {
                    'color': fallback_colors[idx % len(fallback_colors)],
                    'linestyle': '-',
                    'marker': 'o',
                    'linewidth': 1.7,
                }
            y = sub[col].to_numpy(dtype=float) * 100.0
            x = sub['noise_level'].to_numpy(dtype=float)
            ax.plot(
                x, y,
                label=ratio,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=style['linewidth'],
                markersize=5.3,
                markerfacecolor=style['color'],
                markeredgecolor=style['color'],
                markeredgewidth=0.9,
            )
        ax.set_title(panel_titles[eval_type])
        ax.set_ylabel(r'Mean relative $L_1$ error (%)')
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(0.0, y_upper)
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([
            f'{int(v)}' if float(v).is_integer() else f'{v:g}'
            for v in noise_levels
        ])
        ax.set_yticks(y_ticks)
        _apply_axis_style(ax)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    for ax in axes[2:]:
        ax.set_xlabel('Noise level (%)')

    _add_panel_labels(list(axes))
    if handles and labels:
        axes[1].legend(
            handles, labels,
            loc='upper left',
            ncol=min(2, len(representative_ratios)),
            bbox_to_anchor=(0.02, 0.98),
            frameon=False,
            fontsize=9.2,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.4,
            labelspacing=0.25,
            borderaxespad=0.0,
        )

    # fig.suptitle('Representative Training-Ratio Effects Across Test Subsets', y=1.07)
    plt.tight_layout()
    fig_file = output_path / filename
    _save_figure_pair(fig, fig_file, dpi)
    plt.close()
    print(f"Saved representative ratio trends: {fig_file}")
    return str(fig_file)


def plot_noise_mean_std_from_csv(csv_file=None, output_dir=None,
                                 target_methods=None, target_eval_types=None,
                                 target_noise_levels=None,
                                 filename='mean_std_combined.png', dpi=600):
    output_path = _resolve_output_dir(output_dir, 'comparison_results')
    csv_path = Path(csv_file) if csv_file is not None else (output_path / 'noise_statistics.csv')
    if not csv_path.exists():
        print(f"Noise statistics CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    required_cols = {'noise_level', 'eval_type', 'method', 'mean', 'std'}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        print(f"Missing required columns in CSV: {sorted(missing_cols)}")
        return None

    for col in ['noise_level', 'mean', 'std']:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    df['method'] = df['method'].astype(str).str.strip()
    df['eval_type'] = df['eval_type'].astype(str).str.strip().str.lower()
    if 'gamma' not in df.columns:
        df['gamma'] = np.nan
    else:
        df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce')
    df = df.dropna(subset=['noise_level', 'mean', 'std'])

    if target_methods is not None:
        method_set = {str(m).strip() for m in target_methods}
        df = df[df['method'].isin(method_set)]
    if target_eval_types is not None:
        eval_set = {str(e).strip().lower() for e in target_eval_types}
        df = df[df['eval_type'].isin(eval_set)]
    if target_noise_levels is not None:
        noise_vals = np.array([float(v) for v in target_noise_levels], dtype=float)
        df = df[df['noise_level'].apply(lambda x: np.any(np.isclose(float(x), noise_vals, atol=1e-9)))]

    if df.empty:
        print("No rows available for mean-std plotting after filtering")
        return None

    df['label'] = df.apply(
        lambda r: _method_label(
            r['method'],
            None if pd.isna(r['gamma']) else float(r['gamma'])
        ),
        axis=1
    )

    stats_df = (
        df.groupby(['noise_level', 'eval_type', 'label'], as_index=False)
        .agg(mean=('mean', 'mean'), std=('std', 'mean'))
    )
    if stats_df.empty:
        print("No grouped rows available for mean-std plotting")
        return None

    data = {}
    for row in stats_df.itertuples(index=False):
        noise_level = float(row.noise_level)
        eval_type = str(row.eval_type)
        label = str(row.label)
        data.setdefault(noise_level, {}).setdefault(eval_type, {})[label] = {
            'mean_pct': float(row.mean) * 100.0,
            'std_pct': float(row.std) * 100.0,
        }

    noise_levels = sorted(data.keys())
    all_eval_types = sorted({et for nl_data in data.values() for et in nl_data.keys()})
    all_labels = _sorted_labels({lb for nl_data in data.values() for et_data in nl_data.values() for lb in et_data.keys()})
    if not noise_levels or not all_eval_types or not all_labels:
        print("No valid grid data for mean-std plotting")
        return None

    # Row-wise y-limit: share scale within each noise level to improve comparability.
    # Use percentile to avoid one extreme bar flattening all panels in the same row.
    row_ymax = {}
    for noise_level in noise_levels:
        upper_vals = []
        for eval_type in all_eval_types:
            for label in all_labels:
                stats = data.get(noise_level, {}).get(eval_type, {}).get(label)
                if stats is None:
                    continue
                upper_vals.append(stats['mean_pct'] + stats['std_pct'])
        if not upper_vals:
            row_ymax[noise_level] = 1.0
            continue
        p95 = float(np.percentile(np.array(upper_vals, dtype=float), 95))
        max_mean = float(np.max([
            data.get(noise_level, {}).get(et, {}).get(lb, {}).get('mean_pct', 0.0)
            for et in all_eval_types for lb in all_labels
        ]))
        row_ymax[noise_level] = max(1e-6, max(p95 * 1.08, max_mean * 1.10))

    last_noise_level = noise_levels[-1]

    def _draw_mean_std_panel(ax, eval_type, noise_level, panel_data, row_ymax_map, show_legend=True):
        y_max = float(row_ymax_map.get(noise_level, 1.0))
        if noise_level not in panel_data or eval_type not in panel_data[noise_level]:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_xticks(np.arange(len(all_labels)))
            if noise_level == last_noise_level:
                ax.set_xlabel('Method')
                ax.set_xticklabels(all_labels, rotation=15, ha='right')
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            ax.set_ylim(0, y_max)
            _apply_axis_style(ax)
            return

        method_stats = panel_data[noise_level][eval_type]
        x = np.arange(len(all_labels))
        for i, label in enumerate(all_labels):
            if label not in method_stats:
                continue
            mean_pct = method_stats[label]['mean_pct']
            std_pct = method_stats[label]['std_pct']
            color = _method_color(label)
            lower = max(0.0, mean_pct - std_pct)
            upper = mean_pct + std_pct
            upper_clipped = min(upper, y_max)
            yerr_low = mean_pct - lower
            yerr_up = max(0.0, upper_clipped - mean_pct)
            ax.errorbar(
                i, mean_pct, yerr=np.array([[yerr_low], [yerr_up]]),
                fmt='o', markersize=4.0,
                linewidth=1.2, elinewidth=1.1, capsize=2.6,
                color=color, ecolor=color,
                linestyle='none',
                label=label
            )
            if upper > y_max:
                # Indicate clipped error bars and annotate the true upper bound.
                arrow_start = y_max * 0.88 if mean_pct >= y_max * 0.88 else max(mean_pct, y_max * 0.80)
                ax.annotate(
                    '',
                    xy=(i, y_max * 0.995),
                    xytext=(i, arrow_start),
                    arrowprops={'arrowstyle': '-|>', 'color': color, 'linewidth': 1.0},
                )
                ax.text(
                    i + 0.05, y_max * 0.985,
                    f'>{upper:.1f}',
                    color=color, fontsize=6.5, rotation=90,
                    va='top', ha='left'
                )
        ax.set_xticks(x)
        if noise_level == last_noise_level:
            ax.set_xlabel('Method')
            ax.set_xticklabels(all_labels, rotation=15, ha='right')
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        ax.set_ylim(0, y_max)
        _apply_axis_style(ax)
    fig_file = _draw_grid_figure(
        noise_levels, all_eval_types, data, row_ymax,
        _draw_mean_std_panel, r'Relative $L_1$ Error (%)',
        output_path, filename,
        show_noise_label=False,
        shared_legend=False,
    )
    print(f"Saved mean-std plot: {fig_file}")
    return str(fig_file)


def save_noise_statistics(experiments_data, output_dir=None, filename_suffix='', label_mode='method'):
    output_path = _resolve_output_dir(output_dir, 'comparison_results')
    rows = []
    for (config_type, load_type, exp_id, exp_path), results in experiments_data.items():
        if config_type != 'mix':
            continue
        config = results.get('config') or {}
        method = config.get('method', 'Unknown')
        gamma = config.get('gamma', None)
        ratio_info = extract_mix_ratio_info(results, exp_id=exp_id, exp_path=exp_path)
        search_paths = [Path(exp_path) / 'all_samples_test', Path(exp_path) / 'all_samples']
        seen = set()
        for all_samples_dir in search_paths:
            if not all_samples_dir.exists():
                continue
            for test_file in all_samples_dir.glob('all_L1_test*.txt'):
                file_key = str(test_file.resolve())
                if file_key in seen:
                    continue
                seen.add(file_key)
                noise_level, eval_type = _parse_test_stem(test_file.stem)
                L1_errors = np.loadtxt(test_file)
                std = np.std(L1_errors)
                rows.append({
                    'noise_level': int(noise_level) if noise_level.is_integer() else noise_level,
                    'eval_type': eval_type,
                    'method': method,
                    'gamma': gamma,
                    'label': _label_for_results(results, exp_id, exp_path, label_mode),
                    'ratio_tag': ratio_info.get('ratio_tag'),
                    'bil_ratio': ratio_info.get('bil_ratio'),
                    'exp_ratio': ratio_info.get('exp_ratio'),
                    'grf_ratio': ratio_info.get('grf_ratio'),
                    'mean': np.mean(L1_errors),
                    'std': std,
                    'variance': std ** 2,
                    'n_samples': len(L1_errors),
                })

    if not rows:
        print("No data for noise statistics")
        return None
    sort_cols = ['noise_level', 'eval_type', 'method', 'gamma']
    if label_mode == 'ratio':
        sort_cols = ['noise_level', 'eval_type', 'ratio_tag']
    df_long = pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

    # Keep long-form statistics for downstream plotting code.
    csv_long = output_path / f'noise_statistics_long{filename_suffix}.csv'
    df_long.to_csv(csv_long, index=False)
    print(f"Saved noise statistics (long): {csv_long}")

    if label_mode == 'ratio':
        df_raw = df_long.copy()
        raw_csv = output_path / f'noise_statistics_raw{filename_suffix}.csv'
        df_raw.to_csv(raw_csv, index=False)
        print(f"Saved raw noise statistics: {raw_csv}")

        df_raw['ratio'] = df_raw.apply(
            lambda r: _ratio_display_from_values(r['bil_ratio'], r['exp_ratio'], r['grf_ratio']),
            axis=1
        )
        pivot = (
            df_raw
            .pivot_table(
                index=['noise_level', 'ratio'],
                columns='eval_type',
                values=['mean', 'std'],
                aggfunc='first'
            )
            .sort_index(axis=1)
        )
        pivot.columns = [f'{etype}_{stat}' for stat, etype in pivot.columns]
        paper_df = pivot.reset_index()

        desired_cols = [
            'noise_level', 'ratio',
            'bil_mean', 'bil_std',
            'exp_mean', 'exp_std',
            'grf_mean', 'grf_std',
            'mix_mean', 'mix_std',
        ]
        for col in desired_cols:
            if col not in paper_df.columns:
                paper_df[col] = np.nan
        paper_df = paper_df[desired_cols]

        noise_order = sorted(paper_df['noise_level'].dropna().unique())
        ratio_order_map = {ratio: idx for idx, ratio in enumerate(_TABLE_RATIO_ORDER)}
        paper_df['_noise_order'] = paper_df['noise_level'].map({n: i for i, n in enumerate(noise_order)})
        paper_df['_ratio_order'] = paper_df['ratio'].map(ratio_order_map).fillna(len(ratio_order_map))
        paper_df = paper_df.sort_values(['_noise_order', '_ratio_order']).drop(columns=['_noise_order', '_ratio_order'])
        paper_df = paper_df.reset_index(drop=True)
        paper_df['noise_level'] = paper_df['noise_level'].apply(
            lambda v: int(v) if float(v).is_integer() else float(v)
        )

        csv_table = output_path / f'noise_statistics{filename_suffix}.csv'
        paper_df.to_csv(csv_table, index=False, float_format='%.8f')
        print(f"Saved paper-format noise statistics: {csv_table}")
        return str(csv_table)

    # Create a paper-table-friendly layout:
    # noise_level | method | bil_mean | bil_std | exp_mean | exp_std | grf_mean | grf_std | mix_mean | mix_std
    method_display_map = {
        'MSE': 'MSE',
        'LocResloss': 'Loc',
        'GloResloss': 'Glo',
        'LocMixloss': 'LocMix',
        'GloMixloss': 'GloMix',
    }
    method_order = ['MSE', 'LocResloss', 'GloResloss', 'LocMixloss', 'GloMixloss']
    eval_order = ['bil', 'exp', 'grf', 'mix']

    df_table_src = (
        df_long.groupby(['noise_level', 'method', 'eval_type'], as_index=False)
        .agg(mean=('mean', 'mean'), std=('std', 'mean'))
    )

    noise_values = sorted(df_table_src['noise_level'].unique().tolist())
    method_values_present = [m for m in method_order if m in set(df_table_src['method'])]
    # Append any unexpected methods in stable order.
    for m in sorted(set(df_table_src['method']) - set(method_values_present)):
        method_values_present.append(m)

    table_rows = []
    for noise_level in noise_values:
        for method in method_values_present:
            row = {
                'noise_level': int(noise_level) if float(noise_level).is_integer() else float(noise_level),
                'method': method_display_map.get(method, method),
            }
            sub = df_table_src[
                (df_table_src['noise_level'] == noise_level) &
                (df_table_src['method'] == method)
            ]
            for eval_type in eval_order:
                sub_eval = sub[sub['eval_type'].str.lower() == eval_type]
                if sub_eval.empty:
                    row[f'{eval_type}_mean'] = np.nan
                    row[f'{eval_type}_std'] = np.nan
                else:
                    row[f'{eval_type}_mean'] = float(sub_eval['mean'].iloc[0])
                    row[f'{eval_type}_std'] = float(sub_eval['std'].iloc[0])
            table_rows.append(row)

    df_table = pd.DataFrame(table_rows)
    table_cols = [
        'noise_level', 'method',
        'bil_mean', 'bil_std',
        'exp_mean', 'exp_std',
        'grf_mean', 'grf_std',
        'mix_mean', 'mix_std',
    ]
    df_table = df_table.reindex(columns=table_cols)

    csv_table = output_path / f'noise_statistics{filename_suffix}.csv'
    df_table.to_csv(csv_table, index=False)
    print(f"Saved noise statistics (table layout): {csv_table}")
    return str(csv_long)


__all__ = [
    'plot_ecdf_curve',
    'plot_ratio_mix_heatmap_from_csv',
    'plot_ratio_representative_trends_from_csv',
    'plot_noise_mean_std_from_csv',
    'save_noise_statistics',
]
