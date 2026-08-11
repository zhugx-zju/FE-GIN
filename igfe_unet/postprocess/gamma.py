import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator
from pathlib import Path

from .common import (
    _resolve_output_dir,
    _parse_test_stem,
    _dataset_title,
    _SUBTITLE_FONTSIZE,
    _LABEL_FONTSIZE,
    _add_panel_labels,
    _apply_axis_style,
    _figure_legend_with_frame,
    _top_shared_legend_kwargs,
    _collect_unique_legend_items,
    LOSS_COLORS,
    METHOD_DISPLAY,
    _MIX_METHODS,
)

_NCS_HEATMAP_CMAP = 'Blues'
_NCS_HEATMAP_LEVELS = 7


def _save_figure_pair(fig, fig_file, dpi):
    """Save every gamma figure in both raster and vector formats."""
    fig.savefig(fig_file, dpi=dpi, bbox_inches='tight')
    pdf_file = Path(fig_file).with_suffix('.pdf')
    fig.savefig(pdf_file, bbox_inches='tight')
    print(f"Saved PDF: {pdf_file}")


def _float_in_values(value, values, tol=1e-9):
    return any(abs(float(value) - float(v)) <= tol for v in values)


def _gamma_key(method, config):
    gamma = config.get('gamma', None) if method in _MIX_METHODS else None
    if gamma is None:
        return None
    try:
        return float(gamma)
    except (TypeError, ValueError):
        return None


def _gamma_from_exp_id(exp_id):
    if not exp_id:
        return None
    m = re.search(r'_gamma_([0-9p\.eE+-]+)$', str(exp_id))
    if not m:
        return None
    token = m.group(1).replace('p', '.')
    try:
        return float(token)
    except ValueError:
        return None


def _format_gamma(gamma):
    if gamma is None:
        return 'NA'
    return f'{float(gamma):.3g}'


def _select_best_runs_by_method_gamma(experiments_data, target_methods):
    best = {}
    for exp_info, results in experiments_data.items():
        config = results.get('config') or {}
        method = config.get('method', '')
        if method not in target_methods:
            continue

        gamma = _gamma_key(method, config)
        if method in _MIX_METHODS and gamma is None:
            gamma = _gamma_from_exp_id(exp_info[2])
        key = (method, gamma)

        valid_mae = results.get('valid_mae')
        valid_loss = results.get('valid_loss')
        if valid_mae is not None and len(valid_mae) > 0:
            score = float(np.min(valid_mae))
        elif valid_loss is not None and len(valid_loss) > 0:
            score = float(np.min(valid_loss))
        else:
            score = float('inf')

        if key not in best or score < best[key]['score']:
            best[key] = {
                'score': score,
                'exp_info': exp_info,
                'results': results,
            }
    return best


def _collect_selected_test_errors(selected_runs, eval_types=None, noise_levels=None, data_split='test'):
    eval_set = set(eval_types) if eval_types is not None else None
    noise_values = [float(v) for v in noise_levels] if noise_levels is not None else None
    split = str(data_split).strip().lower()
    if split not in {'test', 'val'}:
        raise ValueError(f"Unsupported data_split: {data_split}. Use 'test' or 'val'.")

    rows = []
    for (method, gamma), item in selected_runs.items():
        exp_info = item['exp_info']
        exp_path = Path(exp_info[3])
        search_paths = [exp_path / f'all_samples_{split}', exp_path / 'all_samples', exp_path]
        patterns = [f'all_L1_{split}*.txt', f'L1_{split}*.txt']
        visited_files = set()
        for search_path in search_paths:
            if not search_path.exists():
                continue
            for pattern in patterns:
                for test_file in search_path.glob(pattern):
                    file_key = str(test_file.resolve())
                    if file_key in visited_files:
                        continue
                    visited_files.add(file_key)

                    noise_level, eval_type = _parse_test_stem(test_file.stem)
                    if eval_set is not None and eval_type not in eval_set:
                        continue
                    if noise_values is not None and not _float_in_values(noise_level, noise_values):
                        continue

                    errors = np.atleast_1d(np.loadtxt(test_file)).astype(float)
                    rows.append({
                        'method': method,
                        'gamma': gamma,
                        'noise_level': float(noise_level),
                        'eval_type': eval_type,
                        'errors': errors,
                    })
    return rows


def _aggregate_error_rows(rows):
    grouped = {}
    for row in rows:
        key = (row['method'], row['gamma'], row['noise_level'], row['eval_type'])
        grouped.setdefault(key, []).append(row['errors'])
    return {k: np.concatenate(v) for k, v in grouped.items()}


def _gamma_curve_color(method, idx, total):
    if method == 'LocMixloss':
        cmap = plt.cm.Purples
    else:
        cmap = plt.cm.Oranges
    if total <= 1:
        return cmap(0.65)
    return cmap(0.35 + 0.55 * (idx / (total - 1)))


def _gamma_curve_label(method, gamma):
    method_name = METHOD_DISPLAY.get(method, method)
    return f'{method_name} (\u03b3={_format_gamma(gamma)})'


def save_gamma_statistics(experiments_data, output_dir=None,
                          eval_types=None, noise_levels=None, data_split='test',
                          filename_suffix=''):
    output_path = _resolve_output_dir(output_dir, 'comparison_gamma')
    target_methods = {'MSE', 'LocResloss', 'GloResloss', 'LocMixloss', 'GloMixloss'}

    selected_runs = _select_best_runs_by_method_gamma(experiments_data, target_methods)
    rows = _collect_selected_test_errors(
        selected_runs,
        eval_types=eval_types,
        noise_levels=noise_levels,
        data_split=data_split,
    )
    grouped = _aggregate_error_rows(rows)
    if not grouped:
        print("No data for gamma statistics")
        return None

    stats_rows = []
    for (method, gamma, noise_level, eval_type), errors in grouped.items():
        std = float(np.std(errors))
        p05, p95 = np.percentile(errors, [5.0, 95.0])
        stats_rows.append({
            'method': method,
            'gamma': gamma,
            'noise_level': int(noise_level) if float(noise_level).is_integer() else noise_level,
            'eval_type': eval_type,
            'mean': float(np.mean(errors)),
            'std': std,
            'variance': std ** 2,
            'p05': float(p05),
            'p95': float(p95),
            'mean_pct': float(np.mean(errors) * 100.0),
            'std_pct': float(std * 100.0),
            'variance_pct': float((std * 100.0) ** 2),
            'p05_pct': float(p05 * 100.0),
            'p95_pct': float(p95 * 100.0),
            'n_samples': int(len(errors)),
        })

    df = pd.DataFrame(stats_rows).sort_values(
        ['noise_level', 'eval_type', 'method', 'gamma']
    ).reset_index(drop=True)
    csv_file = output_path / f'gamma_statistics{filename_suffix}.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved gamma statistics: {csv_file}")
    return str(csv_file)


def plot_gamma_ecdf_panels(experiments_data, output_dir=None,
                           eval_types=None, noise_levels=None, data_split='test', dpi=600,
                           filename_suffix=''):
    output_path = _resolve_output_dir(output_dir, 'comparison_gamma')
    target_methods = {'MSE', 'LocResloss', 'GloResloss', 'LocMixloss', 'GloMixloss'}
    baseline_map = {
        'LocMixloss': ['MSE', 'LocResloss'],
        'GloMixloss': ['MSE', 'GloResloss'],
    }

    selected_runs = _select_best_runs_by_method_gamma(experiments_data, target_methods)
    rows = _collect_selected_test_errors(
        selected_runs,
        eval_types=eval_types,
        noise_levels=noise_levels,
        data_split=data_split,
    )
    grouped = _aggregate_error_rows(rows)
    if not grouped:
        print("No data for gamma ECDF plotting")
        return []

    all_noises = sorted({key[2] for key in grouped.keys()})
    all_eval_types = sorted({key[3] for key in grouped.keys()})
    if noise_levels is not None:
        all_noises = [n for n in all_noises if _float_in_values(n, [float(v) for v in noise_levels])]
    if eval_types is not None:
        all_eval_types = [e for e in all_eval_types if e in set(eval_types)]

    saved_files = []
    mix_methods = ['LocMixloss', 'GloMixloss']
    if not all_noises or not all_eval_types:
        print("No data left after gamma ECDF filters")
        return saved_files

    for mix_method in mix_methods:
        n_rows = len(all_noises)
        n_cols = len(all_eval_types)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.9 * n_rows), squeeze=False)
        for row_idx, noise_level in enumerate(all_noises):
            for col_idx, eval_type in enumerate(all_eval_types):
                ax = axes[row_idx, col_idx]
                line_count = 0

                gamma_values = sorted({
                    key[1] for key in grouped.keys()
                    if key[0] == mix_method and key[2] == noise_level and key[3] == eval_type and key[1] is not None
                })

                for gamma_idx, gamma in enumerate(gamma_values):
                    errors = grouped[(mix_method, gamma, noise_level, eval_type)] * 100.0
                    errors = np.sort(errors)
                    y = np.arange(1, len(errors) + 1) / len(errors)
                    ax.plot(
                        errors, y,
                        color=_gamma_curve_color(mix_method, gamma_idx, len(gamma_values)),
                        linewidth=1.4,
                        linestyle='-',
                        label=_gamma_curve_label(mix_method, gamma),
                    )
                    line_count += 1

                for base_method, ls in zip(baseline_map[mix_method], ['--', '-.']):
                    key = (base_method, None, noise_level, eval_type)
                    if key not in grouped:
                        continue
                    errors = grouped[key] * 100.0
                    errors = np.sort(errors)
                    y = np.arange(1, len(errors) + 1) / len(errors)
                    ax.plot(
                        errors, y,
                        color=LOSS_COLORS.get(base_method, '#333333'),
                        linewidth=1.6,
                        linestyle=ls,
                        label=METHOD_DISPLAY.get(base_method, base_method),
                    )
                    line_count += 1

                if line_count == 0:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')

                if row_idx == 0:
                    ax.set_title(
                        _dataset_title(eval_type),
                        fontsize=_SUBTITLE_FONTSIZE,
                        fontweight='normal',
                        pad=8,
                    )

                if col_idx == 0:
                    noise_text = f"Noise {int(noise_level)}%" if float(noise_level).is_integer() else f"Noise {noise_level:g}%"
                    ax.set_ylabel(f'{noise_text}\nECDF', fontsize=_LABEL_FONTSIZE)
                else:
                    ax.set_ylabel('')

                ax.set_xlabel(r'Relative $L_1$ Error (%)', fontsize=_LABEL_FONTSIZE)
                ax.set_xlim(0, 10)
                ax.set_xticks(np.arange(0, 11, 1))
                ax.set_ylim(0, 1)
                _apply_axis_style(ax)

        _add_panel_labels(
            [axes[r, 0] for r in range(n_rows)],
            x=-0.16,
            y=1.06,
            fontsize=18,
        )
        handles, labels = _collect_unique_legend_items([axes[r, c] for r in range(n_rows) for c in range(n_cols)])
        _figure_legend_with_frame(fig, handles, labels, **_top_shared_legend_kwargs(len(labels), fontsize=11))
        plt.tight_layout(rect=(0, 0, 1, 0.945))
        method_tag = 'locmix' if mix_method == 'LocMixloss' else 'glomix'
        fig_file = output_path / f'ecdf_gamma_{method_tag}{filename_suffix}.png'
        _save_figure_pair(fig, fig_file, dpi)
        plt.close()
        print(f"Saved gamma ECDF: {fig_file}")
        saved_files.append(str(fig_file))

    return saved_files


def plot_gamma_mean_std_bars(experiments_data, output_dir=None,
                             eval_types=None, noise_levels=None, data_split='test', dpi=600,
                             filename_suffix=''):
    output_path = _resolve_output_dir(output_dir, 'comparison_gamma')
    target_methods = {'MSE', 'LocResloss', 'GloResloss', 'LocMixloss', 'GloMixloss'}
    baseline_map = {
        'LocMixloss': ['MSE', 'LocResloss'],
        'GloMixloss': ['MSE', 'GloResloss'],
    }
    mix_methods = ['LocMixloss', 'GloMixloss']

    selected_runs = _select_best_runs_by_method_gamma(experiments_data, target_methods)
    rows = _collect_selected_test_errors(
        selected_runs,
        eval_types=eval_types,
        noise_levels=noise_levels,
        data_split=data_split,
    )
    grouped = _aggregate_error_rows(rows)
    if not grouped:
        print("No data for gamma curve plotting")
        return []

    all_noises = sorted({key[2] for key in grouped.keys()})
    all_eval_types = sorted({key[3] for key in grouped.keys()})
    if noise_levels is not None:
        all_noises = [n for n in all_noises if _float_in_values(n, [float(v) for v in noise_levels])]
    if eval_types is not None:
        all_eval_types = [e for e in all_eval_types if e in set(eval_types)]

    saved_files = []
    if not all_noises or not all_eval_types:
        print("No data left after gamma bar filters")
        return saved_files

    for mix_method in mix_methods:
        n_rows = len(all_noises)
        n_cols = len(all_eval_types)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.0 * n_rows), squeeze=False)
        for row_idx, noise_level in enumerate(all_noises):
            for col_idx, eval_type in enumerate(all_eval_types):
                ax = axes[row_idx, col_idx]
                gamma_values = sorted({
                    key[1] for key in grouped.keys()
                    if key[0] == mix_method and key[2] == noise_level and key[3] == eval_type and key[1] is not None
                })

                if gamma_values:
                    means = []
                    p05_values = []
                    p95_values = []
                    for gamma_idx, gamma in enumerate(gamma_values):
                        errors = grouped[(mix_method, gamma, noise_level, eval_type)] * 100.0
                        p05, p95 = np.percentile(errors, [5.0, 95.0])
                        means.append(float(np.mean(errors)))
                        p05_values.append(float(p05))
                        p95_values.append(float(p95))
                    x = np.array(gamma_values, dtype=float)
                    means = np.array(means, dtype=float)
                    p05_values = np.array(p05_values, dtype=float)
                    p95_values = np.array(p95_values, dtype=float)
                    percentile_error = np.vstack([
                        means - p05_values,
                        p95_values - means,
                    ])
                    main_color = LOSS_COLORS.get(mix_method, '#333333')
                    ax.errorbar(
                        x, means, yerr=percentile_error, fmt='-o',
                        markersize=4.2, linewidth=1.2, elinewidth=0.9, capsize=2.6,
                        color=main_color,
                        label=f"{METHOD_DISPLAY.get(mix_method, mix_method)}",
                    )
                    ax.set_xscale('log')
                    major_gamma_values = np.array(
                        [10.0 ** exponent for exponent in (1, 3, 5, 7)],
                        dtype=float,
                    )
                    ax.set_xticks(major_gamma_values)
                    ax.set_xticklabels(
                        _gamma_log10_labels(major_gamma_values), rotation=0
                    )
                    ax.xaxis.set_minor_locator(
                        LogLocator(base=10.0, subs=(2.0, 4.0, 6.0, 8.0), numticks=100)
                    )
                    ax.tick_params(axis='x', which='major', direction='in',
                                   length=4.0, width=0.8)
                    ax.tick_params(axis='x', which='minor', direction='in',
                                   length=2.0, width=0.6)
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')

                for base_method, ls in zip(baseline_map[mix_method], ['--', '-.']):
                    key = (base_method, None, noise_level, eval_type)
                    if key not in grouped:
                        continue
                    errors_base = grouped[key] * 100.0
                    mean_base = float(np.mean(errors_base))
                    p05_base, p95_base = np.percentile(errors_base, [5.0, 95.0])
                    base_color = LOSS_COLORS.get(base_method, '#333333')
                    if gamma_values and p95_base > p05_base:
                        x_band = np.array([min(gamma_values), max(gamma_values)], dtype=float)
                        ax.fill_between(
                            x_band,
                            p05_base,
                            p95_base,
                            color=base_color,
                            alpha=0.10,
                            linewidth=0.0,
                            zorder=1,
                        )
                    ax.axhline(
                        y=mean_base,
                        color=base_color,
                        linewidth=1.2,
                        linestyle=ls,
                        zorder=2,
                        label='_nolegend_',
                    )

                if row_idx == 0:
                    ax.set_title(
                        _dataset_title(eval_type),
                        fontsize=_SUBTITLE_FONTSIZE,
                        fontweight='normal',
                        pad=8,
                    )

                ax.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=19)
                if col_idx == 0:
                    ax.set_ylabel(r'Relative $L_1$ Error (%)', fontsize=15)
                else:
                    ax.set_ylabel('')
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.set_ylim(bottom=0.0)
                _apply_axis_style(ax)

        _add_panel_labels(
            [axes[r, 0] for r in range(n_rows)],
            x=-0.16,
            y=1.16,
            fontsize=18,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.99))
        method_tag = 'locmix' if mix_method == 'LocMixloss' else 'glomix'
        fig_file = output_path / f'gamma_curve_mean_std_{method_tag}{filename_suffix}.png'
        _save_figure_pair(fig, fig_file, dpi)
        plt.close()
        print(f"Saved gamma curve plot: {fig_file}")
        saved_files.append(str(fig_file))

    return saved_files


def plot_gamma_history_2x2(experiments_data, output_dir=None,
                           filename='history_gamma_2x2.png', dpi=600, filename_suffix=''):
    output_path = _resolve_output_dir(output_dir, 'comparison_gamma')
    target_methods = {'MSE', 'LocResloss', 'GloResloss', 'LocMixloss', 'GloMixloss'}
    baseline_map = {
        'LocMixloss': ['MSE', 'LocResloss'],
        'GloMixloss': ['MSE', 'GloResloss'],
    }
    mix_methods = ['LocMixloss', 'GloMixloss']

    selected_runs = _select_best_runs_by_method_gamma(experiments_data, target_methods)
    if not selected_runs:
        print("No runs for gamma history plotting")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for row_idx, mix_method in enumerate(mix_methods):
        ax_loss = axes[row_idx, 0]
        ax_mae = axes[row_idx, 1]

        gamma_values = sorted({
            key[1] for key in selected_runs.keys()
            if key[0] == mix_method and key[1] is not None
        })
        for gamma_idx, gamma in enumerate(gamma_values):
            run = selected_runs.get((mix_method, gamma))
            if run is None:
                continue
            results = run['results']
            train_loss = results.get('train_loss')
            valid_loss = results.get('valid_loss')
            train_mae = results.get('train_mae')
            valid_mae = results.get('valid_mae')
            color = _gamma_curve_color(mix_method, gamma_idx, len(gamma_values))
            label = _gamma_curve_label(mix_method, gamma)

            if train_loss is not None and len(train_loss) > 0:
                ep = np.arange(1, len(train_loss) + 1)
                ax_loss.plot(ep, train_loss, color=color, linewidth=1.0, linestyle='-',
                             alpha=0.85, label=f'{label}, Train.')
            if valid_loss is not None and len(valid_loss) > 0:
                ep = np.arange(1, len(valid_loss) + 1)
                ax_loss.plot(ep, valid_loss, color=color, linewidth=1.3, linestyle='--',
                             alpha=0.95, label=f'{label}, Val.')
            if train_mae is not None and len(train_mae) > 0:
                ep = np.arange(1, len(train_mae) + 1)
                ax_mae.plot(ep, train_mae, color=color, linewidth=1.0, linestyle='-',
                            alpha=0.85, label=f'{label}, Train.')
            if valid_mae is not None and len(valid_mae) > 0:
                ep = np.arange(1, len(valid_mae) + 1)
                ax_mae.plot(ep, valid_mae, color=color, linewidth=1.3, linestyle='--',
                            alpha=0.95, label=f'{label}, Val.')

        for base_method, ls in zip(baseline_map[mix_method], ['--', '-.']):
            run = selected_runs.get((base_method, None))
            if run is None:
                continue
            results = run['results']
            train_loss = results.get('train_loss')
            valid_loss = results.get('valid_loss')
            train_mae = results.get('train_mae')
            valid_mae = results.get('valid_mae')
            label = METHOD_DISPLAY.get(base_method, base_method)
            color = LOSS_COLORS.get(base_method, '#333333')

            if train_loss is not None and len(train_loss) > 0:
                ep = np.arange(1, len(train_loss) + 1)
                ax_loss.plot(ep, train_loss, color=color, linewidth=1.2, linestyle='-',
                             alpha=0.9, label=f'{label}, Train.')
            if valid_loss is not None and len(valid_loss) > 0:
                ep = np.arange(1, len(valid_loss) + 1)
                ax_loss.plot(ep, valid_loss, color=color, linewidth=1.5, linestyle=ls,
                             alpha=0.95, label=f'{label}, Val.')
            if train_mae is not None and len(train_mae) > 0:
                ep = np.arange(1, len(train_mae) + 1)
                ax_mae.plot(ep, train_mae, color=color, linewidth=1.2, linestyle='-',
                            alpha=0.9, label=f'{label}, Train.')
            if valid_mae is not None and len(valid_mae) > 0:
                ep = np.arange(1, len(valid_mae) + 1)
                ax_mae.plot(ep, valid_mae, color=color, linewidth=1.5, linestyle=ls,
                            alpha=0.95, label=f'{label}, Val.')

        ax_loss.set_yscale('log')
        ax_loss.set_title(f"{METHOD_DISPLAY.get(mix_method, mix_method)}: Loss")
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize=8, frameon=False)
        _apply_axis_style(ax_loss)

        ax_mae.set_title(f"{METHOD_DISPLAY.get(mix_method, mix_method)}: MAE")
        ax_mae.set_xlabel('Epoch')
        ax_mae.set_ylabel('MAE')
        ax_mae.legend(fontsize=8, frameon=False)
        _apply_axis_style(ax_mae)

    _add_panel_labels(
        [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        x=-0.16,
        y=1.06,
        fontsize=18,
    )
    plt.tight_layout()
    if filename_suffix and filename.endswith('.png'):
        filename = f"{filename[:-4]}{filename_suffix}.png"
    fig_file = output_path / filename
    _save_figure_pair(fig, fig_file, dpi)
    plt.close()
    print(f"Saved gamma history plot: {fig_file}")
    return str(fig_file)


def _collect_gamma_grouped_data(experiments_data, eval_types=None, noise_levels=None, data_split='test'):
    target_methods = {'MSE', 'LocResloss', 'GloResloss', 'LocMixloss', 'GloMixloss'}
    selected_runs = _select_best_runs_by_method_gamma(experiments_data, target_methods)
    rows = _collect_selected_test_errors(
        selected_runs,
        eval_types=eval_types,
        noise_levels=noise_levels,
        data_split=data_split,
    )
    return _aggregate_error_rows(rows)


def _ordered_eval_types(eval_types_found, preferred=None):
    preferred_order = preferred or ['bil', 'exp', 'grf', 'mix']
    ordered = [e for e in preferred_order if e in eval_types_found]
    ordered.extend([e for e in sorted(eval_types_found) if e not in ordered])
    return ordered


def _noise_tag(noise_level):
    if float(noise_level).is_integer():
        return f"{int(noise_level)}"
    return f"{float(noise_level):g}"


def _gamma_log10_labels(gamma_values):
    labels = []
    for gamma in gamma_values:
        logv = np.log10(float(gamma))
        rounded = np.round(logv)
        if np.isclose(logv, rounded):
            labels.append(f"{int(rounded)}")
        else:
            labels.append(f"{logv:.1f}")
    return labels


def grouped_data_from_gamma_statistics_csv(csv_file, eval_types=None, noise_levels=None):
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Gamma statistics CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    required_cols = {'method', 'gamma', 'noise_level', 'eval_type', 'mean'}
    missing = required_cols.difference(df.columns)
    if missing:
        print(f"Missing required columns in gamma CSV: {sorted(missing)}")
        return {}

    df['method'] = df['method'].astype(str).str.strip()
    df['eval_type'] = df['eval_type'].astype(str).str.strip().str.lower()
    df['noise_level'] = pd.to_numeric(df['noise_level'], errors='coerce')
    df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
    df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce')
    df = df.dropna(subset=['noise_level', 'mean'])

    if eval_types is not None:
        eval_set = {str(v).strip().lower() for v in eval_types}
        df = df[df['eval_type'].isin(eval_set)]
    if noise_levels is not None:
        noise_vals = [float(v) for v in noise_levels]
        df = df[df['noise_level'].apply(lambda x: _float_in_values(x, noise_vals))]

    grouped = {}
    for row in df.itertuples(index=False):
        gamma = None if pd.isna(row.gamma) else float(row.gamma)
        key = (str(row.method), gamma, float(row.noise_level), str(row.eval_type))
        grouped.setdefault(key, []).append(float(row.mean))

    return {k: np.asarray(v, dtype=float) for k, v in grouped.items()}


def plot_gamma_error_heatmaps(experiments_data, output_dir=None,
                              eval_types=None, noise_levels=(1.0, 3.0),
                              data_split='test', mix_methods=None, dpi=600,
                              grouped_data=None, filename_suffix=''):
    output_path = _resolve_output_dir(output_dir, 'comparison_gamma')
    grouped = grouped_data
    if grouped is None:
        grouped = _collect_gamma_grouped_data(
            experiments_data,
            eval_types=eval_types,
            noise_levels=noise_levels,
            data_split=data_split,
        )
    if not grouped:
        print("No data for gamma heatmap plotting")
        return []

    methods_to_plot = list(mix_methods or ['LocMixloss', 'GloMixloss'])
    noises_found = sorted({key[2] for key in grouped.keys() if key[0] in methods_to_plot})
    if noise_levels is not None:
        requested = [float(v) for v in noise_levels]
        noises_found = [n for n in noises_found if _float_in_values(n, requested)]
    if eval_types is not None:
        eval_order = [str(v).strip().lower() for v in eval_types]
    else:
        eval_found = {key[3] for key in grouped.keys() if key[0] in methods_to_plot}
        eval_order = _ordered_eval_types(eval_found)
    if not noises_found or not eval_order:
        print("No data left after heatmap filters")
        return []

    saved_files = []
    for mix_method in methods_to_plot:
        gamma_values = sorted({
            key[1] for key in grouped.keys()
            if key[0] == mix_method and key[1] is not None and key[3] in eval_order and key[2] in noises_found
        })
        if not gamma_values:
            continue

        matrices = {}
        finite_values = []
        for eval_type in eval_order:
            mat = np.full((len(noises_found), len(gamma_values)), np.nan, dtype=float)
            for i, noise_level in enumerate(noises_found):
                for j, gamma in enumerate(gamma_values):
                    key = (mix_method, gamma, noise_level, eval_type)
                    if key in grouped:
                        mat[i, j] = float(np.mean(grouped[key]) * 100.0)
            matrices[eval_type] = mat
            vals = mat[np.isfinite(mat)]
            if vals.size > 0:
                finite_values.append(vals)

        if not finite_values:
            continue
        all_vals = np.concatenate(finite_values)
        vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
        text_threshold = vmin + 0.55 * (vmax - vmin) if vmax > vmin else vmin

        n_cols = len(eval_order)
        panel_width = 4.4
        panel_height = max(3.7, 0.60 * len(noises_found) + 1.4)
        annotation_fontsize = 12.0
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(panel_width * n_cols, panel_height),
            squeeze=False,
        )
        axes_flat = [axes[0, c] for c in range(n_cols)]
        color_ref = None

        for idx, eval_type in enumerate(eval_order):
            ax = axes_flat[idx]
            mat = matrices[eval_type]
            if not np.isfinite(mat).any():
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                _apply_axis_style(ax)
                continue

            color_ref = ax.imshow(
                mat,
                aspect='auto',
                cmap=plt.get_cmap(_NCS_HEATMAP_CMAP, _NCS_HEATMAP_LEVELS),
                vmin=vmin,
                vmax=vmax,
            )

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    if np.isnan(val):
                        continue
                    txt_color = 'white' if val >= text_threshold else '#1a1a1a'
                    ax.text(
                        j,
                        i,
                        f'{val:.2f}',
                        ha='center',
                        va='center',
                        fontsize=annotation_fontsize,
                        color=txt_color,
                    )

            ax.set_title(_dataset_title(eval_type), fontsize=_SUBTITLE_FONTSIZE, fontweight='normal', pad=8)
            ax.set_xticks(np.arange(len(gamma_values)))
            ax.set_xticklabels(_gamma_log10_labels(gamma_values))
            ax.set_yticks(np.arange(len(noises_found)))
            ax.set_yticklabels([_noise_tag(v) for v in noises_found])
            ax.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=_LABEL_FONTSIZE)
            ax.set_ylabel('Noise level (%)' if idx == 0 else '', fontsize=_LABEL_FONTSIZE)
            ax.tick_params(axis='both', which='both', length=0)
            ax.invert_yaxis()
            _apply_axis_style(ax)

        _add_panel_labels(axes_flat, x=-0.16, y=1.06, fontsize=18)

        if color_ref is not None:
            cax = fig.add_axes([0.92, 0.18, 0.012, 0.66])
            cbar = fig.colorbar(color_ref, cax=cax)
            cbar.set_label(r'Mean Relative $L_1$ Error (%)', fontsize=_LABEL_FONTSIZE)

        fig.subplots_adjust(left=0.05, right=0.90, bottom=0.12, top=0.90, wspace=0.28)
        method_tag = 'locmix' if mix_method == 'LocMixloss' else 'glomix'
        noise_tag = "_".join([_noise_tag(v) for v in noises_found])
        fig_file = output_path / f'gamma_heatmap_{method_tag}_noise_{noise_tag}{filename_suffix}.png'
        _save_figure_pair(fig, fig_file, dpi)
        plt.close()
        print(f"Saved gamma heatmap: {fig_file}")
        saved_files.append(str(fig_file))

    return saved_files


__all__ = [
    'save_gamma_statistics',
    'plot_gamma_ecdf_panels',
    'plot_gamma_mean_std_bars',
    'plot_gamma_history_2x2',
    'grouped_data_from_gamma_statistics_csv',
    'plot_gamma_error_heatmaps',
]
