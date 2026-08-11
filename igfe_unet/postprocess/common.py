import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re
from pathlib import Path

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.8

_THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_GROUPS = ('std', 'gamma', 'arch', 'ratio', 'final_model')
OUTPUT_GROUP_NAMES = {
    'std': 'standard_models',
    'gamma': 'gamma_sweep',
    'arch': 'architecture_sweep',
    'ratio': 'loss_ratio_sweep',
    'final_model': 'final_model',
}
_SUBTITLE_FONTSIZE = 20
_LABEL_FONTSIZE = 17

LOSS_COLORS = {
    'MSE': '#e41a1c',
    'LocResloss': '#377eb8',
    'GloResloss': '#4daf4a',
    'LocMixloss': '#984ea3',
    'GloMixloss': '#ff7f00',
}

METHOD_DISPLAY = {
    'MSE': 'MSE-M',
    'LocResloss': 'LE-M',
    'GloResloss': 'GE-M',
    'LocMixloss': 'LM-M',
    'GloMixloss': 'GM-M',
}

_MIX_METHODS = frozenset({'LocMixloss', 'GloMixloss'})


def _use_batch_norm_from_config(config):
    if not isinstance(config, dict):
        return False
    return bool(config.get('use_batch_norm', False))


def _gn_suffix(use_batch_norm=False):
    return '_GN' if bool(use_batch_norm) else ''


def _variant_tag(use_batch_norm=False):
    return f"UNet{_gn_suffix(use_batch_norm)}"


def _method_from_label(label):
    base = label.split('(')[0].strip()
    if base in LOSS_COLORS:
        return base
    for method, display in METHOD_DISPLAY.items():
        if base == display:
            return method
    return None


def _method_label(method, gamma):
    display = METHOD_DISPLAY.get(method, method)
    if method in _MIX_METHODS and gamma is not None:
        return f'{display} (\u03b3={gamma:.2g})'
    return display


def _method_color(label):
    method = _method_from_label(label)
    return LOSS_COLORS.get(method, '#333333')


def _sorted_labels(labels):
    base_order = list(LOSS_COLORS.keys())

    def _key(label):
        method = _method_from_label(label)
        return (base_order.index(method) if method in base_order else len(base_order), label)

    return sorted(labels, key=_key)


def _resolve_output_dir(output_dir, subdir):
    p = Path(output_dir) if output_dir is not None else _THIS_DIR / '..' / '..' / subdir
    p = p.resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_analysis_output_dir(
    experiment_group,
    load_type='force_load',
    use_batch_norm=False,
    data_split=None,
    root_name='results',
):
    """Return the grouped output directory for statistics and figures.

    Outputs are kept separate from model checkpoints while mirroring the
    model layout:
        results/{load_type}/{meaningful_group_name}/{group_norm}/{split}/
    """
    group = str(experiment_group).strip().lower()
    if group not in EXPERIMENT_GROUPS:
        raise ValueError(
            f"Invalid experiment group '{group}'. "
            f"Expected one of: {', '.join(EXPERIMENT_GROUPS)}."
        )
    output_group = OUTPUT_GROUP_NAMES[group]
    output_path = _THIS_DIR / '..' / '..' / root_name / str(load_type) / output_group
    if bool(use_batch_norm):
        output_path = output_path / 'GN'
    if data_split is not None:
        output_path = output_path / str(data_split).strip().lower()
    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _apply_axis_style(ax):
    ax.tick_params(direction='in', which='both', top=False, right=False,
                   labelsize=15)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _legend_with_frame(ax, **kwargs):
    base = {
        'frameon': False,
        'facecolor': 'white',
        'edgecolor': 'black',
        'framealpha': 1.0,
        'fancybox': False,
    }
    base.update(kwargs)
    return ax.legend(**base)


def _figure_legend_with_frame(fig, handles, labels, **kwargs):
    if not handles or not labels:
        return None
    base = {
        'frameon': False,
        'facecolor': 'white',
        'edgecolor': 'black',
        'framealpha': 1.0,
        'fancybox': False,
    }
    base.update(kwargs)
    return fig.legend(handles, labels, **base)


def _dataset_title(eval_type, panel_idx=None):
    mapping = {
        'mix': 'MIX Dataset',
        'bil': 'BIL Dataset',
        'exp': 'EXP Dataset',
        'grf': 'GRF Dataset',
    }
    name = mapping.get(str(eval_type).lower(), f'{str(eval_type).upper()} Dataset')
    if panel_idx is None:
        return name
    return f'({chr(97 + panel_idx)}) {name}'


def _top_shared_legend_kwargs(label_count, fontsize=11):
    return {
        'loc': 'upper left',
        'bbox_to_anchor': (0.06, 0.995, 0.88, 0.001),
        'mode': 'expand',
        'ncol': max(2, min(6, int(label_count))),
        'fontsize': fontsize,
        'borderaxespad': 0.2,
    }


def _add_panel_labels(axes_flat, x=-0.16, y=1.08, fontsize=15,
                      fontweight='normal', fontstyle='normal'):
    for idx, ax in enumerate(axes_flat):
        ax.text(x, y, f'({chr(97 + idx)})',
                transform=ax.transAxes, fontsize=fontsize,
                fontweight=fontweight, fontstyle=fontstyle,
                va='bottom', ha='left')


def _parse_test_stem(stem):
    if 'noise' in stem:
        noise_level = float(stem.split('noise_')[-1].replace('p', '.'))
        type_part = stem.split('_noise_')[0]
    else:
        noise_level = 0.0
        type_part = stem

    eval_type = None
    for prefix in ('all_L1_test', 'all_L1_val', 'L1_test', 'L1_val'):
        if type_part.startswith(prefix):
            eval_type = type_part[len(prefix):].lstrip('_') or 'mix'
            break
    if eval_type is None:
        eval_type = 'mix'

    return noise_level, eval_type


def _format_decimal_token(value, precision=10):
    return f"{float(value):.{precision}f}".rstrip('0').rstrip('.').replace('.', 'p')


def _format_ratio_display(value):
    return f'{float(value):.3g}'


def _ratio_tag_from_values(bil_ratio, exp_ratio, grf_ratio):
    return (
        f"b{_format_decimal_token(bil_ratio)}"
        f"_e{_format_decimal_token(exp_ratio)}"
        f"_g{_format_decimal_token(grf_ratio)}"
    )


def _ratio_values_from_tag(ratio_tag):
    try:
        parts = ratio_tag.split('_')
        if len(parts) != 3:
            return None
        bil = float(parts[0][1:].replace('p', '.'))
        exp = float(parts[1][1:].replace('p', '.'))
        grf = float(parts[2][1:].replace('p', '.'))
    except Exception:
        return None
    return {'bil_ratio': bil, 'exp_ratio': exp, 'grf_ratio': grf}


def extract_mix_ratio_info(results, exp_id=None, exp_path=None):
    """Extract mix ratios and ratio tag from config and/or experiment id."""
    ratio_info = {
        'bil_ratio': None,
        'exp_ratio': None,
        'grf_ratio': None,
        'ratio_tag': None,
    }

    config = results.get('config', {}) if isinstance(results, dict) else {}
    bil = config.get('bil_ratio')
    exp = config.get('exp_ratio')
    grf = config.get('grf_ratio')
    if bil is not None and exp is not None and grf is not None:
        bil = float(bil)
        exp = float(exp)
        grf = float(grf)
        ratio_info['bil_ratio'] = bil
        ratio_info['exp_ratio'] = exp
        ratio_info['grf_ratio'] = grf
        ratio_info['ratio_tag'] = _ratio_tag_from_values(bil, exp, grf)
        return ratio_info

    candidates = [exp_id, str(exp_path) if exp_path is not None else None]
    for text in candidates:
        if not text:
            continue
        match = re.search(r'(b[0-9p]+_e[0-9p]+_g[0-9p]+)', str(text))
        if not match:
            continue
        ratio_tag = match.group(1)
        parsed = _ratio_values_from_tag(ratio_tag)
        if parsed is None:
            continue
        ratio_info.update(parsed)
        ratio_info['ratio_tag'] = ratio_tag
        break

    return ratio_info


def mse_ratio_label(results=None, exp_id=None, exp_path=None, ratio_info=None):
    ratio_info = ratio_info or extract_mix_ratio_info(results, exp_id=exp_id, exp_path=exp_path)
    bil_ratio = ratio_info.get('bil_ratio')
    exp_ratio = ratio_info.get('exp_ratio')
    grf_ratio = ratio_info.get('grf_ratio')

    if bil_ratio is None or exp_ratio is None or grf_ratio is None:
        return 'dat.-driv.'

    return (
        'dat.-driv. '
        f'(B:E:G={_format_ratio_display(bil_ratio)}:'
        f'{_format_ratio_display(exp_ratio)}:'
        f'{_format_ratio_display(grf_ratio)})'
    )


def _collect_unique_legend_items(axes):
    label_to_handle = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in label_to_handle:
                label_to_handle[label] = handle
    return list(label_to_handle.values()), list(label_to_handle.keys())


def _draw_grid_figure(noise_list, all_eval_types, data, global_x_max,
                      draw_fn, ylabel_label, output_path, filename,
                      show_noise_label=True, shared_legend=False, legend_kwargs=None,
                      panel_label_kwargs=None):
    n_rows = len(noise_list)
    n_cols = len(all_eval_types)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for row, noise_level in enumerate(noise_list):
        noise_label = f"Noise level {int(noise_level)}%" if noise_level > 0 else "No Noise"
        for col, eval_type in enumerate(all_eval_types):
            ax = axes[row, col]
            draw_fn(ax, eval_type, noise_level, data, global_x_max,
                    show_legend=not shared_legend)
            if row == 0:
                ax.set_title(_dataset_title(eval_type), fontsize=18,
                             fontweight='normal')
            if col == 0:
                if show_noise_label:
                    if ylabel_label:
                        ax.set_ylabel(f'{noise_label}\n{ylabel_label}', fontsize=16)
                    else:
                        ax.set_ylabel(noise_label, fontsize=16)
                else:
                    ax.set_ylabel(ylabel_label, fontsize=16)
            else:
                ax.set_ylabel('')
    _add_panel_labels([axes[row, 0] for row in range(n_rows)],
                      **(panel_label_kwargs or {}))
    if shared_legend:
        handles, labels = _collect_unique_legend_items(
            [axes[r, c] for r in range(n_rows) for c in range(n_cols)]
        )
        lk = dict(legend_kwargs or {})
        layout_top = float(lk.pop('layout_top', 0.945))
        if 'bbox_to_anchor' not in lk:
            lk.update(_top_shared_legend_kwargs(len(labels), fontsize=9))
        _figure_legend_with_frame(fig=fig, handles=handles, labels=labels, **lk)
        plt.tight_layout(rect=(0, 0, 1, layout_top))
    else:
        plt.tight_layout()
    fig_file = output_path / filename
    plt.savefig(fig_file, dpi=600, bbox_inches='tight')
    pdf_file = fig_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    print(f"Saved PDF: {pdf_file}")
    return fig_file


def experiment_group_from_path(exp_path, default='std'):
    """Infer the grouped experiment folder from a model path."""
    path = Path(exp_path).resolve()
    for parent in (path.parent, *path.parents):
        if parent.name in EXPERIMENT_GROUPS:
            return parent.name
    return default


def _is_experiment_dir(path):
    path = Path(path)
    return (
        path.is_dir()
        and (path / 'config.py').exists()
        and (path / 'model.pt').exists()
    )


def find_all_experiments(base_dir=None, experiment_group=None):
    """Find both legacy and grouped experiment directories.

    Supported layouts:
        trained_models_mix/force_load/<experiment_id>/
        trained_models_mix/force_load/{std,gamma,arch,ratio,final_model}/<experiment_id>/
    """
    base_path = Path(base_dir) if base_dir is not None else _THIS_DIR / '..' / '..'
    requested_group = None if experiment_group is None else str(experiment_group).strip().lower()
    if requested_group is not None and requested_group not in EXPERIMENT_GROUPS:
        raise ValueError(
            f"Invalid experiment group '{requested_group}'. "
            f"Expected one of: {', '.join(EXPERIMENT_GROUPS)}."
        )
    experiments = []
    for trained_dir in base_path.glob('trained_models_*'):
        config_type = trained_dir.name.replace('trained_models_', '')
        for load_dir in trained_dir.iterdir():
            if not load_dir.is_dir():
                continue
            candidates = []
            for child in load_dir.iterdir():
                if not child.is_dir():
                    continue
                if _is_experiment_dir(child):
                    # Legacy layout: force_load/<experiment_id>.
                    candidates.append((child, 'std', False))
                    continue
                if child.name in EXPERIMENT_GROUPS:
                    for exp_dir in child.iterdir():
                        if _is_experiment_dir(exp_dir):
                            candidates.append((exp_dir, child.name, True))

            selected = {}
            for exp_dir, group, is_grouped in candidates:
                if requested_group is not None and group != requested_group:
                    continue
                key = (config_type, load_dir.name, exp_dir.name)
                previous = selected.get(key)
                if previous is None or is_grouped > previous[1]:
                    selected[key] = (exp_dir, is_grouped)

            for (exp_config_type, exp_load_type, exp_id), (exp_dir, _) in selected.items():
                experiments.append((exp_config_type, exp_load_type, exp_id, str(exp_dir)))
    return experiments


def load_experiment_results(exp_path):
    results = {
        'config': {},
        'train_loss': None, 'valid_loss': None,
        'train_mae': None, 'valid_mae': None,
        'test_L1': {},
        'test_metrics': {},
        'training_time': None,
    }
    exp_path = Path(exp_path)

    config_file = exp_path / 'config.py'
    if config_file.exists():
        config_dict = {}
        with open(config_file, 'r') as f:
            exec(f.read(), config_dict)
        results['config'] = config_dict
        results['training_time'] = config_dict.get('time', None)

    for filename, train_key, valid_key in [
        ('history_loss.txt', 'train_loss', 'valid_loss'),
        ('history_mae.txt', 'train_mae', 'valid_mae'),
    ]:
        f = exp_path / filename
        if f.exists():
            d = np.loadtxt(f)
            if d.ndim == 2:
                results[train_key] = d[:, 0]
                results[valid_key] = d[:, 1]

    for search_path in [exp_path / 'all_samples_test', exp_path / 'all_samples', exp_path]:
        if not search_path.exists():
            continue
        for test_file in search_path.glob('*L1_test*.txt'):
            stem = test_file.stem
            noise_level = float(stem.split('noise_')[-1].replace('p', '.')) if 'noise' in stem else 0.0
            if noise_level in results['test_L1']:
                continue
            L1_errors = np.loadtxt(test_file)
            std = np.std(L1_errors)
            results['test_L1'][noise_level] = {
                'mean': np.mean(L1_errors),
                'std': std,
                'variance': std ** 2,
            }

        for metrics_file in search_path.glob('metrics_test*.csv'):
            noise_level, eval_type = _parse_test_stem(metrics_file.stem.replace('metrics_', 'all_L1_'))
            metric_data = _read_metric_csv(metrics_file)
            if metric_data:
                results['test_metrics'][(noise_level, eval_type)] = metric_data

    return results


def _read_metric_csv(metrics_file):
    """Read the shared per-sample metric CSV written by ``Testing``."""
    try:
        data = np.genfromtxt(metrics_file, delimiter=',', names=True)
    except (OSError, ValueError):
        return None
    if data.size == 0 or data.dtype.names is None:
        return None
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    summary = {}
    for name in data.dtype.names:
        values = np.asarray(data[name], dtype=float)
        if name == 'sample':
            continue
        summary[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }
    return summary


def _load_specific_metric_stats(exp_path, eval_type='mix', noise_level=0.0, split='test'):
    exp_path = Path(exp_path)
    eval_suffix = f'_{str(eval_type).strip().lower()}' if str(eval_type).strip() else ''
    noise_suffix = '' if np.isclose(float(noise_level), 0.0) else f"_noise_{_format_decimal_token(noise_level)}"
    filename = f'metrics_{split}{eval_suffix}{noise_suffix}.csv'
    for search_path in [exp_path / f'all_samples_{split}', exp_path / 'all_samples', exp_path]:
        file_path = search_path / filename
        if file_path.exists():
            return _read_metric_csv(file_path)
    return None


def save_unified_metrics_table(experiments_data, output_dir=None, filename='unified_metrics.csv',
                               noise_levels=None, eval_types=None):
    """Save one long-form metrics table for U-Net and FNO experiments."""
    output_path = _resolve_output_dir(output_dir, 'comparison_results')
    noise_levels = [0, 2, 4, 6, 8, 10] if noise_levels is None else list(noise_levels)
    rows = []
    for (config_type, load_type, exp_id, exp_path), results in experiments_data.items():
        config = results.get('config') or {}
        dataset_type = str(config.get('dataset_type', 'mix' if config_type == 'mix' else config_type)).lower()
        if dataset_type != 'mix':
            continue
        requested_types = eval_types
        if requested_types is None:
            requested_types = config.get('eval_types', ['mix', 'bil', 'exp', 'grf'])
        if requested_types == 'all':
            requested_types = ['mix', 'bil', 'exp', 'grf']
        if isinstance(requested_types, str):
            requested_types = [requested_types]
        model_type = str(config.get('model_type', 'unet')).lower()
        backend = str(config.get('fno_backend', '')).lower() if model_type == 'fno' else ''
        if model_type == 'fno':
            architecture = (
                f"width={config.get('width')}, modes={config.get('modes1')}x{config.get('modes2')}, "
                f"layers={config.get('n_layers')}"
            )
        else:
            architecture = str(config.get('filters_list', []))
        for noise_level in noise_levels:
            for eval_type in requested_types:
                stats = _load_specific_metric_stats(
                    exp_path, eval_type=eval_type, noise_level=noise_level, split='test'
                )
                if not stats:
                    continue
                row = {
                    'config_type': config_type,
                    'dataset_type': dataset_type,
                    'model_type': model_type,
                    'fno_backend': backend,
                    'load_type': load_type,
                    'experiment_group': config.get('experiment_group', 'std'),
                    'exp_id': exp_id,
                    'method': config.get('method', 'Unknown'),
                    'label': ('FNO-' + backend if model_type == 'fno' else 'U-Net'),
                    'architecture': architecture,
                    'eval_type': eval_type,
                    'noise_level': noise_level,
                    'training_time': results.get('training_time'),
                }
                for metric_name, metric_stats in stats.items():
                    row[f'{metric_name}_mean'] = metric_stats['mean']
                    row[f'{metric_name}_std'] = metric_stats['std']
                rows.append(row)
    if not rows:
        print('No shared metrics found. Run the final-model test scripts first.')
        return None
    table = pd.DataFrame(rows).sort_values(
        ['noise_level', 'eval_type', 'label', 'exp_id']
    ).reset_index(drop=True)
    csv_file = output_path / filename
    table.to_csv(csv_file, index=False)
    print(f'Saved unified metrics table: {csv_file}')
    return str(csv_file)


def _load_specific_test_l1_stats(exp_path, eval_type='mix', noise_level=0.0, split='test'):
    exp_path = Path(exp_path)
    eval_suffix = f'_{str(eval_type).strip().lower()}' if str(eval_type).strip() else ''
    noise_suffix = '' if np.isclose(float(noise_level), 0.0) else f"_noise_{_format_decimal_token(noise_level)}"
    filename = f'all_L1_{split}{eval_suffix}{noise_suffix}.txt'

    for search_path in [exp_path / f'all_samples_{split}', exp_path / 'all_samples', exp_path]:
        file_path = search_path / filename
        if not file_path.exists():
            continue
        l1_errors = np.loadtxt(file_path)
        std = np.std(l1_errors)
        return {
            'mean': np.mean(l1_errors),
            'std': std,
            'variance': std ** 2,
        }
    return None


def generate_comparison_dataframe(experiments_data):
    columns = [
        'config_type',
        'dataset_type',
        'model_type',
        'fno_backend',
        'load_type',
        'exp_id',
        'method',
        'use_batch_norm',
        'architecture',
        'gamma',
        'training_time',
        'test_L1_mean',
        'test_L1_std',
        'test_L1_var',
        'test_L1_mean_noise_2',
        'test_L1_std_noise_2',
        'test_L1_mean_noise_4',
        'test_L1_std_noise_4',
    ]
    rows = []
    for (config_type, load_type, exp_id, exp_path), results in experiments_data.items():
        if not results['test_L1']:
            continue
        config = results.get('config') or {}
        method = config.get('method', 'Unknown')
        dataset_type = str(
            config.get('dataset_type', 'mix' if config_type == 'mix' else config_type)
        ).lower()
        model_type = str(config.get('model_type', 'unet')).lower()
        test_L1 = results['test_L1']
        if dataset_type == 'mix':
            test_results = _load_specific_test_l1_stats(exp_path, eval_type='mix', noise_level=0.0, split='test')
        else:
            test_results = None
        if test_results is None:
            test_results = test_L1.get(0.0) or (test_L1.get(min(test_L1.keys())) if test_L1 else {})
            test_results = test_results or {}

        row = {
            'config_type': config_type,
            'dataset_type': dataset_type,
            'model_type': model_type,
            'fno_backend': config.get('fno_backend', '') if model_type == 'fno' else '',
            'load_type': load_type,
            'exp_id': exp_id,
            'method': method,
            'use_batch_norm': bool(config.get('use_batch_norm', False)),
            'architecture': _architecture_label(config),
            'gamma': config.get('gamma', None),
            'training_time': results['training_time'],
            'test_L1_mean': test_results.get('mean', np.nan),
            'test_L1_std': test_results.get('std', np.nan),
            'test_L1_var': test_results.get('variance', np.nan),
        }
        for noise_level in [2.0, 4.0]:
            if dataset_type == 'mix':
                noise_stats = _load_specific_test_l1_stats(
                    exp_path, eval_type='mix', noise_level=noise_level, split='test'
                )
                if noise_stats is not None:
                    row[f'test_L1_mean_noise_{int(noise_level)}'] = noise_stats['mean']
                    row[f'test_L1_std_noise_{int(noise_level)}'] = noise_stats['std']
                    continue
            if noise_level in results['test_L1']:
                row[f'test_L1_mean_noise_{int(noise_level)}'] = results['test_L1'][noise_level]['mean']
                row[f'test_L1_std_noise_{int(noise_level)}'] = results['test_L1'][noise_level]['std']
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _architecture_label(config):
    """Return a comparable architecture label for U-Net and FNO configs."""
    if str(config.get('model_type', 'unet')).lower() == 'fno':
        return (
            f"width={config.get('width')}, modes={config.get('modes1')}x{config.get('modes2')}, "
            f"layers={config.get('n_layers')}"
        )
    return str(config.get('filters_list', []))


def generate_paraset_table(df):
    if df is None or df.empty or 'config_type' not in df.columns:
        return None
    if 'dataset_type' in df.columns:
        mix_df = df[df['dataset_type'].astype(str).str.lower() == 'mix'].copy()
    else:
        mix_df = df[df['config_type'] == 'mix'].copy()
    if mix_df.empty:
        return None
    lines = [
        '=' * 100,
        'Table 1: ParaSet Comparison - Architecture Sweep on Mix Test Set',
        '=' * 100,
        f"{'Experiment ID':<30} {'Architecture':<30} {'Time(s)':<10} {'Mean':<12} {'Variance':<12}",
        '-' * 100,
    ]
    for _, row in mix_df[['exp_id', 'architecture', 'training_time',
                          'test_L1_mean', 'test_L1_var']].iterrows():
        lines.append(f"{row['exp_id']:<30} {str(row['architecture']):<30} "
                     f"{row['training_time']:<10.2f} {row['test_L1_mean']:<12.6f} {row['test_L1_var']:<12.6f}")
    lines.append('=' * 100)
    return '\n'.join(lines)


def save_results(df, output_dir=None):
    output_path = _resolve_output_dir(output_dir, 'comparison_results')
    csv_file = output_path / 'all_experiments.csv'
    df.to_csv(csv_file, index=False)
    return str(csv_file)
