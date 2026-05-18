import numpy as np
import matplotlib.pyplot as plt

from .common import (
    _resolve_output_dir,
    _method_label,
    LOSS_COLORS,
    METHOD_DISPLAY,
    _legend_with_frame,
    _apply_axis_style,
    _add_panel_labels,
)


TRAIN_CURVE_COLOR = '#1f77b4'
VALID_CURVE_COLOR = '#ff7f0e'


def _loss_curve_styles(n_points, linewidth=1.5):
    train_style = {
        'linestyle': '-',
        'linewidth': linewidth,
    }
    valid_style = {
        'linestyle': '-',
        'linewidth': linewidth,
    }
    return train_style, valid_style


def plot_mae_history(experiments_data, output_dir=None, filename_suffix=''):
    output_path = _resolve_output_dir(output_dir, 'history')
    fig, ax = plt.subplots(figsize=(10, 6))

    for (config_type, load_type, exp_id, exp_path), results in experiments_data.items():
        if config_type != 'mix':
            continue
        if results['train_mae'] is None or results['valid_mae'] is None:
            continue
        method = results['config'].get('method', 'Unknown')
        gamma = results['config'].get('gamma', None)
        label = _method_label(method, gamma)
        color = LOSS_COLORS.get(method, '#333333')
        epochs = np.arange(1, len(results['train_mae']) + 1)
        ax.plot(epochs, results['train_mae'], color=color, linestyle='-', label=f'{label}, Training set')
        ax.plot(epochs, results['valid_mae'], color=color, linestyle='--', label=f'{label}, Validation set')
        best_ep = int(np.argmin(results['valid_mae'])) + 1
        ax.axvline(x=best_ep, color=color, linestyle=':', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('MAE Training History by Loss Function')
    _legend_with_frame(ax, fontsize=9)
    _apply_axis_style(ax)
    plt.tight_layout()
    fig_file = output_path / f'mae_history{filename_suffix}.png'
    plt.savefig(fig_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved MAE history plot: {fig_file}")
    return str(fig_file)


def plot_loss_history(experiments_data, output_dir=None, filename_suffix=''):
    output_path = _resolve_output_dir(output_dir, 'history')
    valid_exps = [
        (key, results) for key, results in experiments_data.items()
        if key[0] == 'mix' and results['train_loss'] is not None and results['valid_loss'] is not None
    ]
    if not valid_exps:
        print("No loss history data available")
        return None

    n = len(valid_exps)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for idx, ((config_type, load_type, exp_id, exp_path), results) in enumerate(valid_exps):
        ax = axes[idx // n_cols][idx % n_cols]
        method = results['config'].get('method', 'Unknown')
        gamma = results['config'].get('gamma', None)
        label = _method_label(method, gamma)
        epochs = np.arange(1, len(results['train_loss']) + 1)
        train_style, valid_style = _loss_curve_styles(len(epochs))
        ax.plot(epochs, results['train_loss'], color=TRAIN_CURVE_COLOR, label='Train.', **train_style)
        ax.plot(epochs, results['valid_loss'], color=VALID_CURVE_COLOR, label='Val.', **valid_style)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(label)
        if results.get('valid_mae') is not None:
            best_ep = int(np.argmin(results['valid_mae'])) + 1
        elif results.get('valid_loss') is not None:
            best_ep = int(np.argmin(results['valid_loss'])) + 1
        else:
            best_ep = None
        if best_ep is not None:
            ax.axvline(x=best_ep, color='#666666', linestyle=':', linewidth=0.8, label='best epoch')
        _legend_with_frame(ax, fontsize=9)
        _apply_axis_style(ax)

    for idx in range(n, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    axes_flat = [axes[idx // n_cols][idx % n_cols] for idx in range(n)]
    _add_panel_labels(axes_flat)
    plt.tight_layout()
    fig_file = output_path / f'loss_history{filename_suffix}.png'
    plt.savefig(fig_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved loss history plot: {fig_file}")
    return str(fig_file)


def plot_history_row(experiments_data, target_methods=None, output_dir=None,
                     filename='loss_history_row.png', dpi=600):
    output_path = _resolve_output_dir(output_dir, 'history')
    if target_methods is None:
        target_methods = ['MSE', 'LocResloss', 'GloResloss']

    best_by_method = {}
    for (config_type, load_type, exp_id, exp_path), results in experiments_data.items():
        if config_type != 'mix':
            continue
        config = results.get('config') or {}
        method = config.get('method', '')
        if method not in target_methods:
            continue
        if results.get('train_loss') is None or results.get('valid_loss') is None:
            continue

        if results.get('valid_mae') is not None:
            score = float(np.min(results['valid_mae']))
        else:
            score = float(np.min(results['valid_loss']))

        if method not in best_by_method or score < best_by_method[method]['score']:
            best_by_method[method] = {'score': score, 'results': results}

    selected_methods = [m for m in target_methods if m in best_by_method]
    if len(selected_methods) != len(target_methods):
        missing = [m for m in target_methods if m not in selected_methods]
        print("Missing required methods for history-row plot:")
        for method in missing:
            print(f"  - {method}")
        return None

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.2), squeeze=False)

    for col, method in enumerate(target_methods):
        ax = axes[0, col]
        results = best_by_method[method]['results']
        train_loss = results['train_loss']
        valid_loss = results['valid_loss']
        method_name = METHOD_DISPLAY.get(method, method)
        if results.get('valid_mae') is not None:
            best_ep = int(np.argmin(results['valid_mae'])) + 1
        else:
            best_ep = int(np.argmin(valid_loss)) + 1
        epochs = np.arange(1, best_ep + 1)
        train_style, valid_style = _loss_curve_styles(len(epochs))

        ax.plot(epochs, train_loss[:best_ep], color=TRAIN_CURVE_COLOR, label='Train.', **train_style)
        ax.plot(epochs, valid_loss[:best_ep], color=VALID_CURVE_COLOR, label='Val.', **valid_style)
        ax.set_yscale('log')
        ax.set_title(method_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        _legend_with_frame(ax, fontsize=9)
        _apply_axis_style(ax)

    ax = axes[0, 3]
    val_mae_curves = []
    for method in target_methods:
        results = best_by_method[method]['results']
        valid_mae = results.get('valid_mae')
        if valid_mae is None:
            continue
        best_ep = int(np.argmin(valid_mae)) + 1
        epochs = np.arange(1, best_ep + 1)
        color = LOSS_COLORS.get(method, '#333333')
        method_name = METHOD_DISPLAY.get(method, method)
        val_curve = valid_mae[:best_ep]
        ax.plot(epochs, val_curve, color=color, linestyle='-',
                linewidth=1.5, label=method_name)
        val_mae_curves.append((epochs, val_curve, color))

    ax.set_title('MAE Comparison on Validation Set')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    _legend_with_frame(ax, fontsize=9)

    if val_mae_curves:
        end_epochs = [int(epochs[-1]) for epochs, _, _ in val_mae_curves if len(epochs) > 0]
        common_end = min(end_epochs) if end_epochs else None
        if common_end is not None:
            zoom_start = max(1, int(common_end * 0.55))
            if zoom_start >= common_end:
                zoom_start = max(1, common_end - 1)

        zoom_values = []
        for epochs, curve, _ in val_mae_curves:
            if common_end is None:
                continue
            mask = (epochs >= zoom_start) & (epochs <= common_end)
            if np.any(mask):
                zoom_values.extend(curve[mask])

        if zoom_values:
            y_min = float(np.min(zoom_values))
            y_max = float(np.max(zoom_values))
            y_span = y_max - y_min
            y_pad = 0.08 * y_span if y_span > 0 else max(1e-4, 0.05 * max(abs(y_max), 1.0))
            axins = ax.inset_axes([0.52, 0.16, 0.44, 0.40])
            for epochs, curve, color in val_mae_curves:
                if common_end is None:
                    continue
                mask = (epochs >= zoom_start) & (epochs <= common_end)
                if np.any(mask):
                    axins.plot(epochs[mask], curve[mask], color=color, linestyle='-', linewidth=1.2)
            axins.set_xlim(zoom_start, common_end)
            axins.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
            axins.tick_params(labelsize=7, direction='in', top=False, right=False)
            axins.grid(False)
            for spine in axins.spines.values():
                spine.set_linewidth(0.8)
            ax.indicate_inset_zoom(axins, edgecolor='black', alpha=1.0)

    _apply_axis_style(ax)

    _add_panel_labels([axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3]])
    plt.tight_layout()
    fig_file = output_path / filename
    plt.savefig(fig_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved history-row plot: {fig_file}")
    return str(fig_file)


__all__ = [
    'plot_mae_history',
    'plot_loss_history',
    'plot_history_row',
]
