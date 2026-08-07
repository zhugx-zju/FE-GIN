import csv
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

try:
    from .common import (
        METHOD_DISPLAY,
        RELATIVE_ERROR_COLORBAR_MAX_PCT,
        align_panel_by_noise,
        build_unet_file_stem,
        build_noisy_input_by_noise,
        build_asm_context,
        candidate_data_dirs,
        find_all_experiments,
        find_available_mix_gammas,
        find_best_experiment,
        format_decimal_token,
        gamma_close,
        load_unet_from_experiment,
        load_experiment_results,
        load_sample,
        normalize_mix_gamma_map,
        parse_optional_float,
        predict_unet_panel,
        resolve_case_mix_gamma,
        resolve_variant_output_dir,
        resolve_output_root,
        save_measured_displacement,
        select_requested_mix_gamma,
        is_mix_method,
    )
except ImportError:
    from common import (
        METHOD_DISPLAY,
        RELATIVE_ERROR_COLORBAR_MAX_PCT,
        align_panel_by_noise,
        build_unet_file_stem,
        build_noisy_input_by_noise,
        build_asm_context,
        candidate_data_dirs,
        find_all_experiments,
        find_available_mix_gammas,
        find_best_experiment,
        format_decimal_token,
        gamma_close,
        load_unet_from_experiment,
        load_experiment_results,
        load_sample,
        normalize_mix_gamma_map,
        parse_optional_float,
        predict_unet_panel,
        resolve_case_mix_gamma,
        resolve_variant_output_dir,
        resolve_output_root,
        save_measured_displacement,
        select_requested_mix_gamma,
        is_mix_method,
    )


def _extract_saved_gamma(unet_data):
    case_cfg = unet_data.get('case_config', {})
    if isinstance(case_cfg, dict):
        for key in ['unet_gamma_used', 'unet_gamma_requested', 'unet_gamma']:
            if key in case_cfg:
                return parse_optional_float(case_cfg.get(key))
    for key in ['unet_gamma_used', 'unet_gamma_requested', 'unet_gamma']:
        if key in unet_data:
            return parse_optional_float(unet_data.get(key))
    return None


def _normalize_case_cfg(cfg, case_cfg):
    data_type = case_cfg.get('dataset', case_cfg.get('data_type'))
    if data_type is None:
        raise KeyError("Each UNet case must provide 'dataset'.")

    sample_index_map = cfg.get('sample_index', {})
    if 'noise_level' in case_cfg:
        noise_levels = [float(case_cfg['noise_level'])]
    else:
        noise_levels = list(case_cfg.get('noise_levels', cfg['noise_levels']))

    default_method = cfg['unet_methods'][0] if cfg.get('unet_methods') else 'MSE'

    return {
        'dataset': data_type,
        'sample_index': int(case_cfg.get('sample_index', sample_index_map.get(data_type, 0))),
        'noise_levels': noise_levels,
        'unet_method': case_cfg.get('unet_method', case_cfg.get('method', default_method)),
        'unet_config_type': case_cfg.get('unet_config_type', cfg['unet_config_type']),
        'unet_load_type': case_cfg.get('unet_load_type', cfg['unet_load_type']),
        'unet_use_batch_norm': bool(case_cfg.get('unet_use_batch_norm', cfg.get('unet_use_batch_norm', False))),
        'unet_architecture': case_cfg.get('unet_architecture', cfg['unet_architecture']),
        'exp_path': case_cfg.get('exp_path', None),
        'mix_gamma': case_cfg.get('mix_gamma', cfg.get('mix_gamma', None)),
        'mix_gamma_by_method': normalize_mix_gamma_map(
            case_cfg.get('mix_gamma_by_method', cfg.get('mix_gamma_by_method', None))
        ),
        'strict_mix_gamma': bool(case_cfg.get('strict_mix_gamma', cfg.get('strict_mix_gamma', True))),
    }


def _resolve_case_methods(cfg, case_cfg):
    if case_cfg.get('unet_methods', None) is not None:
        return [str(method) for method in case_cfg['unet_methods']]
    if case_cfg.get('unet_method', None) is not None:
        return [str(case_cfg['unet_method'])]
    if case_cfg.get('method', None) is not None:
        return [str(case_cfg['method'])]
    return [str(method) for method in cfg.get('unet_methods', ['MSE'])]


def _resolve_case_exp_path(case_cfg, method):
    exp_paths = case_cfg.get('exp_paths', None)
    if isinstance(exp_paths, dict):
        return exp_paths.get(method, case_cfg.get('exp_path', None))
    return case_cfg.get('exp_path', None)


def _model_cache_key(case_cfg):
    exp_path = case_cfg.get('exp_path', None)
    if exp_path:
        return ('exp_path', os.path.abspath(exp_path))
    return (
        'auto',
        case_cfg['unet_config_type'],
        case_cfg['unet_load_type'],
        case_cfg['unet_use_batch_norm'],
        case_cfg['unet_architecture'],
        case_cfg['unet_method'],
        case_cfg.get('unet_gamma', None),
    )


def _resolve_model(project_root, experiments, model_cache, case_cfg):
    cache_key = _model_cache_key(case_cfg)
    if cache_key in model_cache:
        return model_cache[cache_key]

    requested_gamma = parse_optional_float(case_cfg.get('unet_gamma', None))
    strict_mix_gamma = bool(case_cfg.get('strict_mix_gamma', True))

    exp_path = case_cfg.get('exp_path', None)
    if exp_path:
        resolved_exp_path = exp_path
        if not os.path.isabs(resolved_exp_path):
            resolved_exp_path = os.path.abspath(os.path.join(project_root, resolved_exp_path))
        if not os.path.isdir(resolved_exp_path):
            raise FileNotFoundError(f'UNet experiment folder not found: {resolved_exp_path}')

        results = load_experiment_results(resolved_exp_path)
        exp_cfg = results.get('config') or {}
        exp_id = os.path.basename(resolved_exp_path.rstrip('\\/'))
        exp_use_batch_norm = bool(exp_cfg.get('use_batch_norm', False))
        if exp_use_batch_norm != bool(case_cfg['unet_use_batch_norm']):
            raise ValueError(
                f"Explicit exp_path use_batch_norm mismatch: requested={case_cfg['unet_use_batch_norm']}, "
                f"exp={exp_use_batch_norm}, exp_path={resolved_exp_path}"
            )

        if is_mix_method(case_cfg['unet_method']) and requested_gamma is not None:
            used_gamma = parse_optional_float(exp_cfg.get('gamma', None))
            if used_gamma is None or not gamma_close(used_gamma, requested_gamma):
                msg = (
                    f"Explicit exp_path gamma mismatch for method={case_cfg['unet_method']}: "
                    f"requested_gamma={requested_gamma}, exp_gamma={used_gamma}, exp_path={resolved_exp_path}"
                )
                if strict_mix_gamma:
                    raise ValueError(msg)
                print(f"Warning: {msg}")
    else:
        best = find_best_experiment(
            experiments=experiments,
            config_type=case_cfg['unet_config_type'],
            load_type=case_cfg['unet_load_type'],
            architecture=case_cfg['unet_architecture'],
            method=case_cfg['unet_method'],
            gamma=requested_gamma,
            use_batch_norm=case_cfg['unet_use_batch_norm'],
        )
        if (
            best is None
            and is_mix_method(case_cfg['unet_method'])
            and requested_gamma is not None
            and not strict_mix_gamma
        ):
            best = find_best_experiment(
                experiments=experiments,
                config_type=case_cfg['unet_config_type'],
                load_type=case_cfg['unet_load_type'],
                architecture=case_cfg['unet_architecture'],
                method=case_cfg['unet_method'],
                gamma=None,
                use_batch_norm=case_cfg['unet_use_batch_norm'],
            )
            if best is not None:
                print(
                    f"Warning: requested gamma={requested_gamma} for method={case_cfg['unet_method']} "
                    "not found; strict_mix_gamma=False so fallback to best available gamma."
                )
        if best is None:
            gamma_text = ''
            if is_mix_method(case_cfg['unet_method']) and requested_gamma is not None:
                available = find_available_mix_gammas(
                    experiments=experiments,
                    config_type=case_cfg['unet_config_type'],
                    load_type=case_cfg['unet_load_type'],
                    architecture=case_cfg['unet_architecture'],
                    method=case_cfg['unet_method'],
                    use_batch_norm=case_cfg['unet_use_batch_norm'],
                )
                if available:
                    gamma_text = (
                        f", requested_gamma={requested_gamma}, "
                        f"available_gammas={available}"
                    )
                else:
                    gamma_text = f", requested_gamma={requested_gamma}, available_gammas=[]"
            raise FileNotFoundError(
                "No UNet experiment found for "
                f"method={case_cfg['unet_method']}, "
                f"config_type={case_cfg['unet_config_type']}, "
                f"load_type={case_cfg['unet_load_type']}, "
                f"architecture={case_cfg['unet_architecture']}"
                f"{gamma_text}"
            )
        (_, _, exp_id, resolved_exp_path), results, _ = best
        exp_cfg = results.get('config') or {}

    net, device = load_unet_from_experiment(
        case_cfg['unet_config_type'],
        case_cfg['unet_load_type'],
        resolved_exp_path,
        exp_cfg,
    )
    used_gamma = parse_optional_float(exp_cfg.get('gamma', None))
    file_stem = build_unet_file_stem(
        case_cfg['unet_method'],
        used_gamma if used_gamma is not None else requested_gamma,
    )
    model_info = {
        'net': net,
        'device': device,
        'exp_id': exp_id,
        'exp_path': resolved_exp_path,
        'exp_cfg': exp_cfg,
        'unet_gamma_requested': requested_gamma,
        'unet_gamma_used': used_gamma,
        'file_stem': file_stem,
    }
    model_cache[cache_key] = model_info
    return model_info


def _write_case_config(save_dir, case_cfg, resolved_idx, row, file_stem, model_info):
    config_path = os.path.join(save_dir, f'{file_stem}_config.py')
    ordered_items = [
        ('dataset', case_cfg['dataset']),
        ('sample_index', case_cfg['sample_index']),
        ('resolved_sample_index', resolved_idx),
        ('noise_levels', list(case_cfg['noise_levels'])),
        ('noise_level', float(row['noise'])),
        ('unet_method', case_cfg['unet_method']),
        ('unet_config_type', case_cfg['unet_config_type']),
        ('unet_load_type', case_cfg['unet_load_type']),
        ('unet_use_batch_norm', case_cfg['unet_use_batch_norm']),
        ('unet_architecture', case_cfg['unet_architecture']),
        ('unet_gamma_requested', case_cfg.get('unet_gamma', None)),
        ('unet_gamma_used', model_info.get('unet_gamma_used', None)),
        ('exp_id', model_info['exp_id']),
        ('exp_path', model_info['exp_path']),
        ('elapsed_time', float(row.get('elapsed_time', 0.0))),
        ('l1', float(row['l1'])),
        ('mae_pct', float(row['mae_pct'])),
    ]

    with open(config_path, 'w', encoding='utf-8') as f:
        for key, value in ordered_items:
            f.write(f"{key} = {repr(value)}\n")

    return config_path
def load_saved_unet_panel_data(
    output_root,
    data_type,
    sample_index,
    noise_levels,
    methods,
    mix_gamma=None,
    mix_gamma_by_method=None,
    strict_mix_gamma=True,
):
    method_to_panel = {}
    resolved_idx = None

    normalized_mix_gamma_map = normalize_mix_gamma_map(mix_gamma_by_method)

    for method in methods:
        requested_gamma = select_requested_mix_gamma(
            method=method,
            mix_gamma=mix_gamma,
            mix_gamma_by_method=normalized_mix_gamma_map,
        )
        primary_stem = build_unet_file_stem(method, requested_gamma)
        stem_candidates = [primary_stem]
        legacy_stem = build_unet_file_stem(method, None)
        if legacy_stem not in stem_candidates and (requested_gamma is None or not strict_mix_gamma):
            stem_candidates.append(legacy_stem)

        panel_data = []

        for noise in noise_levels:
            noise_tag = format_decimal_token(noise)
            for data_dir in candidate_data_dirs(output_root, data_type, sample_index):
                found_for_noise = False
                for stem in stem_candidates:
                    pkl_file = os.path.join(data_dir, f'noise_{noise_tag}', f'{stem}_results.pkl')
                    if not os.path.exists(pkl_file):
                        continue
                    with open(pkl_file, 'rb') as f:
                        unet_data = pickle.load(f)
                    case_cfg = unet_data.get('case_config', {})
                    if not isinstance(case_cfg, dict):
                        case_cfg = {}
                    requested_idx = case_cfg.get('sample_index', unet_data.get('sample_index', -1))
                    saved_idx = int(unet_data.get('sample_index', sample_index))
                    if int(requested_idx) != int(sample_index) and saved_idx != int(sample_index):
                        continue

                    if is_mix_method(method) and requested_gamma is not None:
                        saved_gamma = _extract_saved_gamma(unet_data)
                        if saved_gamma is not None and not gamma_close(saved_gamma, requested_gamma):
                            continue
                        if saved_gamma is None and strict_mix_gamma and stem == legacy_stem:
                            continue

                    if resolved_idx is None:
                        resolved_idx = saved_idx
                    for item in unet_data.get('panel_data', []):
                        row = dict(item)
                        if 'unet_gamma_used' not in row:
                            row['unet_gamma_used'] = unet_data.get(
                                'unet_gamma_used',
                                _extract_saved_gamma(unet_data),
                            )
                        if 'unet_gamma_requested' not in row:
                            row['unet_gamma_requested'] = unet_data.get('unet_gamma_requested', None)
                        if 'file_stem' not in row:
                            row['file_stem'] = unet_data.get('file_stem', stem)
                        panel_data.append(row)
                    found_for_noise = True
                    break
                if found_for_noise:
                    break

        if panel_data:
            method_to_panel[method] = panel_data

    return method_to_panel, resolved_idx


def _save_case_outputs(data_dir, data_type, resolved_idx, panel_data, case_cfg, model_info, summary_rows):
    saved_dirs = []
    file_stem = model_info.get(
        'file_stem',
        build_unet_file_stem(case_cfg['unet_method'], case_cfg.get('unet_gamma', None)),
    )

    for row in panel_data:
        noise = float(row['noise'])
        noise_tag = format_decimal_token(noise)
        noise_dir = os.path.join(data_dir, f'noise_{noise_tag}')
        os.makedirs(noise_dir, exist_ok=True)

        row_case_cfg = dict(case_cfg)
        row_case_cfg['noise_levels'] = [noise]

        with open(os.path.join(noise_dir, f'{file_stem}_results.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'data_type': data_type,
                    'sample_index': resolved_idx,
                    'noise_levels': [noise],
                    'method': case_cfg['unet_method'],
                    'unet_gamma_requested': case_cfg.get('unet_gamma', None),
                    'unet_gamma_used': model_info.get('unet_gamma_used', None),
                    'panel_data': [row],
                    'case_config': row_case_cfg,
                    'exp_id': model_info['exp_id'],
                    'exp_path': model_info['exp_path'],
                    'file_stem': file_stem,
                },
                f,
            )

        np.savez(
            os.path.join(noise_dir, f'{file_stem}.npz'),
            target=row['target'],
            pred=row['pred'],
            rel_err=row['rel_err'],
        )

        _write_case_config(
            save_dir=noise_dir,
            case_cfg=row_case_cfg,
            resolved_idx=resolved_idx,
            row=row,
            file_stem=file_stem,
            model_info=model_info,
        )

        summary_rows.append({
            'data_type': data_type,
            'sample_index': resolved_idx,
            'method': case_cfg['unet_method'],
            'unet_gamma_requested': case_cfg.get('unet_gamma', None),
            'unet_gamma_used': model_info.get('unet_gamma_used', None),
            'noise': noise,
            'l1': row['l1'],
            'mae_pct': row['mae_pct'],
            'elapsed_time': row.get('elapsed_time', ''),
            'exp_id': model_info['exp_id'],
        })
        saved_dirs.append(noise_dir)

    return saved_dirs


def _save_displacement_outputs(
    data_dir,
    data_type,
    resolved_idx,
    noise_levels,
    sample_seed,
    noisy_input_by_noise,
    mesh_info=None,
    contour_fn=None,
    dpi=600,
):
    saved_dirs = []
    for noise in noise_levels:
        noise_key = float(noise)
        if noise_key not in noisy_input_by_noise:
            continue
        noise_dir = os.path.join(data_dir, f'noise_{format_decimal_token(noise_key)}')
        save_measured_displacement(
            noise_dir=noise_dir,
            noisy_input=noisy_input_by_noise[noise_key],
            data_type=data_type,
            sample_index=resolved_idx,
            noise_level=noise_key,
            sample_seed=sample_seed,
            dpi=dpi,
            mesh_info=mesh_info,
            contour_fn=contour_fn,
        )
        saved_dirs.append(noise_dir)
    return saved_dirs


def _first_available_target(method_to_panel_data, methods_order, noise_order):
    for method in methods_order:
        if method not in method_to_panel_data:
            continue
        noise_map = align_panel_by_noise(method_to_panel_data[method])
        for noise in noise_order:
            row = noise_map.get(float(noise))
            if row is not None:
                return row['target']
    return None


def _row_tag(row_idx):
    return f"({chr(ord('a') + row_idx)})"


def _draw_field_panel(ax, values, mesh_info, contour_fn, cmap, vmin=None, vmax=None):
    if contour_fn is not None and mesh_info is not None:
        im = contour_fn(mesh_info.plot_x, mesh_info.plot_y, values, ax, levels=128, cmap=cmap)
        if vmin is not None and vmax is not None:
            im.set_clim(vmin, vmax)
        ax.axis('equal')
        ax.axis('off')
    else:
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    return im


def _add_row_labels(fig, first_col_axes, noise_order, x_pad=0.012):
    for row_idx, ax in first_col_axes.items():
        bbox = ax.get_position()
        noise = float(noise_order[row_idx])
        fig.text(
            bbox.x0 - x_pad,
            bbox.y0 + 0.5 * bbox.height,
            f"{_row_tag(row_idx)} Noise {noise:g}%",
            ha='right',
            va='center',
            fontsize=10,
            fontweight='bold',
        )


def _add_panel_title(fig, ax, text, y_pad=0.004):
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + 0.5 * bbox.width,
        bbox.y1 + y_pad,
        text,
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
    )


def _plot_prediction_figure(methods_order, noise_order, method_to_panel_data, save_path, title, mesh_info, contour_fn, dpi):
    methods = [m for m in methods_order if m in method_to_panel_data]
    if not methods:
        return

    target = _first_available_target(method_to_panel_data, methods, noise_order)
    if target is None:
        return

    n_noise = len(noise_order)
    n_methods = len(methods)
    vmin = float(np.min(target))
    vmax = float(np.max(target))

    fig = plt.figure(figsize=(2.55 * n_methods + 0.55, 2.45 * n_noise + 0.60))
    gs = fig.add_gridspec(n_noise, n_methods, wspace=0.045, hspace=0.055)

    im_ref = None
    axes_pred = []
    first_col_axes = {}
    for row_idx, noise in enumerate(noise_order):
        for col_idx, method in enumerate(methods):
            noise_map = align_panel_by_noise(method_to_panel_data[method])
            if float(noise) not in noise_map:
                continue
            ax = fig.add_subplot(gs[row_idx, col_idx])
            pred = noise_map[float(noise)]['pred']
            im = _draw_field_panel(ax, pred, mesh_info, contour_fn, 'viridis', vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(METHOD_DISPLAY.get(method, method), fontsize=11, fontweight='bold', pad=7)
            if row_idx not in first_col_axes:
                first_col_axes[row_idx] = ax
            axes_pred.append(ax)
            im_ref = im

    if im_ref is None or not axes_pred:
        plt.close(fig)
        return

    # fig.suptitle(title, fontsize=14, fontweight='bold', y=0.985)
    fig.subplots_adjust(left=0.115, right=0.895, bottom=0.035, top=0.935, wspace=0.045, hspace=0.055)
    _add_row_labels(fig, first_col_axes, noise_order, x_pad=0.014)
    cax = fig.add_axes([0.915, 0.14, 0.016, 0.74])
    colorbar = fig.colorbar(im_ref, cax=cax)
    colorbar.set_label('Modulus (MPa)', fontsize=10, fontweight='bold')
    colorbar.ax.tick_params(labelsize=9)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def _plot_error_figure(methods_order, noise_order, method_to_panel_data, save_path, title, mesh_info, contour_fn, dpi):
    methods = [m for m in methods_order if m in method_to_panel_data]
    if not methods:
        return

    n_noise = len(noise_order)
    n_methods = len(methods)
    vmax = RELATIVE_ERROR_COLORBAR_MAX_PCT

    fig = plt.figure(figsize=(2.55 * n_methods + 0.55, 2.45 * n_noise + 0.60))
    gs = fig.add_gridspec(n_noise, n_methods, wspace=0.045, hspace=0.055)

    im_ref = None
    axes_err = []
    first_col_axes = {}
    for row_idx, noise in enumerate(noise_order):
        for col_idx, method in enumerate(methods):
            noise_map = align_panel_by_noise(method_to_panel_data[method])
            if float(noise) not in noise_map:
                continue
            ax = fig.add_subplot(gs[row_idx, col_idx])
            err = noise_map[float(noise)]['rel_err']
            im = _draw_field_panel(ax, err, mesh_info, contour_fn, 'Blues', vmin=0, vmax=vmax)
            if row_idx == 0:
                ax.set_title(METHOD_DISPLAY.get(method, method), fontsize=11, fontweight='bold', pad=7)
            if col_idx == 0:
                first_col_axes[row_idx] = ax
            ax.text(
                0.03,
                0.96,
                f"$L_1$={noise_map[float(noise)]['l1']:.2e}",
                transform=ax.transAxes,
                va='top',
                fontsize=8,
                color='white',
                bbox={'facecolor': 'black', 'alpha': 0.35, 'edgecolor': 'none', 'pad': 2},
            )
            axes_err.append(ax)
            im_ref = im

    if im_ref is not None and axes_err:
        cax = fig.add_axes([0.915, 0.14, 0.016, 0.74])
        colorbar = fig.colorbar(im_ref, cax=cax)
        colorbar.set_label('Relative error (%)', fontsize=10, fontweight='bold')
        colorbar.ax.tick_params(labelsize=9)
    # fig.suptitle(title, fontsize=14, fontweight='bold', y=0.985)
    fig.subplots_adjust(left=0.115, right=0.895, bottom=0.035, top=0.935, wspace=0.045, hspace=0.055)
    _add_row_labels(fig, first_col_axes, noise_order, x_pad=0.014)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def _plot_unet_comparison(
    output_root,
    project_root,
    cfg,
    data_type,
    sample_index,
    noise_levels,
    methods_order,
    method_to_panel_data,
    asm_ctx=None,
):
    if not method_to_panel_data:
        return

    if asm_ctx is None:
        asm_ctx = build_asm_context(
            project_root=project_root,
            nodesx=cfg['nodesx'],
            nodesy=cfg['nodesy'],
            asm_gamma=cfg.get('asm_gamma', None),
            asm_max_iter=cfg['asm_max_iter'],
            asm_ftol=cfg['asm_ftol'],
            asm_gtol=cfg['asm_gtol'],
        )
    out_dir = os.path.join(output_root, f"sample_{sample_index}", data_type, 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    filename_suffix = '_GN' if bool(cfg.get('unet_use_batch_norm', False)) else ''

    _plot_prediction_figure(
        methods_order=methods_order,
        noise_order=noise_levels,
        method_to_panel_data=method_to_panel_data,
        save_path=os.path.join(out_dir, f'unet_prediction_sample_{sample_index}_{data_type}{filename_suffix}.png'),
        title=f'{data_type.upper()} Sample {sample_index} - UNet Prediction',
        mesh_info=asm_ctx['mesh'],
        contour_fn=asm_ctx['create_smooth_contour'],
        dpi=cfg['dpi'],
    )
    _plot_error_figure(
        methods_order=methods_order,
        noise_order=noise_levels,
        method_to_panel_data=method_to_panel_data,
        save_path=os.path.join(out_dir, f'unet_error_sample_{sample_index}_{data_type}{filename_suffix}.png'),
        title=f'{data_type.upper()} Sample {sample_index} - UNet Relative Error',
        mesh_info=asm_ctx['mesh'],
        contour_fn=asm_ctx['create_smooth_contour'],
        dpi=cfg['dpi'],
    )


def _write_summary(output_root, summary_rows, use_batch_norm=False):
    summary_suffix = '_GN' if bool(use_batch_norm) else ''
    summary_file = os.path.join(output_root, f'unet_summary{summary_suffix}.csv')
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'data_type',
                'sample_index',
                'method',
                'unet_gamma_requested',
                'unet_gamma_used',
                'noise',
                'l1',
                'mae_pct',
                'elapsed_time',
                'exp_id',
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved UNet inference summary: {summary_file}")


def run_unet_inference_cases(project_root, cfg, cases):
    output_dir = resolve_variant_output_dir(
        cfg['output_dir'],
        use_batch_norm=cfg.get('unet_use_batch_norm', False),
    )
    output_root = resolve_output_root(project_root, output_dir)
    os.makedirs(output_root, exist_ok=True)

    summary_rows = []
    model_cache = {}
    experiments = find_all_experiments(
        experiment_group=cfg.get('unet_experiment_group', 'std'),
    )

    for raw_case in cases:
        case_start_time = time.time()
        base_case_cfg = _normalize_case_cfg(cfg, raw_case)
        methods = _resolve_case_methods(cfg, raw_case)
        data_type = base_case_cfg['dataset']
        sample_index = base_case_cfg['sample_index']

        input_sample, target_sample, resolved_idx = load_sample(
            config_type=base_case_cfg['unet_config_type'],
            load_type=base_case_cfg['unet_load_type'],
            data_type=data_type,
            sample_index=sample_index,
        )

        data_dir = os.path.join(output_root, f"sample_{resolved_idx}", data_type)
        os.makedirs(data_dir, exist_ok=True)

        asm_ctx = build_asm_context(
            project_root=project_root,
            nodesx=cfg['nodesx'],
            nodesy=cfg['nodesy'],
            asm_gamma=cfg.get('asm_gamma', None),
            asm_max_iter=cfg['asm_max_iter'],
            asm_ftol=cfg['asm_ftol'],
            asm_gtol=cfg['asm_gtol'],
        )
        noisy_input_by_noise = build_noisy_input_by_noise(
            input_sample=input_sample,
            noise_levels=base_case_cfg['noise_levels'],
            sample_seed=resolved_idx,
        )

        saved_dirs = []
        saved_dirs.extend(
            _save_displacement_outputs(
                data_dir=data_dir,
                data_type=data_type,
                resolved_idx=resolved_idx,
                noise_levels=base_case_cfg['noise_levels'],
                sample_seed=resolved_idx,
                noisy_input_by_noise=noisy_input_by_noise,
                mesh_info=asm_ctx['mesh'],
                contour_fn=asm_ctx['create_smooth_contour'],
                dpi=cfg['dpi'],
            )
        )
        method_to_panel_data = {}
        for method in methods:
            case_cfg = dict(base_case_cfg)
            case_cfg['unet_method'] = method
            case_cfg['exp_path'] = _resolve_case_exp_path(raw_case, method)
            case_cfg['unet_gamma'] = resolve_case_mix_gamma(base_case_cfg, method)

            model_info = _resolve_model(project_root, experiments, model_cache, case_cfg)
            print(
                f"[UNet] dataset={data_type}, sample_index={sample_index}, "
                f"noise_levels={case_cfg['noise_levels']}, method={case_cfg['unet_method']}, "
                f"gamma={model_info.get('unet_gamma_used', None)}, exp_id={model_info['exp_id']}"
            )

            panel_data = predict_unet_panel(
                net=model_info['net'],
                device=model_info['device'],
                input_sample=input_sample,
                target_sample=target_sample,
                noise_levels=case_cfg['noise_levels'],
                sample_seed=resolved_idx,
                noisy_input_by_noise=noisy_input_by_noise,
            )
            method_to_panel_data[method] = panel_data

            saved_dirs.extend(
                _save_case_outputs(
                    data_dir=data_dir,
                    data_type=data_type,
                    resolved_idx=resolved_idx,
                    panel_data=panel_data,
                    case_cfg=case_cfg,
                    model_info=model_info,
                    summary_rows=summary_rows,
                )
            )

        _plot_unet_comparison(
            output_root=output_root,
            project_root=project_root,
            cfg=cfg,
            data_type=data_type,
            sample_index=resolved_idx,
            noise_levels=base_case_cfg['noise_levels'],
            methods_order=methods,
            method_to_panel_data=method_to_panel_data,
            asm_ctx=asm_ctx,
        )
        total_time = time.time() - case_start_time
        print(
            f"Saved UNet results for dataset={data_type}, sample_index={resolved_idx}, "
            f"noise_levels={base_case_cfg['noise_levels']} (case_time={total_time:.2f}s)"
        )
        for save_dir in sorted(set(saved_dirs)):
            print(f"Saved UNet case outputs: {save_dir}")

    _write_summary(output_root, summary_rows, use_batch_norm=cfg.get('unet_use_batch_norm', False))
