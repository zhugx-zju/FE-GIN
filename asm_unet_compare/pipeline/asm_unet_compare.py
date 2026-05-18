import csv
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from .common import (
        METHOD_DISPLAY,
        RELATIVE_ERROR_COLORBAR_MAX_PCT,
        align_panel_by_noise,
        build_asm_context,
        format_decimal_token,
        get_asm_result_suffix,
        get_asm_result_suffix_candidates,
        load_asm_panel_data,
        normalize_mix_gamma_map,
        resolve_variant_output_dir,
        parse_optional_float,
        resolve_output_root,
        unique_preserve_order,
    )
    from .unet_inference import load_saved_unet_panel_data
except ImportError:
    from common import (
        METHOD_DISPLAY,
        RELATIVE_ERROR_COLORBAR_MAX_PCT,
        align_panel_by_noise,
        build_asm_context,
        format_decimal_token,
        get_asm_result_suffix,
        get_asm_result_suffix_candidates,
        load_asm_panel_data,
        normalize_mix_gamma_map,
        resolve_variant_output_dir,
        parse_optional_float,
        resolve_output_root,
        unique_preserve_order,
    )
    from unet_inference import load_saved_unet_panel_data


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


def _row_tag(row_idx):
    return f"({chr(ord('a') + row_idx)})"


def _add_row_tags(first_col_axes, x=-0.10, y=1.06, fontsize=11):
    for row_idx, ax in sorted(first_col_axes.items()):
        ax.text(
            x,
            y,
            _row_tag(row_idx),
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight='bold',
            va='top',
            ha='left',
        )


def _normalize_case_cfg(cfg, case_cfg):
    data_type = case_cfg.get('dataset', case_cfg.get('data_type'))
    if data_type is None:
        raise KeyError("Each comparison case must provide 'dataset'.")

    sample_index_map = cfg.get('sample_index', {})
    if 'noise_level' in case_cfg:
        noise_levels = [float(case_cfg['noise_level'])]
    else:
        noise_levels = [float(noise) for noise in case_cfg.get('noise_levels', cfg['noise_levels'])]
    noise_levels = unique_preserve_order(noise_levels)

    methods = case_cfg.get('unet_methods', case_cfg.get('methods', cfg['unet_methods']))
    methods = unique_preserve_order([str(method) for method in methods])

    return {
        'dataset': data_type,
        'sample_index': int(case_cfg.get('sample_index', sample_index_map.get(data_type, 0))),
        'noise_levels': noise_levels,
        'unet_methods': methods,
        'mix_gamma': case_cfg.get('mix_gamma', cfg.get('mix_gamma', None)),
        'mix_gamma_by_method': normalize_mix_gamma_map(
            case_cfg.get('mix_gamma_by_method', cfg.get('mix_gamma_by_method', None))
        ),
        'strict_mix_gamma': bool(case_cfg.get('strict_mix_gamma', cfg.get('strict_mix_gamma', True))),
        'use_warm_start': bool(case_cfg.get('use_warm_start', cfg.get('use_warm_start', False))),
        'warm_start_method': case_cfg.get('warm_start_method', cfg.get('warm_start_method', None)),
        'unet_use_batch_norm': bool(case_cfg.get('unet_use_batch_norm', cfg.get('unet_use_batch_norm', False))),
        'nodesx': int(case_cfg.get('nodesx', cfg['nodesx'])),
        'nodesy': int(case_cfg.get('nodesy', cfg['nodesy'])),
        'asm_gamma': case_cfg.get('asm_gamma', cfg.get('asm_gamma', None)),
        'asm_max_iter': int(case_cfg.get('asm_max_iter', cfg['asm_max_iter'])),
        'asm_ftol': float(case_cfg.get('asm_ftol', cfg['asm_ftol'])),
        'asm_gtol': float(case_cfg.get('asm_gtol', cfg['asm_gtol'])),
    }
def _plot_prediction_figure(methods_order, noise_order, method_to_panel_data, save_path, title, mesh_info, contour_fn, dpi):
    methods = [m for m in methods_order if m in method_to_panel_data]
    if not methods:
        return

    target = None
    for method in methods:
        noise_map = align_panel_by_noise(method_to_panel_data[method])
        for noise in noise_order:
            row = noise_map.get(float(noise))
            if row is not None:
                target = row['target']
                break
        if target is not None:
            break
    if target is None:
        return

    n_methods = len(methods)
    n_noise = len(noise_order)
    vmin = float(np.min(target))
    vmax = float(np.max(target))

    fig = plt.figure(figsize=(2.55 * n_methods + 0.55, 2.60 * n_noise + 0.75))
    gs = fig.add_gridspec(n_noise, n_methods, wspace=0.045, hspace=0.085)

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
                ax.set_title(METHOD_DISPLAY.get(method, method), fontsize=11, fontweight='bold', pad=0)
            if col_idx == 0:
                first_col_axes[row_idx] = ax
            axes_pred.append(ax)
            im_ref = im

    if im_ref is None or not axes_pred:
        plt.close(fig)
        return

    fig.subplots_adjust(left=0.070, right=0.895, bottom=0.035, top=0.935, wspace=0.045, hspace=0.085)
    _add_row_tags(first_col_axes, x=-0.10, y=1.06, fontsize=11)
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
    n_methods = len(methods)
    n_noise = len(noise_order)
    vmax = RELATIVE_ERROR_COLORBAR_MAX_PCT

    fig = plt.figure(figsize=(2.55 * n_methods + 0.55, 2.60 * n_noise + 0.75))
    gs = fig.add_gridspec(n_noise, n_methods, wspace=0.045, hspace=0.085)

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
                ax.set_title(METHOD_DISPLAY.get(method, method), fontsize=11, fontweight='bold', pad=2)
            if col_idx == 0:
                first_col_axes[row_idx] = ax
            ax.text(
                0.00,
                0.969,
                f"$L_1$={noise_map[float(noise)]['l1']:.2e}",
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=8,
                color='white',
                bbox={'facecolor': 'black', 'alpha': 0.35, 'edgecolor': 'none', 'boxstyle': 'square,pad=0.0'},
            )
            axes_err.append(ax)
            im_ref = im

    if im_ref is not None and axes_err:
        cax = fig.add_axes([0.915, 0.14, 0.016, 0.74])
        colorbar = fig.colorbar(im_ref, cax=cax)
        colorbar.set_label('Relative error (%)', fontsize=10, fontweight='bold')
        colorbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.070, right=0.895, bottom=0.035, top=0.935, wspace=0.045, hspace=0.085)
    _add_row_tags(first_col_axes, x=-0.10, y=1.06, fontsize=11)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def _filter_panel_by_noise(panel_data, noise_order):
    noise_set = {float(noise) for noise in noise_order}
    return [row for row in panel_data if float(row['noise']) in noise_set]


def _missing_noises(panel_data, noise_order):
    panel_noise_set = {float(row['noise']) for row in panel_data}
    return [float(noise) for noise in noise_order if float(noise) not in panel_noise_set]


def _summary_suffix_for_cases(case_cfgs):
    if not case_cfgs:
        return ''

    warm_flags = {bool(case_cfg.get('use_warm_start', False)) for case_cfg in case_cfgs}
    if warm_flags == {False}:
        return ''
    if warm_flags == {True}:
        warm_methods = {case_cfg.get('warm_start_method', None) for case_cfg in case_cfgs}
        if len(warm_methods) == 1:
            return get_asm_result_suffix(True, next(iter(warm_methods)))
    return '_mixed'


def _write_comparison_summary(output_root, summary_rows, case_cfgs, use_batch_norm=False):
    if not summary_rows:
        print(f"No comparison summary written under {output_root}: no comparable ASM/UNet outputs were found.")
        return

    os.makedirs(output_root, exist_ok=True)
    summary_suffix = _summary_suffix_for_cases(case_cfgs)
    if bool(use_batch_norm):
        summary_suffix = f'{summary_suffix}_GN'
    summary_file = os.path.join(output_root, f'comparison_summary{summary_suffix}.csv')
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'data_type',
                'sample_index',
                'method',
                'unet_gamma_used',
                'noise',
                'l1',
                'mae_pct',
                'use_warm_start',
                'warm_start_method',
                'unet_use_batch_norm',
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved comparison summary: {summary_file}")


def compare_asm_unet_cases(project_root, cfg, cases):
    if not cases:
        print("No ASM-vs-UNet comparison cases were provided.")
        return

    summary_groups = {}
    for raw_case in cases:
        case_cfg = _normalize_case_cfg(cfg, raw_case)
        output_dir = resolve_variant_output_dir(
            cfg['output_dir'],
            use_batch_norm=case_cfg['unet_use_batch_norm'],
        )
        output_root = resolve_output_root(project_root, output_dir)
        load_filename_suffix = get_asm_result_suffix(
            case_cfg['use_warm_start'],
            case_cfg['warm_start_method'],
        )
        filename_suffix = f"{load_filename_suffix}{'_GN' if bool(case_cfg['unet_use_batch_norm']) else ''}"
        filename_suffixes = get_asm_result_suffix_candidates(
            case_cfg['use_warm_start'],
            case_cfg['warm_start_method'],
        )

        data_type = case_cfg['dataset']
        idx_cfg = case_cfg['sample_index']
        asm_dir, asm_panel_data, sample_idx, saved_case_cfg = load_asm_panel_data(
            output_root=output_root,
            data_type=data_type,
            sample_index=idx_cfg,
            noise_levels=case_cfg['noise_levels'],
            filename_suffix=filename_suffixes,
        )
        if not asm_panel_data or asm_dir is None or sample_idx is None:
            print(
                f"Skip dataset={data_type}, sample_index={idx_cfg}: "
                f"missing ASM noise results under {output_root}"
            )
            continue
        missing_asm_noises = _missing_noises(asm_panel_data, case_cfg['noise_levels'])
        if missing_asm_noises:
            print(
                f"Warning: dataset={data_type}, sample_index={sample_idx} is missing ASM results for "
                f"noise levels {missing_asm_noises} under {output_root}"
            )
        missing_asm_noise_set = set(missing_asm_noises)
        plot_noises = [noise for noise in case_cfg['noise_levels'] if float(noise) not in missing_asm_noise_set]
        if not plot_noises:
            print(
                f"Skip dataset={data_type}, sample_index={sample_idx}: "
                "no requested noise levels are available in the saved ASM results"
            )
            continue
        asm_panel_data = _filter_panel_by_noise(asm_panel_data, plot_noises)

        unet_method_to_panel, unet_sample_idx = load_saved_unet_panel_data(
            output_root=output_root,
            data_type=data_type,
            sample_index=idx_cfg,
            noise_levels=plot_noises,
            methods=case_cfg['unet_methods'],
            mix_gamma=case_cfg.get('mix_gamma', None),
            mix_gamma_by_method=case_cfg.get('mix_gamma_by_method', None),
            strict_mix_gamma=case_cfg.get('strict_mix_gamma', True),
        )

        resolved_idx = sample_idx if sample_idx is not None else unet_sample_idx
        if resolved_idx is None:
            resolved_idx = idx_cfg

        method_to_panel = {'ASM': asm_panel_data}
        for method in case_cfg['unet_methods']:
            panel = unet_method_to_panel.get(method, [])
            if not panel:
                print(
                    f"Warning: dataset={data_type}, sample_index={resolved_idx} is missing UNet results "
                    f"for method={method} under {output_root}"
                )
                continue
            missing_method_noises = _missing_noises(panel, plot_noises)
            if missing_method_noises:
                print(
                    f"Warning: dataset={data_type}, sample_index={resolved_idx}, method={method} is missing "
                    f"noise levels {missing_method_noises} under {output_root}"
                )
            filtered_panel = _filter_panel_by_noise(panel, plot_noises)
            if filtered_panel:
                method_to_panel[method] = filtered_panel
        if len(method_to_panel) <= 1:
            print(
                f"Skip dataset={data_type}, sample_index={resolved_idx}: "
                f"missing saved UNet noise results under {output_root}"
            )
            continue

        group = summary_groups.setdefault(
            output_root,
            {
                'rows': [],
                'case_cfgs': [],
                'use_batch_norm': bool(case_cfg.get('unet_use_batch_norm', False)),
            },
        )
        group['case_cfgs'].append(case_cfg)

        mesh_case_cfg = saved_case_cfg or case_cfg
        asm_ctx = build_asm_context(
            project_root=project_root,
            nodesx=int(mesh_case_cfg.get('nodesx', case_cfg['nodesx'])),
            nodesy=int(mesh_case_cfg.get('nodesy', case_cfg['nodesy'])),
            asm_gamma=mesh_case_cfg.get('asm_gamma', case_cfg.get('asm_gamma', cfg.get('asm_gamma', None))),
            asm_max_iter=int(mesh_case_cfg.get('asm_max_iter', case_cfg['asm_max_iter'])),
            asm_ftol=float(mesh_case_cfg.get('asm_ftol', case_cfg['asm_ftol'])),
            asm_gtol=float(mesh_case_cfg.get('asm_gtol', case_cfg['asm_gtol'])),
        )
        contour_fn = asm_ctx['create_smooth_contour']
        mesh_info = asm_ctx['mesh']

        out_dir = os.path.join(output_root, f"sample_{resolved_idx}", data_type, 'comparison')
        os.makedirs(out_dir, exist_ok=True)

        methods_order = ['ASM'] + case_cfg['unet_methods']
        _plot_prediction_figure(
            methods_order=methods_order,
            noise_order=plot_noises,
            method_to_panel_data=method_to_panel,
            save_path=os.path.join(out_dir, f'prediction_sample_{resolved_idx}_{data_type}{filename_suffix}.png'),
            title=f'{data_type.upper()} Sample {resolved_idx} - Prediction',
            mesh_info=mesh_info,
            contour_fn=contour_fn,
            dpi=cfg['dpi'],
        )
        _plot_error_figure(
            methods_order=methods_order,
            noise_order=plot_noises,
            method_to_panel_data=method_to_panel,
            save_path=os.path.join(out_dir, f'error_sample_{resolved_idx}_{data_type}{filename_suffix}.png'),
            title=f'{data_type.upper()} Sample {resolved_idx} - Relative Error',
            mesh_info=mesh_info,
            contour_fn=contour_fn,
            dpi=cfg['dpi'],
        )

        for method, panel in method_to_panel.items():
            for row in panel:
                group['rows'].append({
                    'data_type': data_type,
                    'sample_index': resolved_idx,
                    'method': method,
                    'unet_gamma_used': row.get('unet_gamma_used', ''),
                    'noise': row['noise'],
                    'l1': row['l1'],
                    'mae_pct': row['mae_pct'],
                    'use_warm_start': bool(case_cfg.get('use_warm_start', False)),
                    'warm_start_method': case_cfg.get('warm_start_method', None),
                    'unet_use_batch_norm': bool(case_cfg.get('unet_use_batch_norm', False)),
                })

    for output_root, group in sorted(summary_groups.items()):
        _write_comparison_summary(
            output_root=output_root,
            summary_rows=group['rows'],
            case_cfgs=group['case_cfgs'],
            use_batch_norm=group['use_batch_norm'],
        )


def compare_asm_unet_batch(project_root, cfg):
    cases = []
    for data_type in cfg['datasets']:
        cases.append({
            'dataset': data_type,
            'sample_index': cfg['sample_index'].get(data_type, 0),
            'noise_levels': cfg['noise_levels'],
            'unet_methods': cfg['unet_methods'],
            'mix_gamma': cfg.get('mix_gamma', None),
            'mix_gamma_by_method': cfg.get('mix_gamma_by_method', None),
            'strict_mix_gamma': cfg.get('strict_mix_gamma', True),
            'use_warm_start': cfg.get('use_warm_start', False),
            'warm_start_method': cfg.get('warm_start_method', None),
            'unet_use_batch_norm': cfg.get('unet_use_batch_norm', False),
        })
    compare_asm_unet_cases(project_root=project_root, cfg=cfg, cases=cases)
