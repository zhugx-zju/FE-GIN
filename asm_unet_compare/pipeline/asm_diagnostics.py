import os

import matplotlib.pyplot as plt

try:
    from .asm_inversion import _normalize_case_cfg
    from .common import (
        RELATIVE_ERROR_COLORBAR_MAX_PCT,
        align_panel_by_noise,
        build_asm_context,
        format_decimal_token,
        get_asm_result_suffix,
        get_asm_result_suffix_candidates,
        load_asm_panel_data,
        plot_iteration_png,
        plot_lcurve_png,
        resolve_variant_output_dir,
        resolve_output_root,
        save_lcurve_data,
    )
except ImportError:
    from asm_inversion import _normalize_case_cfg
    from common import (
        RELATIVE_ERROR_COLORBAR_MAX_PCT,
        align_panel_by_noise,
        build_asm_context,
        format_decimal_token,
        get_asm_result_suffix,
        get_asm_result_suffix_candidates,
        load_asm_panel_data,
        plot_iteration_png,
        plot_lcurve_png,
        resolve_variant_output_dir,
        resolve_output_root,
        save_lcurve_data,
    )


def _plot_asm_prediction_figure(noise_order, panel_data, save_path, title, mesh_info, contour_fn, dpi):
    noise_map = align_panel_by_noise(panel_data)
    plotted_noises = [float(noise) for noise in noise_order if float(noise) in noise_map]
    if not plotted_noises:
        return

    target = noise_map[plotted_noises[0]]['target']
    vmin = float(target.min())
    vmax = float(target.max())

    fig = plt.figure(figsize=(5.6, 2.8 * len(plotted_noises)))
    gs = fig.add_gridspec(len(plotted_noises), 2, wspace=0.05, hspace=0.06)

    ax_true = fig.add_subplot(gs[:, 0])
    if contour_fn is not None and mesh_info is not None:
        im_ref = contour_fn(mesh_info.plot_x, mesh_info.plot_y, target, ax_true, levels=100, cmap='viridis')
        im_ref.set_clim(vmin, vmax)
        ax_true.axis('equal')
        ax_true.axis('off')
    else:
        im_ref = ax_true.imshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_true.set_xticks([])
        ax_true.set_yticks([])
    ax_true.set_title('True', fontsize=12, fontweight='bold')

    pred_axes = []
    for row_idx, noise in enumerate(plotted_noises):
        ax = fig.add_subplot(gs[row_idx, 1])
        pred = noise_map[noise]['pred']
        if contour_fn is not None and mesh_info is not None:
            im = contour_fn(mesh_info.plot_x, mesh_info.plot_y, pred, ax, levels=100, cmap='viridis')
            im.set_clim(vmin, vmax)
            ax.axis('equal')
            ax.axis('off')
        else:
            im = ax.imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f'ASM ({noise:g}%)', fontsize=11, fontweight='bold')
        pred_axes.append(ax)
        im_ref = im

    fig.colorbar(im_ref, ax=[ax_true] + pred_axes, fraction=0.03, pad=0.01)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _plot_asm_error_figure(noise_order, panel_data, save_path, title, mesh_info, contour_fn, dpi):
    noise_map = align_panel_by_noise(panel_data)
    plotted_noises = [float(noise) for noise in noise_order if float(noise) in noise_map]
    if not plotted_noises:
        return

    vmax = RELATIVE_ERROR_COLORBAR_MAX_PCT

    fig = plt.figure(figsize=(3.0, 2.8 * len(plotted_noises)))
    gs = fig.add_gridspec(len(plotted_noises), 1, wspace=0.05, hspace=0.06)

    im_ref = None
    axes = []
    for row_idx, noise in enumerate(plotted_noises):
        ax = fig.add_subplot(gs[row_idx, 0])
        err = noise_map[noise]['rel_err']
        if contour_fn is not None and mesh_info is not None:
            im = contour_fn(mesh_info.plot_x, mesh_info.plot_y, err, ax, levels=100, cmap='Blues')
            im.set_clim(0, vmax)
            ax.axis('equal')
            ax.axis('off')
        else:
            im = ax.imshow(err, cmap='Blues', vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f'ASM Error ({noise:g}%)', fontsize=11, fontweight='bold')
        ax.text(0.02, 0.98, f"L1={noise_map[noise]['l1']:.2e}", transform=ax.transAxes, va='top', fontsize=8)
        axes.append(ax)
        im_ref = im

    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes, fraction=0.05, pad=0.02)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_asm_diagnostics_batch(project_root, cfg):
    output_dir = resolve_variant_output_dir(
        cfg['output_dir'],
        use_batch_norm=cfg.get('unet_use_batch_norm', False),
    )
    output_root = resolve_output_root(project_root, output_dir)
    filename_suffix = get_asm_result_suffix(
        cfg.get('use_warm_start', False),
        cfg.get('warm_start_method', None),
    )
    filename_suffix = f"{filename_suffix}{'_GN' if bool(cfg.get('unet_use_batch_norm', False)) else ''}"
    filename_suffixes = get_asm_result_suffix_candidates(
        cfg.get('use_warm_start', False),
        cfg.get('warm_start_method', None),
    )
    for data_type in cfg['datasets']:
        idx_cfg = cfg['sample_index'].get(data_type, 0)
        data_dir, panel_data, _, _ = load_asm_panel_data(
            output_root=output_root,
            data_type=data_type,
            sample_index=idx_cfg,
            noise_levels=cfg['noise_levels'],
            filename_suffix=filename_suffixes,
        )
        if not panel_data or data_dir is None:
            print(f"Skip {data_type}: ASM results not found under {output_root}")
            continue
        for row in panel_data:
            noise = row['noise']
            noise_dir = os.path.join(data_dir, f"noise_{format_decimal_token(noise)}")
            os.makedirs(noise_dir, exist_ok=True)

            plot_iteration_png(
                results=row.get('solver_results', {}),
                save_dir=noise_dir,
                noise_percent=noise,
                dpi=cfg['dpi'],
                filename_suffix=filename_suffix,
            )
            lcurve_results = row.get('lcurve_results', None)
            if lcurve_results is not None:
                save_lcurve_data(lcurve_results, noise_dir, filename_suffix=filename_suffix)
                plot_lcurve_png(lcurve_results, noise_dir, dpi=cfg['dpi'], filename_suffix=filename_suffix)

        print(f"Saved ASM diagnostic plots: {data_dir}")


def plot_asm_results_cases(project_root, cfg, cases):
    output_dir = resolve_variant_output_dir(
        cfg['output_dir'],
        use_batch_norm=cfg.get('unet_use_batch_norm', False),
    )
    output_root = resolve_output_root(project_root, output_dir)
    asm_ctx_cache = {}

    for raw_case in cases:
        case_cfg = _normalize_case_cfg(cfg, raw_case)
        filename_suffix = get_asm_result_suffix(
            case_cfg.get('use_warm_start', False),
            case_cfg.get('warm_start_method', None),
        )
        filename_suffix = f"{filename_suffix}{'_GN' if bool(case_cfg.get('unet_use_batch_norm', False)) else ''}"
        filename_suffixes = get_asm_result_suffix_candidates(
            case_cfg.get('use_warm_start', False),
            case_cfg.get('warm_start_method', None),
        )
        data_type = case_cfg['dataset']
        sample_index = case_cfg['sample_index']
        data_dir, panel_data, resolved_idx, saved_case_cfg = load_asm_panel_data(
            output_root=output_root,
            data_type=data_type,
            sample_index=sample_index,
            noise_levels=case_cfg['noise_levels'],
            filename_suffix=filename_suffixes,
        )
        if not panel_data or data_dir is None:
            print(
                f"Skip dataset={data_type}, sample_index={sample_index}: "
                f"ASM results not found under {output_root}"
            )
            continue

        plot_cfg = dict(case_cfg)
        if isinstance(saved_case_cfg, dict):
            plot_cfg.update({k: v for k, v in saved_case_cfg.items() if k in plot_cfg})

        ctx_key = (
            int(plot_cfg['nodesx']),
            int(plot_cfg['nodesy']),
            plot_cfg.get('asm_gamma', None),
            int(plot_cfg['asm_max_iter']),
            float(plot_cfg['asm_ftol']),
            float(plot_cfg['asm_gtol']),
        )
        if ctx_key not in asm_ctx_cache:
            asm_ctx_cache[ctx_key] = build_asm_context(
                project_root=project_root,
                nodesx=plot_cfg['nodesx'],
                nodesy=plot_cfg['nodesy'],
                asm_gamma=plot_cfg.get('asm_gamma', None),
                asm_max_iter=plot_cfg['asm_max_iter'],
                asm_ftol=plot_cfg['asm_ftol'],
                asm_gtol=plot_cfg['asm_gtol'],
            )
        asm_ctx = asm_ctx_cache[ctx_key]

        selected_noises = [float(noise) for noise in plot_cfg['noise_levels']]
        filtered_panel = [row for row in panel_data if float(row['noise']) in selected_noises]
        if not filtered_panel:
            print(
                f"Skip dataset={data_type}, sample_index={resolved_idx}: "
                f"no matching noise levels found in the selected ASM result file"
            )
            continue

        for row in filtered_panel:
            noise = row['noise']
            noise_dir = os.path.join(data_dir, f"noise_{format_decimal_token(noise)}")
            os.makedirs(noise_dir, exist_ok=True)

            single_panel = [row]
            _plot_asm_prediction_figure(
                noise_order=[noise],
                panel_data=single_panel,
                save_path=os.path.join(noise_dir, f'prediction_sample_{resolved_idx}_{data_type}{filename_suffix}.png'),
                title=f'{data_type.upper()} Sample {resolved_idx} - ASM Prediction',
                mesh_info=asm_ctx['mesh'],
                contour_fn=asm_ctx['create_smooth_contour'],
                dpi=cfg['dpi'],
            )
            _plot_asm_error_figure(
                noise_order=[noise],
                panel_data=single_panel,
                save_path=os.path.join(noise_dir, f'error_sample_{resolved_idx}_{data_type}{filename_suffix}.png'),
                title=f'{data_type.upper()} Sample {resolved_idx} - ASM Relative Error',
                mesh_info=asm_ctx['mesh'],
                contour_fn=asm_ctx['create_smooth_contour'],
                dpi=cfg['dpi'],
            )

            plot_iteration_png(
                results=row.get('solver_results', {}),
                save_dir=noise_dir,
                noise_percent=noise,
                dpi=cfg['dpi'],
                filename_suffix=filename_suffix,
            )
            lcurve_results = row.get('lcurve_results', None)
            if lcurve_results is not None:
                save_lcurve_data(lcurve_results, noise_dir, filename_suffix=filename_suffix)
                plot_lcurve_png(lcurve_results, noise_dir, dpi=cfg['dpi'], filename_suffix=filename_suffix)

        print(f"Saved ASM result plots: {data_dir}")
