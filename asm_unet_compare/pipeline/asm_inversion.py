import csv
import os
import pickle
import time

import numpy as np

try:
    from .common import (
        asm_field_to_vector,
        build_unet_file_stem,
        build_asm_context,
        candidate_data_dirs,
        format_decimal_token,
        get_asm_result_suffix,
        load_asm_noise_result,
        load_measured_displacement,
        load_sample,
        normalize_mix_gamma_map,
        parse_optional_float,
        predict_asm_panel,
        resolve_case_mix_gamma,
        resolve_variant_output_dir,
        resolve_output_root,
    )
    from .unet_inference import load_saved_unet_panel_data
except ImportError:
    from common import (
        asm_field_to_vector,
        build_unet_file_stem,
        build_asm_context,
        candidate_data_dirs,
        format_decimal_token,
        get_asm_result_suffix,
        load_asm_noise_result,
        load_measured_displacement,
        load_sample,
        normalize_mix_gamma_map,
        parse_optional_float,
        predict_asm_panel,
        resolve_case_mix_gamma,
        resolve_variant_output_dir,
        resolve_output_root,
    )
    from unet_inference import load_saved_unet_panel_data


def _normalize_case_cfg(cfg, case_cfg):
    data_type = case_cfg.get('dataset', case_cfg.get('data_type'))
    if data_type is None:
        raise KeyError("Each ASM case must provide 'dataset'.")

    sample_index_map = cfg.get('sample_index', {})
    if 'noise_level' in case_cfg:
        noise_levels = [float(case_cfg['noise_level'])]
    else:
        noise_levels = list(case_cfg.get('noise_levels', cfg['noise_levels']))

    use_warm_start = bool(case_cfg.get('use_warm_start', cfg.get('use_warm_start', False)))
    if 'use_cold_start_gamma' in case_cfg:
        use_cold_start_gamma = bool(case_cfg['use_cold_start_gamma'])
    else:
        use_cold_start_gamma = bool(cfg.get('use_cold_start_gamma', False))

    if not use_warm_start:
        use_cold_start_gamma = False

    return {
        'dataset': data_type,
        'sample_index': int(case_cfg.get('sample_index', sample_index_map.get(data_type, 0))),
        'noise_levels': noise_levels,
        'nodesx': int(case_cfg.get('nodesx', cfg['nodesx'])),
        'nodesy': int(case_cfg.get('nodesy', cfg['nodesy'])),
        'asm_dof_order': case_cfg.get('asm_dof_order', cfg['asm_dof_order']),
        'asm_gamma': case_cfg.get('asm_gamma', cfg.get('asm_gamma', None)),
        'asm_max_iter': int(case_cfg.get('asm_max_iter', cfg['asm_max_iter'])),
        'asm_ftol': float(case_cfg.get('asm_ftol', cfg['asm_ftol'])),
        'asm_gtol': float(case_cfg.get('asm_gtol', cfg['asm_gtol'])),
        'enable_lcurve': bool(case_cfg.get('enable_lcurve', cfg['enable_lcurve'])),
        'lcurve_points': int(case_cfg.get('lcurve_points', cfg['lcurve_points'])),
        'lcurve_gamma_min': case_cfg.get('lcurve_gamma_min', cfg['lcurve_gamma_min']),
        'lcurve_gamma_max': case_cfg.get('lcurve_gamma_max', cfg['lcurve_gamma_max']),
        'unet_config_type': case_cfg.get('unet_config_type', cfg['unet_config_type']),
        'unet_load_type': case_cfg.get('unet_load_type', cfg['unet_load_type']),
        'unet_use_batch_norm': bool(case_cfg.get('unet_use_batch_norm', cfg.get('unet_use_batch_norm', False))),
        'use_warm_start': use_warm_start,
        'warm_start_method': case_cfg.get('warm_start_method', cfg.get('warm_start_method', None)),
        'warm_start_output_dir': case_cfg.get(
            'warm_start_output_dir',
            cfg.get('warm_start_output_dir', cfg['output_dir']),
        ),
        'use_cold_start_gamma': use_cold_start_gamma,
        'cold_start_gamma_source_dir': case_cfg.get(
            'cold_start_gamma_source_dir',
            cfg.get('cold_start_gamma_source_dir', cfg['output_dir']),
        ),
        'mix_gamma': case_cfg.get('mix_gamma', cfg.get('mix_gamma', None)),
        'mix_gamma_by_method': normalize_mix_gamma_map(
            case_cfg.get('mix_gamma_by_method', cfg.get('mix_gamma_by_method', None))
        ),
        'strict_mix_gamma': bool(case_cfg.get('strict_mix_gamma', cfg.get('strict_mix_gamma', True))),
    }


def _asm_context_key(case_cfg):
    return (
        case_cfg['nodesx'],
        case_cfg['nodesy'],
        case_cfg['asm_gamma'],
        case_cfg['asm_max_iter'],
        case_cfg['asm_ftol'],
        case_cfg['asm_gtol'],
    )


def _build_saved_displacement_map(output_root, case_cfg, resolved_idx):
    displacement_by_noise = {}

    for noise in case_cfg['noise_levels']:
        noise_key = float(noise)
        noise_tag = format_decimal_token(noise_key)
        found = False
        for data_dir in candidate_data_dirs(output_root, case_cfg['dataset'], resolved_idx):
            npz_file = os.path.join(data_dir, f'noise_{noise_tag}', 'measured_displacement.npz')
            if not os.path.exists(npz_file):
                continue

            displacement_info = load_measured_displacement(
                npz_file=npz_file,
                asm_dof_order=case_cfg['asm_dof_order'],
            )
            saved_idx = displacement_info.get('sample_index', None)
            if saved_idx is not None and int(saved_idx) != int(resolved_idx):
                continue

            displacement_by_noise[noise_key] = {
                'input': displacement_info['input'],
                'U_measured': displacement_info['U_measured'],
                'source': 'saved_displacement',
                'source_file': npz_file,
            }
            found = True
            break

        if not found:
            raise FileNotFoundError(
                f"Saved displacement field not found for dataset={case_cfg['dataset']}, "
                f"sample_index={resolved_idx}, noise={noise_key:g}% under {output_root}. "
                "Please run asm_unet_compare/run_sample_unet.py first."
            )

    return displacement_by_noise


def _build_warm_start_map(project_root, case_cfg, resolved_idx, target_shape, E_min, E_max):
    if not case_cfg['use_warm_start']:
        return None

    method = case_cfg.get('warm_start_method', None)
    if not method:
        raise ValueError("Warm-start is enabled, but 'warm_start_method' is not set.")
    warm_start_gamma = resolve_case_mix_gamma(case_cfg, method)

    warm_start_output_dir = resolve_variant_output_dir(
        case_cfg['warm_start_output_dir'],
        use_batch_norm=case_cfg.get('unet_use_batch_norm', False),
    )
    warm_start_root = resolve_output_root(project_root, warm_start_output_dir)
    method_to_panel, warm_start_idx = load_saved_unet_panel_data(
        output_root=warm_start_root,
        data_type=case_cfg['dataset'],
        sample_index=resolved_idx,
        noise_levels=case_cfg['noise_levels'],
        methods=[method],
        mix_gamma_by_method=case_cfg.get('mix_gamma_by_method', None),
        mix_gamma=case_cfg.get('mix_gamma', None),
        strict_mix_gamma=case_cfg.get('strict_mix_gamma', True),
    )
    if method not in method_to_panel:
        gamma_text = '' if warm_start_gamma is None else f", gamma={warm_start_gamma}"
        raise FileNotFoundError(
            f"Warm-start UNet results not found for dataset={case_cfg['dataset']}, "
            f"sample_index={resolved_idx}, method={method}{gamma_text} under {warm_start_root}"
        )

    panel_by_noise = {float(row['noise']): row for row in method_to_panel[method]}
    source_idx = resolved_idx if warm_start_idx is None else int(warm_start_idx)
    init_by_noise = {}
    missing_noise = []
    expected_shape = tuple(target_shape)

    for noise in case_cfg['noise_levels']:
        noise_key = float(noise)
        row = panel_by_noise.get(noise_key)
        if row is None:
            missing_noise.append(noise_key)
            continue

        pred = np.asarray(row['pred'], dtype=np.float64)
        if tuple(pred.shape) != expected_shape:
            raise ValueError(
                f"Warm-start field shape mismatch for dataset={case_cfg['dataset']}, "
                f"sample_index={resolved_idx}, noise={noise_key:g}: "
                f"expected {expected_shape}, got {tuple(pred.shape)}"
            )
        pred_clipped = np.clip(pred, float(E_min), float(E_max))
        if not np.allclose(pred, pred_clipped):
            clipped_count = int(np.count_nonzero(~np.isclose(pred, pred_clipped)))
            print(
                f"Warning: clipped {clipped_count} warm-start modulus values to "
                f"[{float(E_min):.3e}, {float(E_max):.3e}] for noise={noise_key:g}%"
            )

        file_stem = row.get(
            'file_stem',
            build_unet_file_stem(method, row.get('unet_gamma_used', warm_start_gamma)),
        )
        init_by_noise[noise_key] = {
            'E_init': asm_field_to_vector(pred_clipped),
            'method': method,
            'source_file': os.path.join(
                warm_start_root,
                f"sample_{source_idx}",
                case_cfg['dataset'],
                f"noise_{format_decimal_token(noise_key)}",
                f"{file_stem}.npz",
            ),
            'l1': float(row['l1']),
            'mae_pct': float(row['mae_pct']),
            'elapsed_time': float(row.get('elapsed_time', 0.0)),
            'unet_gamma_used': parse_optional_float(row.get('unet_gamma_used', warm_start_gamma)),
        }

    if missing_noise:
        missing_text = ', '.join(f'{noise:g}%' for noise in missing_noise)
        raise FileNotFoundError(
            f"Warm-start UNet results are missing for noise levels [{missing_text}] "
            f"under {warm_start_root}"
        )

    return init_by_noise


def _build_cold_start_gamma_map(project_root, case_cfg, resolved_idx):
    if not case_cfg.get('use_cold_start_gamma', False):
        return None

    cold_start_output_dir = resolve_variant_output_dir(
        case_cfg['cold_start_gamma_source_dir'],
        use_batch_norm=case_cfg.get('unet_use_batch_norm', False),
    )
    source_root = resolve_output_root(project_root, cold_start_output_dir)
    gamma_by_noise = {}
    missing_noise = []

    for noise in case_cfg['noise_levels']:
        noise_key = float(noise)
        source_file, asm_data = load_asm_noise_result(
            output_root=source_root,
            data_type=case_cfg['dataset'],
            sample_index=resolved_idx,
            noise_level=noise_key,
            filename_suffix='',
        )
        if asm_data is None:
            missing_noise.append(noise_key)
            continue

        panel_data = asm_data.get('panel_data', [])
        row = None
        for item in panel_data:
            if float(item.get('noise', noise_key)) == noise_key:
                row = item
                break
        if row is None and panel_data:
            row = panel_data[0]
        if row is None:
            raise ValueError(
                f"Cold-start ASM results exist but contain no panel data for "
                f"dataset={case_cfg['dataset']}, sample_index={resolved_idx}, noise={noise_key:g}%"
            )

        gamma_value = row.get('gamma_selected_from_lcurve', None)
        if gamma_value in [None, '']:
            gamma_value = row.get('asm_gamma', None)
        if gamma_value in [None, '']:
            raise ValueError(
                f"Cold-start ASM results do not contain a usable gamma for "
                f"dataset={case_cfg['dataset']}, sample_index={resolved_idx}, noise={noise_key:g}%"
            )

        gamma_by_noise[noise_key] = {
            'gamma': float(gamma_value),
            'source_file': source_file,
            'selection_mode': row.get('selection_mode', None),
        }

    if missing_noise:
        missing_text = ', '.join(f'{noise:g}%' for noise in missing_noise)
        raise FileNotFoundError(
            f"Cold-start ASM results are missing for noise levels [{missing_text}] "
            f"under {source_root}"
        )

    return gamma_by_noise


def _write_case_config(save_dir, case_cfg, resolved_idx, panel_data, total_time):
    suffix = get_asm_result_suffix(
        case_cfg.get('use_warm_start', False),
        case_cfg.get('warm_start_method', None),
    )
    config_path = os.path.join(save_dir, f'config{suffix}.py')
    noise_times = {float(row['noise']): float(row.get('elapsed_time', 0.0)) for row in panel_data}
    noise_lcurve_times = {float(row['noise']): float(row.get('lcurve_elapsed_time', 0.0)) for row in panel_data}
    noise_final_run_times = {float(row['noise']): float(row.get('final_run_elapsed_time', 0.0)) for row in panel_data}
    noise_warm_start_times = {float(row['noise']): float(row.get('warm_start_elapsed_time', 0.0)) for row in panel_data}
    noise_end_to_end_times = {float(row['noise']): float(row.get('end_to_end_elapsed_time', 0.0)) for row in panel_data}
    noise_iterations = {float(row['noise']): int(row.get('n_iterations', -1)) for row in panel_data}
    noise_gammas = {float(row['noise']): float(row.get('asm_gamma', 0.0)) for row in panel_data}
    gamma_selected_from_lcurve = {float(row['noise']): row.get('gamma_selected_from_lcurve', None) for row in panel_data}
    selection_mode = {float(row['noise']): row.get('selection_mode', 'fixed_gamma') for row in panel_data}
    gamma_sources = {float(row['noise']): row.get('gamma_source', None) for row in panel_data}
    gamma_source_files = {float(row['noise']): row.get('gamma_source_file', None) for row in panel_data}
    displacement_sources = {float(row['noise']): row.get('displacement_source', 'saved_displacement') for row in panel_data}
    displacement_source_files = {float(row['noise']): row.get('displacement_source_file', None) for row in panel_data}
    converged = {float(row['noise']): bool(row.get('converged', False)) for row in panel_data}
    warm_start_sources = {float(row['noise']): row.get('warm_start_source', None) for row in panel_data}
    warm_start_unet_gamma = {float(row['noise']): row.get('warm_start_unet_gamma', None) for row in panel_data}
    warm_start_init_l1 = {float(row['noise']): row.get('warm_start_init_l1', None) for row in panel_data}
    warm_start_init_mae_pct = {float(row['noise']): row.get('warm_start_init_mae_pct', None) for row in panel_data}

    ordered_items = [
        ('dataset', case_cfg['dataset']),
        ('sample_index', case_cfg['sample_index']),
        ('resolved_sample_index', resolved_idx),
        ('noise_levels', list(case_cfg['noise_levels'])),
        ('noise_level', float(case_cfg['noise_levels'][0]) if len(case_cfg['noise_levels']) == 1 else None),
        ('nodesx', case_cfg['nodesx']),
        ('nodesy', case_cfg['nodesy']),
        ('asm_dof_order', case_cfg['asm_dof_order']),
        ('asm_gamma', case_cfg['asm_gamma']),
        ('asm_max_iter', case_cfg['asm_max_iter']),
        ('asm_ftol', case_cfg['asm_ftol']),
        ('asm_gtol', case_cfg['asm_gtol']),
        ('enable_lcurve', case_cfg['enable_lcurve']),
        ('lcurve_points', case_cfg['lcurve_points']),
        ('lcurve_gamma_min', case_cfg['lcurve_gamma_min']),
        ('lcurve_gamma_max', case_cfg['lcurve_gamma_max']),
        ('unet_config_type', case_cfg['unet_config_type']),
        ('unet_load_type', case_cfg['unet_load_type']),
        ('unet_use_batch_norm', case_cfg.get('unet_use_batch_norm', False)),
        ('use_warm_start', case_cfg['use_warm_start']),
        ('warm_start_method', case_cfg['warm_start_method']),
        ('warm_start_output_dir', case_cfg['warm_start_output_dir']),
        ('use_cold_start_gamma', case_cfg['use_cold_start_gamma']),
        ('cold_start_gamma_source_dir', case_cfg['cold_start_gamma_source_dir']),
        ('mix_gamma', case_cfg.get('mix_gamma', None)),
        ('mix_gamma_by_method', case_cfg.get('mix_gamma_by_method', None)),
        ('strict_mix_gamma', case_cfg.get('strict_mix_gamma', True)),
        ('result_suffix', suffix),
        ('warm_start_sources', warm_start_sources),
        ('warm_start_unet_gamma', warm_start_unet_gamma),
        ('warm_start_init_l1', warm_start_init_l1),
        ('warm_start_init_mae_pct', warm_start_init_mae_pct),
        ('selection_mode', selection_mode),
        ('gamma_sources', gamma_sources),
        ('gamma_source_files', gamma_source_files),
        ('displacement_sources', displacement_sources),
        ('displacement_source_files', displacement_source_files),
        ('noise_times', noise_times),
        ('noise_lcurve_times', noise_lcurve_times),
        ('noise_final_run_times', noise_final_run_times),
        ('noise_warm_start_times', noise_warm_start_times),
        ('noise_end_to_end_times', noise_end_to_end_times),
        ('noise_iterations', noise_iterations),
        ('noise_gammas', noise_gammas),
        ('gamma_selected_from_lcurve', gamma_selected_from_lcurve),
        ('converged', converged),
        ('time', float(total_time)),
    ]

    with open(config_path, 'w', encoding='utf-8') as f:
        for key, value in ordered_items:
            f.write(f"{key} = {repr(value)}\n")

    return config_path


def _save_case_outputs(data_dir, data_type, resolved_idx, panel_data, case_cfg, summary_rows):
    saved_dirs = []
    suffix = get_asm_result_suffix(
        case_cfg.get('use_warm_start', False),
        case_cfg.get('warm_start_method', None),
    )
    run_type = 'warm_start' if case_cfg.get('use_warm_start', False) else 'cold_start'

    for row in panel_data:
        noise = float(row['noise'])
        noise_tag = format_decimal_token(noise)
        noise_dir = os.path.join(data_dir, f"noise_{noise_tag}")
        os.makedirs(noise_dir, exist_ok=True)

        row_case_cfg = dict(case_cfg)
        row_case_cfg['noise_levels'] = [noise]

        with open(os.path.join(noise_dir, f'asm_results{suffix}.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'data_type': data_type,
                    'sample_index': resolved_idx,
                    'noise_levels': [noise],
                    'panel_data': [row],
                    'case_config': row_case_cfg,
                    'run_type': run_type,
                },
                f,
            )

        np.savez(
            os.path.join(noise_dir, f'asm{suffix}.npz'),
            target=row['target'],
            pred=row['pred'],
            rel_err=row['rel_err'],
        )

        _write_case_config(
            save_dir=noise_dir,
            case_cfg=row_case_cfg,
            resolved_idx=resolved_idx,
            panel_data=[row],
            total_time=float(row.get('elapsed_time', 0.0)),
        )
        summary_rows.append({
            'data_type': data_type,
            'sample_index': resolved_idx,
            'run_type': run_type,
            'noise': noise,
            'l1': row['l1'],
            'mae_pct': row['mae_pct'],
            'n_iterations': row.get('n_iterations', ''),
            'converged': row.get('converged', ''),
            'asm_gamma': row.get('asm_gamma', ''),
            'gamma_selected_from_lcurve': row.get('gamma_selected_from_lcurve', ''),
            'selection_mode': row.get('selection_mode', ''),
            'gamma_source': row.get('gamma_source', ''),
            'gamma_source_file': row.get('gamma_source_file', ''),
            'displacement_source': row.get('displacement_source', ''),
            'displacement_source_file': row.get('displacement_source_file', ''),
            'elapsed_time': row.get('elapsed_time', ''),
            'lcurve_elapsed_time': row.get('lcurve_elapsed_time', ''),
            'final_run_elapsed_time': row.get('final_run_elapsed_time', ''),
            'warm_start_elapsed_time': row.get('warm_start_elapsed_time', ''),
            'end_to_end_elapsed_time': row.get('end_to_end_elapsed_time', ''),
            'warm_start_method': row.get('warm_start_method', ''),
            'warm_start_unet_gamma': row.get('warm_start_unet_gamma', ''),
            'warm_start_init_l1': row.get('warm_start_init_l1', ''),
            'warm_start_init_mae_pct': row.get('warm_start_init_mae_pct', ''),
        })
        saved_dirs.append(noise_dir)

    return saved_dirs


def _write_summary(output_root, summary_rows, use_warm_start=False, warm_start_method=None, use_batch_norm=False):
    suffix = get_asm_result_suffix(use_warm_start, warm_start_method)
    summary_gn_suffix = '_GN' if bool(use_batch_norm) else ''
    summary_file = os.path.join(output_root, f'asm_summary{suffix}{summary_gn_suffix}.csv')
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'data_type',
                'sample_index',
                'run_type',
                'noise',
                'l1',
                'mae_pct',
                'n_iterations',
                'converged',
                'asm_gamma',
                'gamma_selected_from_lcurve',
                'selection_mode',
                'gamma_source',
                'gamma_source_file',
                'displacement_source',
                'displacement_source_file',
                'elapsed_time',
                'lcurve_elapsed_time',
                'final_run_elapsed_time',
                'warm_start_elapsed_time',
                'end_to_end_elapsed_time',
                'warm_start_method',
                'warm_start_unet_gamma',
                'warm_start_init_l1',
                'warm_start_init_mae_pct',
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved ASM inversion summary: {summary_file}")


def run_asm_inversion_cases(project_root, cfg, cases):
    output_dir = resolve_variant_output_dir(
        cfg['output_dir'],
        use_batch_norm=cfg.get('unet_use_batch_norm', False),
    )
    output_root = resolve_output_root(project_root, output_dir)
    os.makedirs(output_root, exist_ok=True)

    summary_rows = []
    asm_ctx_cache = {}

    for raw_case in cases:
        case_start_time = time.time()
        case_cfg = _normalize_case_cfg(cfg, raw_case)
        data_type = case_cfg['dataset']
        sample_index = case_cfg['sample_index']

        if (
            case_cfg['use_warm_start']
            and case_cfg['use_cold_start_gamma']
            and case_cfg['enable_lcurve']
        ):
            print(
                "Warm-start is configured to reuse the corresponding cold-start gamma. "
                "Current-run L-curve will be skipped."
            )
            case_cfg['enable_lcurve'] = False

        ctx_key = _asm_context_key(case_cfg)
        if ctx_key not in asm_ctx_cache:
            asm_ctx_cache[ctx_key] = build_asm_context(
                project_root=project_root,
                nodesx=case_cfg['nodesx'],
                nodesy=case_cfg['nodesy'],
                asm_gamma=case_cfg['asm_gamma'],
                asm_max_iter=case_cfg['asm_max_iter'],
                asm_ftol=case_cfg['asm_ftol'],
                asm_gtol=case_cfg['asm_gtol'],
            )
        asm_ctx = asm_ctx_cache[ctx_key]

        print(
            f"[ASM] dataset={data_type}, sample_index={sample_index}, "
            f"noise_levels={case_cfg['noise_levels']}, gamma={case_cfg['asm_gamma']}, "
            f"max_iter={case_cfg['asm_max_iter']}, "
            f"use_warm_start={case_cfg['use_warm_start']}, "
            f"use_cold_start_gamma={case_cfg['use_cold_start_gamma']}"
        )

        input_sample, target_sample, resolved_idx = load_sample(
            config_type=case_cfg['unet_config_type'],
            load_type=case_cfg['unet_load_type'],
            data_type=data_type,
            sample_index=sample_index,
        )
        data_dir = os.path.join(output_root, f"sample_{resolved_idx}", data_type)
        os.makedirs(data_dir, exist_ok=True)
        displacement_by_noise = _build_saved_displacement_map(
            output_root=output_root,
            case_cfg=case_cfg,
            resolved_idx=resolved_idx,
        )
        warm_start_map = _build_warm_start_map(
            project_root=project_root,
            case_cfg=case_cfg,
            resolved_idx=resolved_idx,
            target_shape=tuple(target_sample.squeeze().detach().cpu().numpy().shape),
            E_min=asm_ctx['E_min'],
            E_max=asm_ctx['E_max'],
        )
        gamma_by_noise = _build_cold_start_gamma_map(
            project_root=project_root,
            case_cfg=case_cfg,
            resolved_idx=resolved_idx,
        )
        if case_cfg['use_warm_start']:
            warm_start_gamma = resolve_case_mix_gamma(case_cfg, case_cfg['warm_start_method'])
            gamma_text = '' if warm_start_gamma is None else f", gamma={warm_start_gamma}"
            print(
                f"Using UNet warm-start: method={case_cfg['warm_start_method']}, "
                f"source_root={resolve_output_root(project_root, resolve_variant_output_dir(case_cfg['warm_start_output_dir'], use_batch_norm=case_cfg.get('unet_use_batch_norm', False)))}"
                f"{gamma_text}"
            )
        if gamma_by_noise is not None:
            gamma_text = ', '.join(
                f"{noise:g}%->{info['gamma']:.3e}" for noise, info in sorted(gamma_by_noise.items())
            )
            print(f"Reusing cold-start gamma by noise: {gamma_text}")
        print(f"Reusing saved displacement fields from {output_root}")

        panel_data = predict_asm_panel(
            asm_ctx=asm_ctx,
            asm_dof_order=case_cfg['asm_dof_order'],
            input_sample=input_sample,
            target_sample=target_sample,
            noise_levels=case_cfg['noise_levels'],
            sample_seed=resolved_idx,
            enable_lcurve=case_cfg['enable_lcurve'],
            lcurve_points=case_cfg['lcurve_points'],
            lcurve_gamma_min=case_cfg['lcurve_gamma_min'],
            lcurve_gamma_max=case_cfg['lcurve_gamma_max'],
            gamma_by_noise=gamma_by_noise,
            init_by_noise=warm_start_map,
            displacement_by_noise=displacement_by_noise,
            return_raw=True,
        )

        saved_dirs = _save_case_outputs(
            data_dir=data_dir,
            data_type=data_type,
            resolved_idx=resolved_idx,
            panel_data=panel_data,
            case_cfg=case_cfg,
            summary_rows=summary_rows,
        )
        total_time = time.time() - case_start_time
        print(
            f"Saved ASM results for dataset={data_type}, sample_index={resolved_idx}, "
            f"noise_levels={case_cfg['noise_levels']} (case_time={total_time:.2f}s)"
        )
        for save_dir in saved_dirs:
            config_suffix = get_asm_result_suffix(
                case_cfg.get('use_warm_start', False),
                case_cfg.get('warm_start_method', None),
            )
            print(f"Saved ASM case config: {os.path.join(save_dir, f'config{config_suffix}.py')}")

    normalized_cases = [_normalize_case_cfg(cfg, case) for case in cases]
    warm_flags = {bool(case.get('use_warm_start', False)) for case in normalized_cases}
    use_warm_start = warm_flags.pop() if len(warm_flags) == 1 else False
    warm_methods = {
        case.get('warm_start_method', None)
        for case in normalized_cases
        if case.get('use_warm_start', False)
    }
    if len(warm_methods) == 1:
        warm_start_method = warm_methods.pop()
    elif len(warm_methods) > 1:
        warm_start_method = 'mixed'
    else:
        warm_start_method = None
    _write_summary(
        output_root,
        summary_rows,
        use_warm_start=use_warm_start,
        warm_start_method=warm_start_method,
        use_batch_norm=cfg.get('unet_use_batch_norm', False),
    )


def run_asm_inversion_batch(project_root, cfg):
    cases = []
    for data_type in cfg['datasets']:
        cases.append({
            'dataset': data_type,
            'sample_index': cfg['sample_index'].get(data_type, 0),
        })
    run_asm_inversion_cases(project_root=project_root, cfg=cfg, cases=cases)
