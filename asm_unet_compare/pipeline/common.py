import csv
import importlib
import importlib.util
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# Minimal import bootstrap for igfe_unet modules.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
UNET_ROOT = os.path.join(PROJECT_ROOT, 'igfe_unet')
if not os.path.isdir(UNET_ROOT):
    raise FileNotFoundError(f'igfe_unet folder not found: {UNET_ROOT}')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if UNET_ROOT not in sys.path:
    sys.path.insert(0, UNET_ROOT)

from igfe_unet.model.train import Training
from igfe_unet.postprocess.common import find_all_experiments, load_experiment_results
from igfe_unet.utils.utils_process import (
    Config,
    batch_norm_enabled,
    batch_norm_suffix,
    format_decimal_token,
    network_variant_name,
)
from igfe_unet.utils.utils_test import (
    fixed_test_set_exists,
    generate_noise_data,
    get_fixed_test_data_dir,
    load_test_data,
)


METHOD_DISPLAY = {
    'ASM': 'Adjoint-State Method',
    'MSE': 'MSE-M',
    'GloResloss': 'GE-M',
    'LocResloss': 'LE-M',
    'GloMixloss': 'GM-M',
    'LocMixloss': 'LM-M',
}

MIX_METHODS = frozenset({'LocMixloss', 'GloMixloss'})
RELATIVE_ERROR_COLORBAR_MAX_PCT = 10.0
MEASURED_DISPLACEMENT_STEM = 'measured_displacement'


def sanitize_method_tag(method):
    import re

    tag = re.sub(r'[^0-9A-Za-z]+', '_', str(method)).strip('_').lower()
    return tag or 'unet'


def resolve_runtime_device(requested_device=None):
    device_text = 'cpu' if requested_device is None else str(requested_device).strip().lower()
    if device_text == '':
        device_text = 'cpu'

    if device_text.startswith('cuda') and not torch.cuda.is_available():
        print(
            f"Warning: requested device '{requested_device}' is unavailable because this PyTorch build "
            "does not provide CUDA. Falling back to CPU."
        )
        return 'cpu'

    mps_available = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    if device_text == 'mps' and not mps_available:
        print(
            f"Warning: requested device '{requested_device}' is unavailable on this machine. "
            "Falling back to CPU."
        )
        return 'cpu'

    return requested_device if requested_device is not None else device_text


def is_mix_method(method):
    return str(method) in MIX_METHODS


def parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == '':
            return None
        value = text
    return float(value)


def gamma_to_token(gamma):
    gamma_val = parse_optional_float(gamma)
    if gamma_val is None:
        return None
    return format_decimal_token(gamma_val)


def gamma_close(lhs, rhs, rtol=1e-8, atol=1e-12):
    lhs_val = parse_optional_float(lhs)
    rhs_val = parse_optional_float(rhs)
    if lhs_val is None or rhs_val is None:
        return False
    return bool(np.isclose(lhs_val, rhs_val, rtol=rtol, atol=atol))


def select_requested_mix_gamma(method, mix_gamma=None, mix_gamma_by_method=None):
    method_name = str(method)
    if not is_mix_method(method_name):
        return None

    if isinstance(mix_gamma_by_method, dict):
        if method_name in mix_gamma_by_method:
            return parse_optional_float(mix_gamma_by_method[method_name])
        for key, value in mix_gamma_by_method.items():
            if str(key) == method_name:
                return parse_optional_float(value)

    return parse_optional_float(mix_gamma)


def normalize_mix_gamma_map(mix_gamma_by_method):
    if not isinstance(mix_gamma_by_method, dict):
        return None
    normalized = {}
    for key, value in mix_gamma_by_method.items():
        normalized[str(key)] = parse_optional_float(value)
    return normalized


def resolve_case_mix_gamma(case_cfg, method):
    return select_requested_mix_gamma(
        method=method,
        mix_gamma=case_cfg.get('mix_gamma', None),
        mix_gamma_by_method=case_cfg.get('mix_gamma_by_method', None),
    )


def build_unet_file_stem(method, gamma=None):
    method_tag = sanitize_method_tag(method)
    stem = f'unet_{method_tag}'
    if is_mix_method(method):
        gamma_token = gamma_to_token(gamma)
        if gamma_token:
            stem = f'{stem}_gamma_{gamma_token}'
    return stem


def unique_preserve_order(values):
    unique_values = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def get_asm_result_suffix(use_warm_start=False, warm_start_method=None):
    if not bool(use_warm_start):
        return ''
    if warm_start_method:
        return f"_warm_start_{sanitize_method_tag(warm_start_method)}"
    return '_warm_start'


def get_model_variant_suffix(use_batch_norm=False):
    if not bool(use_batch_norm):
        return ''
    return batch_norm_suffix({'use_batch_norm': bool(use_batch_norm)})


def resolve_variant_output_dir(output_dir, use_batch_norm=False):
    if not bool(use_batch_norm):
        return output_dir
    suffix = get_model_variant_suffix(use_batch_norm=use_batch_norm)
    if str(output_dir).endswith(suffix):
        return output_dir
    return f"{output_dir}{suffix}"


def get_asm_result_suffix_candidates(use_warm_start=False, warm_start_method=None):
    if not bool(use_warm_start):
        return ['']

    candidates = [get_asm_result_suffix(True, warm_start_method)]
    generic = '_warm_start'
    if generic not in candidates:
        candidates.append(generic)
    return candidates


def resolve_data_path_for_type(mix_data_path, data_type):
    if data_type == 'mix':
        return mix_data_path

    new_path = mix_data_path.replace('/data_mix/', f'/data_{data_type}/')
    if new_path == mix_data_path:
        new_path = mix_data_path.replace('\\data_mix\\', f'\\data_{data_type}\\')
    if new_path == mix_data_path:
        new_path = mix_data_path.replace('data_mix', f'data_{data_type}')
    return new_path


def load_sample(config_type, load_type, data_type, sample_index):
    cfg = Config(config_type)
    cfg.config_type = config_type
    cfg.load_type = load_type
    cfg.device = resolve_runtime_device(getattr(cfg, 'device', 'cpu'))
    cfg.data_path = resolve_data_path_for_type(cfg.data_path, data_type)
    if not fixed_test_set_exists(cfg):
        fixed_dir = get_fixed_test_data_dir(cfg)
        raise FileNotFoundError(
            f"Shared fixed test set is required for asm_unet_compare but was not found.\n"
            f"Expected directory: {fixed_dir}\n"
            f"Please run: python igfe_unet/script/generate_fixed_test_set.py"
        )
    inputs, targets = load_test_data(cfg)
    idx = min(max(sample_index, 0), inputs.shape[0] - 1)
    return inputs[idx], targets[idx], idx


def _to_numpy_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_displacement_components(input_data):
    input_np = np.asarray(_to_numpy_array(input_data), dtype=np.float64)
    if input_np.ndim == 4:
        if input_np.shape[0] != 1:
            raise ValueError(
                f'Expected batched displacement input with batch size 1, got shape {tuple(input_np.shape)}'
            )
        input_np = input_np[0]
    if input_np.ndim != 3 or input_np.shape[0] < 2:
        raise ValueError(
            f'Expected displacement input shaped like [2, H, W], got {tuple(input_np.shape)}'
        )
    ux = np.asarray(input_np[0], dtype=np.float64)
    uy = np.asarray(input_np[1], dtype=np.float64)
    return input_np, ux, uy


def build_noisy_input_by_noise(input_sample, noise_levels, sample_seed):
    noisy_input_by_noise = {}
    for noise in noise_levels:
        noise_key = float(noise)
        input_tmp = input_sample.clone()
        if noise_key > 0:
            input_tmp = generate_noise_data(input_tmp, noise_key, seed=sample_seed)
        noisy_input_by_noise[noise_key] = input_tmp.detach().cpu().clone()
    return noisy_input_by_noise


def _draw_displacement_panel(ax, values, mesh_info=None, contour_fn=None, cmap='viridis'):
    if contour_fn is not None and mesh_info is not None:
        im = contour_fn(mesh_info.plot_x, mesh_info.plot_y, values, ax, levels=128, cmap=cmap)
        ax.axis('equal')
        ax.axis('off')
    else:
        im = ax.imshow(values, cmap=cmap, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    return im


def save_measured_displacement(
    noise_dir,
    noisy_input,
    data_type,
    sample_index,
    noise_level,
    sample_seed=None,
    dpi=600,
    mesh_info=None,
    contour_fn=None,
):
    os.makedirs(noise_dir, exist_ok=True)

    input_np, ux, uy = extract_displacement_components(noisy_input)
    npz_path = os.path.join(noise_dir, f'{MEASURED_DISPLACEMENT_STEM}.npz')
    np.savez(
        npz_path,
        input=input_np,
        ux=ux,
        uy=uy,
        U_measured_C=input_to_dof_vector(input_np, order='C'),
        U_measured_F=input_to_dof_vector(input_np, order='F'),
        data_type=np.asarray(str(data_type)),
        sample_index=int(sample_index),
        noise_level=float(noise_level),
        sample_seed=-1 if sample_seed is None else int(sample_seed),
    )

    png_path = os.path.join(noise_dir, f'{MEASURED_DISPLACEMENT_STEM}.png')
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, field, title in zip(
        axes,
        [ux, uy],
        ['X-Displacement $u_x$', 'Y-Displacement $u_y$'],
    ):
        im = _draw_displacement_panel(ax, field, mesh_info=mesh_info, contour_fn=contour_fn, cmap='viridis')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=7)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f'{str(data_type).upper()} Sample {int(sample_index)} - Noise {float(noise_level):g}%',
        fontsize=12,
        fontweight='bold',
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

    return {
        'npz_path': npz_path,
        'png_path': png_path,
    }


def load_measured_displacement(npz_path=None, asm_dof_order='C', npz_file=None):
    if npz_path is None:
        npz_path = npz_file
    if npz_path is None:
        raise ValueError("Either 'npz_path' or 'npz_file' must be provided.")

    with np.load(npz_path, allow_pickle=False) as data:
        input_np = np.asarray(data['input'], dtype=np.float64)
        order_key = f'U_measured_{str(asm_dof_order).upper()}'
        if order_key in data.files:
            U_measured = np.asarray(data[order_key], dtype=np.float64)
        else:
            U_measured = input_to_dof_vector(input_np, order=asm_dof_order)

        if 'sample_index' in data.files:
            sample_index = int(np.asarray(data['sample_index']).item())
        else:
            sample_index = None
        if 'noise_level' in data.files:
            noise_level = float(np.asarray(data['noise_level']).item())
        else:
            noise_level = None
        if 'sample_seed' in data.files:
            sample_seed = int(np.asarray(data['sample_seed']).item())
            if sample_seed < 0:
                sample_seed = None
        else:
            sample_seed = None

    return {
        'input': input_np,
        'ux': np.asarray(input_np[0], dtype=np.float64),
        'uy': np.asarray(input_np[1], dtype=np.float64),
        'U_measured': U_measured,
        'sample_index': sample_index,
        'noise_level': noise_level,
        'sample_seed': sample_seed,
    }


def _purge_module_tree(module_name):
    """Remove a module and all its children from sys.modules."""
    to_delete = [name for name in sys.modules if name == module_name or name.startswith(f'{module_name}.')]
    for name in to_delete:
        del sys.modules[name]


def find_best_experiment(
    experiments,
    config_type,
    load_type,
    architecture,
    method,
    gamma=None,
    use_batch_norm=False,
):
    best = None
    best_valid_mae = np.inf
    requested_gamma = parse_optional_float(gamma)

    for exp_info in experiments:
        exp_config_type, exp_load_type, _, exp_path = exp_info
        if exp_config_type != config_type or exp_load_type != load_type:
            continue

        results = load_experiment_results(exp_path)
        cfg = results.get('config') or {}
        if cfg.get('method') != method:
            continue
        if batch_norm_enabled(cfg) != bool(use_batch_norm):
            continue
        if str(cfg.get('filters_list', '')) != architecture:
            continue
        if requested_gamma is not None and is_mix_method(method):
            cfg_gamma = parse_optional_float(cfg.get('gamma', None))
            if cfg_gamma is None or not gamma_close(cfg_gamma, requested_gamma):
                continue
        if not os.path.exists(os.path.join(exp_path, 'model.pt')):
            continue

        valid_mae = results.get('valid_mae')
        if valid_mae is None or len(valid_mae) == 0:
            score = np.inf
        else:
            score = float(np.min(valid_mae))

        if score < best_valid_mae:
            best_valid_mae = score
            best = (exp_info, results, score)

    return best


def find_available_mix_gammas(
    experiments,
    config_type,
    load_type,
    architecture,
    method,
    use_batch_norm=False,
):
    if not is_mix_method(method):
        return []

    gamma_values = set()
    for exp_info in experiments:
        exp_config_type, exp_load_type, _, exp_path = exp_info
        if exp_config_type != config_type or exp_load_type != load_type:
            continue
        if not os.path.exists(os.path.join(exp_path, 'model.pt')):
            continue

        results = load_experiment_results(exp_path)
        cfg = results.get('config') or {}
        if cfg.get('method') != method:
            continue
        if batch_norm_enabled(cfg) != bool(use_batch_norm):
            continue
        if str(cfg.get('filters_list', '')) != architecture:
            continue

        gamma_value = parse_optional_float(cfg.get('gamma', None))
        if gamma_value is not None:
            gamma_values.add(float(gamma_value))

    return sorted(gamma_values)


def load_unet_from_experiment(config_type, load_type, exp_path, exp_cfg):
    cfg = Config(config_type)
    cfg.config_type = config_type
    cfg.load_type = load_type
    cfg.device = 'cpu'
    cfg.method = exp_cfg.get('method', cfg.method)
    if 'filters_list' in exp_cfg:
        cfg.filters_list = exp_cfg['filters_list']
    if 'kernel_size' in exp_cfg:
        cfg.kernel_size = exp_cfg['kernel_size']
    if 'use_batch_norm' in exp_cfg:
        cfg.use_batch_norm = exp_cfg['use_batch_norm']

    training = Training(cfg)
    net = training.select_network(cfg.filters_list, cfg.kernel_size)
    ckpt_path = os.path.join(exp_path, 'model.pt')
    net.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=True))
    net.to(cfg.device)
    net.eval()
    return net, cfg.device


def compute_error_fields(target_np, pred_np):
    abs_err = np.abs(target_np - pred_np)
    rel_err = np.zeros_like(target_np)
    mask = ~np.isclose(target_np, 0)
    rel_err[mask] = (abs_err[mask] / target_np[mask]) * 100.0
    target_flat = np.asarray(target_np).reshape(-1)
    pred_flat = np.asarray(pred_np).reshape(-1)
    denom = np.sum(np.abs(target_flat))
    if np.isclose(denom, 0.0):
        l1 = 0.0
    else:
        l1 = np.sum(np.abs(target_flat - pred_flat)) / denom
    mae_pct = float(np.mean(rel_err))
    return rel_err, float(l1), mae_pct


def _synchronize_for_timing(device):
    device_text = str(device).strip().lower()
    if device_text.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_forward_time(net, input_tmp, device):
    _synchronize_for_timing(device)
    start_time = time.perf_counter()
    output_tmp = net(input_tmp)
    _synchronize_for_timing(device)
    return output_tmp, float(time.perf_counter() - start_time)


def predict_unet_panel(
    net,
    device,
    input_sample,
    target_sample,
    noise_levels,
    sample_seed,
    noisy_input_by_noise=None,
):
    target_np = target_sample.squeeze().detach().cpu().numpy()
    panel_data = []

    with torch.no_grad():
        for noise in noise_levels:
            noise_key = float(noise)
            if noisy_input_by_noise is not None and noise_key in noisy_input_by_noise:
                saved_input = noisy_input_by_noise[noise_key]
                if isinstance(saved_input, torch.Tensor):
                    input_tmp = saved_input.clone().to(device)
                else:
                    input_tmp = torch.tensor(
                        np.asarray(saved_input),
                        dtype=input_sample.dtype,
                        device=device,
                    )
            else:
                input_tmp = input_sample.clone().to(device)
                if noise_key > 0:
                    input_tmp = generate_noise_data(input_tmp, noise_key, seed=sample_seed)
            if input_tmp.dim() == 3:
                input_tmp = input_tmp.unsqueeze(0)

            output_tmp, elapsed_time = _measure_forward_time(
                net=net,
                input_tmp=input_tmp,
                device=device,
            )
            pred_np = output_tmp.squeeze().detach().cpu().numpy()
            rel_err, l1, mae_pct = compute_error_fields(target_np, pred_np)

            panel_data.append({
                'noise': noise,
                'target': target_np,
                'pred': pred_np,
                'rel_err': rel_err,
                'l1': l1,
                'mae_pct': mae_pct,
                'elapsed_time': float(elapsed_time),
            })

    return panel_data


def load_asm_modules(project_root):
    asm_root = os.path.join(project_root, 'asm_log')
    if not os.path.isdir(asm_root):
        raise FileNotFoundError(f'asm_log folder not found: {asm_root}')

    _purge_module_tree('fgm_asm')
    if asm_root in sys.path:
        sys.path.remove(asm_root)
    sys.path.insert(0, asm_root)

    asm_config_path = os.path.join(asm_root, 'config.py')
    spec = importlib.util.spec_from_file_location('asm_log_config', asm_config_path)
    asm_cfg = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f'Cannot load asm_log config from {asm_config_path}')
    spec.loader.exec_module(asm_cfg)

    asm_pkg = importlib.import_module('fgm_asm')
    asm_lcurve_mod = importlib.import_module('fgm_asm.l_curve')
    asm_mesh_mod = importlib.import_module('fgm_asm.mesh')
    asm_vis_mod = importlib.import_module('fgm_asm.visualization')

    return {
        'cfg_module': asm_cfg,
        'MeshInfo': asm_pkg.MeshInfo,
        'setup_boundary_conditions': asm_mesh_mod.setup_boundary_conditions,
        'lbfgs_inverse_solver_scipy': asm_pkg.lbfgs_inverse_solver_scipy,
        'find_optimal_gamma_lcurve': asm_lcurve_mod.find_optimal_gamma_lcurve,
        'create_smooth_contour': asm_vis_mod.create_smooth_contour,
    }


def build_asm_context(project_root, nodesx, nodesy, asm_gamma, asm_max_iter, asm_ftol, asm_gtol):
    asm_mods = load_asm_modules(project_root)
    asm_cfg = asm_mods['cfg_module']
    forward_cfg = asm_cfg.get_forward_config()
    inverse_cfg = asm_cfg.get_inverse_config()
    lcurve_cfg = asm_cfg.get_lcurve_config()

    nel_x = int(nodesx) - 1
    nel_y = int(nodesy) - 1
    if nel_x != int(forward_cfg.nel_x) or nel_y != int(forward_cfg.nel_y):
        print(
            "Warning: UNet mesh and asm_log config mesh mismatch. "
            f"Using UNet mesh {nel_x}x{nel_y} for asm_log inversion."
        )

    mesh = asm_mods['MeshInfo'](
        float(forward_cfg.geo_l),
        float(forward_cfg.geo_h),
        nel_x,
        nel_y,
    )
    bc_info = asm_mods['setup_boundary_conditions'](
        mesh,
        float(forward_cfg.geo_l),
        float(forward_cfg.geo_h),
        float(forward_cfg.f_tot),
    )
    mesh.assemble_mass_matrix()

    return {
        'mesh': mesh,
        'bc_info': bc_info,
        'forward_cfg': forward_cfg,
        'nu': float(forward_cfg.nu),
        'gamma': float(inverse_cfg.gamma) if asm_gamma is None else float(asm_gamma),
        'E_min': float(inverse_cfg.E_min),
        'E_max': float(inverse_cfg.E_max),
        'max_iter': int(asm_max_iter),
        'ftol': float(asm_ftol),
        'gtol': float(asm_gtol),
        'solver': asm_mods['lbfgs_inverse_solver_scipy'],
        'nodesx': int(nodesx),
        'nodesy': int(nodesy),
        'lcurve_cfg': lcurve_cfg,
        'find_optimal_gamma_lcurve': asm_mods['find_optimal_gamma_lcurve'],
        'create_smooth_contour': asm_mods['create_smooth_contour'],
    }


def input_to_dof_vector(input_np, order='C'):
    ux = input_np[0]
    uy = input_np[1]
    ux_flat = ux.reshape(-1, order=order)
    uy_flat = uy.reshape(-1, order=order)

    dof = np.empty(2 * ux_flat.size, dtype=np.float64)
    dof[0::2] = ux_flat
    dof[1::2] = uy_flat
    return dof


def asm_vector_to_field(E_vec, nodesx, nodesy):
    return E_vec.reshape(nodesx, nodesy, order='F').T


def asm_field_to_vector(E_field):
    return np.asarray(E_field, dtype=np.float64).T.reshape(-1, order='F')


def save_lcurve_data(lcurve_results, save_dir, filename_suffix=''):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f'lcurve_analysis{filename_suffix}.pkl'), 'wb') as f:
        pickle.dump(lcurve_results, f)

    csv_path = os.path.join(save_dir, f'lcurve_curve{filename_suffix}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gamma', 'residual_norm', 'regularization_norm', 'curvature'])
        for g, r, reg, c in zip(
            lcurve_results['gamma_values'],
            lcurve_results['residual_norms'],
            lcurve_results['regularization_norms'],
            lcurve_results['curvature'],
        ):
            writer.writerow([float(g), float(r), float(reg), float(c)])
    return csv_path


def predict_asm_panel(
    asm_ctx,
    asm_dof_order,
    input_sample,
    target_sample,
    noise_levels,
    sample_seed,
    enable_lcurve=False,
    lcurve_points=25,
    lcurve_gamma_min=None,
    lcurve_gamma_max=None,
    gamma_by_noise=None,
    init_by_noise=None,
    displacement_by_noise=None,
    return_raw=False,
):
    target_np = target_sample.squeeze().detach().cpu().numpy()
    panel_data = []

    for noise in noise_levels:
        total_start_time = time.time()
        noise_key = float(noise)
        displacement_info = None if displacement_by_noise is None else displacement_by_noise.get(noise_key)
        if displacement_info is not None:
            if 'input' not in displacement_info:
                raise KeyError(
                    f"Saved displacement entry for noise={noise_key:g}% does not contain 'input'."
                )
            input_np = np.asarray(displacement_info['input'], dtype=np.float64)
            if 'U_measured' in displacement_info and displacement_info['U_measured'] is not None:
                U_measured = np.asarray(displacement_info['U_measured'], dtype=np.float64)
            else:
                U_measured = input_to_dof_vector(input_np, order=asm_dof_order)
            displacement_source = displacement_info.get('source', 'saved')
            displacement_source_file = displacement_info.get('source_file', None)
        else:
            input_tmp = input_sample.clone()
            if noise_key > 0:
                input_tmp = generate_noise_data(input_tmp, noise_key, seed=sample_seed)
            input_np = input_tmp.detach().cpu().numpy()
            U_measured = input_to_dof_vector(input_np, order=asm_dof_order)
            displacement_source = 'generated'
            displacement_source_file = None
        init_info = None if init_by_noise is None else init_by_noise.get(float(noise))
        E_init = None if init_info is None else np.asarray(init_info['E_init'], dtype=np.float64).copy()
        gamma_info = None if gamma_by_noise is None else gamma_by_noise.get(float(noise))
        if isinstance(gamma_info, dict):
            gamma_override = gamma_info.get('gamma', None)
            gamma_source_file = gamma_info.get('source_file', None)
            gamma_source_selection_mode = gamma_info.get('selection_mode', None)
        else:
            gamma_override = gamma_info
            gamma_source_file = None
            gamma_source_selection_mode = None

        used_gamma = asm_ctx['gamma']
        lcurve_results = None
        results = None
        gamma_selected_from_lcurve = None
        lcurve_elapsed_time = 0.0
        final_run_elapsed_time = 0.0
        warm_start_elapsed_time = 0.0 if init_info is None else float(init_info.get('elapsed_time', 0.0))
        selection_mode = 'fixed_gamma'
        gamma_source = 'config_fixed'

        if gamma_override is not None:
            used_gamma = float(gamma_override)
            selection_mode = 'fixed_gamma_from_cold_start'
            gamma_source = 'cold_start_case'

            final_run_start_time = time.time()
            results = asm_ctx['solver'](
                mesh_info=asm_ctx['mesh'],
                bc_info=asm_ctx['bc_info'],
                U_measured=U_measured,
                E_init=E_init,
                gamma=used_gamma,
                E_min=asm_ctx['E_min'],
                E_max=asm_ctx['E_max'],
                max_iter=asm_ctx['max_iter'],
                ftol=asm_ctx['ftol'],
                gtol=asm_ctx['gtol'],
                nu=asm_ctx['nu'],
            )
            final_run_elapsed_time = float(time.time() - final_run_start_time)
        elif enable_lcurve:
            gamma_min_raw = asm_ctx['lcurve_cfg'].gamma_min if lcurve_gamma_min is None else float(lcurve_gamma_min)
            gamma_max_raw = asm_ctx['lcurve_cfg'].gamma_max if lcurve_gamma_max is None else float(lcurve_gamma_max)

            if float(gamma_min_raw) <= 0.0 or float(gamma_max_raw) <= 0.0:
                raise ValueError(
                    f"L-curve gamma bounds must be positive, got gamma_min={gamma_min_raw}, "
                    f"gamma_max={gamma_max_raw}"
                )

            # Enforce valid bounds so the downstream L-curve routine always scans gamma
            # from large to small (warm-start continuation remains stable).
            gamma_min = float(min(gamma_min_raw, gamma_max_raw))
            gamma_max = float(max(gamma_min_raw, gamma_max_raw))
            if gamma_min != float(gamma_min_raw) or gamma_max != float(gamma_max_raw):
                print(
                    "Warning: lcurve_gamma_min/lcurve_gamma_max were reversed; "
                    f"using gamma_max={gamma_max:.3e} -> gamma_min={gamma_min:.3e}"
                )

            lcurve_start_time = time.time()
            gamma_opt, lcurve_results = asm_ctx['find_optimal_gamma_lcurve'](
                mesh_info=asm_ctx['mesh'],
                bc_info=asm_ctx['bc_info'],
                U_measured=U_measured,
                config=asm_ctx['forward_cfg'],
                gamma_min=float(gamma_min),
                gamma_max=float(gamma_max),
                n_gamma=int(lcurve_points),
                E_min=asm_ctx['E_min'],
                E_max=asm_ctx['E_max'],
                max_iter=asm_ctx['max_iter'],
                ftol=asm_ctx['ftol'],
                gtol=asm_ctx['gtol'],
                E_init=E_init,
            )
            lcurve_elapsed_time = float(time.time() - lcurve_start_time)
            gamma_selected_from_lcurve = float(gamma_opt)
            used_gamma = gamma_selected_from_lcurve
            selection_mode = 'lcurve_then_rerun'
            gamma_source = 'current_lcurve'

            # After selecting gamma from the scan, rerun the inverse solve with
            # that fixed gamma using the same initialization policy as the scan.
            final_run_start_time = time.time()
            results = asm_ctx['solver'](
                mesh_info=asm_ctx['mesh'],
                bc_info=asm_ctx['bc_info'],
                U_measured=U_measured,
                E_init=E_init,
                gamma=used_gamma,
                E_min=asm_ctx['E_min'],
                E_max=asm_ctx['E_max'],
                max_iter=asm_ctx['max_iter'],
                ftol=asm_ctx['ftol'],
                gtol=asm_ctx['gtol'],
                nu=asm_ctx['nu'],
            )
            final_run_elapsed_time = float(time.time() - final_run_start_time)
            lcurve_results['selection_mode'] = selection_mode
            lcurve_results['gamma_selected_from_lcurve'] = gamma_selected_from_lcurve
            lcurve_results['lcurve_elapsed_time'] = lcurve_elapsed_time
            lcurve_results['final_run_gamma'] = used_gamma
            lcurve_results['final_run_elapsed_time'] = final_run_elapsed_time

            print(
                f"  Re-ran ASM with selected gamma = {used_gamma:.6e} "
                f"using the same initialization policy"
            )
        else:
            final_run_start_time = time.time()
            results = asm_ctx['solver'](
                mesh_info=asm_ctx['mesh'],
                bc_info=asm_ctx['bc_info'],
                U_measured=U_measured,
                E_init=E_init,
                gamma=asm_ctx['gamma'],
                E_min=asm_ctx['E_min'],
                E_max=asm_ctx['E_max'],
                max_iter=asm_ctx['max_iter'],
                ftol=asm_ctx['ftol'],
                gtol=asm_ctx['gtol'],
                nu=asm_ctx['nu'],
            )
            final_run_elapsed_time = float(time.time() - final_run_start_time)

        pred_np = asm_vector_to_field(results['E_final'], asm_ctx['nodesx'], asm_ctx['nodesy'])
        rel_err, l1, mae_pct = compute_error_fields(target_np, pred_np)

        row = {
            'noise': noise,
            'target': target_np,
            'pred': pred_np,
            'rel_err': rel_err,
            'l1': l1,
            'mae_pct': mae_pct,
            'n_iterations': int(results.get('n_iterations', -1)),
            'converged': bool(results.get('converged', False)),
            'asm_gamma': float(used_gamma),
            'gamma_selected_from_lcurve': gamma_selected_from_lcurve,
            'selection_mode': selection_mode,
            'gamma_source': gamma_source,
            'gamma_source_file': gamma_source_file,
            'gamma_source_selection_mode': gamma_source_selection_mode,
            'lcurve_elapsed_time': lcurve_elapsed_time,
            'final_run_elapsed_time': final_run_elapsed_time,
            'warm_start_elapsed_time': warm_start_elapsed_time,
            'end_to_end_elapsed_time': float(time.time() - total_start_time) + warm_start_elapsed_time,
            'elapsed_time': float(time.time() - total_start_time),
            'used_warm_start': bool(init_info is not None),
            'warm_start_method': None if init_info is None else init_info.get('method', None),
            'warm_start_source': None if init_info is None else init_info.get('source_file', None),
            'warm_start_unet_gamma': None if init_info is None else init_info.get('unet_gamma_used', None),
            'warm_start_init_l1': None if init_info is None else init_info.get('l1', None),
            'warm_start_init_mae_pct': None if init_info is None else init_info.get('mae_pct', None),
            'displacement_source': displacement_source,
            'displacement_source_file': displacement_source_file,
        }
        if return_raw:
            row['solver_results'] = results
            row['lcurve_results'] = lcurve_results
        panel_data.append(row)

    return panel_data


def align_panel_by_noise(panel_data):
    aligned = {}
    for row in panel_data:
        aligned[float(row['noise'])] = row
    return aligned


def resolve_output_root(project_root, output_dir):
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.abspath(os.path.join(project_root, output_dir))


def candidate_data_dirs(output_root, data_type, sample_index):
    dirs = [os.path.join(output_root, f"sample_{sample_index}", data_type)]
    if os.path.isdir(output_root):
        for sample_name in sorted(os.listdir(output_root)):
            candidate_dir = os.path.join(output_root, sample_name, data_type)
            if candidate_dir not in dirs:
                dirs.append(candidate_dir)
    return dirs


def candidate_asm_dirs(output_root, data_type, sample_index):
    base_dir = os.path.join(output_root, f"sample_{sample_index}", data_type)
    dirs = [base_dir, os.path.join(base_dir, 'asm')]
    if os.path.isdir(output_root):
        for sample_name in sorted(os.listdir(output_root)):
            sample_base_dir = os.path.join(output_root, sample_name, data_type)
            for candidate_dir in [sample_base_dir, os.path.join(sample_base_dir, 'asm')]:
                if candidate_dir not in dirs:
                    dirs.append(candidate_dir)
    return dirs


def load_asm_noise_result(output_root, data_type, sample_index, noise_level, filename_suffix=''):
    noise_tag = format_decimal_token(noise_level)
    if isinstance(filename_suffix, (list, tuple)):
        filename_suffixes = list(filename_suffix)
    else:
        filename_suffixes = [filename_suffix]

    for asm_dir in candidate_asm_dirs(output_root, data_type, sample_index):
        for suffix in filename_suffixes:
            pkl_file = os.path.join(asm_dir, f"noise_{noise_tag}", f'asm_results{suffix}.pkl')
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as f:
                asm_data = pickle.load(f)
            case_cfg = asm_data.get('case_config', {})
            if not isinstance(case_cfg, dict):
                case_cfg = {}
            requested_idx = case_cfg.get('sample_index', asm_data.get('sample_index', -1))
            saved_idx = int(asm_data.get('sample_index', sample_index))
            if int(requested_idx) == int(sample_index) or saved_idx == int(sample_index):
                return pkl_file, asm_data

    return None, None


def load_asm_panel_data(output_root, data_type, sample_index, noise_levels, filename_suffix=''):
    panel_data = []
    asm_root = None
    resolved_idx = None
    saved_case_cfg = None

    for noise in noise_levels:
        pkl_file, asm_data = load_asm_noise_result(
            output_root=output_root,
            data_type=data_type,
            sample_index=sample_index,
            noise_level=noise,
            filename_suffix=filename_suffix,
        )
        if asm_data is None:
            continue
        if asm_root is None:
            asm_root = os.path.dirname(os.path.dirname(pkl_file))
            resolved_idx = int(asm_data.get('sample_index', sample_index))
            saved_case_cfg = asm_data.get('case_config', None)
        panel_data.extend(asm_data.get('panel_data', []))

    return asm_root, panel_data, resolved_idx, saved_case_cfg


def plot_lcurve_png(lcurve_results, save_dir, dpi=600, filename_suffix=''):
    gamma_values = np.asarray(lcurve_results['gamma_values'], dtype=float)
    residual_norms = np.asarray(lcurve_results['residual_norms'], dtype=float)
    regularization_norms = np.asarray(lcurve_results['regularization_norms'], dtype=float)
    curvature = np.asarray(lcurve_results['curvature'], dtype=float)
    gamma_opt = float(lcurve_results['gamma_optimal'])
    idx_opt = int(lcurve_results['optimal_idx'])

    order = np.argsort(gamma_values)
    gv = gamma_values[order]
    rn = residual_norms[order]
    reg = regularization_norms[order]
    curv = curvature[order]
    idx_opt_sorted = int(np.where(order == idx_opt)[0][0])

    fig1 = plt.figure(figsize=(7.5, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.loglog(rn, reg, '-o', linewidth=1.8, markersize=4.5, label='L-curve')
    ax1.loglog(rn[idx_opt_sorted], reg[idx_opt_sorted], 'r*', markersize=14, label=f'opt gamma={gamma_opt:.2e}')
    ax1.set_xlabel(r'Data mismatch $||u-u_{obs}||_2$')
    ax1.set_ylabel(r'Regularization $||\nabla E||_2$')
    ax1.set_title('L-curve')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.legend(loc='best', fontsize=9, frameon=True, edgecolor='black')
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, f'lcurve{filename_suffix}.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plt.figure(figsize=(7.5, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    valid = np.isfinite(curv)
    ax2.plot(gv[valid], curv[valid], '-o', linewidth=1.8, markersize=4.5, label='Curvature')
    if np.isfinite(curv[idx_opt_sorted]):
        ax2.plot(gamma_opt, curv[idx_opt_sorted], 'r*', markersize=14, label=f'opt gamma={gamma_opt:.2e}')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Regularization Parameter $\gamma$')
    ax2.set_ylabel('Curvature')
    ax2.set_title('Curvature vs gamma')
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend(loc='best', fontsize=9, frameon=True, edgecolor='black')
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f'curvature_vs_gamma{filename_suffix}.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig2)


def plot_iteration_png(results, save_dir, noise_percent, dpi=600, filename_suffix=''):
    cost = np.asarray(results.get('cost_history', []), dtype=float)
    cost_tar = np.asarray(results.get('cost_tar_history', []), dtype=float)
    cost_reg = np.asarray(results.get('cost_reg_history', []), dtype=float)
    grad = np.asarray(results.get('grad_norm_history', []), dtype=float)

    if len(cost) == 0:
        return

    it = np.arange(len(cost))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].semilogy(it, cost, 'b-', linewidth=1.8, label='Total')
    if len(cost_tar) == len(cost):
        axes[0].semilogy(it, cost_tar, 'r--', linewidth=1.4, label='Data')
    if len(cost_reg) == len(cost):
        axes[0].semilogy(it, cost_reg, 'g-.', linewidth=1.4, label='Reg')
    axes[0].set_title('Convergence')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend(loc='best', fontsize=8, frameon=True, edgecolor='black')

    if len(grad) > 0:
        axes[1].semilogy(np.arange(len(grad)), grad, 'k-', linewidth=1.8)
    axes[1].set_title('Gradient Norm')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('||grad||')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    if len(grad) > 0:
        rel = grad / (grad[0] + 1e-15)
        axes[2].semilogy(np.arange(len(rel)), rel, 'm-', linewidth=1.8)
    axes[2].set_title('Relative Gradient Norm')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('||grad|| / ||grad||0')
    axes[2].grid(True, linestyle='--', alpha=0.3)

    fig.suptitle(f'ASM Iteration History (Noise={noise_percent:g}%)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        os.path.join(save_dir, f'iteration_history_noise_{format_decimal_token(noise_percent)}{filename_suffix}.png'),
        dpi=dpi,
        bbox_inches='tight',
    )
    plt.close(fig)
