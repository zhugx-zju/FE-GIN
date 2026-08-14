"""Shared model, data, and metric helpers."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNET_ROOT = PROJECT_ROOT / 'igfe_unet'
for import_root in (PROJECT_ROOT, UNET_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from architectures.unet import UNet
from .metrics import apply_relative_noise, field_metrics


METHOD_ORDER = ('MSE-M', 'LM-M', 'GM-M')
METHOD_COLORS = {'MSE-M': '#d62728', 'LM-M': '#9467bd', 'GM-M': '#ff7f0e'}
METHOD_MARKERS = {'MSE-M': 'o', 'LM-M': 's', 'GM-M': '^'}
GRF_CONDITION_ORDER = ('grf_l25', 'grf_l20', 'grf_l15', 'grf_l10', 'grf_l8')
CONDITION_DISPLAY = {
    'grf_l25': 'GRF-$l=25$ mm',
    'grf_l20': 'GRF-$l=20$ mm',
    'grf_l15': 'GRF-$l=15$ mm',
    'grf_l10': 'GRF-$l=10$ mm',
    'grf_l8': 'GRF-$l=8$ mm',
    'grf_l5': 'GRF-$l=5$ mm',
    'steep_sigmoid': 'Continuous steep gradient',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_manifest(data_root):
    manifest_path = Path(data_root).resolve() / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'Generalization dataset manifest not found: {manifest_path}\n'
            'Run grf_generalization/run_generate_cases.py first.'
        )
    with manifest_path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('training_or_validation_use_permitted') is not False:
        raise ValueError('The generalization dataset is not explicitly marked test-only.')
    geometry = manifest['geometry']
    if (int(geometry['nodes_x']), int(geometry['nodes_y'])) != (40, 40):
        raise ValueError('The selected final U-Net models require 40 x 40 nodal fields.')
    return manifest_path, manifest


def load_condition(data_root, condition):
    condition_dir = Path(data_root).resolve() / condition['directory']
    inputs = np.asarray(loadmat(condition_dir / 'input.mat')['U'], dtype=np.float32)
    targets = np.asarray(loadmat(condition_dir / 'output.mat')['E'], dtype=np.float32)
    if inputs.ndim != 4 or inputs.shape[1] != 2:
        raise ValueError(f"Invalid input shape for {condition['condition_id']}: {inputs.shape}")
    expected_target_shape = (inputs.shape[0], inputs.shape[2], inputs.shape[3])
    if targets.shape != expected_target_shape:
        raise ValueError(
            f"Invalid target shape for {condition['condition_id']}: "
            f'{targets.shape}, expected {expected_target_shape}'
        )
    return inputs, targets


def resolve_device(requested_device):
    requested = str(requested_device).strip().lower()
    if requested.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: requested device '{requested_device}' is unavailable; using CPU.")
        return torch.device('cpu')
    return torch.device(requested or 'cpu')


def load_final_model(model_dir, device):
    model_dir = Path(model_dir).resolve()
    config_path = model_dir / 'config.py'
    checkpoint_path = model_dir / 'model.pt'
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f'Missing config.py or model.pt under {model_dir}')
    config = runpy.run_path(str(config_path))
    filters = list(config.get('filters_list', [2, 32, 64, 128]))
    kernel_size = int(config.get('kernel_size', 3))
    use_normalization = bool(config.get('use_batch_norm', False))
    network = UNet(filters, kernel_size, use_batch_norm=use_normalization)
    network.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    network.to(device)
    network.eval()
    return network, {
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'filters_list': filters,
        'kernel_size': kernel_size,
        'use_batch_norm': use_normalization,
        'training_method': config.get('method'),
        'gamma': config.get('gamma'),
    }


def predict_condition(network, device, inputs, noise_level, batch_size, noise_seed):
    predictions = []
    for start in range(0, len(inputs), int(batch_size)):
        stop = min(start + int(batch_size), len(inputs))
        noisy_batch = np.stack(
            [
                apply_relative_noise(
                    inputs[sample_id],
                    noise_level,
                    seed=int(noise_seed) + sample_id,
                )
                for sample_id in range(start, stop)
            ]
        )
        with torch.no_grad():
            output = network(torch.from_numpy(noisy_batch).float().to(device))[:, 0]
        predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def build_metric_row(model_label, condition, noise_level, sample_id, target, prediction):
    return {
        'model': model_label,
        'condition_id': condition['condition_id'],
        'field_type': condition['field_type'],
        'distribution_status': condition['distribution_status'],
        'correlation_length_mm': condition.get('correlation_length_mm', ''),
        'transition_width_10_90_mm': condition.get('transition_width_10_90_mm', ''),
        'noise_level_percent': float(noise_level),
        'sample_id': int(sample_id),
        **field_metrics(target, prediction),
    }
