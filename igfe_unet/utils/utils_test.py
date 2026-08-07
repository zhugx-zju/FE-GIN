import torch
import scipy.io as scio
import numpy as np
from pathlib import Path
from utils.utils_training import load_data

def generate_noise_data(org_data, noise_level=1, seed=None):
    """
    Generate noisy data following Equation 19 in the paper.

    The noise is added such that:
    ||noise_data - org_data||_2 / ||org_data||_2 = noise_level / 100

    Args:
        org_data: Original data (torch.Tensor or numpy.ndarray)
        noise_level: Noise level in percentage (e.g., 1 for 1%)
        seed: Random seed for reproducibility

    Returns:
        noise_data: Noisy data with the same type as input
    """
    is_torch = isinstance(org_data, torch.Tensor)
    device = org_data.device if is_torch else None

    # Convert to numpy for computation
    if is_torch:
        data_np = org_data.cpu().numpy()
    else:
        data_np = org_data

    # Compute L2 norm of original data
    denom = np.linalg.norm(data_np)

    # Noise level ratio
    r = noise_level / 100.0

    # Generate random noise
    if seed is not None:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(data_np.shape)
    else:
        z = np.random.randn(*data_np.shape)

    z_norm = np.linalg.norm(z)

    # Scale noise: alpha = r * ||org_data||_2 / ||z||_2
    alpha = r * denom / z_norm

    # Add scaled noise: noise_data = org_data + alpha * z
    e = alpha * z
    noise_data_np = data_np + e

    # Convert back to original type
    if is_torch:
        noise_data = torch.tensor(noise_data_np, dtype=org_data.dtype).to(device)
    else:
        noise_data = noise_data_np

    return noise_data

def _load_array_from_dir(data_dir, filename, device):
    data_dir = Path(data_dir)
    path_mat = data_dir / f'{filename}.mat'
    path_npy = data_dir / f'{filename}.npy'

    if path_mat.exists():
        data = scio.loadmat(path_mat)
        if filename in ['input', 'input_bil', 'input_exp']:
            return torch.tensor(data['U'], dtype=torch.float32).to(device)
        if filename in ['output', 'output_bil', 'output_exp']:
            return torch.tensor(data['E'], dtype=torch.float32).to(device)
        return torch.tensor(data['F'], dtype=torch.float32).to(device)

    if path_npy.exists():
        return torch.from_numpy(np.load(path_npy)).float().to(device)

    raise FileNotFoundError(
        f"Data file not found: {filename}\n"
        f"Searched paths:\n"
        f"  - {path_mat}\n"
        f"  - {path_npy}"
    )


def get_fixed_test_data_dir(cfg):
    data_path = Path(cfg.data_path).resolve()
    if len(data_path.parents) < 2:
        raise ValueError(f'cfg.data_path has unexpected structure: {cfg.data_path}')

    dataset_dir = data_path.parent.name
    load_type = data_path.name
    data_root = data_path.parent.parent
    return data_root / 'fixed_test_sets' / load_type / dataset_dir


def fixed_test_set_exists(cfg):
    fixed_dir = get_fixed_test_data_dir(cfg)
    input_exists = (fixed_dir / 'input.mat').exists() or (fixed_dir / 'input.npy').exists()
    output_exists = (fixed_dir / 'output.mat').exists() or (fixed_dir / 'output.npy').exists()
    return input_exists and output_exists


def load_test_data(cfg):
    if fixed_test_set_exists(cfg):
        fixed_dir = get_fixed_test_data_dir(cfg)
        print(f'Using fixed test set from: {fixed_dir}')
        inputs = _load_array_from_dir(fixed_dir, 'input', cfg.device)
        targets = _load_array_from_dir(fixed_dir, 'output', cfg.device)
        return inputs, targets

    if _split_data_exists(cfg, 'test'):
        inputs = load_data('input', cfg, split='test')
        targets = load_data('output', cfg, split='test')
        return inputs, targets

    inputs = load_data('input', cfg)
    targets = load_data('output', cfg)
    total = inputs.shape[0]
    test_start = int(total * cfg.train_rto) + int(total * cfg.valid_rto)
    print('Fixed test set not found. Falling back to sliced test split from cfg.data_path.')
    return inputs[test_start:], targets[test_start:]


def load_validation_data(cfg):
    if _split_data_exists(cfg, 'val'):
        inputs = load_data('input', cfg, split='val')
        targets = load_data('output', cfg, split='val')
        return inputs, targets

    inputs = load_data('input', cfg)
    targets = load_data('output', cfg)
    total = inputs.shape[0]
    valid_start = int(total * cfg.train_rto)
    valid_end = valid_start + int(total * cfg.valid_rto)
    return inputs[valid_start:valid_end], targets[valid_start:valid_end]


def _split_data_exists(cfg, split):
    split_dir = Path(cfg.data_path).resolve() / split
    return all(
        (split_dir / filename).exists()
        for filename in ['input.mat', 'output.mat']
    )


# ------------------------------
# Optional self-check utilities
# ------------------------------
def test_noise_generation():
    """Quick check for Equation-19 style noise scaling."""
    org_data = np.random.randn(10, 10)
    noise_level = 1.0
    noise_data = generate_noise_data(org_data, noise_level, seed=42)
    error = noise_data - org_data
    actual_noise = np.linalg.norm(error) / np.linalg.norm(org_data) * 100.0
    return {
        'target_noise_pct': float(noise_level),
        'actual_noise_pct': float(actual_noise),
        'abs_diff_pct': float(abs(actual_noise - noise_level)),
    }


def test_split_loading(cfg):
    """Quick check for val/test split sizes."""
    test_x, test_y = load_test_data(cfg)
    val_x, val_y = load_validation_data(cfg)
    return {
        'test_size': int(test_x.shape[0]),
        'val_size': int(val_x.shape[0]),
        'test_target_size': int(test_y.shape[0]),
        'val_target_size': int(val_y.shape[0]),
    }


__all__ = [
    'fixed_test_set_exists',
    'get_fixed_test_data_dir',
    'generate_noise_data',
    'load_test_data',
    'load_validation_data',
    'test_noise_generation',
    'test_split_loading',
]
