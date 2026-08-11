import importlib
import os
import torch
from torch.utils.data import Dataset


EXPERIMENT_GROUPS = ('std', 'gamma', 'arch', 'ratio', 'final_model')


def normalize_experiment_group(group, default='std'):
    """Normalize the model/analysis experiment group name."""
    value = default if group is None else str(group).strip().lower()
    if value not in EXPERIMENT_GROUPS:
        raise ValueError(
            f"Invalid experiment group '{value}'. "
            f"Expected one of: {', '.join(EXPERIMENT_GROUPS)}."
        )
    return value


def resolve_experiment_group(cfg, override=None):
    """Resolve the output group, keeping mixed-loss runs separate by default."""
    group = override
    if group is None:
        group = getattr(cfg, 'experiment_group', None)
    if group is None:
        method = str(getattr(cfg, 'method', '')).strip().lower()
        group = 'gamma' if method in ('locmixloss', 'glomixloss') else 'std'
    return normalize_experiment_group(group)

class Config:
    def __init__(self, config_type):
        # Dynamically import the correct configuration module
        if config_type == 'layer':
            cfg = importlib.import_module('configs.config_layer')
        elif config_type == 'mix':
            cfg = importlib.import_module('configs.config_mix')
        elif config_type == 'exp':
            cfg = importlib.import_module('configs.config_exp')
        elif config_type == 'bil':
            cfg = importlib.import_module('configs.config_bil')
        elif config_type == 'grf':
            cfg = importlib.import_module('configs.config_grf')
        elif config_type == 'fno':
            cfg = importlib.import_module('configs.config_fno')
        elif config_type == 'fno_neuralop':
            cfg = importlib.import_module('configs.config_fno_neuralop')
        else:
            raise ValueError("Invalid configuration type")

        # Load variables from the imported module
        self.load_config_variables(cfg)
        self.config_type = config_type
        # Keep dataset identity separate from the model family. This allows
        # FNO experiments to use the same mix/bil/exp/grf evaluation workflow.
        if not hasattr(self, 'dataset_type'):
            self.dataset_type = config_type

        # Fix data_path to use absolute path
        if hasattr(self, 'data_path') and not os.path.isabs(self.data_path):
            # Get the path to the configs directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            configs_dir = os.path.join(current_dir, '..', 'configs')
            # Convert relative path to absolute path
            self.data_path = os.path.abspath(os.path.join(configs_dir, self.data_path))

        # Keep config files portable across GPU and CPU-only environments.
        # The configured device remains unchanged when the requested backend
        # is available.
        if getattr(self, 'device', None) == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
        elif getattr(self, 'device', None) == 'mps':
            mps = getattr(torch.backends, 'mps', None)
            if mps is None or not mps.is_available():
                self.device = 'cpu'

    def load_config_variables(self, cfg):
        # Iterate through the module's attributes and set them to this class
        for attr in dir(cfg):
            # Skip built-in attributes and methods
            if not attr.startswith("__"):
                setattr(self, attr, getattr(cfg, attr))

def generate_arch_id(filters_list):
    # Generate a short architecture identifier from filters_list/nfilters.
    # Supports both legacy symmetric lists and UNet encoder-style lists.
    if not filters_list or len(filters_list) < 2:
        return 'unknown'

    layers = list(filters_list[1:])
    if len(layers) > 1 and float(layers[-1]) == 1.0:
        layers = layers[:-1]
    if not layers:
        return 'unknown'

    max_idx = layers.index(max(layers))
    key_layers = layers[:max_idx + 1]
    return '-'.join(map(str, key_layers))

def format_decimal_token(value, precision=10):
    # Format numeric values for filenames, e.g., 0.01 -> "0p01", 1.0 -> "1"
    return f"{float(value):.{precision}f}".rstrip('0').rstrip('.').replace('.', 'p')


def batch_norm_enabled(cfg_or_dict):
    if cfg_or_dict is None:
        return False
    if isinstance(cfg_or_dict, dict):
        value = cfg_or_dict.get('use_batch_norm', False)
    else:
        value = getattr(cfg_or_dict, 'use_batch_norm', False)
    return bool(value)


def batch_norm_suffix(cfg_or_dict=None):
    return '_GN' if batch_norm_enabled(cfg_or_dict) else ''


def network_variant_name(cfg_or_dict=None, use_batch_norm=None):
    if use_batch_norm is None:
        use_batch_norm = batch_norm_enabled(cfg_or_dict)
    suffix = '_GN' if bool(use_batch_norm) else ''
    return f"UNet{suffix}"


def generate_mix_ratio_tag(cfg):
    # Build ratio tag from mix dataset ratios, e.g., b0p33_e0p33_g0p34
    bil_ratio = getattr(cfg, 'bil_ratio', None)
    exp_ratio = getattr(cfg, 'exp_ratio', None)
    grf_ratio = getattr(cfg, 'grf_ratio', None)
    if bil_ratio is None or exp_ratio is None or grf_ratio is None:
        return None
    return (
        f"b{format_decimal_token(bil_ratio)}"
        f"_e{format_decimal_token(exp_ratio)}"
        f"_g{format_decimal_token(grf_ratio)}"
    )

def generate_experiment_id(cfg):
    # Allow task scripts to pin a custom experiment id without changing core flow.
    exp_id_override = getattr(cfg, 'exp_id_override', None)
    if exp_id_override:
        return str(exp_id_override)

    if str(getattr(cfg, 'model_type', 'unet')).lower() == 'fno':
        backend = str(getattr(cfg, 'fno_backend', 'custom')).lower()
        return (
            f"FNO_{backend}_w{int(cfg.width)}"
            f"_m{int(cfg.modes1)}x{int(cfg.modes2)}"
            f"_l{int(cfg.n_layers)}"
        )

    # Generate unique experiment identifier based on key parameters
    method_short = get_filepath(cfg.method)
    arch_id = generate_arch_id(cfg.filters_list)
    network_variant = network_variant_name(cfg)

    # Build experiment ID
    exp_id = f"{method_short}_{network_variant}_arch_{arch_id}"

    # Add gamma for mixed loss methods
    if 'Mix' in cfg.method:
        gamma_str = format_decimal_token(cfg.gamma)
        exp_id += f"_gamma_{gamma_str}"

    return exp_id

def get_filepath(method):
    if method == 'EleResloss':
        filepath = 'EleRes'
    elif method == 'TotResloss':
        filepath = 'TotRes'
    elif method == 'MSE':
        filepath = 'MSE'
    elif method == 'LocMixloss':
        filepath = 'LocMix'
    elif method == 'GloMixloss':
        filepath = 'GloMix'
    elif method == 'LocResloss':
        filepath = 'LocRes'
    elif method == 'GloResloss':
        filepath = 'GloRes'
    return filepath


def _model_root_for_config(config_type):
    if config_type == 'layer':
        return 'trained_models_layer'
    if config_type == 'mix':
        return 'trained_models_mix'
    if config_type == 'bil':
        return 'trained_models_bil'
    if config_type == 'exp':
        return 'trained_models_exp'
    if config_type == 'grf':
        return 'trained_models_grf'
    if config_type in ('fno', 'fno_neuralop'):
        return 'trained_models_fno'
    raise ValueError(f"Unsupported configuration type: {config_type}")


def experiment_dir_for_config(cfg):
    """Return the grouped model directory for a training configuration.

    New experiments are stored as:
        trained_models_{config_type}/{load_type}/{group}/{experiment_id}

    Explicit groups take precedence. Without one, mixed-loss methods use
    ``gamma`` and the standard physical-loss methods use ``std``.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_root_override = getattr(cfg, 'model_root_override', None)
    model_root = (
        os.path.abspath(model_root_override)
        if model_root_override
        else os.path.join(project_root, _model_root_for_config(cfg.config_type))
    )
    load_type = str(getattr(cfg, 'load_type', 'force_load')).strip()
    group = resolve_experiment_group(cfg)
    exp_id = generate_experiment_id(cfg)
    return os.path.join(model_root, load_type, group, exp_id)

def write_config(cfg, filepath):
    # Generate experiment ID
    exp_id = generate_experiment_id(cfg)

    dir_path = experiment_dir_for_config(cfg)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_filename = 'config.py'
    config_file_path = os.path.join(dir_path, config_filename)

    # Write selected configuration variables to the file
    keys_to_write = list(cfg.variable_names)
    for key in ('parameter_count', 'parameter_tensor_count'):
        if hasattr(cfg, key) and key not in keys_to_write:
            keys_to_write.append(key)
    if 'experiment_group' not in keys_to_write:
        keys_to_write.append('experiment_group')
    experiment_group = resolve_experiment_group(cfg)
    is_mix_method = isinstance(getattr(cfg, 'method', None), str) and ('Mix' in cfg.method)

    if is_mix_method:
        if 'gamma' not in keys_to_write:
            keys_to_write.append('gamma')
    else:
        keys_to_write = [key for key in keys_to_write if key != 'gamma']

    with open(config_file_path, 'w') as file:
        for key in keys_to_write:
            value = experiment_group if key == 'experiment_group' else getattr(cfg, key, None)
            file.write(f"{key} = {repr(value)}\n")

    return config_file_path

def construct_paths(cfg):
    # Generate experiment ID based on all key parameters
    exp_id = generate_experiment_id(cfg)

    train_path = experiment_dir_for_config(cfg)

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    ckpt_name = os.path.join(train_path, 'model.pt')
    loss_filename = os.path.join(train_path, 'history')

    return ckpt_name, loss_filename, train_path

class InvSet(Dataset):
    def __init__(self, *data):
        self.data = data
        self.Num = self.data[0].shape[0]

    def __len__(self):
        return self.Num

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)

# Separate DataSet for training, validation and testing
def PartSet(batch_Num, train_rto, valid_rto, *data):
    train_size = int(batch_Num * train_rto)
    valid_size = int(batch_Num * valid_rto)

    # Slice Function
    def slice_data(data, start, end):
        return data[start:end] if data is not None else None

    # Generate Slicing Index
    splits = [
        (0, train_size),
        (train_size, train_size + valid_size),
        (train_size + valid_size, None)
    ]
    return [InvSet(*[slice_data(d, s, e) for d in data]) for s, e in splits]

import json

def config_from_json():
    # Get the absolute path to the config.json file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, '..', 'script', 'config.json')
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)
    return config_data['config_type']
