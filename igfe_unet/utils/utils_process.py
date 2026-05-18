import importlib
import os
from torch.utils.data import Dataset

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
        else:
            raise ValueError("Invalid configuration type")

        # Load variables from the imported module
        self.load_config_variables(cfg)
        self.config_type = config_type

        # Fix data_path to use absolute path
        if hasattr(self, 'data_path') and not os.path.isabs(self.data_path):
            # Get the path to the configs directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            configs_dir = os.path.join(current_dir, '..', 'configs')
            # Convert relative path to absolute path
            self.data_path = os.path.abspath(os.path.join(configs_dir, self.data_path))

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

def write_config(cfg, filepath):
    # Generate experiment ID
    exp_id = generate_experiment_id(cfg)

    if cfg.config_type == 'layer':
        dir_name = 'trained_models_layer'
    elif cfg.config_type == 'mix':
        dir_name = 'trained_models_mix'
    elif cfg.config_type == 'bil':
        dir_name = 'trained_models_bil'
    elif cfg.config_type == 'exp':
        dir_name = 'trained_models_exp'
    elif cfg.config_type == 'grf':
        dir_name = 'trained_models_grf'

    # Get the path to the igfe_loss directory (parent of igfe_cnn)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    igfe_loss_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    dir_path = os.path.join(igfe_loss_dir, dir_name)
    dir_path = os.path.join(dir_path, cfg.load_type, exp_id)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_filename = 'config.py'
    config_file_path = os.path.join(dir_path, config_filename)

    # Write selected configuration variables to the file
    keys_to_write = list(cfg.variable_names)
    is_mix_method = isinstance(getattr(cfg, 'method', None), str) and ('Mix' in cfg.method)

    if is_mix_method:
        if 'gamma' not in keys_to_write:
            keys_to_write.append('gamma')
    else:
        keys_to_write = [key for key in keys_to_write if key != 'gamma']

    with open(config_file_path, 'w') as file:
        for key in keys_to_write:
            value = getattr(cfg, key, None)
            file.write(f"{key} = {repr(value)}\n")

    return config_file_path

def construct_paths(cfg):
    # Generate experiment ID based on all key parameters
    exp_id = generate_experiment_id(cfg)

    if cfg.config_type == 'layer':
        dir_name = 'trained_models_layer'
    elif cfg.config_type == 'mix':
        dir_name = 'trained_models_mix'
    elif cfg.config_type == 'bil':
        dir_name = 'trained_models_bil'
    elif cfg.config_type == 'exp':
        dir_name = 'trained_models_exp'
    elif cfg.config_type == 'grf':
        dir_name = 'trained_models_grf'

    # Construct directory with experiment ID
    # Get the path to the igfe_loss directory (parent of igfe_cnn)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    igfe_loss_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    train_path = os.path.join(igfe_loss_dir, dir_name)
    train_path = os.path.join(train_path, cfg.load_type, exp_id)

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
