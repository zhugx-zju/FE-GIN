import scipy.io as scio
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from .utils_process import InvSet, PartSet

def load_data(filename, cfg, split=None):
    data_path = cfg.data_path
    if split:
        split_path = os.path.join(data_path, split)
        if os.path.isdir(split_path):
            data_path = split_path
    device = cfg.device
    path_mat = os.path.join(data_path, f"{filename}.mat")
    path_npy = os.path.join(data_path, f"{filename}.npy")
    if os.path.exists(path_mat):
        data = scio.loadmat(path_mat)
        if filename in ['input', 'input_bil', 'input_exp']:
                return torch.tensor(data['U'],dtype=torch.float32).to(device)
        elif filename in ['output', 'output_bil', 'output_exp']:
                return torch.tensor(data['E'],dtype=torch.float32).to(device)
        else:
                return torch.tensor(data['F'],dtype=torch.float32).to(device)
    elif os.path.exists(path_npy):
        return torch.from_numpy(np.load(path_npy)).float().to(device)
    else:
        raise FileNotFoundError(
            f"Data file not found: {filename}\n"
            f"Searched paths:\n"
            f"  - {path_mat}\n"
            f"  - {path_npy}\n"
            f"Please check that the data files exist in: {data_path}"
        )

def load_mse_data(cfg):
    train_rto = cfg.train_rto
    valid_rto = cfg.valid_rto
    batch_size = cfg.batch_size

    # Unified loading for all config types
    input_name = 'input'
    output_name = 'output'
    if _has_split_data(cfg, 'train') and _has_split_data(cfg, 'val'):
        train_input = load_data(input_name, cfg, split='train')
        train_output = load_data(output_name, cfg, split='train')
        valid_input = load_data(input_name, cfg, split='val')
        valid_output = load_data(output_name, cfg, split='val')
        train_set = InvSet(train_input, train_output)
        valid_set = InvSet(valid_input, valid_output)
    else:
        input = load_data(input_name, cfg)
        output = load_data(output_name, cfg)
        train_set, valid_set, _ = PartSet(input.shape[0], train_rto, valid_rto,
                                          input, output)
    # Create DataLoader
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=False)
    return train_loader, valid_loader

def load_eleres_data(cfg):
    train_rto = cfg.train_rto
    valid_rto = cfg.valid_rto
    batch_size = cfg.batch_size

    # Unified loading for all config types
    input_name = 'input'
    output_name = 'output'
    if (_has_split_data(cfg, 'train', ['input.mat', 'output.mat', 'dof.npy', 'force_ele.npy'])
            and _has_split_data(cfg, 'val', ['input.mat', 'output.mat', 'dof.npy', 'force_ele.npy'])):
        train_set = InvSet(
            load_data(input_name, cfg, split='train'),
            load_data(output_name, cfg, split='train'),
            load_data('dof', cfg, split='train'),
            load_data('force_ele', cfg, split='train'),
        )
        valid_set = InvSet(
            load_data(input_name, cfg, split='val'),
            load_data(output_name, cfg, split='val'),
            load_data('dof', cfg, split='val'),
            load_data('force_ele', cfg, split='val'),
        )
    else:
        input = load_data(input_name, cfg)
        output = load_data(output_name, cfg)
        dof = load_data('dof', cfg)
        force_ele = load_data('force_ele', cfg)
        train_set, valid_set, _ = PartSet(input.shape[0], train_rto, valid_rto,
                                          input, output, dof, force_ele)
    # Create DataLoader
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=False)
    return train_loader, valid_loader

def load_totres_data(cfg):
    train_rto = cfg.train_rto
    valid_rto = cfg.valid_rto
    batch_size = cfg.batch_size

    # Unified loading for all config types
    input_name = 'input'
    output_name = 'output'
    if (_has_split_data(cfg, 'train', ['input.mat', 'output.mat', 'dof.npy', 'force.npy'])
            and _has_split_data(cfg, 'val', ['input.mat', 'output.mat', 'dof.npy', 'force.npy'])):
        train_set = InvSet(
            load_data(input_name, cfg, split='train'),
            load_data(output_name, cfg, split='train'),
            load_data('dof', cfg, split='train'),
            load_data('force', cfg, split='train'),
        )
        valid_set = InvSet(
            load_data(input_name, cfg, split='val'),
            load_data(output_name, cfg, split='val'),
            load_data('dof', cfg, split='val'),
            load_data('force', cfg, split='val'),
        )
    else:
        input = load_data(input_name, cfg)
        output = load_data(output_name, cfg)
        dof = load_data('dof', cfg)
        force = load_data('force', cfg)
        train_set, valid_set, _ = PartSet(input.shape[0], train_rto, valid_rto,
                                          input, output, dof, force)
    # Create DataLoader
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=False)
    return train_loader, valid_loader

def save_train(train, valid, filename):
    training_data = np.hstack(
        (np.array(train).reshape((-1,1)),
         np.array(valid).reshape((-1,1)))
         )
    np.savetxt(filename, training_data)


def _has_split_data(cfg, split, required_files=None):
    split_path = os.path.join(cfg.data_path, split)
    required = required_files or ['input.mat', 'output.mat']
    return os.path.isdir(split_path) and all(
        os.path.exists(os.path.join(split_path, filename)) for filename in required
    )

