"""Small FNO compatibility helpers.

Training, testing, checkpoint paths, and result serialization live in the
shared U-Net modules. This file only keeps FNO-specific utility imports stable
for older callers.
"""

import random

import numpy as np
import torch


def count_parameters(model):
    """Count trainable real scalar values, including complex weights."""
    return int(sum(
        (2 if parameter.is_complex() else 1) * parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ))


def count_tensor_parameters(model):
    return int(sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ))


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["count_parameters", "count_tensor_parameters", "set_seed"]
