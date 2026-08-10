"""Run a dependency and shape smoke test without requiring training data."""

import os
import sys
import time

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from architectures.fno import FNO2d
from utils.utils_fno import count_parameters, set_seed


def main():
    set_seed(42)
    device = torch.device("cpu")
    model = FNO2d(width=16, modes1=4, modes2=4, n_layers=2).to(device)
    inputs = torch.randn(2, 2, 40, 40, device=device)
    targets = torch.randn(2, 40, 40, device=device)
    start = time.perf_counter()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    elapsed = time.perf_counter() - start
    assert outputs.shape == targets.shape, (outputs.shape, targets.shape)
    assert torch.isfinite(loss), loss
    print(f"FNO smoke test passed: shape={tuple(outputs.shape)}")
    print(f"parameters={count_parameters(model)}, forward_backward_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
