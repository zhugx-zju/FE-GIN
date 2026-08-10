"""Train the optional NeuralOperator-backed FNO-MSE baseline."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fno.config import get_config
from fno.neuraloperator import build_neuralop_fno_model
from fno.train import train_fno


train_fno(get_config("neuralop"), model_builder=build_neuralop_fno_model)
