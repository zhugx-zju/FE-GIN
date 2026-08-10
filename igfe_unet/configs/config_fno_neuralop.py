"""Configuration defaults for the optional NeuralOperator FNO comparison."""

from configs.config_fno import *


model_tag = "fno_neuralop_mse"
method_label = "FNO-neuraloperator-MSE"
output_dir = os.path.join(
    project_root, "results", "force_load", "fno_neuralop"
)
