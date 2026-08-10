"""Configuration defaults for the isolated FNO comparison experiment."""

import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.environ.get(
    "FNO_DATA_PATH",
    os.path.join(project_root, "data", "data_mix", "force_load"),
)
load_type = "force_load"
config_type = "fno"
model_tag = "fno_mse"
method_label = "FNO-MSE"
output_dir = os.path.join(
    project_root, "results", "force_load", "fno_custom"
)

lr_start = 3e-4
n_epochs = 1500
batch_size = 32
train_rto = 0.70
valid_rto = 0.20
patience_lr = 10
patience_stop = 25
weight_decay = 0.0
seed = 42
device = "cuda"

input_channels = 2
output_channels = 1
width = 21
modes1 = 8
modes2 = 8
n_layers = 4
use_coordinates = True

nodesx = 40
nodesy = 40
num = "all"
noise_levels = [0, 2, 4, 6, 8, 10]
dataset_types = ["mix", "bil", "exp", "grf"]
sample_index = 0
