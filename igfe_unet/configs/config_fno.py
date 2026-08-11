"""Configuration for the custom FNO using the shared U-Net workflow."""

data_path = "../../data/data_mix"
load_type = "force_load"
data_path = data_path + "/" + load_type

model_type = "fno"
fno_backend = "custom"
dataset_type = "mix"

# Keep the supervised training protocol aligned with config_mix.py.
lr_start = 3e-4
n_epochs = 1500
batch_size = 32
train_rto = 0.70
valid_rto = 0.20
patience_lr = 10
patience_stop = 25
preload = False
method = "MSE"
device = "cuda"
seed = 42
num = "all"
noise_level = 0.0
noise_levels = [0, 2, 4, 6, 8, 10]
sample_index = 0
save_format = "both"
eval_split = "test"

# FNO-only structural parameters. The two FNO backends intentionally share
# this exact structural configuration.
input_channels = 2
output_channels = 1
width = 32
modes1 = 16
modes2 = 16
n_layers = 4
use_coordinates = True

geoX = 9.0
geoY = 9.0
nodesx = 40
nodesy = 40

eval_types = "all"
variable_names = [
    "lr_start", "patience_lr", "patience_stop", "batch_size", "n_epochs",
    "model_type", "fno_backend", "input_channels", "output_channels",
    "dataset_type",
    "width", "modes1", "modes2", "n_layers", "use_coordinates",
    "geoX", "geoY", "nodesx", "nodesy", "method", "seed", "time",
    "parameter_count", "parameter_tensor_count",
]
