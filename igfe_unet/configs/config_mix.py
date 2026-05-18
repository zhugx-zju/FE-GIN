############### Configuration file for Deep learning CHTC scripts###############
data_path = "../../data/data_mix"
load_type = "force_load" # options: 'force_load' 'disp_load' 'all'
data_path = data_path + "/" + load_type
# %% ------------------ General NN optimization parameters ------------------
lr_start = 3e-4  # unified default LR for fair comparison across loss types
# %% ------------------ NN architecture parameters ------------------
filters_list = [2, 32, 64, 128]
kernel_size = 3  # kernel size
use_batch_norm = True
# %% ------------------ Tunable parameters/  ------------------
n_epochs = 1500  # reduce max training time; avoid very long tail iterations
batch_size = 32  # faster and more stable gradient estimate than very small batch
train_rto = 0.70
valid_rto = 0.20
patience_lr = 10
patience_stop = 25
preload = False  # Whether to load pretrained model
gamma = 100000000  # Default gamma for mixed loss; tune around this value if needed
method = 'LocMixloss'  # options: 'LocMixloss', 'GloMixloss', 'LocResloss', 'GloResloss', 'MSE'
device = "cuda" # options: 'cuda', 'mps', 'cpu'
num = 'all' # options: 'all', or int for number of samples to use (for quick testing)
# Backward-compatible single-noise setting (percentage format)
noise_level = 0.0
# Multi-noise settings (percentage format)
noise_levels = [0, 2, 4, 6, 8, 10]
# Fixed sample index for cross-noise visualization
sample_index = 0
save_format = 'both'  # options: 'npz', 'mat', 'both'
# %% ------------------ Mix dataset parameters ------------------
# Dataset mixing ratios (must sum to 1.0)
exp_ratio = 0.60  # Proportion of exponential case samples
bil_ratio = 0.10  # Proportion of bilinear case samples
grf_ratio = 0.30  # Proportion of GRF case samples
# Evaluation types for mix dataset
# Options: 'all' (evaluate all types), or list like ['bil', 'exp'] or ['grf']
eval_types = 'all'  # Evaluate all types: bil, exp, grf
# Note: total_samples will be determined from the actual data files
# %% ------------------ FEM parameters ------------------
geoX = 9.0
geoY = 9.0
nodesx = 40
nodesy = 40
# %% ------------------ Parameter to be monitored ------------------
variable_names = [
    'lr_start', 'patience_lr', 'patience_stop', 'batch_size', 'n_epochs',
    'kernel_size', 'filters_list', 'use_batch_norm',
    'geoX', 'geoY', 'nodesx', 'nodesy',
    'method', 'time']
