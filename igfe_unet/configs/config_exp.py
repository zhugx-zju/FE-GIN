############### Configuration file for Deep learning CHTC scripts###############
data_path = "../../data/data_exp"
load_type = "force_load" # options: 'force_load' 'disp_load'
data_path = data_path + "/" + load_type
# %% ------------------ General NN optimization parameters ------------------
lr_start = 0.001  # optimal for the input output

# %% ------------------ NN architecture parameters ------------------
filters_list = [2, 32, 64, 128, 256]
kernel_size = 3  # kernel size
use_batch_norm = False

# %% ------------------ Tunable parameters/  ------------------
n_epochs = 1000  # number of epochs
batch_size = 16  # number of batches
train_rto = 0.7
valid_rto = 0.2
patience_lr = 10
patience_stop = 50
preload = False  # Whether to load pretrained model
method = 'GloResloss'  # options: 'LocResloss', 'GloResloss', 'MSE'
device = "cuda" # options: 'cuda', 'mps', 'cpu'
num = 10
save_format = 'both'  # options: 'npz', 'mat', 'both'
# %% ------------------ FEM parameters ------------------
geoX = 9.0
geoY = 9.0
nodesx = 40
nodesy = 40
# %% ------------------ Parameter to be monitored ------------------
variable_names = [
    'patience_lr', 'patience_stop', 'batch_size', 'n_epochs',
    'kernel_size', 'filters_list', 'use_batch_norm',
    'geoX', 'geoY', 'nodesx', 'nodesy',
    'method', 'time']
