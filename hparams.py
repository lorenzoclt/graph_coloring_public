import torch
from itertools import product

num_trials = 1 # number of trials per hparam combination

# hparams can be defined in combinations
num_epochs = [25000]
batch_size = [64]
learning_rate = [2e-4]
patience = [50]
patience_factor = [0.98]
num_colors = [5]
latent_dim = [32]
num_layers = [5]
temp = [10] # softmax temperature
entropy_factor = [0.05] # importance of entropy in loss function
entropy_time = [1] # number of epochs to keep entropy_factor constant
dropout_rate = [0.] # controls dropout rate in phi and gamma
self_loops = [True] # whether to add self loops to the graph
gamma = [0.5]
t = [0]
alpha_min = [0.4]
alpha_max = [0.9]
hparams_combinations = list(product(num_epochs, batch_size, learning_rate, patience, patience_factor, num_colors, latent_dim, num_layers, temp, entropy_factor, entropy_time, dropout_rate, self_loops,gamma,alpha_min,alpha_max,t))

