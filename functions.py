import os
import torch
import matplotlib.pyplot as plt

def plot_autocorrelation(autocorrelations, alphas, correlation_dist, folder = 'Logs'):
    if os.path.exists(folder) == False: os.makedirs(folder)
    plt.style.use('seaborn-v0_8-paper')
    plt.plot(alphas)
    plt.plot(autocorrelations)
    plt.grid(True, which="both", ls="--")
    plt.legend(['alpha_min', 'Autocorrelation'])
    plt.title(f'Correlation dist: {correlation_dist}')
    plt.savefig(f'{folder}/autocorrelation{correlation_dist}.png')
    plt.close()

def find_checkpoints(directory):
    '''returns first checkpoint in directory'''
    for file in os.listdir(directory):
        if file.endswith('.ckpt'): 
            return os.path.join(directory, file)
    return None

# computes potts energy on graph
def discrete_energy(graph, num_colors=5):

    # compute one-hot features
    max_indices   = torch.argmax(graph.x[:,:num_colors], dim=1)
    rows          = torch.arange(len(graph.x[:,:num_colors]), device = graph.x.device) # rows indexes
    one_hot_graph = torch.zeros_like(graph.x[:,:num_colors], device = graph.x.device)
    one_hot_graph[rows, max_indices] = 1

    # compute percentage of conflicting edges
    conflicts_tensor = torch.sum(one_hot_graph[graph.edge_index[0],:num_colors] * one_hot_graph[graph.edge_index[1],:num_colors], dim = 1)
    avg_conflicts    = torch.sum(conflicts_tensor, dim = 0)/conflicts_tensor.shape[0]

    return avg_conflicts

def autocorrelation_func(graph1, graph2):

    # measures autocorrelation between output of the network and the input
    overlap = torch.sum(graph1 * graph2, dim = 1)
    autocorrelation = torch.mean(overlap)

    return autocorrelation