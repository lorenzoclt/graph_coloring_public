import os
import time
import math
import torch
from torch.cuda.amp import autocast
import ctypes
import matplotlib.pyplot as plt


import torch_tensorrt
 
from GNN import MMPN_denoiser, MMPN_torch
from functions import find_checkpoints
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N=10000
q=5
# computes potts energy on graph
def discrete_energy(graph):
    
    # compute one-hot features
    max_indices   = torch.argmax(graph.x[:,:5], dim=1)
    rows          = torch.arange(len(graph.x[:,:5]), device = graph.x.device) # rows indexes
    one_hot_graph = torch.zeros_like(graph.x[:,:5], device = graph.x.device)
    one_hot_graph[rows, max_indices] = 1 

    # compute percentage of conflicting edges
    conflicts_tensor = torch.sum(one_hot_graph[graph.edge_index[0],:5] * one_hot_graph[graph.edge_index[1],:5], dim = 1)
    avg_conflicts    = torch.sum(conflicts_tensor, dim = 0)/conflicts_tensor.shape[0]

    return avg_conflicts
def continuous_energy( graph):
        
        # compute the energy of the batch
        scalar_product_tensor = torch.sum((graph.x[graph.edge_index[0],:num_colors] * graph.x[graph.edge_index[1],:num_colors]), dim=1)
        continuous_energy     = torch.mean(scalar_product_tensor)

        return continuous_energy
# path to load data
data_dir = os.getcwd()+'/data_test/'
parent_dir = os.getcwd() + f'/training_inpainting/'

# network params
batch_size = 1
latent_dim = 64
num_colors = 5
num_layers = 5
dropout_rate = 0.
temp = 10
self_loops = True
alpha_min_arr=torch.linspace(0.01,0.3, 10, device=device)
alpha_max_arr=torch.linspace(0.65,0.99, 10, device=device)
stopping_values = [5]
iter_array=[5000]
alpha_array=torch.linspace(0.001,1.0, 10, device=device)

lib = ctypes.CDLL('./computeoverlap.so')
IntArray = ctypes.c_int * N
lib.computeOverlap.restype = ctypes.c_double
conn_array=[15.0]
for trained_model in ['t0_20240827-075801','t0_20240827-075911','t0_20240827-075926']:
    for conn in conn_array:
        edges = int(N*conn/2)
        list_graphs=[]
        data_path = data_dir+f'planted/num_nodes_10000-edges_{edges}-num_col_5-conn_{conn}-T_0/test/graph.pt'

        # load graphs
        dataloader = DataLoader(torch.load(data_path), batch_size=batch_size, pin_memory=True, num_workers=39)

        best_energies = []
        num_graphs = 0

        
    
        
        # load torch model
        pwd = os.getcwd()
        
        checkpoint_path = pwd+f'/training_softmax_unsupervised/{trained_model}/'
        #verify file that start with checkpoint
        checkpoint_path = find_checkpoints(checkpoint_path)
        model = MMPN_torch(latent_dim, num_colors, num_layers, dropout_rate, self_loops, 10)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model = model.to(device)
        model.eval()


        # load jit model
        
        try: model = model.to_torchscript()
        except: raise Exception('JIT model could not be loaded')
        mean_energy = []
        mean_overlap = []
        mean_corr=[]
        mean_continuous_energy=[]
        mean_energy_post = []
        mean_overlap_post = []
        mean_corr_post=[]
        mean_continuous_energy_post=[]
        
        
        for alpha in alpha_array:

            all_energy = [] 
            all_overlap = []
            all_correlation=[]
            all_continuous_energy=[]
            all_energy_post = [] 
            all_overlap_post = []
            all_correlation_post=[]
            all_continuous_energy_post=[]
            with torch.no_grad():
                    with autocast():
                        for batch_idx, random_graph in enumerate(dataloader):  #test_dataloader
                                        # create an array of alpha_min values
                                        num_graphs += 1
                                        random_graph = random_graph.to(device) 
                                        perfect_graph = random_graph.clone()
                                        random_graph.x[:,:5]=torch.nn.functional.one_hot(torch.randint(0,5,(10000,),device=device),num_classes=5).float()
                                        #transform max_indices to one-hot 
                                        for i in range(10):
                                            random_graph.x[:,:5]=torch.sqrt(alpha)*random_graph.x[:,:5]+torch.sqrt(1-alpha)*torch.randn((10000,5),device=device)  
                                            #apply softmax
                                            random_graph.x[:,:5]=torch.nn.functional.softmax(random_graph.x[:,:5],dim=1)
                                            continuous=continuous_energy(random_graph)
                                            correlation=torch.sum(perfect_graph.x[:,:5]*random_graph.x[:,:5])/N
                                            max_indices = torch.argmax(random_graph.x[:,:5], dim=1)
                                            color_array = IntArray(*max_indices.cpu().numpy())
                                            '''overlap_planted=lib.computeOverlap(color_array)
                                            all_overlap.append(overlap_planted)
                                            e0 = discrete_energy(random_graph)
                                            all_energy.append(e0)
                                            all_correlation.append(correlation)
                                            all_continuous_energy.append(continuous)'''

                                            random_graph.x[:,:num_colors] = model(random_graph.x, random_graph.edge_index)
                                        continuous=continuous_energy(random_graph)
                                        correlation=torch.sum(perfect_graph.x[:,:5]*random_graph.x[:,:5])/N
                                        max_indices = torch.argmax(random_graph.x[:,:5], dim=1)
                                        color_array = IntArray(*max_indices.cpu().numpy())
                                        overlap_planted=lib.computeOverlap(color_array)
                                        all_overlap_post.append(overlap_planted)
                                        e0 = discrete_energy(random_graph)
                                        all_energy_post.append(e0)
                                        all_correlation_post.append(correlation)
                                        all_continuous_energy_post.append(continuous)

                                        
                        mean_energy.append(torch.mean(torch.tensor(all_energy)))
                        mean_overlap.append(torch.mean(torch.tensor(all_overlap)))
                        mean_corr.append(torch.mean(torch.tensor(all_correlation)))
                        mean_continuous_energy.append(torch.mean(torch.tensor(all_continuous_energy)))
                        mean_energy_post.append(torch.mean(torch.tensor(all_energy_post)).cpu())
                        mean_overlap_post.append(torch.mean(torch.tensor(all_overlap_post)).cpu())
                        mean_corr_post.append(torch.mean(torch.tensor(all_correlation_post)).cpu())
                        mean_continuous_energy_post.append(torch.mean(torch.tensor(all_continuous_energy_post)).cpu())

                        print(f'theta={alpha},energy={mean_energy[-1]}, overlap={mean_overlap[-1]}')
        #plt.plot(alpha_array.cpu(),mean_energy,label=f'energy {trained_model}')    
        #plt.plot(alpha_array.cpu(),mean_overlap,label=f'overlap {trained_model}')  
        #plt.plot(alpha_array.cpu(),mean_corr,label=f'correlation {trained_model}')
        #plt.plot(alpha_array.cpu(),mean_continuous_energy,label=f'continuous_energy {trained_model}')
        plt.plot(alpha_array.cpu(),mean_energy_post,label=f'energy_post {trained_model}')
        plt.plot(alpha_array.cpu(),mean_overlap_post,label=f'overlap_post {trained_model}')
        plt.plot(alpha_array.cpu(),mean_corr_post,label=f'correlation_post {trained_model}')
        #plt.plot(alpha_array.cpu(),mean_continuous_energy_post,label=f'continuous_energy_post {trained_model}')
        plt.legend()

        plt.yscale('log')   
plt.savefig(f'visualization_all_models_softmax_unsupervised_random_it10.png')
plt.close()
                                    
                                    