import os
import re
import math
import torch
from time import time
from tqdm import tqdm
from collections import deque
from torch.cuda.amp import autocast

from GNN import MMPN_torch
from torch_geometric.loader import DataLoader
from functions import discrete_energy, autocorrelation_func, plot_autocorrelation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network params
batch_size = 1
correlation_dist = 100
latent_dim = 32
num_colors = 5
num_layers = 5
dropout_rate = 0.
temp = 10
self_loops = True
plt_corr = False
alg = "GNN"

# training params
alpha_min = 0.4#0.999
alpha_max = 0.9#0.9999
nums_iterations = [1000]

G = 10 # number of graphs
root_plt_1k = "data/1k/plt/G_10-N_1000-"
root_plt_3k = "data/3k/plt/G_10-N_3000-"
root_plt_10k = "data/10k/plt/G_10-N_10000-"
root_plt_30k = "data/30k/plt/G_10-N_30000-"
root_plt_100k = "data/100k/plt/G_10-N_100000-"

leaf = "/test/graph.pt"
log_type = "tempo_esecuzione"

filenames = [
    root_plt_1k+"E_5750-C_11.50"+leaf,
    root_plt_1k+"E_6500-C_13.00"+leaf,
    root_plt_1k+"E_6750-C_13.50"+leaf,
    root_plt_1k+"E_7750-C_15.50"+leaf,
    root_plt_3k+"E_17250-C_11.50"+leaf,
    root_plt_3k+"E_19500-C_13.00"+leaf,
    root_plt_3k+"E_20250-C_13.50"+leaf,
    root_plt_3k+"E_23250-C_15.50"+leaf,
    root_plt_10k+"E_57500-C_11.50"+leaf,
    root_plt_10k+"E_65000-C_13.00"+leaf,
    root_plt_10k+"E_67500-C_13.50"+leaf,
    root_plt_10k+"E_77500-C_15.50"+leaf,
    root_plt_30k+"E_172500-C_11.50"+leaf,
    root_plt_30k+"E_195000-C_13.00"+leaf,
    root_plt_30k+"E_202500-C_13.50"+leaf,
    root_plt_30k+"E_232500-C_15.50"+leaf,
    root_plt_100k+"E_575000-C_11.50"+leaf,
    root_plt_100k+"E_650000-C_13.00"+leaf,
    root_plt_100k+"E_675000-C_13.50"+leaf,
    root_plt_100k+"E_775000-C_15.50"+leaf
    ]

# check if folder Logs exists and save logs
if not os.path.exists(f'Logs/{log_type}'): os.makedirs(f'Logs/{log_type}')
with open(f'Logs/{log_type}/logfile.csv', 'a+') as f:
    f.write('Conn,Iter,E_mean,E_std,Type,Algorithm,Time,Num_solved,Num_nodes\n')
    
# iterate over all files in data_dir
for num_iterations in nums_iterations:
    for filename in filenames:

        type = "planted" if "plt" in filename else "random"
        num_nodes = re.search(r'(?<=N_)\d+', filename).group(0)
        conn = re.search(r'(?<=C_)\d+\.\d+', filename).group(0)

        print("\n\n========================================================================================\n")
        print("Processing file: ", filename)

        # load graphs
        dataloader = DataLoader(torch.load(filename), batch_size=batch_size, pin_memory=True, num_workers=39)
 
        best_energies = []
        energies = []

        # create an array of alpha_min values
        alpha_array = torch.linspace(alpha_min, alpha_max, num_iterations, device=device)

        # load torch model
        pwd = os.getcwd()
        checkpoint_path = pwd+'/weights/checkpoint-epoch=1219-val_loss=0.00.ckpt'
        model = MMPN_torch(latent_dim, num_colors, num_layers, dropout_rate, self_loops, temp)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model = model.to(device)
        model.eval()
        
        solved_counter = 0
        avg_time = 0
        
        # inference
        with torch.no_grad():
            with autocast():
                for batch_idx, random_graph in enumerate(dataloader): #test_dataloader

                    if plt_corr:
                        graphs_window = deque(maxlen=correlation_dist+1)
                        autocorrelations = []
                    alphas = []
                    e_best = 1.
                    
                    # set alpha values
                    t0 = time() # start time denoising

                    # move graph to device
                    random_graph = random_graph.to(device)
                    
                    # initialize random graph
                    random_graph.x[:,:num_colors] = torch.randn_like(random_graph.x[:,:num_colors].float(), device=device)

                    # iterate
                    t0 = time()
                    for alpha in alpha_array:

                        # apply diffusion
                        random_graph.x[:,:num_colors] = math.sqrt(alpha)*random_graph.x[:,:num_colors] + math.sqrt(1-alpha)*torch.randn_like(random_graph.x[:,:num_colors])

                        # unpack data
                        x, edges = random_graph.x, random_graph.edge_index   

                        # apply model and compute energy
                        random_graph.x[:,:num_colors] = model(random_graph.x, random_graph.edge_index)
                        e0 = discrete_energy(random_graph)

                        energies.append(e0.item())

                        # save best energy
                        if e0<e_best: e_best=e0
                        if e0 == 0:
                            e_best = e0 
                            solved_counter += 1
                            break # if energy is zero, exit denoising loop

                        if plt_corr:
                            graphs_window.append(random_graph.x[:,:num_colors].clone())
                            if len(graphs_window) == correlation_dist+1: autocorr = autocorrelation_func(graphs_window[0], graphs_window[-1])
                            else: autocorr = torch.tensor(0.)
                            autocorrelations.append(autocorr.detach().cpu().numpy())
                            alphas.append(alpha.detach().cpu().numpy())

                    print(f"Time: {time()-t0:.1f} - Best energy: {e_best.item():.8f} - Num solved: {solved_counter} - Num graphs: {G} - Iterations: {num_iterations}")

                    # append symbol to distinguish between different graphs
                    energies.append(-1)

                    # compute time and best energy
                    best_energies.append(e_best)
                    t1 = time() # end time denoising
                    avg_time += t1-t0

                    # plot autocorrelation
                    if plt_corr: plot_autocorrelation(autocorrelations, alphas, correlation_dist)

                # write log
                with open(f'Logs/{log_type}/logfile.csv', 'a+') as f:
                    f.write(f'{conn},{num_iterations},{torch.mean(torch.tensor(best_energies)).item():.8f},{torch.std(torch.tensor(best_energies)).item():.8f},{type},{alg},{avg_time/G:.2f},{solved_counter},{num_nodes}\n')