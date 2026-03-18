import torch
from torch_geometric.data import Data
import numpy as np
import random
from tqdm import tqdm
N=100
q=5
def parse_graph_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    graph_data = data.strip().split('\n\n\n')
    
    graphs = []

    for graph_str in graph_data:
         
        lines = graph_str.strip().split('\n')
        edges = []
        for i in range(len(lines)):
            nodes = list(map(int, lines[i].split()))
            for node in nodes:
                edges.append([i, node])
         
        graphs.append(edges)

    return graphs
 
def create_torch_geometric_graphs(graphs,N):
    torch_graphs = []
    for edges in graphs:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        num_nodes = max(max(edges)) + 1
        x= torch.rand((N, q))        
        torch_graph = Data(x=x, edge_index=edge_index)
        torch_graphs.append(torch_graph)
        print(torch_graph)
        
    return torch_graphs

# Let's assume your file is named 'graph_data.txt'
N_array = [100,500,1000]
for N in tqdm(N_array):
    file_path = f'neigh_dataset_N{N}.txt'

    # Parse the graph data
    graphs = parse_graph_data(file_path)

    # Create PyTorch Geometric graphs
    torch_graphs = create_torch_geometric_graphs(graphs,N)
    torch.save(torch_graphs,f'validation_graphs_N{N}.pt')

    # Now you have a list of PyTorch Geometric graph objects ready for further processing or training.
