import torch
import random
from tqdm import tqdm
from pathlib import Path
from torch_geometric import data
from ortools.sat.python import cp_model
from torch_geometric.utils import to_undirected, to_dense_adj

from time import time

# generates a random permutation of numbers from 0 to size-1
def generate_permutation(size):
    numbers = list(range(size))
    random.shuffle(numbers)
    return numbers

# graph coloring solver
def g_coloring(graph,num_colors):
    col=0
    model=cp_model.CpModel()
    num_nodes=len(graph)
    colors = [model.NewIntVar(0, num_colors - 1, f'color_{i}') for i in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            if graph[i][j] == 1:
                model.Add(colors[i] != colors[j])
    
    solver=cp_model.CpSolver()
    status=solver.Solve(model)
    if status==cp_model.FEASIBLE or status == cp_model.OPTIMAL:
         
        col=1
     
    return col

def create_2_hop_edges(graph):

    edge_index = to_undirected(graph.edge_index)
    adj = to_dense_adj(edge_index)[0]
    edge_index_2 = torch.matmul(adj, adj).nonzero().t()

    return edge_index_2

def create_graph(num_nodes, num_edges, num_colors, type):
     
    t0 = time()

    mapped_edge_index = []
    color = [0]*num_nodes
    adj_mat = torch.zeros(num_nodes, num_nodes)
     
    for i in range(num_nodes):
         
        color[i] = i%num_colors

    # permute color  
    random.shuffle(color)
    
    if type == 'rdm':
        for i in range(num_edges):
            while True:
                var1 = random.randint(0, num_nodes - 1)
                var2 = random.randint(0, num_nodes - 1)
                
                if adj_mat[var1][var2] == 0 and var1 != var2: 
                    break
                
            adj_mat[var1][var2] = 1
            adj_mat[var2][var1] = 1
            
            # add edges to make it undirected
            mapped_edge_index.append([var1, var2])
            mapped_edge_index.append([var2, var1])

    elif type == 'plt':
        for i in range(num_edges):
            while True:
                var1 = random.randint(0, num_nodes - 1)
                var2 = random.randint(0, num_nodes - 1)
                
                if color[var1] != color[var2] and adj_mat[var1][var2] == 0:
                    break

            adj_mat[var1][var2] = 1
            adj_mat[var2][var1] = 1
            
            # add edges to make it undirected
            mapped_edge_index.append([var1,var2])
            mapped_edge_index.append([var2,var1])

    else: raise ValueError("Invalid type. Must be either 'rdm' or 'plt'")
    
    # compute degree of nodes
    degree = adj_mat.sum(dim=1)

    # transform color to one-hot encoding
    color = torch.nn.functional.one_hot(torch.tensor(color), num_classes=num_colors)   

    # add degree as last feature
    nodes_features = torch.cat([color, degree.unsqueeze(1)], dim=1).float()

    # fix edge index
    mapped_edge_index = torch.tensor(mapped_edge_index).t()             
    
    graph  = data.Data(x=nodes_features, edge_index=mapped_edge_index) # (N, num_colors+1)
    
    return graph

def generate_dataset(all_nodes, num_graphs, num_colors, connectivities, split, type):

    """
    Generates a dataset of graphs between min_nodes and max_nodes, for a given number of iterations (iters)
    """

    assert type in ['plt', 'rdm'], "type must be either 'rdm' or 'plt'"

    assert split[0] + split[1] + split[2] == 1, "split must be a list of three numbers that sum to 1"

    # iterates over each number of nodes and each connectivity degree for the given chromatic number
    for num_nodes in all_nodes:
        for conn in connectivities:

            # compute number of edges
            num_edges = int(num_nodes * conn / 2)
            
            # number of edges based on nodes and connectivity
            print('---------------------------------------------------------------------------------')
            print(f'Number of nodes: {num_nodes} - Connectivity: {conn} - Number of edges: {num_edges}')

            # checks if the number of edges exceeds the maximum possible for a simple graph
            if num_edges > num_nodes * (num_nodes - 1) / 2:
                print(f"Excessive number of edges for N={num_nodes}, conn={conn}")
                continue
            
            list_graphs = []

            for _ in tqdm(range(num_graphs)):

                # create graph
                graph = create_graph(num_nodes, num_edges, num_colors, type)

                # add graph to list
                list_graphs.append(graph)

                N = num_nodes//1000

            train_path = Path(f'data/{N}k/{type}/G_{num_graphs}-E_{num_edges}-C_{conn:.2f}/train')
            val_path   = Path(f'data/{N}k/{type}/G_{num_graphs}-E_{num_edges}-C_{conn:.2f}/val')
            test_path  = Path(f'data/{N}k/{type}/G_{num_graphs}-E_{num_edges}-C_{conn:.2f}/test')
            
            if not train_path.is_dir(): train_path.mkdir(parents=True, exist_ok=True)
            if not val_path.is_dir():   val_path.mkdir(parents=True, exist_ok=True)
            if not test_path.is_dir():  test_path.mkdir(parents=True, exist_ok=True)

            # print number of graphs generated
            print(f'Generated {len(list_graphs)} colorable graphs')

            # split graphs into train, validation and test sets
            train = int(split[0] * len(list_graphs))
            val = int(split[1] * len(list_graphs))

            print("Saving graphs...")
            print('---------------------------------------------------------------------------------')
            
            # save graphs
            torch.save(list_graphs[:train],f'{train_path}/graph.pt')
            torch.save(list_graphs[train:train+val],f'{val_path}/graph.pt')
            torch.save(list_graphs[train+val:],f'{test_path}/graph.pt')

generate_dataset(all_nodes=[1000, 3000, 10000, 30000, 100000], num_graphs=10, num_colors=5, connectivities=[11.25, 13.25, 15.25], split=[0.,0.,1.], type='plt')