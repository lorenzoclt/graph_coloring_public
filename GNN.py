from time import time
import torch
from torch.optim import Adam
from torch_scatter import scatter_mean
from torch.nn.functional import softmax
from torch_geometric.utils import add_self_loops
from torch.optim.lr_scheduler import StepLR
from torch.nn import ModuleList
import pytorch_lightning as pl

class multiscale_layer(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_colors):
        super(multiscale_layer, self).__init__()
        
        # create NN for aggregating intermediate node features with dropout
        self.multiscale = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, num_colors)
        )

    def forward(self, intermediate_node_features):
        return  self.multiscale(intermediate_node_features)

class phi_layer(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate):
        super(phi_layer, self).__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, input):
        return self.phi(input)

class gamma_layer(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate):
        super(gamma_layer, self).__init__()
        self.gamma = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, input):
        return self.gamma(input)

class MMPN_denoiser(pl.LightningModule):
    def __init__(self, batch_size, latent_dim, num_colors, num_layers, dropout_rate, self_loops, temp, learning_rate, patience_factor, patience, entropy_time, entropy_factor, gamma,alpha_min, alpha_max):
        super(MMPN_denoiser, self).__init__()

        self.save_hyperparameters()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.self_loops = self_loops
        self.latent_dim = latent_dim
        self.temp = temp
        self.learning_rate = learning_rate
        self.patience_factor = patience_factor
        self.patience = patience
        self.entropy_time = entropy_time
        self.entropy_factor = entropy_factor
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_colors = num_colors

        latent_dim = latent_dim + 1 # add one dimension for number of neighbors
        self.table_alpha = torch.linspace(alpha_min, alpha_max, 1000)

        # define phi NNs
        phi_layers = [phi_layer(2*(num_colors+1), latent_dim, dropout_rate)]
        phi_layers.extend([phi_layer(2*latent_dim, latent_dim, dropout_rate) for _ in range(num_layers-1)])
        self.phi_modules = ModuleList(phi_layers)

        # define gamma NNs
        gamma_layers = [gamma_layer(latent_dim+num_colors+1, latent_dim, dropout_rate)]
        gamma_layers.extend([gamma_layer(2*latent_dim, latent_dim, dropout_rate) for _ in range(num_layers-1)])
        self.gamma_modules = ModuleList(gamma_layers)

        # define last NN
        self.multiscale = multiscale_layer(num_layers*latent_dim+num_colors+1, latent_dim, num_colors)

    def forward(self, data):

        # unpack data
        x, edge_index = data.x, data.edge_index

        # start accumulating features
        intermediate_node_features = x
        
        # add self-loops to each node
        if self.self_loops: edge_index, _ = add_self_loops(edge_index, num_nodes = x.shape[0])

        for phi, gamma in zip(self.phi_modules, self.gamma_modules):

            # message
            x_i = x[edge_index[0]] # x_i is the source node (sends messages)
            x_j = x[edge_index[1]] # x_j is the target node (receives messages)
            joint_input = torch.cat([x_i, x_j], dim=-1)
            messages = phi(joint_input)

            # aggregate
            target_nodes = edge_index[1]
            aggregated_messages = scatter_mean(messages, target_nodes, dim=0, dim_size=x.size(0))

            # update
            joint_input = torch.cat([x, aggregated_messages], dim=-1)
            x = gamma(joint_input)

            # save intermediate node features
            intermediate_node_features = torch.cat([intermediate_node_features, x], dim = 1)

        # use a NN to aggregate intermediate node features
        x = self.multiscale(intermediate_node_features)

        # delete intermediate node features to save memory
        del intermediate_node_features

        # softmax with adjustable temperature
        x = softmax(x/self.temp, dim=1)

        return x

    # computes potts energy on graph
    def continuous_energy(self, graph):
        
        # compute the energy of the batch
        scalar_product_tensor = torch.sum((graph.x[graph.edge_index[0],:self.num_colors] * graph.x[graph.edge_index[1],:self.num_colors]), dim=1)
        continuous_energy     = torch.mean(scalar_product_tensor)

        return continuous_energy
    
    def overlap(self, graph, input):

        # measures overlap between output of the network and the input
        overlap = torch.sum(graph.x[:,:self.num_colors] * input[:,:self.num_colors], dim = 1).mean()

        return overlap
    
    # computes potts energy on graph
    def discrete_energy(self, graph):
        
        # compute one-hot features
        max_indices   = torch.argmax(graph.x[:,:self.num_colors], dim=1)
        rows          = torch.arange(len(graph.x[:,:self.num_colors]), device = graph.x.device) # rows indexes
        one_hot_graph = torch.zeros_like(graph.x[:,:self.num_colors], device = graph.x.device)
        one_hot_graph[rows, max_indices] = 1 

        # compute percentage of conflicting edges
        conflicts_tensor = torch.sum(one_hot_graph[graph.edge_index[0],:self.num_colors] * one_hot_graph[graph.edge_index[1],:self.num_colors], dim = 1)
        avg_conflicts    = torch.sum(conflicts_tensor, dim = 0)/conflicts_tensor.shape[0]

        return avg_conflicts

    # used to compute one-hotness
    def entropy(self, batch):

        # calculate entropy for each vector, using clamp to avoid log(0)
        log_probs      = torch.log2(batch[:,:self.num_colors].clamp(min=1e-10))
        entropy_values = -torch.sum(batch[:,:self.num_colors]*log_probs, dim=1)

        return torch.mean(entropy_values)
    
    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=self.patience, gamma=self.patience_factor, verbose=True)

        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx):

        # exclude degree from features
        target = train_batch.x[:,:self.num_colors].clone()
        num_graphs = train_batch.num_graphs
        self.alpha_t = self.table_alpha[torch.randint(0, 1000,(num_graphs,))].to(train_batch.x.device)
        self.alpha_t = self.alpha_t[train_batch.batch]
        train_batch.x[:,:self.num_colors] = torch.sqrt(self.alpha_t).unsqueeze(1)*train_batch.x[:,:self.num_colors] + torch.sqrt(1-self.alpha_t).unsqueeze(1)*torch.randn_like(train_batch.x[:,:self.num_colors])
  
        # forward pass
        train_batch.x[:,:self.num_colors] = self.forward(train_batch) 
                 
        # compute energy and entropy
        batch_continuous_energy = self.continuous_energy(train_batch) # continuous and discrete energy of the graph
        batch_entropy           = self.entropy(train_batch.x) # entropy of the batch
        overlap                 = self.overlap(train_batch, target)

        # compute loss
        batch_loss = 0
        batch_loss += batch_continuous_energy
        batch_loss -= self.gamma*overlap
        
        if self.current_epoch < self.entropy_time: batch_loss += self.entropy_factor * batch_entropy

        # log metrics
        self.log('energy/train', batch_continuous_energy, prog_bar = True, on_epoch=True, sync_dist=True)
        self.log('loss/train', batch_loss, prog_bar = True, on_epoch=True, sync_dist=True)
        self.log('entropy/train', batch_entropy, prog_bar = True, on_epoch=True, sync_dist=True)
        self.log('overlap/train', overlap, prog_bar = True, on_epoch=True, sync_dist=True)
          
        return batch_loss
    
    def validation_step(self, val_batch, batch_idx):

        target      = val_batch.x.clone()
        val_batch.x[:,:self.num_colors] = self.forward(val_batch)
        overlap     = self.overlap(val_batch, target)
        
        # compute conflicts and entropy
        batch_avg_conflicts = self.discrete_energy(val_batch) # continuous and discrete energy of the graphv
        entropy             = self.entropy(val_batch.x)
        
        # log metrics
        self.log('energy/val', batch_avg_conflicts, prog_bar = True, on_epoch=True, sync_dist=True)
        self.log('entropy/val', entropy, prog_bar = True, on_epoch=True, sync_dist=True)
        self.log('overlap/val', overlap, prog_bar = True, on_epoch=True, sync_dist=True)
         
        return batch_avg_conflicts
    
class MMPN_torch(pl.LightningModule):
    def __init__(self, latent_dim, num_colors, num_layers, dropout_rate, self_loops, temp):
        super(MMPN_torch, self).__init__()

        self.save_hyperparameters()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.self_loops = self_loops
        self.latent_dim = latent_dim
        self.temp = temp
        self.num_colors = num_colors

        # define phi NNs
        phi_layers = [phi_layer(2*(num_colors+1), (latent_dim+1), dropout_rate)]
        phi_layers.extend([phi_layer(2*(latent_dim+1), (latent_dim+1), dropout_rate) for _ in range(num_layers-1)])
        self.phi_modules = ModuleList(phi_layers)

        # define gamma NNs
        gamma_layers = [gamma_layer((latent_dim+1)+num_colors+1, (latent_dim+1), dropout_rate)]
        gamma_layers.extend([gamma_layer(2*(latent_dim+1), (latent_dim+1), dropout_rate) for _ in range(num_layers-1)])
        self.gamma_modules = ModuleList(gamma_layers)

        # define last NN
        self.multiscale = multiscale_layer(num_layers*(latent_dim+1)+num_colors+1, (latent_dim+1), num_colors)

    def forward(self, x, edge_index):

        # start accumulating features
        intermediate_node_features = x
        
        # add self-loops to each node
        if self.self_loops: edge_index, _ = add_self_loops(edge_index, num_nodes = x.shape[0])

        edge_index_0 = edge_index[0]
        edge_index_1 = edge_index[1]

        for phi, gamma in zip(self.phi_modules, self.gamma_modules):

            # message
            x_i = x[edge_index_0] # x_i is the source node (sends messages)
            x_j = x[edge_index_1] # x_j is the target node (receives messages)
            joint_input = torch.cat([x_i, x_j], dim=-1)
            messages = phi(joint_input)

            # aggregate
            target_nodes = edge_index_1
            aggregated_messages = scatter_mean(messages, target_nodes, dim=0, dim_size=x.size(0))

            # update
            joint_input = torch.cat([x, aggregated_messages], dim=-1)
            x = gamma(joint_input)

            # save intermediate node features
            intermediate_node_features = torch.cat([intermediate_node_features, x], dim = 1)

        # use a NN to aggregate intermediate node features
        x = self.multiscale(intermediate_node_features)

        # delete intermediate node features to save memory
        del intermediate_node_features

        # softmax with adjustable temperature
        x = softmax(x/self.temp, dim=1)

        return x
    