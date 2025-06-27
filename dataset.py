import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Subset, ConcatDataset
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Dataset

# class dataset
class CustomGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        super(CustomGraphDataset, self).__init__()

    def len(self): 
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        return graph
    
    def device(self):
        return self.graphs[0].x.device

def load_dataset(folder, graph_type):
    # Function to process files based on type (train, val, test)
    def process_files(file_type, graph_type=graph_type):
        graphs = []
        for file in os.listdir(os.path.join(folder, graph_type)):
            subfolder_path = os.path.join(folder, graph_type, file, file_type)
            for filename in os.listdir(subfolder_path):
                
                name = 'graph.pt'
                 
                if filename.endswith(name):
                    file_path = os.path.join(subfolder_path, filename)
                    print(file_path)
                    graph = torch.load(file_path)
                    graphs.extend(graph)
                     
        print(len(graphs))
        return graphs
             
    # Process files for each dataset type
    train_graphs = process_files('train')
    val_graphs   = process_files('val')
    test_graphs  = process_files('test')

    return CustomGraphDataset(train_graphs), CustomGraphDataset(val_graphs), CustomGraphDataset(test_graphs)

# data module from lightning
class LitDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size,graph_type, target = False, index = None, copies = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.index = index
        self.copies = copies
        self.target = target
        self.graph_type = graph_type


    def setup(self, stage = None):
        train_set, val_set, test_set = load_dataset(self.data_dir, self.graph_type)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        if self.index is not None and self.copies is not None:
            self.train_set = Subset(self.train_set, [self.index])
            self.train_set = ConcatDataset([self.train_set]*self.copies)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=39, pin_memory = True,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=39, pin_memory = True,drop_last=True,shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=39, pin_memory = True,drop_last=True, shuffle = False)