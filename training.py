'''train models given hyperparameters'''

import os
import torch
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from GNN import MMPN_denoiser
from dataset import LitDataModule
from hparams import hparams_combinations, num_trials

# run num_trials for each hparams combination
for _ in range(num_trials):

    # try every combination of hparams
    for hparams in hparams_combinations:
         
        # unpack hparams
        num_epochs, batch_size, learning_rate, patience, patience_factor, num_colors, latent_dim, num_layers, temp, entropy_factor, entropy_time, dropout_rate, self_loops, gamma,alpha_min,alpha_max,t= hparams
        
        # set batch size per gpu
        num_gpus = max(1, torch.cuda.device_count())
        batch_size_per_gpu = batch_size//num_gpus

        # create path to save weights and load data
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
         
        log_dir = os.getcwd() + '/training_log/' +f't{t}_{timestr}' + '/'
        data_dir = os.getcwd()+'/data/paper10k/'
        
        # load dataset
        data = LitDataModule(data_dir, batch_size_per_gpu, graph_type='planted/', target = True) 

        # define callbacks
        early_stopping = EarlyStopping(monitor='energy/val', min_delta=0.00001, patience=100, verbose=True,  mode='min')
        checkpoint     = ModelCheckpoint(monitor='energy/train_epoch', dirpath=log_dir, filename='checkpoint-{epoch:02d}-{val_loss:.2f}', save_top_k=1, save_last=True,mode='min')

        # create TensorBoard logger
        logger  = TensorBoardLogger(log_dir, name = "training")
        trainer = Trainer(max_epochs = num_epochs, logger = logger, callbacks=[early_stopping, checkpoint])

        # instantiate lightning model
        model = MMPN_denoiser(batch_size, latent_dim, num_colors, num_layers, dropout_rate, self_loops, temp, learning_rate, patience_factor, patience, entropy_time, entropy_factor, gamma, alpha_min, alpha_max)

        # train model
        trainer.fit(model, data)