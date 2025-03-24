import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

class LightningUnsupervised(LightningModule):
    """Training module for unsupervised learning"""
    # Optimizer parameters
    lr = 0.001
    inner_lr = 0.01
    scheduler_patience = 10
    scheduler_decay = 0.1
    t0 = 10
    eta_min = 1e-5

    def __init__(self, net, **kwargs):
        super().__init__()
        # Optimizer parameters
        self.lr = kwargs.get('lr', LightningUnsupervised.lr)
        self.inner_lr = kwargs.get('inner_lr', LightningUnsupervised.inner_lr)
        self.lr_strat = kwargs.get('lr_strat', 'none')
        self.scheduler_decay = kwargs.get(
            'scheduler_decay', LightningUnsupervised.scheduler_decay)
        self.scheduler_patience = kwargs.get(
            'scheduler_patience', LightningUnsupervised.scheduler_patience)
        self.t0 = kwargs.get('t0', LightningUnsupervised.t0)
        self.eta_min = kwargs.get('eta_min', LightningUnsupervised.eta_min)

        if self.lr_strat == 'none':
            self.scheduler_patience = float('inf')
            print('No lr scheduler')
        elif self.lr_strat == 'lrop':
            print(f'lrop - patience:{self.scheduler_patience}, decay:{self.scheduler_decay}')
        elif self.lr_strat == 'cawr':
            print(f'cawr - T0:{self.t0}, eta min:{self.eta_min}')
        else:
            raise ValueError('Unsupported lr strategy')
        
        self.net = net
        self.is_unsupervised = True  # Add unsupervised learning flag
        self.save_hyperparameters()

        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.net(x)
    

    def training_step(self, batch, batch_idx):
        x, y = batch  
        x_recon, logits = self(x)
        recon_loss = self.criterion1(x_recon, x)
        ce_loss = self.criterion2(logits, y)
        loss = 0.9*recon_loss + 0.1*ce_loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_recon, logits = self(x)
        recon_loss = self.criterion1(x_recon, x)
        ce_loss = self.criterion2(logits, y)
        loss = 0.9*recon_loss + 0.1*ce_loss
        self.log('val_loss', loss)
        return loss
        

    def training_epoch_end(self, _):
        # Save model weights
        saved_weights_path = f'{self.logger.log_dir}/pretrain_models'
        os.makedirs(saved_weights_path, exist_ok=True)
        checkpoint = {
            'encoder': self.net.model.state_dict(),
            'head': self.net.head.state_dict(),
        }   
        torch.save(checkpoint, 
                   f'{saved_weights_path}/autoencoder_ep{self.trainer.current_epoch}.pt')
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_strat == 'cawr':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.t0, T_mult=1, eta_min=self.eta_min
            )
            return [optimizer], [lr_scheduler]
        elif self.lr_strat in ['none', 'lrop']:
            # If it is 'none' the patience is inf
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=self.scheduler_patience, mode='min',
                        factor=self.scheduler_decay
                    ),
                    'interval': 'epoch',  # The unit of the scheduler's step size
                    'monitor': 'val_loss',  # Metric to monitor
                    'frequency': 1,  # How many epochs/steps should pass between calls to `scheduler.step()`
                    'name': 'ReduceLROnPlateau'  # Needed for logging
                }}
        else:
            raise ValueError('Unsupported lr strategy')




    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    #     return optimizer 