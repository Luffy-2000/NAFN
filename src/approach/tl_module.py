import os
import json
import torch
import importlib
import numpy as np
from torch import optim
from argparse import ArgumentParser
from collections import defaultdict
from pytorch_lightning import LightningModule

appr_module = {
    'rfs': 'LightningRFS'
}


class LightningTLModule(LightningModule):
    """
    LightningTLModule is a base class for implementing transfer-learning approaches.
    """
    # Adaptation parameters
    shots = 5
    queries = 5
    
    # Optimizer parameters
    lr = 0.001
    inner_lr = 0.01
    scheduler_patience = 10
    scheduler_decay = 0.1
    t0 = 10
    eta_min = 1e-5
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Adaptation parameters
        self.shots = kwargs.get('shots', LightningTLModule.shots) # Used for adaptation
        self.queries = kwargs.get('queries', LightningTLModule.queries)
        
        # Optimizer parameters
        self.lr = kwargs.get('lr', LightningTLModule.lr)
        self.inner_lr = kwargs.get('inner_lr', LightningTLModule.inner_lr)
        self.lr_strat = kwargs.get('lr_strat', 'none')
        self.scheduler_decay = kwargs.get(
            'scheduler_decay', LightningTLModule.scheduler_decay)
        self.scheduler_patience = kwargs.get(
            'scheduler_patience', LightningTLModule.scheduler_patience)
        self.t0 = kwargs.get('t0', LightningTLModule.t0)
        self.eta_min = kwargs.get('eta_min', LightningTLModule.eta_min)
        if self.lr_strat == 'none':
            self.scheduler_patience = float('inf')
            print('No lr scheduler')
        elif self.lr_strat == 'lrop':
            print(f'lrop - patience:{self.scheduler_patience}, decay:{self.scheduler_decay}')
        elif self.lr_strat == 'cawr':
            print(f'cawr - T0:{self.t0}, eta min:{self.eta_min}')
        else:
            raise ValueError('Unsupported lr strategy')
        
        # Other
        self._device = kwargs.get('device', 'cpu')
        self.is_fscil = kwargs.get('is_fscil', False)

    @staticmethod
    def factory_approach(approach_name, net, **kwargs):
        Approach = getattr(importlib.import_module(
                name=f'approach.{approach_name}'), appr_module[approach_name])
        
        ckpt_path = kwargs.get('ckpt_path', None) #None

        if ckpt_path is None:
            # Create a new approach
            return Approach(net=net, **kwargs)
        
        print(f'Recovering from checkpoint {ckpt_path}')
        return Approach.load_from_checkpoint(ckpt_path, net=net, **kwargs)
          
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        parser.add_argument(
            '--shots', type=int, default=LightningTLModule.shots
        )
        parser.add_argument(
            '--queries', type=int, default=LightningTLModule.queries
        )
        parser.add_argument(
            '--lr', type=float, default=LightningTLModule.lr
        )
        parser.add_argument(
            '--inner-lr', type=float, default=LightningTLModule.inner_lr
        )
        parser.add_argument(
            '--lr-strat', type=str, nargs='?', choices=['lrop', 'cawr', 'none'],
            default='none'
        )
        parser.add_argument(
            '--scheduler_patience', type=int, default=LightningTLModule.scheduler_patience
        )
        parser.add_argument(
            '--scheduler_decay', type=float, default=LightningTLModule.scheduler_decay
        )
        parser.add_argument(
            '--t0', type=int, default=LightningTLModule.t0
        )
        parser.add_argument(
            '--eta-min', type=float, default=LightningTLModule.eta_min
        )
        parser.add_argument('--ckpt-path', type=str, default=None)
        
        return parser
    
    ####
    ## ON STEP CALLBACKS
    ####
    
    def training_step(self, batch, batch_idx):
        outputs = self.pt_step(batch, batch_idx, is_train=True)
        
        self.log(
            'train_loss',
            outputs['loss'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            'train_accuracy',
            outputs['accuracy'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        outputs = self.pt_step(batch, batch_idx)
        
        self.log(
            'valid_loss',
            outputs['loss'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            'valid_accuracy',
            outputs['accuracy'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs['loss'].item()
    
    def test_step(self, batch, batch_idx):
        outputs = self.pt_step(batch, batch_idx)
        # print(outputs)
        self.log(
            'pt_test_loss',
            outputs['loss'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            'pt_test_accuracy',
            outputs['accuracy'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs
    
    def adaptation_step(self, dist_calibrator, batch):
        outputs = self.ft_step(dist_calibrator, batch)
        return outputs
    
    ####
    ## ON EPOCH END CALLBACKS
    ####    
    
    def test_epoch_end(self, outputs):
        self._save_outputs('pt_test_data', outputs)
        
    def adaptation_loop_end(self, outputs, path):
        self._save_outputs('adaptation_data', outputs, path)
        
    def _save_outputs(self, folder_name, outputs, path=None):
        save_path = path or self.logger.log_dir
        save_path = f'{save_path}/{folder_name}/'
        os.makedirs(save_path, exist_ok=True)
        
        save_outs = defaultdict(list)
        for output in outputs:
            for k, v in output.items():
                save_outs[k].append(v.detach().cpu().numpy() 
                                    if torch.is_tensor(v) else v)
        
        # Save specific outputs from the test on the pre-trained model                      
        if folder_name == 'pt_test_data':
            
            for k, v in save_outs.items():
                if k in ['labels', 'logits']:
                    save_outs[k] = np.concatenate(v)
                    
            np.savez_compressed(f'{save_path}labels.npz',
                                query_labels=save_outs['labels'])
        # Save specific outputs from the test on the fine-tuned model   
        elif folder_name == 'adaptation_data':
            
            np.savez_compressed(f'{save_path}labels.npz',
                                query_labels=save_outs['query_labels'],
                                support_labels=save_outs['support_labels'])
            np.savez_compressed(f'{save_path}supports.npz',
                                supports=save_outs['support'])
            np.savez_compressed(f'{save_path}queries.npz',
                                queries=save_outs['query'])
            if 'preds' in save_outs:
                np.savez_compressed(f'{save_path}preds.npz', 
                                    preds=save_outs['preds'])
        else:
            raise ValueError('bad folder_name')
        
        # Common outputs
        np.savez_compressed(f'{save_path}logits.npz',
                            logits=save_outs['logits'])
        np.savez_compressed(f'{save_path}losses.npz', 
                            loss=save_outs['loss'])
        with open(f'{save_path}le.json', 'w') as f:
            json.dump(save_outs['le_map'], f)
        
    ####
    ## OPTIMIZER SETUP
    ####
        
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
                    'monitor': 'valid_loss',  # Metric to monitor
                    'frequency': 1,  # How many epochs/steps should pass between calls to `scheduler.step()`
                    'name': 'ReduceLROnPlateau'  # Needed for logging
                }}
        else:
            raise ValueError('Unsupported lr strategy')
        
    ####
    ## UTILITY FUNCTIONS
    ####
    
    def label_encoding(self, labels):
        """ Label encoding from 0 to Ways """
        le = dict(zip(labels.unique().tolist(), range(labels.unique().size(0))))
        labels = torch.tensor([le[int(label)]
                              for label in labels], device=self._device)
        return labels, le