import os
import json
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from glob import glob
from pytorch_lightning.callbacks import ModelCheckpoint

from data.dataset_config import ClassInfo
import util.rng
import util.logger
import util.cleanup


class TLTrainer:
    """
    Adds functionalities for transfer-learning training procedure
    """
    def __init__(self, args, dict_args):
        self.args = args
        self.dict_args = dict_args
        self.log_path = args.default_root_dir
        self.device = self.args.device
            
        self._setup_first_trainer()
        
    def set_finetune_taskset(self, finetune_taskset):
        """Sets the finetuning dataset."""
        self.finetune_taskset = finetune_taskset
        
    def _setup_first_trainer(self):
        self.callbacks = [self.create_callbacks(self.args.callbacks)]
        self.trainers = [self.create_trainer()]
        
    def create_trainer(self):
        """ 
        Create a PL Trainer given args dict 
        """
        trainer_args = {
            'gpus': self.args.gpus,
            'deterministic': True,
            'accumulate_grad_batches': self.args.accumulate_grad_batches,
            'callbacks': self.callbacks[-1]
        }
        
        # If it is unsupervised learning, use the validation set to monitor training
        if hasattr(self.args, 'is_unsupervised') and self.args.is_unsupervised:
            trainer_args['checkpoint_callback'] = ModelCheckpoint(
                mode='min', 
                monitor='val_loss'
            )
        else:
            trainer_args['checkpoint_callback'] = ModelCheckpoint(
                mode=self.args.mode, 
                monitor=self.args.monitor
            )
            
        return pl.Trainer.from_argparse_args(self.args, **trainer_args)
        
        
    def create_callbacks(self, callbacks=[]):
        """ 
        Create a set of default callbacks with the one passed in 'callbacks' 
        """
        # If it is unsupervised learning, use the validation set loss as the monitoring indicator
        if hasattr(self.args, 'is_unsupervised') and self.args.is_unsupervised:
            monitor = 'val_loss'
            mode = 'min'
        else:
            monitor = self.args.monitor
            mode = self.args.mode

        default_callbacks = [
            util.callbacks.NoLeaveProgressBar(),
            util.callbacks.LearningRateMonitorOnLog(logging_interval='epoch'),
            util.callbacks.EarlyStoppingDoubleMetric(
                monitor=monitor, 
                min_delta=self.args.min_delta,
                patience=self.args.patience, 
                mode=mode, 
                verbose=True,
                double_monitor=self.args.double_monitor)
        ]
        return default_callbacks + callbacks
    
    
    def fit(self, approach, datamodule=None, train_dataloader=None, val_dataloader=None):
        """ 
        Wrapper function to Pytorch Lightning Trainer fit 
        """
        self.save_args()
        
        # If it is unsupervised learning, use datamodule
        if hasattr(approach, 'is_unsupervised') and approach.is_unsupervised:
            self.trainers[0].fit(
                model=approach,
                datamodule=datamodule
            )
        else:
            self.trainers[0].fit(
                model=approach,
                datamodule=datamodule,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader,
            )
    
    
    def test(self):  
        """ 
        Wraps Pytorch Lightning 'test()' 
        """
        best_model_path = glob(f'{self.trainers[-1].logger.log_dir}/checkpoints/*')[0]
        eval_res = self.trainers[-1].test(ckpt_path=best_model_path)
        return eval_res  
    
    
    def adaptation(self, approach, dataloader, already_resumed=False):
        """ 
        Implementation of the fine-tuning stage with a FSL/FSCIL procedure 
        """
        if self.args.pretrained_autoencoder or self.args.ft_only or already_resumed:
            # Use current model weights
            best_approach = approach
        else:
            # Resuming best weights
            best_ckpt_path = glob(f'{self.trainers[-1].logger.log_dir}/checkpoints/*')[0]
            print(f'Resuming {best_ckpt_path} for fine-tuning')
            print('-'*80)
            best_approach = type(approach).load_from_checkpoint(
                net=approach.net, checkpoint_path=best_ckpt_path, **vars(self.args))

        # Adaptation on the Support Set and evaluation on the Query Set
        losses = []
        accuracies = []
        outputs = []
        ft_loop = tqdm(dataloader)
        
        best_approach.on_adaptation_start()
            
        for episode in ft_loop:
            # Adaptation step
            episode = [episode[0].to(self.device), episode[1].to(self.device)]
            output = best_approach.adaptation_step(episode) 
            
            # Collect metrics from the last epoch
            loss = output['loss'].item()
            acc = output['accuracy'].item()
            ft_loop.set_postfix(loss=loss, acc=acc)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(output)
            
        best_approach.adaptation_loop_end(
            outputs=outputs, 
            path=f'{self.trainers[-1].logger.log_dir}', 
        )
        
        # Return the avg per episode metrics     
        eval_res = {
            'adaptation_loss': np.array(losses).mean(),
            'adaptation_accuracy': np.array(accuracies).mean()
        }
        print('\n'+'-'*80)
        print(eval_res)
        return eval_res
    
    
    def save_args(self):
        """ 
        Storing initial argument dict
        """
        os.makedirs(f'{self.trainers[-1].logger.log_dir}', exist_ok=True)
        with open(f'{self.trainers[-1].logger.log_dir}/dict_args.json', 'w') as f:
            json.dump(self.dict_args, f)
    
    
    def save_results(self, eval_res, trainer_idx=-1):
        # If it is unsupervised learning, only clean the model file
        if hasattr(self.args, 'is_unsupervised') and self.args.is_unsupervised:
            util.cleanup.cleanup_autoencoder_models(self.trainers[trainer_idx].logger.log_dir)
            return
            
        f1_scores = util.logger.get_metric(
            exp_path=self.trainers[trainer_idx].logger.log_dir,
            folders=['adaptation_data', 'pt_test_data'],
            wanted_metrics=['f1_all_macro']
        )
        
        util.logger.plot_confusion_matrix(exp_path=self.trainers[trainer_idx].logger.log_dir)
        with open(f'{self.trainers[trainer_idx].logger.log_dir}/test_results.json', 'w') as f:
            json.dump({**eval_res, **f1_scores}, f)
        ci = ClassInfo()
        ci.save_data(self.trainers[trainer_idx].logger.log_dir)
        
        util.cleanup.cleanup_distill_models(self.trainers[trainer_idx].logger.log_dir)
        # WARNING: it deletes all the adaptation feature vectors 
        # util.cleanup.cleanup_embeddings(self.trainers[trainer_idx].logger.log_dir)
