from argparse import ArgumentParser
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import numpy as np
import sys
import tqdm
import torch
import logging


TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()
torch_inf = torch.tensor(np.Inf)


def _get_logger(name, log_file, formatter, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_file)        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level) 
    return logger


class NoLeaveProgressBar(pl.callbacks.ProgressBar):

    def init_test_tqdm(self):
        bar = tqdm.tqdm(
            desc='Testing',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar


class TrackTestAccuracyCallback(pl.callbacks.Callback):

    def on_validation_end(self, trainer, pl_module):
        trainer.test(model=pl_module, verbose=False)

        
class EarlyStoppingDoubleMetric(EarlyStopping):
    
    def __init__(
        self, double_monitor: bool = False, monitor: str = 'early_stop_on', 
        min_delta: float = 0.0, patience: int = 3, verbose: bool = False,
        mode: str = 'auto', strict: bool = True
    ):
        
        super().__init__(monitor, min_delta, patience, verbose, mode, strict)
        self.double_monitor = double_monitor
        self.min_acc_delta = self.min_delta*1
        self.min_loss_delta = self.min_delta*-1
        self.best_acc_score = -torch_inf
        self.best_loss_score = torch_inf 
        self.init_patience = self.patience
        self.reset_patience = False

        self.first_reset_wait = True

        if self.double_monitor:
            print('Using double monitor early stopping')
            
            
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler="resolve",
        )
        parser.add_argument("--monitor", type=str, default="valid_accuracy")
        parser.add_argument("--min_delta", type=float, default=0.01)
        parser.add_argument("--patience", type=int, default=17)
        parser.add_argument("--mode", type=str, default='auto')
        parser.add_argument("--double-monitor", action='store_true', default=False)
        return parser
        
    
    def on_save_checkpoint(self, trainer, pl_module):
        # Override
        return {
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
            'best_score': self.best_score,
            'best_acc_score': self.best_acc_score,
            'best_loss_score': self.best_loss_score,
            'patience': self.patience
        }


    def on_load_checkpoint(self, checkpointed_state):
        # Override
        self.wait_count = checkpointed_state['wait_count']
        self.stopped_epoch = checkpointed_state['stopped_epoch']
        self.best_score = checkpointed_state['best_score']
        self.best_acc_score = checkpointed_state['best_acc_score']
        self.best_loss_score = checkpointed_state['best_loss_score']
        self.patience = checkpointed_state['patience']
    
        
    def _run_early_stopping_check(self, trainer, pl_module):
        """
        If double_monitor is True, both validation loss and accuracy are monitored, 
        else classic pl early stopping is called.
        In the former case, the experiment stops when loss and accuracy are not 
        improving for x epochs (x=patience).
        """
        # Override
        if not self.double_monitor:
            super()._run_early_stopping_check(trainer, pl_module)
            return

        logs = trainer.logger_connector.callback_metrics

        if not self._validate_condition_metric(logs):
            return  # short circuit if metric not present

        loss_current = logs.get('valid_loss')
        acc_current = logs.get('valid_accuracy')

        if not isinstance(loss_current, torch.Tensor) and not isinstance(acc_current, torch.Tensor):
            loss_current = torch.tensor(loss_current, device=pl_module.device)
            acc_current = torch.tensor(acc_current, device=pl_module.device)

        if trainer.use_tpu and TPU_AVAILABLE:
            loss_current = loss_current.cpu()
            acc_current = acc_current.cpu()
            
        reset_wait = False
        if torch.gt(acc_current - self.min_acc_delta, self.best_acc_score):
            self.best_acc_score = acc_current
            reset_wait = True
        if torch.lt(loss_current - self.min_loss_delta, self.best_loss_score):
            self.best_loss_score = loss_current
            reset_wait = True

        if reset_wait:
            self.wait_count = 0
            if self.reset_patience and not self.first_reset_wait:
                self.patience = self.init_patience
                self.reset_patience = False
            self.first_reset_wait = False
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience
        
            if bool(should_stop):
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True          
        # Logging
        logger = _get_logger( 
            f'{trainer.logger.log_dir}_es', f'{trainer.logger.log_dir}/early_stopping.log',
            logging.Formatter('%(asctime)s - %(message)s')
        )
        logger.info(
            f'epoch{trainer.current_epoch}, acc_current:{acc_current}, acc_best:{self.best_acc_score}, '+
            f'loss_current:{loss_current}, loss_best:{self.best_loss_score}, ' +
            f'wait_count:{self.wait_count}, patience:{self.patience}'
        )
        # stop every ddp process if any world process decides to stop
        should_stop = trainer.accelerator_backend.early_stopping_should_stop(pl_module)
        trainer.should_stop = should_stop

    def set_patience(self, new_patience):
        self.patience = new_patience
        self.reset_patience = True
        
        
class LearningRateMonitorOnLog(LearningRateMonitor):
    def on_batch_start(self, trainer, pl_module):
        # Override
        if self.logging_interval != 'epoch':
            print('here')
            interval = 'step' if self.logging_interval is None else 'any'
            latest_stat = self._extract_lr(trainer, interval)

            if trainer.logger is not None and latest_stat:
                # Custom log file
                logger = _get_logger( 
                    f'{trainer.logger.log_dir}_lr', f'{trainer.logger.log_dir}/lr_monitor.log',
                    logging.Formatter('%(asctime)s - %(message)s')
                )
                logger.info(f'step:{trainer.global_step}, lr_current:{latest_stat}')


    def on_epoch_start(self, trainer, pl_module):
        # Override
        if self.logging_interval != 'step':
            interval = 'epoch' if self.logging_interval is None else 'any'
            latest_stat = self._extract_lr(trainer, interval)

            if trainer.logger is not None and latest_stat:
                # Custom log file
                logger = _get_logger( 
                    f'{trainer.logger.log_dir}_lr', f'{trainer.logger.log_dir}/lr_monitor.log',
                    logging.Formatter('%(asctime)s - %(message)s')
                )
                logger.info(f'epoch:{trainer.current_epoch}, lr_current:{latest_stat}')