import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import copy

# 导入流量变换函数用于对比学习
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.traffic_transformations import permutation, pkt_translating, wrap


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    SimCLR (https://arxiv.org/abs/2002.05709)
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z_i, z_j):
        """
        z_i, z_j: 两个视角的特征表示 [N, D]
        """
        batch_size = z_i.shape[0]
        
        # 特征归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 将特征堆叠为 [2*N, D]
        features = torch.cat([z_i, z_j], dim=0)
        
        # 计算相似度矩阵 [2*N, 2*N]
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / self.temperature
        # 去除自身的相似度 (对角线元素)
        mask = torch.eye(2 * batch_size, dtype=bool, device=similarity_matrix.device)    
        similarity_matrix[mask] = -float('inf')     # [2*N, 2*N]
        # similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        # print(similarity_matrix.shape)
        #exit()
        # 正样本对的索引
        pos_idx = torch.arange(batch_size, device=similarity_matrix.device)

        pos_idx = torch.cat([pos_idx + batch_size, pos_idx], dim=0)
        
        # 计算对比损失
        logits = similarity_matrix
        labels = pos_idx

        loss = self.criterion(logits, labels) / (2 * batch_size)
        return loss

class LightningUnsupervised(LightningModule):
    """Training module for unsupervised learning"""
    # Optimizer parameters
    lr = 0.001
    inner_lr = 0.01
    scheduler_patience = 10
    scheduler_decay = 0.1
    t0 = 10
    eta_min = 1e-5
    transform_method = 0

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

        # 对比学习参数
        self.mode = kwargs.get('pre_mode', 'recon')  # 'recon'或'contrastive'或'hybrid'
        self.temperature = kwargs.get('temperature', 0.5)
        self.transform_strength = kwargs.get('transform_strength', 0.8)
        
        # 损失权重
        self.recon_weight = kwargs.get('recon_weight', 0.5)
        self.ce_weight = kwargs.get('ce_weight', 0.5)
        self.contrastive_weight = kwargs.get('contrastive_weight', 0.5)

        # EMA参数
        self.ema_decay = kwargs.get('ema_decay', 0.999)

        if self.lr_strat == 'none':
            self.scheduler_patience = float('inf')
            print('No lr scheduler')
        elif self.lr_strat == 'lrop':
            print(f'lrop - patience:{self.scheduler_patience}, decay:{self.scheduler_decay}')
        elif self.lr_strat == 'cawr':
            print(f'cawr - T0:{self.t0}, eta min:{self.eta_min}')
        else:
            raise ValueError('Unsupported lr strategy')
        
        print(f'Training mode: {self.mode}')
        
        self.net = net
        self.is_unsupervised = True  # Add unsupervised learning flag
        self.save_hyperparameters()

        # #双塔模型复制
        # self.copy_net = copy.deepcopy(net)

        self.criterion_reconstructive = nn.MSELoss()
        self.criterion_classify = nn.CrossEntropyLoss()
        self.criterion_contrastive = NTXentLoss(temperature=self.temperature)
        
    def forward(self, x):
        return self.net(x)

    # def forward_dual_tower(self, x):
    #     return self.copy_net(x)
    
    def _update_ema(self):
        """使用EMA更新目标模型的参数"""
        with torch.no_grad():
            for param_q, param_k in zip(self.net.parameters(), self.copy_net.parameters()):
                param_k.data = param_k.data * self.ema_decay + param_q.data * (1. - self.ema_decay)

    def _apply_transform(self, x, transform_method=None):
        """Apply the same random transformation to all samples in the batch"""
        # Convert tensor to numpy for transformation
        device = x.device
        x_np = x.detach().cpu().numpy()
        
       # If transform_method is not specified, randomly choose one
        if transform_method is None:
            transform_method = np.random.choice([0, 1, 2])
        elif transform_method not in [0, 1, 2]:
            raise ValueError("transform_method must be 0, 1, or 2")

        # Apply the same transformation to all samples
        if transform_method == 0:
            transformed = np.array([permutation(sample, a=self.transform_strength) for sample in x_np])
        elif transform_method == 1:
            transformed = np.array([pkt_translating(sample, a=self.transform_strength) for sample in x_np])
        else:
            transformed = np.array([wrap(sample, a=self.transform_strength) for sample in x_np])
        # Convert back to PyTorch tensor
        return torch.from_numpy(transformed).to(device)



    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == 'recon':
            # Reconstruction mode
            x_recon, _, logits = self(x)
            recon_loss = self.criterion_reconstructive(x_recon, x)
            ce_loss = self.criterion_classify(logits, y)
            loss = self.recon_weight * recon_loss + self.ce_weight * ce_loss
            
            self.log('train_recon_loss', recon_loss)
            self.log('train_ce_loss', ce_loss)
            

        elif self.mode == 'contrastive':
            # Contrastive learning mode
            # Generate positive samples
            x_augmented = self._apply_transform(x, transform_method=0)
            
            # Get feature representations from two views
            _, z_i, _ = self(x)
            _, z_j, _ = self(x_augmented)
            
            # Calculate contrastive loss
            contrastive_loss = self.criterion_contrastive(z_i, z_j)
            
            # Optional: Calculate classification loss
            _, _, logits = self(x)
            ce_loss = self.criterion_classify(logits, y)
            
            # Combine losses
            loss = self.contrastive_weight * contrastive_loss + self.ce_weight * ce_loss
            
            self.log('train_contrastive_loss', contrastive_loss)
            self.log('train_ce_loss', ce_loss)
            
        elif self.mode == 'hybrid':
            self.recon_weight, self.contrastive_weight, self.ce_weight = 0.33, 0.33, 0.33
            # Hybrid mode: Combine reconstruction and contrastive learning
            # Generate positive samples
            x_augmented = self._apply_transform(x, transform_method=0) 
            
            # Reconstruction loss
            x_recon, _, logits = self(x)
            recon_loss = self.criterion_reconstructive(x_recon, x)
            
            # Contrastive loss
            _, z_i, _ = self(x)
            _, z_j, _ = self(x_augmented)
            contrastive_loss = self.criterion_contrastive(z_i, z_j)
            
            # Classification loss
            ce_loss = self.criterion_classify(logits, y)
            
            # Combine losses
            loss = (self.recon_weight * recon_loss + 
                   self.contrastive_weight * contrastive_loss + 
                   self.ce_weight * ce_loss)
            
            self.log('train_recon_loss', recon_loss)
            self.log('train_contrastive_loss', contrastive_loss)
            self.log('train_ce_loss', ce_loss)
        
        self.log('train_loss', loss)
        return loss
        

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        if self.mode == 'recon':
            # Reconstruction mode
            x_recon, _, logits = self(x)
            recon_loss = self.criterion_reconstructive(x_recon, x)
            ce_loss = self.criterion_classify(logits, y)

            loss = self.recon_weight * recon_loss + self.ce_weight * ce_loss
            self.log('val_recon_loss', recon_loss)
            self.log('val_ce_loss', ce_loss)
            
        elif self.mode == 'contrastive':
            # Contrastive learning mode
            # Generate positive samples
            x_augmented = self._apply_transform(x, transform_method=None)

            # Get feature representations from two views
            _, z_i, _ = self(x)
            _, z_j, _ = self(x_augmented)

            # Calculate contrastive loss
            contrastive_loss = self.criterion_contrastive(z_i, z_j)
            # Calculate classification loss
            _, _, logits = self(x)
            ce_loss = self.criterion_classify(logits, y)
            
            # Combine losses
            loss = self.contrastive_weight * contrastive_loss + self.ce_weight * ce_loss
            
            self.log('val_contrastive_loss', contrastive_loss)
            self.log('val_ce_loss', ce_loss)
            
        elif self.mode == 'hybrid':
            self.recon_weight, self.contrastive_weight, self.ce_weight = 0.33, 0.33, 0.33
            # Hybrid mode: Combine reconstruction and contrastive learning
            # Generate positive samples
            x_augmented = self._apply_transform(x, transform_method=None)
            
            # Reconstruction loss
            x_recon, _, logits = self(x)
            recon_loss = self.criterion_reconstructive(x_recon, x)
            
            # Contrastive loss
            _, z_i, _ = self(x)
            _, z_j, _ = self(x_augmented)
            contrastive_loss = self.criterion_contrastive(z_i, z_j)
            
            # Classification loss
            ce_loss = self.criterion_classify(logits, y)
            
            # Combine losses
            loss = (self.recon_weight * recon_loss + 
                   self.contrastive_weight * contrastive_loss + 
                   self.ce_weight * ce_loss)
            
            self.log('val_recon_loss', recon_loss)
            self.log('val_contrastive_loss', contrastive_loss)
            self.log('val_ce_loss', ce_loss)
        
        self.log('val_loss', loss)
        return loss
        

    def training_epoch_end(self, _):
        # Save model weights
        saved_weights_path = f'{self.logger.log_dir}/pretrain_models'
        os.makedirs(saved_weights_path, exist_ok=True)
        checkpoint = {
            'encoder': self.net.encoder.state_dict() if hasattr(self.net, 'encoder') else self.net.model.state_dict(),
            'head': self.net.head.state_dict() if hasattr(self.net, 'head') else None,
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