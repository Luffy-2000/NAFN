import importlib
import torch
from torch import nn
from copy import deepcopy
from networks.autoencoder import Autoencoder
from networks.UNet import UNet
from networks.UNet1D2D import UNet1D2D

class LLL_Net(nn.Module):
    """Basic class for implementing networks"""
    def __init__(self, model, activate_features=True, weights_path=None):
        super().__init__()
        self.model = model
        self.is_unsupervised = getattr(self.model, 'is_unsupervised', False)
        self.is_bayesian = hasattr(self.model, 'is_bayesian')
        if activate_features:
            self.feat_activation = nn.functional.relu
        else:
            self.feat_activation = nn.Identity()

        self.head = None
        self.initialize_weights(weights_path)


    @staticmethod
    def factory_network(**kwargs):
        model = getattr(importlib.import_module(name='networks'), kwargs['network'])
        init_model = model(
            num_pkts=kwargs['num_pkts'],
            num_fields=len(kwargs['fields']),
            out_features_size=kwargs['out_features_size'],
            scale=kwargs['scale'],
        )
        setattr(init_model, "is_unsupervised", kwargs.get("is_unsupervised", False))
        # setattr(init_model, "out_features_size", kwargs['num_pkts'] * len(kwargs['fields']))
        net = LLL_Net(init_model)
        net.add_head(num_outputs=kwargs['num_outputs'])
        return net
    
    def add_head(self, num_outputs):
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.model.out_features_size, num_outputs),
        )

    def forward(self, x, return_features=False):
        """Applies the forward pass
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        # 如果是自编码器，直接使用其forward方法
        if self.is_unsupervised:
            recon_x, logits = self.model(x)
            y = self.head(logits)
            y = self.feat_activation(y)
            return recon_x, logits, y
        else:
            x = self.model.extract_features(x)
            x = self.head(x)
            y = self.feat_activation(x)
            return (y, x) if return_features else y


    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def unfreeze_all(self, freezing=None, verbose=True):
        """Unfreeze all parameters from the model, including the head"""
        for name, param in self.named_parameters():
            if not freezing or not sum(nf in name for nf in freezing):
                param.requires_grad = True
        if verbose:
            self.trainability_info()

    def freeze_all(self, non_freezing=None, verbose=True):
        """Freeze all parameters from the model, including the head"""
        for name, param in self.named_parameters():
            if not non_freezing or not sum(nf in name for nf in non_freezing):
                param.requires_grad = False
        if verbose:
            self.trainability_info()

    def freeze_backbone(self, non_freezing=None, verbose=True):
        """Freeze all parameters from the main model, but not the heads"""
        for name, param in self.model.named_parameters():
            if not non_freezing or not sum(nf in name for nf in non_freezing):
                param.requires_grad = False
        if verbose:
            self.trainability_info()
        
    def trainability_info(self):
        print('\nTRAINABILITY INFO')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print('')

    def freeze_bn(self, verbose=True):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_head(self, index):
        for i in index:
            for param in self.head.parameters():
                param.requires_grad = False


    def initialize_weights(self, path, **kwargs):
        """Initialize weights using different strategies"""
        if path is None:
            return
        
        state_dict = torch.load(path)
            
        self.load_state_dict(state_dict)
        print(f'Model loaded from {path}')
