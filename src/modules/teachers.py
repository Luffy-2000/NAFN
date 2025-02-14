import torch
from torch import nn
from copy import deepcopy


class Teacher(nn.Module):
    """
    From "Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?"
    Implemented from: https://github.com/RL-VIG/LibFewShot
    """

    def __init__(self, net, is_distill, teacher_path=None):
        super().__init__()
        self.net = self._load_state_dict(net, teacher_path, is_distill)
    
    def _load_state_dict(self, net, teacher_path, is_distill):
        new_net = None
        if is_distill and teacher_path is not None:
            new_net = deepcopy(net)
            new_net.initialize_weights(teacher_path)
        return new_net
        
    @torch.no_grad()
    def forward(self, x):
        logits = None
        if self.net is not None:
            logits = self.net(x)
        return logits