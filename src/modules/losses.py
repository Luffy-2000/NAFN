from torch import nn


class DistillKLLoss(nn.Module):
    """
    Implementation of the Kullback-Leibler divergence for distilliation
    """
    def __init__(self, T):
        super(DistillKLLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, already_soft_values=False):
        if y_t is None:
            return 0.0

        p_s = nn.functional.log_softmax(y_s / self.T, dim=1)
        p_t = nn.functional.softmax(y_t / self.T, dim=1) if not already_soft_values else y_t
        # FIXME: Deprecation warning on size_average
        loss = nn.functional.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.size(0)
        return loss