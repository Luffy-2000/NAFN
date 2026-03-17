"""
Transition matrix utilities for instance-dependent label noise.
Based on "Confidence Scores Make Instance-dependent Label-noise Learning Possible" (Berthon et al., 2021).
"""
import torch
import torch.nn.functional as F


def get_mus_hat(r: float, y_vec: torch.Tensor, diag_hat: torch.Tensor) -> torch.Tensor:
    """
    Compute diagonal of transition matrix for one sample.
    mus_hat[observed] = r (confidence), mus_hat[other] = diag_hat.

    Args:
        r: Confidence = P(observed_label | x)
        y_vec: One-hot of observed label, shape (C,)
        diag_hat: Diagonal for non-observed classes; typically ro (noise-scale hyperparameter)
    """
    if isinstance(diag_hat, (int, float)):
        diag_hat = torch.full_like(y_vec, float(diag_hat), dtype=y_vec.dtype)
    return (1.0 - y_vec) * diag_hat + r * y_vec


def S_to_T(mus_hat: torch.Tensor, n_class: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Build transition matrix P from diagonal (symmetric structure).
    P[i,j] = P(observe j | true i).
    Diagonal: P[i,i] = mus_hat[i]. Off-diagonal: uniform over other classes.

    Args:
        mus_hat: Diagonal values, shape (C,)
        n_class: Number of classes
        eps: Small constant for numerical stability
    Returns:
        P: (C, C) transition matrix, each row sums to 1
    """
    device = mus_hat.device
    P = torch.zeros(n_class, n_class, device=device, dtype=mus_hat.dtype)
    for i in range(n_class):
        P[i, i] = mus_hat[i].clamp(eps, 1.0 - eps)
        remainder = 1.0 - P[i, i]
        n_other = n_class - 1
        if n_other > 0:
            for j in range(n_class):
                if j != i:
                    P[i, j] = remainder / n_other
    return P


def backward_corrected_label(P: torch.Tensor, observed_class: int, restrict_to_new: bool = True,
                            num_old_classes: int = 0, num_new_classes: int = 0) -> torch.Tensor:
    """
    Backward correction: P(true | obs) ∝ P(obs | true).
    soft_label[c] = P[c, observed] / sum_c P[c, observed].

    If restrict_to_new: zero out old classes to avoid contaminating old-class prototypes (prevents F1-old drop).
    """
    col = P[:, observed_class].clamp(min=1e-12)
    if restrict_to_new and num_old_classes > 0 and num_new_classes > 0:
        col = col.clone()
        col[:num_old_classes] = 0.0
        col = col.clamp(min=1e-12)
    return col / col.sum()
