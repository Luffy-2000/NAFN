"""
NCR: Neighbourhood Consistency Regularisation.
基于 K-NN 近邻得到环境标签，与脏标签融合用于去噪.
Ref: https://arxiv.org/pdf/2202.02200.pdf
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

EPSILON = 1e-16


def l2_normalize(x: torch.Tensor, dim: int = -1, epsilon: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + epsilon)


def get_knn(
    queries: torch.Tensor,
    dataset: torch.Tensor,
    k: int,
    zero_negative_similarities: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K 近邻检索（L2 归一化后余弦相似度 = 内积）.

    Args:
        queries: [q, d]
        dataset: [n, d]
        k: 近邻数
    Returns:
        indices: [q, k]
        similarities: [q, k]
    """
    if k <= 0:
        k = dataset.shape[0]
    k = min(k, dataset.shape[0])

    queries = l2_normalize(queries, dim=-1)
    dataset = l2_normalize(dataset, dim=-1)

    sim = queries @ dataset.t()
    similarities, indices = torch.topk(sim, k, dim=-1)
    if zero_negative_similarities:
        similarities = F.relu(similarities)
    return indices, similarities


def pairwise_kl_loss(
    logits: torch.Tensor,
    neighbourhood_logits: torch.Tensor,
    knn_indices: torch.Tensor,
    knn_similarities: torch.Tensor,
    temperature: float,
    epsilon: float = EPSILON,
    example_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    KL 散度损失，按近邻相似度加权.

    Args:
        logits: [n, d] 当前样本 logits
        neighbourhood_logits: [m, d] 近邻池的 logits（如 one-hot）
        knn_indices: [n, k]
        knn_similarities: [n, k]
        temperature: 温度
    """
    n, d = logits.shape
    k = knn_indices.shape[1]

    knn_logits = neighbourhood_logits[knn_indices.reshape(-1)].reshape(n, k, d)
    t_softmax = F.softmax(knn_logits / temperature, dim=-1) + epsilon
    s_log_softmax = F.log_softmax(logits / temperature, dim=-1)

    norm_sim = knn_similarities / (knn_similarities.sum(dim=-1, keepdim=True) + epsilon)
    weighted_t = (norm_sim.unsqueeze(-1) * t_softmax).sum(dim=1)

    kldiv_per_pair = weighted_t * (torch.log(weighted_t + epsilon) - s_log_softmax)
    kldiv_per_example = (temperature ** 2) * kldiv_per_pair.sum(dim=-1)

    if example_weights is not None:
        norm = example_weights.sum()
    else:
        norm = n
    return kldiv_per_example.sum() / (norm + epsilon)


def ncr_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    batch_logits: torch.Tensor,
    batch_features: torch.Tensor,
    k: int,
    smoothing_gamma: float,
    temperature: float = 1.0,
    example_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NCR 损失：logits 与近邻加权 logits 的 KL 散度.
    """
    indices, similarities = get_knn(features, batch_features, k + 1)
    indices = indices[:, 1:]
    similarities = similarities[:, 1:]
    similarities = similarities.pow(smoothing_gamma)
    return pairwise_kl_loss(
        logits, batch_logits, indices, similarities,
        temperature, example_weights=example_weights,
    )


def build_ncr_soft_labels(
    s_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    way: int,
    k: int = 5,
    smoothing_gamma: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    用 K-NN 构建环境软标签，与脏标签融合.

    Args:
        s_embeddings: [n_support, dim]
        support_labels: [n_support]
        way: 类别数
        k: 近邻数
        smoothing_gamma: 相似度平滑
        alpha: 环境标签权重，final = (1-alpha)*dirty + alpha*env
    Returns:
        support_soft_labels: [n_support, way]
    """
    device = s_embeddings.device
    n = s_embeddings.size(0)
    k_actual = min(k, max(1, n - 1))
    if k_actual <= 0 or n <= 1:
        return F.one_hot(support_labels, num_classes=way).float()

    dirty_onehot = F.one_hot(support_labels, num_classes=way).float().to(device)
    indices, similarities = get_knn(s_embeddings, s_embeddings, k_actual + 1)
    indices = indices[:, 1:]
    similarities = similarities[:, 1:]
    k_neighbors = indices.shape[1]
    similarities = similarities.pow(smoothing_gamma)
    norm_sim = similarities / (similarities.sum(dim=-1, keepdim=True) + EPSILON)

    neighbour_onehot = dirty_onehot[indices.reshape(-1)].reshape(n, k_neighbors, way)
    env_soft = (norm_sim.unsqueeze(-1) * neighbour_onehot).sum(dim=1)

    support_soft_labels = (1 - alpha) * dirty_onehot + alpha * env_soft
    support_soft_labels = support_soft_labels / (support_soft_labels.sum(dim=-1, keepdim=True) + EPSILON)
    return support_soft_labels
