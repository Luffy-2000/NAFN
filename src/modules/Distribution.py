import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple

class RepresentationDistributionCalibrator:
    def __init__(self, top_q=3, gamma=0.7, alpha=1e-2):
        """
        Args:
            top_q: 用于新类分布校准的 base 类数量
            gamma: 融合 base 分布的比例
            alpha: 协方差扰动项，避免奇异性
        """
        self.top_q = top_q
        self.gamma = gamma
        self.alpha = alpha
        self.class_distributions = {}  # {class_id: (mean, cov)}
        self.class_sizes = {}  # {class_id: num_samples}

    def update_class_distribution(self, class_id: int, features: np.ndarray):
        """
        更新或记录某个类的表示分布（特征需为 numpy.ndarray 形状为 [N, D]）
        """
        assert len(features.shape) == 2
        mu = np.mean(features, axis=0)
        cov = np.cov(features.T)
        self.class_distributions[class_id] = (mu, cov)
        self.class_sizes[class_id] = len(features)

    def calibrate(self, target_class_id: int, target_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用已有 base 类分布信息，校准新类 target_class_id 的特征分布，并采样补充数据。
        返回：(sampled_features, sampled_labels)
        """
        mu_k = target_features.mean(0)
        sigma_k = np.cov(target_features.T)

        # 计算与 base 类的欧式距离
        distances = {
            base_cls: np.linalg.norm(mu_k - mu_base)
            for base_cls, (mu_base, _) in self.class_distributions.items()
        }

        # 选 top_q 最相似的 base 类
        top_base = sorted(distances.items(), key=lambda x: x[1])[:self.top_q]
        C_q = [cls_id for cls_id, _ in top_base]

        # 计算融合权重
        weights = np.array([
            self.class_sizes[c] * distances[c] for c in C_q
        ])
        weights = weights / weights.sum()

        # 融合均值 & 协方差
        mu_calibrated = self.gamma * sum(
            w * self.class_distributions[c][0] for w, c in zip(weights, C_q)
        ) + (1 - self.gamma) * mu_k

        sigma_calibrated = self.gamma * sum(
            w * self.class_distributions[c][1] for w, c in zip(weights, C_q)
        ) + (1 - self.gamma) * sigma_k + self.alpha * np.ones_like(sigma_k)

        # 保存新类分布
        self.class_distributions[target_class_id] = (mu_calibrated, sigma_calibrated)
        self.class_sizes[target_class_id] = len(target_features)

        return mu_calibrated, sigma_calibrated

    def sample_from_class(self, class_id: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        从记录的某个类的分布中采样特征向量
        """
        assert class_id in self.class_distributions, f"Class {class_id} not recorded"
        mu, sigma = self.class_distributions[class_id]
        mvn = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
        samples = mvn.rvs(size=n_samples)
        if samples.ndim == 1:
            samples = samples[None, :]
        labels = np.full((samples.shape[0],), class_id)
        return samples, labels
