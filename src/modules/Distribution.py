import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple

class RepresentationDistributionCalibrator:
    def __init__(self, top_q=3, gamma=0.7, alpha=1e-2):
        """
        Args:
            top_q: Number of base classes used for new class distribution calibration
            gamma: Proportion for fusing base distributions
            alpha: Covariance perturbation term to avoid singularity
        """
        self.top_q = top_q
        self.gamma = gamma
        self.alpha = alpha
        self.class_distributions = {}  # {class_id: (mean, cov)}
        self.class_sizes = {}  # {class_id: num_samples}

    def update_class_distribution(self, class_id: int, features: np.ndarray):
        """
        Update or record the representation distribution of a class (features should be numpy.ndarray with shape [N, D])
        """
        assert len(features.shape) == 2
        mu = np.mean(features, axis=0)
        cov = np.cov(features.T)
        self.class_distributions[class_id] = (mu, cov)
        self.class_sizes[class_id] = len(features)

    def calibrate(self, target_class_id: int, target_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use existing base class distribution information to calibrate the feature distribution of a new class (target_class_id), and sample supplementary data.
        Returns: (sampled_features, sampled_labels)
        """
        mu_k = target_features.mean(0)
        sigma_k = np.cov(target_features.T)

        # Calculate Euclidean distance to base classes
        distances = {
            base_cls: np.linalg.norm(mu_k - mu_base)
            for base_cls, (mu_base, _) in self.class_distributions.items()
        }

        # Select top_q most similar base classes
        top_base = sorted(distances.items(), key=lambda x: x[1])[:self.top_q]
        C_q = [cls_id for cls_id, _ in top_base]

        # Calculate fusion weights
        weights = np.array([
            self.class_sizes[c] * distances[c] for c in C_q
        ])
        weights = weights / weights.sum()

        # Fuse mean & covariance
        mu_calibrated = self.gamma * sum(
            w * self.class_distributions[c][0] for w, c in zip(weights, C_q)
        ) + (1 - self.gamma) * mu_k

        sigma_calibrated = self.gamma * sum(
            w * self.class_distributions[c][1] for w, c in zip(weights, C_q)
        ) + (1 - self.gamma) * sigma_k + self.alpha * np.ones_like(sigma_k)

        # Save new class distribution
        self.class_distributions[target_class_id] = (mu_calibrated, sigma_calibrated)
        self.class_sizes[target_class_id] = len(target_features)

        return mu_calibrated, sigma_calibrated

    def sample_from_class(self, class_id: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample feature vectors from the recorded distribution of a class
        """
        assert class_id in self.class_distributions, f"Class {class_id} not recorded"
        mu, sigma = self.class_distributions[class_id]
        mvn = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
        samples = mvn.rvs(size=n_samples)
        if samples.ndim == 1:
            samples = samples[None, :]
        labels = np.full((samples.shape[0],), class_id)
        return samples, labels
