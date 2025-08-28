# denoising.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, List

import numpy as np
import torch

@dataclass
class DenoiseConfig:
    strategy: Literal["none", "LOF", "IF", "proto_margin"] = "proto_margin"
    noise_ratio: float = 0.2              # 噪声占比
    lof_k: int = 10                       # LOF邻居数
    if_random_state: int = 42             # IF随机种子
    proto_iters: int = 2                  # 原型迭代次数（鲁棒原型）
    metric: Literal["cosine", "euclidean"] = "cosine"

class Denoiser:
    def __init__(self, cfg: DenoiseConfig):
        self.cfg = cfg

    @torch.no_grad()
    def __call__(
        self,
        s_embeddings: torch.Tensor,          # (N, D)
        support_labels: torch.Tensor,        # (N,)
        num_old_classes: int,
        num_new_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回: (s_embeddings_filtered, support_labels_filtered, mask_bool)
        仅对[新类]去噪；老类样本全部保留。
        """
        device = s_embeddings.device
        labels_np = support_labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)

        old_class_ids = set(range(num_old_classes))
        new_class_ids = set(range(num_old_classes, num_old_classes + num_new_classes))

        mask = torch.zeros_like(support_labels, dtype=torch.bool, device=device)

        # 预先为原型法准备每类原型（避免循环中重复算）
        proto_map: Dict[int, torch.Tensor] = {}
        if self.cfg.strategy == "proto_margin":
            # 使用“观测类”的全部样本先做初原型（含噪，后续鲁棒迭代会修）
            for cls in unique_labels:
                idx = (labels_np == cls)
                X = s_embeddings[idx]
                proto_map[int(cls)] = self._robust_proto(
                    X, keep_ratio=max(0.0, 1.0 - self.cfg.noise_ratio),
                    iters=self.cfg.proto_iters, metric=self.cfg.metric
                )

        for cls in unique_labels:
            idx = (labels_np == cls)
            idx_tensor = torch.from_numpy(idx).to(device)

            # 老类：不过滤
            if int(cls) in old_class_ids:
                mask[idx_tensor] = True
                continue

            # 新类：按策略去噪
            X = s_embeddings[idx_tensor]

            if self.cfg.strategy == "none":
                keep_mask_local = torch.ones(X.size(0), dtype=torch.bool, device=device)

            elif self.cfg.strategy in ["LOF", "IF"]:
                keep_mask_local = self._denoise_outlier_sklearn(X, strategy=self.cfg.strategy)

            elif self.cfg.strategy == "proto_margin":
                # 目标类原型
                proto_self = proto_map[int(cls)]

                # 其他类原型（仅新类，其它观测类）
                other_protos = [
                    proto_map[int(c)] for c in unique_labels
                    if (int(c) != int(cls)) and (int(c) in new_class_ids)
                ]
                other_protos = torch.stack(other_protos, dim=0) if len(other_protos) > 0 else None

                keep_mask_local = self._denoise_proto_margin(
                    X, proto_self=proto_self, other_protos=other_protos
                )
            else:
                raise ValueError(f"Unknown strategy: {self.cfg.strategy}")

            # 写回全局mask
            idx_indices = torch.from_numpy(np.where(idx)[0]).to(device)
            mask[idx_indices[keep_mask_local]] = True

        return s_embeddings[mask], support_labels[mask], mask

    # ----------------- sklearn分支：LOF / IF -----------------
    def _denoise_outlier_sklearn(self, X: torch.Tensor, strategy: str) -> torch.Tensor:
        """
        按 contamination=self.cfg.noise_ratio 去掉异常点; 返回本类内的 keep_mask(bool)
        """
        X_np = X.detach().cpu().numpy()

        n = len(X_np)
        n_out = int(self.cfg.noise_ratio * n)
        if n_out == 0 and n > 1:
            n_out = 1

        if strategy == "LOF":
            from sklearn.neighbors import LocalOutlierFactor
            model = LocalOutlierFactor(n_neighbors=self.cfg.lof_k, contamination=self.cfg.noise_ratio)
            # 注意：fit_predict 不返回分数；我们取 negative_outlier_factor_（越小越异常）
            model.fit(X_np)
            scores = model.negative_outlier_factor_  # 小 → 异常
            # 取最小的 n_out 做异常
            out_idx = np.argsort(scores)[:n_out]

        elif strategy == "IF":
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=self.cfg.noise_ratio, random_state=self.cfg.if_random_state)
            model.fit(X_np)
            scores = model.decision_function(X_np)  # 大 → 干净
            # 取最小的 n_out 做异常（因为小=更异常）
            out_idx = np.argsort(scores)[:n_out]
        else:
            raise ValueError(strategy)

        keep_local = np.ones(n, dtype=bool)
        keep_local[out_idx] = False
        return torch.from_numpy(keep_local).to(X.device)

    # ----------------- 原型 + margin 分支 -----------------
    def _denoise_proto_margin(
        self,
        X: torch.Tensor,                      # (m, D) 当前类的样本（含噪）
        proto_self: torch.Tensor,             # (D,)
        other_protos: Optional[torch.Tensor]  # (C-1, D) or None
    ) -> torch.Tensor:
        """
        分数= sim(x, proto_self) - max_j sim(x, proto_other_j)；选 top-(1-noise_ratio) 作为干净
        """
        m = X.size(0)
        n_keep = int(round((1.0 - self.cfg.noise_ratio) * m))
        n_keep = min(max(n_keep, 1), m)  # 至少留1个

        # 归一化（cosine）或保留原值（euclidean）
        if self.cfg.metric == "cosine":
            Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
            ps = proto_self / (proto_self.norm() + 1e-12)
            if other_protos is not None:
                po = other_protos / (other_protos.norm(dim=1, keepdim=True) + 1e-12)
            # 正相似
            s_pos = (Xn @ ps)
            # 负相似（最近他类）
            if other_protos is None or other_protos.numel() == 0:
                scores = s_pos
            else:
                s_neg = (Xn @ po.T).max(dim=1).values
                scores = s_pos - s_neg
        else:
            # 欧氏：用负距离 margin
            d_pos = ((X - proto_self) ** 2).sum(dim=1)
            if other_protos is None or other_protos.numel() == 0:
                scores = -d_pos
            else:
                d_neg = torch.cdist(X, other_protos).min(dim=1).values
                scores = -(d_pos - d_neg)

        keep_idx = torch.topk(scores, k=n_keep, largest=True).indices
        keep_local = torch.zeros(m, dtype=torch.bool, device=X.device)
        keep_local[keep_idx] = True
        return keep_local

    def _robust_proto(
        self,
        X: torch.Tensor, keep_ratio: float = 0.8, iters: int = 2, metric: str = "cosine"
    ) -> torch.Tensor:
        """
        迭代式鲁棒原型：每次保留与原型最近的 top-k，再均值；cosine 默认在球面上做。
        """
        if X.size(0) == 0:
            raise ValueError("Empty class samples for prototype.")
        k = max(1, int(round(keep_ratio * X.size(0))))

        if metric == "cosine":
            Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
            proto = Xn.mean(dim=0)
            proto = proto / (proto.norm() + 1e-12)
            for _ in range(max(iters, 0)):
                dist = 1.0 - (Xn @ proto)          # 小=近
                keep = torch.topk(-dist, k).indices
                proto = Xn[keep].mean(dim=0)
                proto = proto / (proto.norm() + 1e-12)
            return proto
        else:
            proto = X.mean(dim=0)
            for _ in range(max(iters, 0)):
                dist = ((X - proto) ** 2).sum(dim=1)
                keep = torch.topk(-dist, k).indices
                proto = X[keep].mean(dim=0)
            return proto
