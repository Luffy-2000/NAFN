# denoising.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class _LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

class _CosineHead(nn.Module):
    """weight-normalized cosine classifier: logits = s * cos(x, w_c)"""
    def __init__(self, in_dim: int, out_dim: int, scale: float = 16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=1)
        Wn = F.normalize(self.W, dim=1)
        logits = self.scale * (x @ Wn.t())   # (N, C)
        return logits


@dataclass
class DenoiseConfig:
    strategy: Literal["none", "LOF", "IF", "proto_margin", "DCML"] = "proto_margin"
    noise_ratio: float = 0.2              # Noise ratio
    lof_k: int = 10                       # LOF neighbor count
    if_random_state: int = 42             # IF random seed
    proto_iters: int = 2                  # Prototype iteration count (robust prototype)
    metric: Literal["cosine", "euclidean"] = "cosine"

    # ===== DCML相关 =====
    dcml_threshold: Optional[float] = None   # 若为 None，按 noise_ratio 分位自适应
    dcml_head_type: Literal["linear", "cosine"] = "linear"
    dcml_head_epochs: int = 5
    dcml_head_lr: float = 1e-2
    dcml_head_weight_decay: float = 0.0
    dcml_head_batch_size: int = 256          # 小样本可全量；放个上限以兼容较大类
    dcml_head_temp: float = 1.0              # 对 logits 的温度缩放
    dcml_min_keep: int = 1                   # 每类至少保留的样本数
    dcml_use_class_balance: bool = True      # 交叉熵 class weight（按样本数反比）

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
        Returns: (s_embeddings_filtered, support_labels_filtered, mask_bool)
        Only denoise [new classes]; old class samples are all preserved.
        """
        device = s_embeddings.device
        labels_np = support_labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)

        old_class_ids = set(range(num_old_classes))
        new_class_ids = set(range(num_old_classes, num_old_classes + num_new_classes))

        mask = torch.zeros_like(support_labels, dtype=torch.bool, device=device)

        # ===== 若使用 DCML：先基于全部“可见类”（旧+新）训练一个临时分类头 =====
        head_fn = None
        class_id_list: List[int] = []
        if self.cfg.strategy == "DCML":
            class_id_list = [int(c) for c in sorted(unique_labels)]
            head_fn = self._train_ephemeral_head(
                X_all=s_embeddings, y_all=support_labels,
                class_ids_all=class_id_list
            )

        # ===== 若使用 proto_margin：预计算原型 =====
        # Pre-compute prototypes for each class for prototype method (avoid repeated computation in loop)
        proto_map: Dict[int, torch.Tensor] = {}
        if self.cfg.strategy == "proto_margin":
            # Use all samples of "observed classes" to make initial prototypes (noisy, will be refined by robust iteration later)
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

            # Old classes: no filtering
            if int(cls) in old_class_ids:
                mask[idx_tensor] = True
                continue

            # New classes: denoise according to strategy
            X = s_embeddings[idx_tensor]
            y_local = support_labels[idx_tensor]

            if self.cfg.strategy == "none":
                keep_mask_local = torch.ones(X.size(0), dtype=torch.bool, device=device)

            elif self.cfg.strategy in ["LOF", "IF"]:
                keep_mask_local = self._denoise_outlier_sklearn(X, strategy=self.cfg.strategy)

            elif self.cfg.strategy == "proto_margin":
                # Target class prototype
                proto_self = proto_map[int(cls)]

                # Other class prototypes (only new classes, other observed classes)
                other_protos = [
                    proto_map[int(c)] for c in unique_labels
                    if (int(c) != int(cls)) and (int(c) in new_class_ids)
                ]
                other_protos = torch.stack(other_protos, dim=0) if len(other_protos) > 0 else None

                keep_mask_local = self._denoise_proto_margin(
                    X, proto_self=proto_self, other_protos=other_protos
                )
            elif self.cfg.strategy == "DCML":
                # 用临时分类头得到 logits → CE → 阈值/分位数过滤
                keep_mask_local = self._denoise_dcml_head(
                    X=X, y=y_local, head_fn=head_fn, class_ids_all=class_id_list
                )
            else:
                raise ValueError(f"Unknown strategy: {self.cfg.strategy}")

            # Write back to global mask
            idx_indices = torch.from_numpy(np.where(idx)[0]).to(device)
            mask[idx_indices[keep_mask_local]] = True

        return s_embeddings[mask], support_labels[mask], mask

    # ----------------- sklearn branch: LOF / IF -----------------
    def _denoise_outlier_sklearn(self, X: torch.Tensor, strategy: str) -> torch.Tensor:
        """
        Remove outliers according to contamination=self.cfg.noise_ratio; returns keep_mask(bool) within this class
        """
        X_np = X.detach().cpu().numpy()

        n = len(X_np)
        n_out = int(self.cfg.noise_ratio * n)
        if n_out == 0 and n > 1:
            n_out = 1

        if strategy == "LOF":
            from sklearn.neighbors import LocalOutlierFactor
            model = LocalOutlierFactor(n_neighbors=self.cfg.lof_k, contamination=self.cfg.noise_ratio)
            # Note: fit_predict doesn't return scores; we use negative_outlier_factor_ (smaller = more anomalous)
            model.fit(X_np)
            scores = model.negative_outlier_factor_  # Small → anomalous
            # Take the smallest n_out as outliers
            out_idx = np.argsort(scores)[:n_out]

        elif strategy == "IF":
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=self.cfg.noise_ratio, random_state=self.cfg.if_random_state)
            model.fit(X_np)
            scores = model.decision_function(X_np)  # Large → clean
            # Take the smallest n_out as outliers (because small = more anomalous)
            out_idx = np.argsort(scores)[:n_out]
        else:
            raise ValueError(strategy)

        keep_local = np.ones(n, dtype=bool)
        keep_local[out_idx] = False
        return torch.from_numpy(keep_local).to(X.device)

    # ----------------- Prototype + margin branch -----------------
    def _denoise_proto_margin(
        self,
        X: torch.Tensor,                      # (m, D) Current class samples (with noise)
        proto_self: torch.Tensor,             # (D,)
        other_protos: Optional[torch.Tensor]  # (C-1, D) or None
    ) -> torch.Tensor:
        """
        Score = sim(x, proto_self) - max_j sim(x, proto_other_j); select top-(1-noise_ratio) as clean
        """
        m = X.size(0)
        n_keep = int(round((1.0 - self.cfg.noise_ratio) * m))
        n_keep = min(max(n_keep, 1), m)  # At least keep 1

        # Normalize (cosine) or keep original values (euclidean)
        if self.cfg.metric == "cosine":
            Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
            ps = proto_self / (proto_self.norm() + 1e-12)
            if other_protos is not None:
                po = other_protos / (other_protos.norm(dim=1, keepdim=True) + 1e-12)
            # Positive similarity
            s_pos = (Xn @ ps)
            # Negative similarity (closest other class)
            if other_protos is None or other_protos.numel() == 0:
                scores = s_pos
            else:
                s_neg = (Xn @ po.T).max(dim=1).values
                scores = s_pos - s_neg
        else:
            # Euclidean: use negative distance margin
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
        Iterative robust prototype: each time keep top-k closest to prototype, then average; cosine defaults to sphere.
        """
        if X.size(0) == 0:
            raise ValueError("Empty class samples for prototype.")
        k = max(1, int(round(keep_ratio * X.size(0))))

        if metric == "cosine":
            Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
            proto = Xn.mean(dim=0)
            proto = proto / (proto.norm() + 1e-12)
            for _ in range(max(iters, 0)):
                dist = 1.0 - (Xn @ proto)          # Small = close
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

    # ----------------- DCML-HEAD branch -----------------
    def _denoise_dcml_head(
        self,
        X: torch.Tensor,                  # (m, D) 当前类样本
        y: torch.Tensor,                  # (m,)
        head_fn,                          # 可调用：X -> logits(N,C)
        class_ids_all: List[int],         # 头部训练时使用的全部类（用于 label->列索引）
    ) -> torch.Tensor:
        device = X.device
        m = X.size(0)
        if m == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        if m <= 2:
            return torch.ones(m, dtype=torch.bool, device=device)

        # label → 列索引
        cls_to_col = {int(c): i for i, c in enumerate(class_ids_all)}
        y_idx = torch.tensor([cls_to_col[int(t.item())] for t in y], device=device, dtype=torch.long)

        logits = head_fn(X)          # (m, C)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(logits, y_idx)  # (m,)

        if (self.cfg.dcml_threshold is not None) and (self.cfg.dcml_threshold > 0):
            keep_local = losses < self.cfg.dcml_threshold
        else:
            rho = float(self.cfg.noise_ratio)
            n_keep = max(self.cfg.dcml_min_keep, int(round((1.0 - rho) * m)))
            n_keep = min(max(n_keep, 1), m)
            keep_idx = torch.topk(-losses, k=n_keep, largest=True).indices  # 等价于按 loss 升序取前 n_keep
            keep_local = torch.zeros(m, dtype=torch.bool, device=device)
            keep_local[keep_idx] = True

        if not keep_local.any():
            keep_local[torch.argmin(losses)] = True

        return keep_local

    def _train_ephemeral_head(
        self,
        X_all: torch.Tensor,              # (N, D) 全部支持集嵌入（旧+新）
        y_all: torch.Tensor,              # (N,)
        class_ids_all: List[int],         # 本 episode 中观测到的所有类（保持稳定次序）
    ):
        """
        在当前 episode 内训练一个临时分类头（只训几轮），返回一个 head_fn(X)->logits 的可调用对象。
        """
        device = X_all.device
        N, D = X_all.size(0), X_all.size(1)
        # 将标签映射到 [0, C-1] 的列索引
        cls_to_col = {int(c): i for i, c in enumerate(class_ids_all)}
        y_idx = torch.tensor([cls_to_col[int(t.item())] for t in y_all], device=device, dtype=torch.long)
        C = len(class_ids_all)

        # 头部
        if self.cfg.dcml_head_type == "cosine":
            head = _CosineHead(in_dim=D, out_dim=C, scale=max(self.cfg.dcml_head_temp, 1.0)).to(device)
        else:
            head = _LinearHead(in_dim=D, out_dim=C).to(device)

        # class weight（可选）——按样本数反比，缓解类不平衡
        if self.cfg.dcml_use_class_balance:
            counts = torch.bincount(y_idx, minlength=C).float()
            inv = 1.0 / (counts + 1e-12)
            class_weight = (inv / inv.sum() * C).to(device)
        else:
            class_weight = None

        criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="mean")
        optim = torch.optim.SGD(
            head.parameters(),
            lr=self.cfg.dcml_head_lr,
            momentum=0.9,
            weight_decay=self.cfg.dcml_head_weight_decay
        )

        # 简单的小批量迭代（few-shot 可直接全量）
        bs = min(self.cfg.dcml_head_batch_size, N)

        # IMPORTANT: 训练需要开启梯度
        with torch.enable_grad():
            head.train()
            for _ in range(max(self.cfg.dcml_head_epochs, 1)):
                # 这里为了稳妥，做个打乱
                perm = torch.randperm(N, device=device)
                X_shuf = X_all[perm]
                y_shuf = y_idx[perm]
                for start in range(0, N, bs):
                    end = min(start + bs, N)
                    xb = X_shuf[start:end]
                    yb = y_shuf[start:end]
                    optim.zero_grad(set_to_none=True)
                    logits = head(xb)
                    # 线性头可选温度缩放
                    if self.cfg.dcml_head_type == "linear" and self.cfg.dcml_head_temp != 1.0:
                        logits = logits / max(self.cfg.dcml_head_temp, 1e-6)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optim.step()

        head.eval()

        # 返回一个“推理函数”，供后续各类局部过滤使用（no grad）
        @torch.no_grad()
        def head_fn(X: torch.Tensor) -> torch.Tensor:
            logits = head(X)
            if self.cfg.dcml_head_type == "linear" and self.cfg.dcml_head_temp != 1.0:
                logits = logits / max(self.cfg.dcml_head_temp, 1e-6)
            return logits

        return head_fn