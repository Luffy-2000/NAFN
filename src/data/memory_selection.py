import torch
import torch.nn.functional as F
import numpy as np
import time
from copy import deepcopy
from typing import Iterable, Optional, Union, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math
from networks.network import LLL_Net
from scipy.ndimage import gaussian_filter1d

class ExemplarsSelector:
    """Exemplar selector for approaches with an interface of Dataset"""

    def __init__(self, exemplars_dataset, max_num_exemplars=100, max_num_exemplars_per_class=10):
        self.exemplars_dataset = exemplars_dataset
        self.already_added_classes = ()
        self.max_num_exemplars = max_num_exemplars
        self.max_num_exemplars_per_class = max_num_exemplars_per_class

    def __call__(self, model: LLL_Net, trn_loader: DataLoader, transform, t=None,
                 from_inputs=False, clean_memory=False, **kwargs):
        model = deepcopy(model)
        # Management of the pre-allocated-output: the non used heads info are removed
        if t is not None:
            model.heads = model.heads[:t + 1]
            model.task_cls = model.task_cls[:t + 1]
            model.task_offset = model.task_offset[:t + 1]
        clock0 = time.time()
        exemplars_per_class = self._get_exemplars_per_class(model)
        sel_loader = DataLoader(
            trn_loader.dataset,
            batch_size=trn_loader.batch_size,
            shuffle=False,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory
        )
        selected_indices = self._select_indices(model, sel_loader, exemplars_per_class, from_inputs, clean_memory)
        x, y = zip(*(trn_loader.dataset[idx] for idx in selected_indices))
        clock1 = time.time()
        print('| Selected {:d} train exemplars, time={:5.1f}s'.format(len(x), clock1 - clock0))
        return torch.tensor(np.array(x)), torch.stack(y)

    def _get_exemplars_per_class(self, model: LLL_Net) -> int:
        if self.max_num_exemplars_per_class:
            return self.max_num_exemplars_per_class
        num_classes = model.task_cls.sum().item()
        return max(1, self.max_num_exemplars // num_classes)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform,
                        from_inputs=None, clean_memory=False) -> Iterable:
        pass


class HerdingExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """
    def __init__(self, exemplars_dataset, max_num_exemplars=1000, max_num_exemplars_per_class=10):
        self.exemplars_dataset = exemplars_dataset
        self.max_num_exemplars = max_num_exemplars
        self.max_num_exemplars_per_class = max_num_exemplars_per_class


    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int,
                        from_inputs=False, clean_memory=False) -> Iterable[int]:
        model_device = next(model.parameters()).device
        features, targets, clean_mask = [], [], []

        model.eval()
        with torch.no_grad():
            for images, labels in sel_loader:
                images = images.to(model_device)
                labels = labels.to(model_device)
                if from_inputs:
                    feats = images
                else:
                    logits, feats, _ = model(images, return_features=True)
                    if clean_memory:
                        preds = logits.argmax(dim=1)
                        clean_mask.extend((preds == labels).cpu().tolist())
                    feats = F.normalize(feats, dim=1)
                features.append(feats.cpu())
                targets.append(labels.cpu())

        features = torch.cat(features)
        targets = torch.cat(targets)

        if not clean_mask:
            clean_mask = torch.ones(len(targets), dtype=torch.bool)
        else:
            clean_mask = torch.tensor(clean_mask, dtype=torch.bool)

        selected_indices = []
        for cls in torch.unique(targets):
            cls_mask = (targets == cls) & clean_mask
            cls_indices = torch.where(cls_mask)[0]

            # assert len(cls_indices) > 0, f"No samples for class {cls}"
            if len(cls_indices) == 0:
                print(f"[Fallback] No clean samples for class {cls}. Using top-k fallback.")
                cls_indices = torch.where(targets == cls)[0]

                # with torch.no_grad():
                #     raw_logits = model.head(raw_feats)
                #     probs = torch.softmax(raw_logits, dim=1)
                #     topk_scores, topk_idxs = probs[:, cls].topk(min(exemplars_per_class, len(raw_cls_indices)))
                #     cls_indices = raw_cls_indices[topk_idxs.cpu()]
            if len(cls_indices) < exemplars_per_class:
                deficit = exemplars_per_class - len(cls_indices)  # ★修改
                raw_cls_indices = torch.where(targets == cls)[0]   # ★修改
                extra = raw_cls_indices[torch.randint(len(raw_cls_indices), (deficit,))]  # ★修改
                cls_indices = torch.cat([cls_indices, extra])      # ★修改


            cls_feats = features[cls_indices].to(model_device)

            if len(cls_indices) > exemplars_per_class:
                sel_local = self._herding(cls_feats, exemplars_per_class)
                chosen = cls_indices[sel_local]
            else:
                chosen = cls_indices  # ★修改：直接用补齐后的索引
        
            selected_indices.extend(chosen.tolist())

        return selected_indices

    def _herding(self, cls_feats: torch.Tensor, k: int) -> torch.Tensor:
        cls_feats = F.normalize(cls_feats, dim=1)
        cls_mu = F.normalize(cls_feats.mean(dim=0), dim=0)
        selected = []
        w_t = torch.zeros_like(cls_mu)

        for _ in range(k):
            scores = torch.matmul(cls_feats, cls_mu - w_t)
            scores[selected] = -1e9  # Avoid duplicate selection
            new_idx = scores.argmax().item()
            w_t += cls_feats[new_idx]
            selected.append(new_idx)

        return torch.tensor(selected)


def augment_data(images: np.ndarray, jitter=True, shift=True, smooth=True,
                 jitter_std=0.05, shift_range=5, time_axis: int = -1) -> Tuple[np.ndarray, int]:
    """
    Apply perturbations to network traffic or time series feature data:
    - jitter: add Gaussian noise
    - shift: time series shift (circular)
    - smooth: Gaussian filter, simulating compression/noise masking
    Returns augmented samples and the augmentation multiplier
    """
    augmented = []
    for x in images:
        augmented.append(x) 

        if jitter:
            noise = np.random.normal(0, jitter_std, size=x.shape)
            augmented.append(x + noise)

        if shift:
            s = np.random.randint(1, shift_range + 1)
            augmented.append(np.roll(x, s, axis=time_axis))

        if smooth:
            if x.ndim == 1:
                smoothed = gaussian_filter1d(x, sigma=1)
            else:
                smoothed = np.stack([gaussian_filter1d(xi, sigma=1) for xi in x])
            augmented.append(smoothed)

    multiplier = len(augmented) // len(images)
    return np.stack(augmented), multiplier


class UncertaintyExemplarsSelector(ExemplarsSelector):
    """
    Rainbow Memory style uncertainty exemplar strategy:
    - Apply perturbations (jitter/shift/smooth) to each class sample and evaluate prediction consistency
    - After obtaining uncertainty scores, select the top k most uncertain samples
    """

    def __init__(self, exemplars_dataset, max_num_exemplars=1000, max_num_exemplars_per_class=10):
        self.exemplars_dataset = exemplars_dataset
        self.max_num_exemplars = max_num_exemplars
        self.max_num_exemplars_per_class = max_num_exemplars_per_class


    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform,
                        from_inputs=None, clean_memory=False, alternate=False, step_selection=False) -> Iterable:
        model_device = next(model.parameters()).device
        model.eval()

        extracted_logits = []
        extracted_targets = []
        extracted_images = np.array(sel_loader.dataset.images)
        clean_index = []

        with torch.no_grad():
            for images, targets in sel_loader:
                images = images.to(model_device)
                targets = targets.to(model_device)
                logits = model(images)
                if clean_memory:
                    preds = torch.argmax(logits, dim=1)
                    clean_index.extend((preds == targets).detach().cpu().tolist())
                extracted_logits.append(logits.cpu())
                extracted_targets.extend(targets.cpu().tolist())

        extracted_logits = torch.cat(extracted_logits, dim=0).cpu().numpy()
        extracted_targets = np.array(extracted_targets)

        if not clean_index:
            clean_index = np.ones(len(extracted_targets), dtype=bool)
        else:
            clean_index = np.array(clean_index, dtype=bool)

        result = []

        for curr_cls in np.unique(extracted_targets):
            cls_mask = (extracted_targets == curr_cls) & clean_index
            cls_indices = np.where(cls_mask)[0]

            if len(cls_indices) == 0:
                print(f"[Fallback] No clean samples for class {curr_cls}, using all available.")
                cls_indices = np.where(extracted_targets == curr_cls)[0]

            if exemplars_per_class < len(cls_indices):
                cls_imgs = extracted_images[cls_indices]
                base_labels = [curr_cls] * len(cls_imgs)

                # Apply augmentation operations (jitter/shift/smooth) to construct perturbed samples
                augmented_imgs, multiplier = augment_data(cls_imgs)
                N = len(cls_imgs)
                T = multiplier - 1  # Number of perturbations

                uncertainty_scores = self._estimate_uncertainty(
                    model, augmented_imgs, base_labels, repeat=T, batch_size=N, device=model_device
                )

                uncertainty_scores = np.array(uncertainty_scores)
                sorted_idx = np.argsort(uncertainty_scores)[::-1]  # From most uncertain to most certain

                if step_selection:
                    step = math.ceil(len(sorted_idx) / exemplars_per_class)
                    selected = [cls_indices[sorted_idx[i]] for i in range(0, len(sorted_idx), step)]
                    while len(selected) < exemplars_per_class:
                        selected.append(cls_indices[sorted_idx[len(selected)]])
                    result.extend(selected[:exemplars_per_class])
                else:
                    selected = cls_indices[sorted_idx[:exemplars_per_class]]
                    result.extend(selected.tolist())
            else:
                print(f"[Warning] Class {curr_cls} only has {len(cls_indices)} samples. Selecting all.")
                result.extend(cls_indices.tolist())

        return result

    def _estimate_uncertainty(self, model, augmented_imgs, labels, repeat=3, batch_size=64, device='cpu'):
        """
        Given perturbed samples, calculate the prediction consistency of each original sample as confidence:
        - The lower the consistency, the higher the uncertainty
        Returns: uncertainty_scores: List[float], higher means more uncertain
        """
        model.eval()
        N = len(labels)
        consistency = np.zeros(N)

        aug_imgs = torch.tensor(augmented_imgs).float().to(device)
        labels = np.array(labels)

        with torch.no_grad():
            for r in range(repeat):
                start = r * N
                end = (r + 1) * N
                r_batch = aug_imgs[start:end]
                
                loader = torch.utils.data.DataLoader(r_batch, batch_size=32, shuffle=False)
                pred_labels = []

                for batch in loader:
                    logits = model(batch)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    pred_labels.extend(pred)
                    
                consistency += (np.array(pred_labels) == labels).astype(np.int32)
        uncertainty_scores = [1 - (c / repeat) for c in consistency]
        return uncertainty_scores
