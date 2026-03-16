import os
import torch
import learn2learn as l2l
from torch import nn
from learn2learn.utils import accuracy
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import copy
from approach.tl_module import LightningTLModule
from modules.losses import DistillKLLoss
from modules.teachers import Teacher
from modules.classifiers import NN, LR, NN_soft, LR_soft
from util.traffic_transformations import permutation, pkt_translating, wrap
from util.denoising import Denoiser, DenoiseConfig
from util.noise_utils import get_mus_hat, S_to_T, backward_corrected_label
from sklearn.neighbors import NearestNeighbors
EPSILON = 1e-8


def _mixup_supplement(features: np.ndarray, n_synthetic: int, lam_range=(0.2, 0.8), rng=None) -> np.ndarray:
    """Generate synthetic samples by Mixup: convex combination of two same-class samples."""
    N, D = features.shape
    if N < 2 or n_synthetic <= 0:
        return np.zeros((0, D))
    rng = rng or np.random.default_rng()
    lam_lo, lam_hi = lam_range
    syn = []
    for _ in range(n_synthetic):
        i, j = rng.choice(N, 2, replace=False)
        lam = rng.uniform(lam_lo, lam_hi)
        syn.append(lam * features[i] + (1.0 - lam) * features[j])
    return np.array(syn, dtype=features.dtype)


def _smote_supplement(features: np.ndarray, n_synthetic: int, k: int = 5, rng=None) -> np.ndarray:
    """Generate synthetic samples by SMOTE: interpolate between a sample and a random k-NN neighbor."""
    N, D = features.shape
    if N <= 1 or n_synthetic <= 0:
        return np.zeros((0, D))
    k = min(k, N - 1)
    rng = rng or np.random.default_rng()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(features)
    _, indices = nn.kneighbors(features)  # (N, k+1), first column is self
    syn = []
    for _ in range(n_synthetic):
        i = rng.integers(0, N)
        j_local = rng.integers(1, k + 1)
        j = indices[i, j_local]
        t = rng.uniform(0.0, 1.0)
        syn.append(features[i] + t * (features[j] - features[i]))
    return np.array(syn, dtype=features.dtype)


class LightningRFS(LightningTLModule):
    """
    [[Link to Source Code]](https://github.com/RL-VIG/LibFewShot)
    LightningRFS is a class that implements the RFS (Rethinking Few-Shot) algorithm for few-shot learning,
    as described in "Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?".

    Args:
        - net (``nn.Module``): The base neural network architecture.
        - loss (``nn.Module``, optional): The loss function for the algorithm. 
         Defaults to ``nn.CrossEntropyLoss(reduction="mean")``.
        - kd_T (float, optional): Temperature for knowledge distillation loss. Defaults to 1.
        - alpha (float, optional): Weight for CE loss. Defaults to 0.5.
        - gamma (float, optional): Weight for KL loss. Defaults to 0.5.
        - is_distill (bool, optional): Whether to use knowledge distillation. Defaults to False.
        - teacher_path (str, optional): Path to the teacher model. Defaults to None.
        - base_learner (str, optional): Type of base learner ('lr' or 'nn'). Defaults to 'nn'.
        - ``**kwargs`` (dict): Additional keyword arguments.

    Attributes:
        - kd_T (float): Temperature for knowledge distillation loss.
        - ce_loss (``nn.Module``): The cross-entropy loss function.
        - kl_loss (``DistillKLLoss``): The knowledge distillation KL loss function.
        - nl_loss (``nn.Module``): The negative log-likelihood loss function.
        - net (``nn.Module``): The base neural network architecture.
        - base_learner (str): Type of base learner ('lr' or 'nn').
        - is_distill (bool): Whether to use knowledge distillation.
        - teacher_path (str): Path to the teacher model.
        - alpha (float): Weight for CE loss.
        - gamma (float): Weight for KL loss.
        - teacher (``src.modules.teacher.Teacher``): Teacher model instance.
    """
    
    alpha = 0.5
    gamma = 0.5
    is_distill = False
    kd_T = 1
    teacher_path = None
    base_learner = 'nn'

    def __init__(self, net, loss=None, **kwargs):
        super().__init__(**kwargs)
        
        # Transfer-Learning specific parameters 
        self.kd_T = kwargs.get('kd_t', LightningRFS.kd_T)
        self.ce_loss = loss or nn.CrossEntropyLoss(reduction="mean")
        self.kl_loss = DistillKLLoss(T=self.kd_T)
        self.nl_loss = nn.NLLLoss()
        self.net = net
        self.base_learner = kwargs.get('base_learner', LightningRFS.base_learner)
        self.is_distill = kwargs.get('is_distill', LightningRFS.is_distill)
        self.teacher_path = kwargs.get('teacher_path', LightningRFS.teacher_path)
        self.alpha = kwargs.get('alpha', LightningRFS.alpha) if self.is_distill else 0
        self.gamma = kwargs.get('gamma', LightningRFS.gamma) if self.is_distill else 1

        # Mode: 'recon' or 'contrastive' or 'hybrid'
        self.mode = kwargs.get('pre_mode', 'none')
        # self.ways = kwargs.get('ways', 5)
        # Parameter of contrasctive learning
        self.temperature = kwargs.get('temperature', 0.5)
        self.transform_strength = kwargs.get('transform_strength', 0.8)
        self.mes_loss = nn.MSELoss()
        # self.ntx_loss = NTXentLoss(temperature=self.temperature)
        # Loss weights
        self.recon_weight = kwargs.get('recon_weight', 0.2)
        self.ce_weight = kwargs.get('ce_weight', 0.8)
        self.contrastive_weight = kwargs.get('contrastive_weight', 0.2)
        self.noise_ratio = kwargs.get('noise_ratio', 0.0)
        self.ro = kwargs.get('ro', 0.2)

        assert (
            self.alpha + self.gamma == 1.0
        ), 'alpha + gamma should be equal to 1'

        self.denoising = kwargs.get('denoising', 'none')
        self.supplement_method = kwargs.get('supplement_method', 'sample')  # none | sample | mixup | smote
        self.classes_per_set = kwargs.get('classes_per_set', [10, 5])
        self.num_old_classes = self.classes_per_set[0]
        self.num_new_classes = self.classes_per_set[1]

        self.teacher = Teacher(
            net=self.net, 
            is_distill=self.is_distill, 
            teacher_path=self.teacher_path
        )
        
        self.save_hyperparameters({
            "shots": self.shots,
            "queries": self.queries,
            "kd_T": self.kd_T,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "is_distill": self.is_distill,
            "teacher_path": self.teacher_path,
            "base_learner": self.base_learner,
            "pre_mode": self.mode,
            "denoising": self.denoising,
            "supplement_method": self.supplement_method,
            "noise_ratio": self.noise_ratio,
            "ro": self.ro
        })

        # ========== MOCO related ==========
        self.moco_m = 0.999
        self.moco_K = 4096
        self.moco_dim = 320
        # Momentum encoder
        self.moco_encoder = copy.deepcopy(self.net)
        for param in self.moco_encoder.parameters():
            param.requires_grad = False
        # Queue and pointer
        self.register_buffer('moco_queue', torch.randn(self.moco_dim, self.moco_K))
        self.moco_queue = F.normalize(self.moco_queue, dim=0)
        self.register_buffer('moco_queue_ptr', torch.zeros(1, dtype=torch.long))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningTLModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--alpha",
            type=float,
            default=LightningRFS.alpha
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=LightningRFS.gamma
        )
        parser.add_argument(
            "--is-distill", 
            action='store_true', 
            default=LightningRFS.is_distill
        )
        parser.add_argument(
            "--kd-t", 
            type=float, 
            default=LightningRFS.kd_T
        )
        parser.add_argument(
            "--teacher-path", 
            type=str, 
            default=LightningRFS.teacher_path
        )
        parser.add_argument(
            "--base-learner", 
            nargs='?', 
            choices=['lr', 'nn'],
            default=LightningRFS.base_learner
        )
        return parser
    
    def on_adaptation_start(self):
        self.net.freeze_backbone()
        
    def training_epoch_end(self, _):
        # Save the network state dict since it can be used to inizialize the Teacher
        saved_weights_path = f'{self.logger.log_dir}/distill_models'
        os.makedirs(saved_weights_path, exist_ok=True)
        torch.save(self.net.state_dict(), 
                   f'{saved_weights_path}/teacher_ep{self.trainer.current_epoch}.pt')
    

    def _apply_transform(self, x, transform_method=None):
        """Apply the same random transformation to all samples in the batch"""

        device = x.device
        x_np = x.detach().cpu().numpy()
        
       # If transform_method is not specified, randomly choose one
        if transform_method is None:
            transform_method = np.random.choice([0, 1, 2])
        elif transform_method not in [0, 1, 2]:
            raise ValueError("transform_method must be 0, 1, or 2")

        # Apply the same transformation to all samples
        if transform_method == 0:
            transformed = np.array([permutation(sample, a=self.transform_strength) for sample in x_np])
        elif transform_method == 1:
            transformed = np.array([pkt_translating(sample, a=self.transform_strength) for sample in x_np])
        else:
            transformed = np.array([wrap(sample, a=self.transform_strength) for sample in x_np])
        # Convert back to PyTorch tensor
        return torch.from_numpy(transformed).to(device)

    def _moco_momentum_update_encoder(self):
        """Momentum update for encoder parameters"""
        for param_q, param_k in zip(self.net.parameters(), self.moco_encoder.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)

    @torch.no_grad()
    def _moco_dequeue_and_enqueue(self, keys):
        """keys: [B, C], enqueue and dequeue for the queue"""
        keys = keys.detach()
        batch_size = keys.shape[0]
        ptr = int(self.moco_queue_ptr)
        K = self.moco_K
        # Replace data in the queue
        if ptr + batch_size <= K:
            self.moco_queue[:, ptr:ptr + batch_size] = keys.T
        else:
            first = K - ptr
            self.moco_queue[:, ptr:] = keys[:first].T
            self.moco_queue[:, :batch_size-first] = keys[first:].T
        ptr = (ptr + batch_size) % K
        self.moco_queue_ptr[0] = ptr

    def pt_step(self, batch, batch_idx, **kwargs):
        data, labels = batch
        if self.is_distill:
            student_logits = self.net(data)
            teacher_logits = self.teacher(data)    # None
            # CE on actual label and student logits
            gamma_loss = self.ce_loss(student_logits, labels) 
            # KL on teacher and student logits
            alpha_loss = self.kl_loss(student_logits, teacher_logits)
            eval_accuracy = accuracy(student_logits, labels)
            loss = gamma_loss * self.gamma + alpha_loss * self.alpha  # tensor(5.3006, device='cuda:0')
        else:
            if self.mode == 'none':
                student_logits = self.net(data)
                teacher_logits = self.teacher(data)    # None
                # CE on actual label and student logits
                gamma_loss = self.ce_loss(student_logits, labels) 
                # KL on teacher and student logits
                alpha_loss = self.kl_loss(student_logits, teacher_logits)
                eval_accuracy = accuracy(student_logits, labels)
                loss = gamma_loss * self.gamma + alpha_loss * self.alpha  # tensor(5.3006, device='cuda:0')
            elif self.mode == 'recon':
                student_logits, _, recon_x = self.net(data, return_features=True)
                recon_loss = self.mes_loss(recon_x, data)
                ce_loss = self.ce_loss(student_logits, labels)
                eval_accuracy = accuracy(student_logits, labels)
                loss = self.recon_weight * recon_loss + self.ce_weight * ce_loss
            elif self.mode == 'contrastive':
                # MOCO contrastive learning
                print('data.shape', data.shape)
                data_aug = self._apply_transform(data, transform_method=0)
                student_logits, q, _ = self.net(data, return_features=True)
                with torch.no_grad():
                    self._moco_momentum_update_encoder()
                    _, k, _ = self.moco_encoder(data_aug, return_features=True)
                q = F.normalize(q, dim=1)
                k = F.normalize(k, dim=1)         # [B, 320]
                # Dynamically adapt MOCO queue feature dimension
                if q.shape[1] != self.moco_queue.shape[0]:
                    device = q.device
                    self.moco_dim = q.shape[1]
                    self.moco_queue = torch.randn(self.moco_dim, self.moco_K, device=device)
                    self.moco_queue = F.normalize(self.moco_queue, dim=0)
                    self.moco_queue_ptr = torch.zeros(1, dtype=torch.long, device=device)
                # Positive samples
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B,1]
                # Negative samples
                queue = self.moco_queue.clone().detach()
                l_neg = torch.einsum('nc,ck->nk', [q, queue])           # [B,K]
                logits = torch.cat([l_pos, l_neg], dim=1)               # [B,1+K]
                logits /= self.temperature
                labels_con = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
                contrastive_loss = F.cross_entropy(logits, labels_con)
                # Enqueue and dequeue
                self._moco_dequeue_and_enqueue(k)
                ce_loss = self.ce_loss(student_logits, labels)
                eval_accuracy = accuracy(student_logits, labels)
                loss = self.ce_weight * ce_loss + self.contrastive_weight * contrastive_loss

            elif self.mode == 'hybrid':
                self.ce_weight, self.recon_weight, self.contrastive_weight = 0.7, 0.15, 0.15
                # Generate positive samples
                data_aug = self._apply_transform(data, transform_method=0)
                
                # Get the output of the original samples
                student_logits, q, recon_x = self.net(data, return_features=True)
                
                # MOCO contrastive learning
                with torch.no_grad():
                    self._moco_momentum_update_encoder()
                    _, k, _ = self.moco_encoder(data_aug, return_features=True)
                q = F.normalize(q, dim=1)
                k = F.normalize(k, dim=1)         # [B, 320]
                # Dynamically adapt MOCO queue feature dimension
                if q.shape[1] != self.moco_queue.shape[0]:
                    device = q.device
                    self.moco_dim = q.shape[1]
                    self.moco_queue = torch.randn(self.moco_dim, self.moco_K, device=device)
                    self.moco_queue = F.normalize(self.moco_queue, dim=0)
                    self.moco_queue_ptr = torch.zeros(1, dtype=torch.long, device=device)
                # Positive samples
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B,1]
                # Negative samples
                queue = self.moco_queue.clone().detach()
                l_neg = torch.einsum('nc,ck->nk', [q, queue])           # [B,K]
                logits = torch.cat([l_pos, l_neg], dim=1)               # [B,1+K]
                logits /= self.temperature
                labels_con = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
                contrastive_loss = F.cross_entropy(logits, labels_con)
                # Enqueue and dequeue
                self._moco_dequeue_and_enqueue(k)
                
                # Calculate the classification loss
                ce_loss = self.ce_loss(student_logits, labels)
                # Calculate the reconstruction loss
                recon_loss = self.mes_loss(recon_x, data)
                eval_accuracy = accuracy(student_logits, labels)
                loss = self.ce_weight * ce_loss + self.recon_weight * recon_loss + self.contrastive_weight * contrastive_loss
            else:
                raise ValueError(f"Mode '{self.mode}' does not exist! Available modes are: 'none', 'recon', 'contrastive', 'hybrid'")

        return {
            'loss': loss,
            'accuracy': eval_accuracy,
            'labels': labels,
            'logits': student_logits
        }
    

    @torch.no_grad()
    def _build_transition_soft_labels(self, s_embeddings, support_labels, way, diag_hat: float = 0.5):
        """
        Build soft labels via paper's transition matrix (Berthon et al., 2021).
        For each new-class sample: r=confidence from leave-one-out -> mus_hat -> P -> backward P(true|obs).
        diag_hat: 1-ro, noise-scale hyperparameter; higher noise -> smaller diag -> less trust in non-observed labels.
        """
        support_soft_labels = F.one_hot(support_labels, num_classes=way).float().to(s_embeddings.device)
        class_counts = torch.bincount(support_labels, minlength=way)
        new_class_range = range(self.num_old_classes, self.num_old_classes + self.num_new_classes)

        for idx, label in enumerate(support_labels.tolist()):
            if label not in new_class_range or class_counts[label].item() <= 1:
                continue

            keep_mask = torch.ones(s_embeddings.size(0), dtype=torch.bool, device=s_embeddings.device)
            keep_mask[idx] = False

            if self.base_learner == 'lr':
                soft_values, _ = LR(
                    x_support=s_embeddings[keep_mask],
                    y_support=support_labels[keep_mask],
                    x_query=s_embeddings[idx].unsqueeze(0),
                )
            elif self.base_learner == 'nn':
                soft_values, _ = NN(
                    x_support=s_embeddings[keep_mask],
                    y_support=support_labels[keep_mask],
                    x_query=s_embeddings[idx].unsqueeze(0),
                )
            else:
                raise ValueError('Bad base learner')

            probs = soft_values.squeeze(0).to(s_embeddings.device)
            r = probs[label].item()
            r = max(1e-6, min(1.0 - 1e-6, r))

            y_vec = F.one_hot(torch.tensor([label], device=s_embeddings.device), num_classes=way).float().squeeze(0)
            diag_val = max(1e-6, min(1.0 - 1e-6, float(diag_hat)))
            mus_hat = get_mus_hat(r, y_vec, diag_val)
            P = S_to_T(mus_hat, way)
            soft_label = backward_corrected_label(
                P, label,
                restrict_to_new=True,
                num_old_classes=self.num_old_classes,
                num_new_classes=self.num_new_classes,
            )
            support_soft_labels[idx] = soft_label

        return support_soft_labels

    @torch.no_grad()
    def ft_step(self, dist_calibrator, batch):
        # Grad disabled since the backbone is frozen and a fully-connected head is not used
        self.net.eval()
        self.dist_calibrator = dist_calibrator
        data, labels = batch
        labels, le = self.label_encoding(labels)
        # print('labels', labels)

        way = labels.unique().size(0)
        
        # Embed the input
        _, feats, _ = self.net(data, return_features=True)
        # print('feats.shape', feats.shape)
        
        s_embeddings, support_labels = feats[:way * self.shots], labels[:way * self.shots]
        q_embeddings, query_labels = feats[way * self.shots:], labels[way * self.shots:]
        # print('support_labels', support_labels)
        # print('query_labels', query_labels)
        
        support_soft_labels = None
        sample_weights = None
        if self.denoising == 'CSIDN':
            # CSIDN: Confidence Scores for Instance-dependent Noise; keeps all support, softens new-class targets.
            support_soft_labels = self._build_transition_soft_labels(
                s_embeddings=s_embeddings,
                support_labels=support_labels,
                way=way,
                diag_hat=1.0 - self.ro,
            )
        else:
            cfg = DenoiseConfig(
                strategy=self.denoising,
                noise_ratio=self.ro,
                lof_k=10,
                if_random_state=42,
                proto_iters=2,
                metric="cosine",
            )
            denoiser = Denoiser(cfg)
            s_embeddings, support_labels, mask, sample_weights = denoiser(
                s_embeddings=s_embeddings,
                support_labels=support_labels,
                num_old_classes=self.num_old_classes,
                num_new_classes=self.num_new_classes,
            )

        # 消融：去噪后可选 不补充 / 分布采样 / Mixup / SMOTE（CSIDN 不参与补充）
        if self.denoising not in ('none', 'CSIDN') and self.supplement_method != 'none':
            new_class_ids = set(range(self.num_old_classes, self.num_old_classes + self.num_new_classes))
            new_support_features = []
            new_support_labels = []
            n_supplement = max(1, int(self.ro * self.shots))
            for cls in new_class_ids:
                idx = (support_labels.cpu().numpy() == cls)
                if np.sum(idx) == 0:
                    continue
                features = s_embeddings[idx].detach().cpu().numpy()
                if self.supplement_method == 'sample':
                    # 原逻辑：记录分布 → 校准 → 从校准分布采样
                    self.dist_calibrator.update_class_distribution(int(cls), features)
                    mu, sigma = self.dist_calibrator.calibrate(int(cls), features)
                    samples, labels = self.dist_calibrator.sample_from_class(int(cls), n_supplement)
                elif self.supplement_method == 'mixup':
                    samples = _mixup_supplement(features, n_supplement)
                    labels = np.full((samples.shape[0],), cls, dtype=np.int64)
                elif self.supplement_method == 'smote':
                    samples = _smote_supplement(features, n_supplement, k=5)
                    labels = np.full((samples.shape[0],), cls, dtype=np.int64)
                else:
                    continue
                if samples.shape[0] > 0:
                    new_support_features.append(samples)
                    new_support_labels.append(labels)
            if new_support_features:
                new_support_features = np.concatenate(new_support_features, axis=0)
                new_support_labels = np.concatenate(new_support_labels, axis=0)
                s_embeddings = torch.cat([s_embeddings, torch.tensor(new_support_features, device=s_embeddings.device, dtype=s_embeddings.dtype)], dim=0)
                support_labels = torch.cat([support_labels, torch.tensor(new_support_labels, device=support_labels.device, dtype=support_labels.dtype)], dim=0)

        if self.denoising == 'CSIDN' and self.base_learner == 'lr':
            soft_values, y_pred = LR_soft(
                x_support=s_embeddings,
                support_soft_labels=support_soft_labels,
                x_query=q_embeddings,
            )
            soft_values = soft_values.clamp(EPSILON, 1 - EPSILON).to(self._device)
            eval_loss = self.nl_loss(soft_values.log(), query_labels)
        elif self.denoising == 'CSIDN' and self.base_learner == 'nn':
            soft_values, y_pred = NN_soft(
                x_support=s_embeddings,
                support_soft_labels=support_soft_labels,
                x_query=q_embeddings,
            )
            soft_values = soft_values.clamp(EPSILON, 1 - EPSILON).to(self._device)
            eval_loss = self.nl_loss(soft_values.log(), query_labels)
        elif self.base_learner == 'lr':
            soft_values, y_pred = LR(
                x_support=s_embeddings,
                y_support=support_labels,
                x_query=q_embeddings,
                sample_weight=sample_weights,
            )
            soft_values = soft_values.clamp(EPSILON, 1 - EPSILON).to(self._device)
            eval_loss = self.nl_loss(soft_values.log(), query_labels)
        elif self.base_learner == 'nn':
            soft_values, y_pred = NN(
                x_support=s_embeddings,
                y_support=support_labels,
                x_query=q_embeddings,
                sample_weight=sample_weights,
            )
            soft_values = soft_values.clamp(EPSILON, 1 - EPSILON).to(self._device)
            eval_loss = self.nl_loss(soft_values.log(), query_labels)
        
        else:
            ValueError('Bad base learner')

        # if self.base_learner == 'lr': 
        #     # Since 'lr' and 'h' do not produce soft values, 
        #     # one-hot encoding of predictions is performed.
        #     soft_values = torch.nn.functional.one_hot(y_pred, num_classes=way)  
        #     assert torch.equal(y_pred, torch.argmax(soft_values, dim=1))
        #     eval_loss = torch.zeros(1)
        
        y_pred = y_pred.to(self._device)
        query_accuracy = (y_pred == query_labels).sum().float() / y_pred.size(0)  
        
        return {
            'loss': eval_loss,
            'accuracy': query_accuracy,
            'query_labels': query_labels,
            'support_labels': support_labels,
            'preds': y_pred,
            'logits': soft_values,
            'le_map': le,
            'support': s_embeddings,
            'query': q_embeddings
        }