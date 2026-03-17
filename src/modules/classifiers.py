import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



def NN(x_support, y_support, x_query, sample_weight=None, support_soft_labels=None):
    """ Classify queries with a nearest-neighbour head.
    When sample_weight is provided, use weighted prototype per class instead of min-distance.
    """
    device = torch.device("cuda:0" if x_query.is_cuda else "cpu")

    query_size = x_query.size()[0]
    if support_soft_labels is not None:
        support_soft_labels = support_soft_labels.to(device=device, dtype=x_support.dtype)
        distances = torch.cdist(x_query, x_support)
        support_weights = torch.softmax(-distances, dim=1)
        soft_values = support_weights @ support_soft_labels
        y_pred = torch.argmax(soft_values, dim=1)
        return soft_values, y_pred

    N = int(y_support.max().item()) + 1

    # Initialization of per-class support sets
    per_class_supports = {}
    per_class_cnt = {}
    per_class_weights = {} if sample_weight is not None else None
    for label, num_samples in zip(*torch.unique(y_support, return_counts=True)):
        key = label.item()
        per_class_cnt[key] = 0
        per_class_supports[key] = torch.zeros(
            (num_samples, ) + x_support.size()[1:]).to(device)
        if sample_weight is not None:
            per_class_weights[key] = []
        
    # Populating per-class support sets (and weights)
    for idx, (sample, label) in enumerate(zip(x_support, y_support)):
        key = label.item()
        per_class_supports[key][per_class_cnt[key]] = sample
        if sample_weight is not None:
            per_class_weights[key].append(sample_weight[idx].item() if sample_weight[idx].dim() == 0 else sample_weight[idx].cpu().item())
        per_class_cnt[key] += 1
    del per_class_cnt
    
    min_distances = torch.full((x_query.size()[0], N), float("inf"), device=device)

    if sample_weight is not None:
        # Weighted prototype: p_c = sum(w_i * x_i) / sum(w_i)
        for key in per_class_supports:
            X = per_class_supports[key]
            w = torch.tensor(per_class_weights[key], dtype=X.dtype, device=device)
            w = w / (w.sum() + 1e-12)
            proto = (X * w.view(-1, 1)).sum(dim=0)
            distances = torch.cdist(x_query, proto.unsqueeze(0))
            min_distances[:, key] = distances.squeeze(1)
    else:
        # Original: min distance to each class
        for key in per_class_supports:
            distances = torch.cdist(x_query, per_class_supports[key])
            min_distances[:, key], _ = torch.min(distances, dim=1)

    soft_values = torch.softmax(-min_distances, dim=1)
    
    # Assert if all soft values vectors sum to 1
    assert torch.eq(
        torch.round(soft_values.sum(dim=1), decimals=max(1, math.ceil(np.log10(max(N, 2))))),
        torch.ones(query_size, device=device),
    ).all()

    y_pred = torch.argmax(soft_values, dim=1)
    
    # Assert if the right number of predictions has been computed
    assert len(y_pred) == query_size
    
    return soft_values, y_pred


def LR(x_support, y_support, x_query, sample_weight=None, support_soft_labels=None):
    """ Classify queries with a logistic regressor head.
    When sample_weight is provided, use weighted fitting (noise-aware).
    """
    device = x_query.device
    x_support = F.normalize(x_support, p=2, dim=1)
    x_query = F.normalize(x_query, p=2, dim=1)

    if support_soft_labels is not None:
        support_soft_labels = support_soft_labels.to(device=device, dtype=x_support.dtype)
        classifier = nn.Linear(x_support.size(1), support_soft_labels.size(1)).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05, weight_decay=1e-4)

        with torch.enable_grad():
            classifier.train()
            for _ in range(200):
                optimizer.zero_grad(set_to_none=True)
                logits = classifier(x_support)
                loss = -(support_soft_labels * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss.backward()
                optimizer.step()

        classifier.eval()
        with torch.no_grad():
            logits = classifier(x_query)
            soft_values = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(soft_values, dim=1)
        return soft_values, y_pred

    classifier = LogisticRegression(
        penalty="l2",
        random_state=0,
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
    )
    x_support_np = x_support.detach().cpu().numpy()
    y_support_np = y_support.detach().cpu().numpy()

    if sample_weight is not None:
        sw = sample_weight.detach().cpu().numpy()
        classifier.fit(x_support_np, y_support_np, sample_weight=sw)
    else:
        classifier.fit(x_support_np, y_support_np)

    x_query_np = x_query.detach().cpu().numpy()

    # Get softmax probabilities
    y_probs = classifier.predict_proba(x_query_np)  # shape = [n_query, n_present_classes]
    num_classes = int(np.max(y_support_np)) + 1
    full_probs = np.zeros((y_probs.shape[0], num_classes), dtype=np.float32)
    full_probs[:, classifier.classes_.astype(np.int64)] = y_probs
    y_pred = np.argmax(full_probs, axis=1)

    # Convert to torch
    soft_values = torch.tensor(full_probs, dtype=torch.float32, device=device)
    y_pred = torch.tensor(y_pred, dtype=torch.long, device=device)

    # Returns the predicted labels
    return soft_values, y_pred


def NN_soft(x_support, support_soft_labels, x_query):
    """Classify queries with soft-label prototypes aggregated from support targets."""
    device = torch.device("cuda:0" if x_query.is_cuda else "cpu")
    query_size = x_query.size(0)
    num_classes = support_soft_labels.size(1)

    x_support = x_support.to(device)
    x_query = x_query.to(device)
    support_soft_labels = support_soft_labels.to(device=device, dtype=x_support.dtype)

    class_mass = support_soft_labels.sum(dim=0, keepdim=True).t().clamp_min(1e-12)
    prototypes = torch.matmul(support_soft_labels.t(), x_support) / class_mass
    distances = torch.cdist(x_query, prototypes)
    soft_values = torch.softmax(-distances, dim=1)

    assert torch.eq(
        torch.round(soft_values.sum(dim=1), decimals=math.ceil(np.log10(num_classes))),
        torch.ones(query_size, device=device),
    ).all()

    y_pred = torch.argmax(soft_values, dim=1)
    assert len(y_pred) == query_size

    return soft_values, y_pred


def NN_proto(x_query, prototypes):
    """Classify by distance to given prototypes. prototypes: (way, dim)."""
    device = x_query.device
    prototypes = prototypes.to(device)
    distances = torch.cdist(x_query, prototypes)
    soft_values = torch.softmax(-distances, dim=1)
    y_pred = torch.argmax(soft_values, dim=1)
    return soft_values, y_pred


def LR_weighted(x_support, y_support, x_query, sample_weight, lr=0.05, max_steps=200, weight_decay=1e-4):
    """可微分的 LR，sample_weight 参与 loss，支持 backprop。"""
    device = x_query.device
    x_support = F.normalize(x_support, p=2, dim=1)
    x_query = F.normalize(x_query, p=2, dim=1)
    sample_weight = sample_weight.to(device=device, dtype=x_support.dtype)
    num_classes = int(y_support.max().item()) + 1

    classifier = nn.Linear(x_support.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    with torch.enable_grad():
        classifier.train()
        for _ in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x_support)
            ce_per_sample = F.cross_entropy(logits, y_support, reduction='none')
            loss = (sample_weight * ce_per_sample).sum() / (sample_weight.sum() + 1e-12)
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        logits = classifier(x_query)
        soft_values = F.softmax(logits, dim=1)
        y_pred = torch.argmax(logits, dim=1)
    return soft_values, y_pred


def LR_soft(x_support, support_soft_labels, x_query, lr=0.1, max_steps=100, weight_decay=1e-4):
    """Fit a linear soft-label head on support embeddings and classify queries."""
    device = x_support.device
    x_support = F.normalize(x_support, p=2, dim=1)
    x_query = F.normalize(x_query, p=2, dim=1)
    support_soft_labels = support_soft_labels.to(device=device, dtype=x_support.dtype)

    classifier = nn.Linear(x_support.size(1), support_soft_labels.size(1)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    with torch.enable_grad():
        classifier.train()
        for _ in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x_support)
            loss = -(support_soft_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        logits = classifier(x_query)
        soft_values = F.softmax(logits, dim=1)
        y_pred = torch.argmax(soft_values, dim=1)

    return soft_values, y_pred
