import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

def NN(x_support, y_support, x_query):
    """Classify queries with nearest-neighbour (min-distance to support per class)."""
    device = torch.device("cuda:0" if x_query.is_cuda else "cpu")

    query_size = x_query.size()[0]
    N = int(y_support.max().item()) + 1

    per_class_supports = {}
    per_class_cnt = {}
    for label, num_samples in zip(*torch.unique(y_support, return_counts=True)):
        key = label.item()
        per_class_cnt[key] = 0
        per_class_supports[key] = torch.zeros(
            (num_samples, ) + x_support.size()[1:]).to(device)
        
    for idx, (sample, label) in enumerate(zip(x_support, y_support)):
        key = label.item()
        per_class_supports[key][per_class_cnt[key]] = sample
        per_class_cnt[key] += 1
    del per_class_cnt
    
    min_distances = torch.full((x_query.size()[0], N), float("inf"), device=device)
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


def LR(x_support, y_support, x_query):
    """Classify queries with sklearn LogisticRegression (hard labels)."""
    device = x_query.device
    x_support = F.normalize(x_support, p=2, dim=1)
    x_query = F.normalize(x_query, p=2, dim=1)

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
    """Differentiable LR; sample_weight participates in loss, supports backprop."""
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
