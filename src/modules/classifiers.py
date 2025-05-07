import torch
import math
import numpy as np
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



def NN(x_support, y_support, x_query):
    """ Classify queries with a nearest-neighbour head """
    device = torch.device("cuda:0" if x_query.is_cuda else "cpu")
    
    query_size = x_query.size()[0]
    # N = query_labels.unique().size(0)
    N = y_support.unique().size(0) # Number of ways

    # Initialization of per-class support sets
    per_class_supports = {}
    per_class_cnt = {}
    for label, num_samples in zip(*torch.unique(y_support, return_counts=True)):
        key = label.item()
        per_class_cnt[key] = 0
        per_class_supports[key] = torch.zeros(
            (num_samples, ) + x_support.size()[1:]).to(device)
        
    # Check if keys are sorted
    assert list(per_class_supports.keys()) == list(range(len(per_class_supports)))
    # assert len(per_class_supports) == N

    # Populating per-class support sets
    for sample, label in zip(x_support, y_support):
        key = label.item()
        per_class_supports[key][per_class_cnt[key]] = sample
        per_class_cnt[key] += 1
    del per_class_cnt
    
    # Initialization of minimum distances tensor of size num. query samples x num classes
    min_distances = torch.zeros((x_query.size()[0], N))

    # Retrieving query embeddings
    #_, q_embeddings = net(query, return_features=True)

    # Foreach suport class
    for key in per_class_supports:

        # Computing min distances for the current class
        distances = torch.cdist(x_query, per_class_supports[key])
        min_distances[:, key], _ = torch.min(distances, dim=1)

    # Computing soft values
    soft_values = torch.softmax(-min_distances, dim=1)
    
    # Assert if all soft values vectors sum to 1
    assert torch.eq(torch.round(soft_values.sum(dim=1), decimals=math.ceil(np.log10(N))),
                    torch.ones(query_size)).all()

    y_pred = torch.argmax(soft_values, dim=1)
    
    # Assert if the right number of predictions has been computed
    assert len(y_pred) == query_size
    
    return soft_values, y_pred


def LR(x_support, y_support, x_query):
    """ Classify queries with a logistic regressor head """
    
    classifier = LogisticRegression(
        penalty="l2",
        random_state=0,
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
    )
    
    x_support = nn.functional.normalize(
        x_support, p=2, dim=1).detach().cpu().numpy()
    y_support = y_support.detach().cpu().numpy()
    
    classifier.fit(x_support, y_support)
    
    x_query = nn.functional.normalize(
        x_query, p=2, dim=1).detach().cpu().numpy()
    
    y_pred = classifier.predict(x_query)
    # Returns the predicted labels
    return y_pred
