from glob import glob
from functools import partial
from sklearn import metrics
import numpy as np
import torch
import re


def accuracy(y_pred, y_true):
    acc = (y_pred == y_true.long()).sum().float()
    return (acc / y_pred.size(0)).numpy()


def f1_score(y_pred, y_true, **kwargs):
    return metrics.f1_score(y_true, y_pred, **kwargs)
    

def silhouette_score(features, labels):
    return metrics.silhouette_score(features, labels, metric='euclidean')


def davies_bouldin_score(features, labels):
    return metrics.davies_bouldin_score(features, labels)


def calinski_harabasz_score(features, labels):
    return metrics.calinski_harabasz_score(features, labels)


logit_based_metrics = {
    'acc': accuracy,
    'f1_all_macro': partial(f1_score, average='macro', labels=None),
    'f1_all_micro': partial(f1_score, average='micro', labels=None),
    'f1_new_macro': partial(f1_score, average='macro'),
    'f1_new_micro': partial(f1_score, average='micro'),
    'f1_old_macro': partial(f1_score, average='macro'),
    'f1_old_micro': partial(f1_score, average='micro'),
}

cluster_based_metrics = {
    'sc' : silhouette_score,
    'db' : davies_bouldin_score,
    'ch' : calinski_harabasz_score
}

def _compute_logit_based_metrics(preds, labels, wanted_metrics, class_pool):
    value = dict() 
    for metric in logit_based_metrics:
        if metric in wanted_metrics:
            metric_func = logit_based_metrics[metric]
            
            if 'new' in metric:
                value[metric] = metric_func(preds, labels, labels=class_pool['new'])
            elif 'old' in metric:
                value[metric] = metric_func(preds, labels, labels=class_pool['old'])
            else:
                value[metric] = metric_func(preds, labels)
    return value


def _compute_cluster_based_metrics(feature, label, wanted_metrics):
    value = dict()
    for metric in cluster_based_metrics:
        if metric in wanted_metrics: 
            value[metric] = cluster_based_metrics[metric](feature, label)
    return value


def compute_confusion_matrix(path, files):  
    """
    This function computes the confusion matrix for each episode.
    The confusion matrix is normalized by the number of episodes.

    Parameters:
        - path (``str``): The path to the directory containing the files.
        - files (``list``): A list of file names to read.
    """
    cms = []
    cms_raw = []  # 存储未归一化的混淆矩阵
    npz_data = _read_npz_files(path, files)
    labels = npz_data['query_labels']
    logits = npz_data['logits']
            
    # Add an extra dimension, used for the pre-training phase of TL appr.
    if logits.dim() == 2:
        logits = torch.unsqueeze(logits, 0)
    if labels.dim() == 1:
        labels = torch.unsqueeze(labels, 0)
        
    for logit, label in zip(logits, labels):
        pred = logit.argmax(dim=1).long()
        # 计算归一化的混淆矩阵
        cms.append(metrics.confusion_matrix(label, pred, normalize='true'))
        # 计算未归一化的混淆矩阵
        cms_raw.append(metrics.confusion_matrix(label, pred))
    
    # 返回归一化和未归一化的混淆矩阵
    return sum(cms) / len(cms), sum(cms_raw) / len(cms_raw)

        
def compute_logit_based_metrics(path, files, wanted_metrics, class_pool, is_averaged=False):
    """
    This function computes the specified metrics for each pair of logits and labels in the specified
    files. The results are returned as a dictionary where the keys are the metric names.

    Parameters:
        - path (``str``): The path to the directory containing the files.
        - files (``list``): A list of files to compute the metrics on.
        - wanted_metrics (``list``): A list of metrics to compute.
        - is_averaged (``bool``): If True, returns the average and standard deviation of the metrics. 
          Defaults to False.
          
    Returns:
        - dict: If `is_averaged` is False, returns a dictionary with the metric as the key and a list 
          of scores for each episode as values. If `is_averaged` is True, returns a dictionary with 
          the metric as the key and a tuple with mean and standard deviation as values.
    """
    metrics = dict()
    npz_data = _read_npz_files(path, files)
    labels = npz_data['query_labels']
    logits = npz_data['logits']
        
    # Add an extra dimension, used for the pre-training phase of TL appr.
    if logits.dim() == 2:
        logits = torch.unsqueeze(logits, 0)
    if labels.dim() == 1:
        labels = torch.unsqueeze(labels, 0)
    
    for logit, label in zip(logits, labels):
        pred = logit.argmax(dim=1).long()
        
        # Computer the metric for each episode, append the computed values in metrics
        metric = _compute_logit_based_metrics(pred, label, wanted_metrics, class_pool)
        for k, v in metric.items():
            metrics.setdefault(k, []).append(v)
    if not is_averaged:
        # Metric as key and a list of scores for each episode as values
        return metrics   
    else:
        # Metric as key and a tuple with mean and std dev as values  
        return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    
    
def compute_cluster_based_metrics(path, files, wanted_metrics, is_averaged=False):
    """
    Computes specified metrics to quantify the quality of a cluster for each file in the provided path.
    The results are returned as a dictionary where the keys are the metric names.

    Parameters:
        - path (``str``): The path to the directory containing the files.
        - files (``list``): A list of files to compute the metrics on.
        - wanted_metrics (``list``): A list of metrics to compute.
        - is_averaged (``bool``): If True, returns the average and standard deviation of the metrics. 
          Defaults to False.

    Returns:
        - dict: If `is_averaged` is False, returns a dictionary with the metric as the key and a 
          list of scores for each episode as values. If `is_averaged` is True, returns a dictionary
          with the metric as the key and a tuple with mean and standard deviation as values.
    """
    npz_data = _read_npz_files(path, files)
    queries = npz_data['queries']
    supports = npz_data['supports']
    query_labels = npz_data['query_labels']
    support_labels = npz_data['support_labels']
    metrics = dict()
    
    if len(support_labels.shape) == 3:
        support_labels = support_labels.squeeze(axis=2)
    
    for query, support, query_label, support_label in zip(queries, supports, query_labels, support_labels):
        # Computer the metric for each episode, append the computed values in metrics
        q_metric = _compute_cluster_based_metrics(query, query_label, wanted_metrics)
        s_metric = _compute_cluster_based_metrics(support, support_label, wanted_metrics)
        for metric in [q_metric, s_metric]:
            for k, v in metric.items():
                metrics.setdefault(k, []).append(v)
    if not is_averaged:
        # Metric as key and a list of scores for each episode as values
        return metrics   
    else:
        # Metric as key and a tuple with mean and std dev as values  
        return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


def _read_npz_files(path, files):
    """
    This function reads .npz files from a specified path and returns a dictionary where the keys are 
    the file names and the values are the data read from the files. If multiple files are found with 
    the same name, an Exception is raised.

    Parameters:
        - path (``str``): The path to the directory containing the files.
        - files (``list``): A list of file names to read.
    """
    npz_data = dict()

    for file in files:
        data_path = glob(f'{path}/{file}.*')
        if len(data_path) != 1:
            raise Exception(f'Multiple or no files found in {data_path}, must be one')
        
        if 'ep' in file:
            # If the file is in the format label_epINT remove _epINT
            file = re.sub(r'_ep\d+', '', file)

        data = np.load(data_path[0])
        for key in data.files:
            value = data[key]
            npz_data[key] = torch.tensor(value)
    
    return npz_data 