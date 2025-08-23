from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import json
import os
import re
import torch
from sklearn.metrics import roc_auc_score

import util.metrics 
import util.logger


default_metrics = [ 'acc', 'sc', 'db', 'f1_all_macro']
fscil_metric = ['f1_new_macro', 'f1_old_macro']
exp_args = [
    'shots', 'queries', 'pre_mode','network', 'fields', 'num_pkts', 'seed', 'memory_selector', 'dataset', 'base_learner', 'noise_label', 'noise_ratio'
]

def compute_train_time(path):
    time_per_epoch = []

    with open(f'{path}early_stopping.log') as log:
        for i, line in enumerate(log):
            date = " ".join(line.split(' ')[:2])
            date_format = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S,%f")
            unix_time = datetime.datetime.timestamp(date_format)
            time_per_epoch.append(unix_time)
                
    return time_per_epoch[-1] - time_per_epoch[0]


def load_label_map(label_conv_path):
    """Load label mapping, return mapping from label names to encodings"""
    with open(label_conv_path, 'r') as f:
        label_map = eval(f.read())
    return label_map


def extract_pre_mode_from_path(path):
    """
    Extract pre_mode from folder path name
    
    Parameters:
        - path (str): folder path
    
    Returns:
        - str: pre_mode value ('none', 'recon', 'contrastive', 'hybrid') or 'unknown'
    """
    # Split path and find the folder that contains pre_mode
    path_parts = path.split('/')
    
    # Define possible pre_mode values
    pre_modes = ['none', 'recon', 'contrastive', 'hybrid']
    
    # Check each part of the path for pre_mode
    for part in path_parts:
        for mode in pre_modes:
            if mode in part:
                return mode
    
    # If not found, return 'unknown'
    return 'unknown'


def compute_pauc_sklearn(y_true, y_scores, max_fpr=0.01):
    """
    Standardized pAUC via sklearn (random≈0.5, perfect=1.0).
    Positive class must have higher scores.
    """
    return roc_auc_score(y_true, y_scores, max_fpr=max_fpr)


def make_scores_from_logits(logits, benign_idx, method="logmeanexp_margin"):
    """
    logits: (N, C) 原始多类logits（来自你模型/NN head）
    benign_idx: 良性类在本地(episode)的列索引
    method:
      - 'logmeanexp_margin'  # 默认：log(恶意平均e^z) - z_benign  (对“恶意类多”不敏感)
      - 'logsumexp_margin'   # logsumexp(恶意) - z_benign       (更“乐观”，略偏向恶意)
      - 'max_margin'         # max(恶意) - z_benign             (看最像的一个恶意类)
      - 'mal_sum_prob'       # sum softmax(恶意) = 1 - p_benign  (等价二元化的边际概率)
      - 'mal_mean_prob'      # mean softmax(恶意)                (对类数更中性)
      - 'mal_max_prob'       # max softmax(恶意)
    """
    z = torch.as_tensor(logits, dtype=torch.float32)         # (N, C)
    N, C = z.shape
    device = z.device
    all_idx = torch.arange(C, device=device)
    mal_mask = all_idx != benign_idx
    z_b = z[:, benign_idx]
    z_m = z[:, mal_mask]                                     # (N, C-1)

    if method == "logmeanexp_margin":
        # log(mean exp(z_m)) - z_b = (logsumexp(z_m) - log(K)) - z_b
        K = z_m.shape[1]
        scores = torch.logsumexp(z_m, dim=1) - np.log(K) - z_b
        return scores.cpu().numpy()

    if method == "logsumexp_margin":
        scores = torch.logsumexp(z_m, dim=1) - z_b
        return scores.cpu().numpy()

    if method == "max_margin":
        scores = z_m.max(dim=1).values - z_b
        return scores.cpu().numpy()

    # softmax 概率系
    probs = torch.softmax(z, dim=1)
    p_b = probs[:, benign_idx]
    p_m_all = probs[:, mal_mask]                             # (N, C-1)

    if method == "mal_sum_prob":
        scores = p_m_all.sum(dim=1)                          # = 1 - p_b
        return scores.cpu().numpy()

    if method == "mal_mean_prob":
        scores = p_m_all.mean(dim=1)                         # 对类数中性
        return scores.cpu().numpy()

    if method == "mal_max_prob":
        scores = p_m_all.max(dim=1).values
        return scores.cpu().numpy()

    raise ValueError(f"Unknown method: {method}")

def _mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))

def compute_binary_pauc(exp_path, dataset, max_fprs=(0.01, 0.02, 0.05), score_method="logmeanexp_margin", return_per_task=False,):
    """
    Calculate binary classification (benign vs malicious) pAUC (max_fpr=0.01)
    
    Parameters:
        - exp_path: experiment path
        - dataset: dataset name
    
    Returns:
        - dict: contains pAUC values for each phase
    """
    label_conv_path = f'../data/{dataset}/classes_map_rename.txt'
    label_map = load_label_map(label_conv_path)


    classes_info_path = os.path.join(exp_path, 'classes_info.json')
    with open(classes_info_path, 'r') as f:
        class_info = json.load(f)

    # ---------- load class names ----------

    old_keys = list(map(int, class_info['old_classes'].keys()))
    new_keys = list(map(int, class_info['new_classes'].keys()))
    inv_label_map = {v: k for k, v in label_map.items()}
    old_labels = [inv_label_map[k] for k in old_keys]
    new_labels = [inv_label_map[k] for k in new_keys]
    labels = old_labels + new_labels
    # pauc_results = {}
    
    # ---------- load tensors ----------
    folder = 'adaptation_data'
    folder_path = os.path.join(exp_path, folder)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f'{folder_path} does not exist')

    logits_data = np.load(os.path.join(folder_path, 'logits.npz'))
    labels_data = np.load(os.path.join(folder_path, 'labels.npz'))

    if 'query_labels' in labels_data:
        true_labels = labels_data['query_labels']  # (T, N)
    else:
        true_labels = labels_data['labels']        # (T, N)

    logits = logits_data['logits']                 # (T, N, C)

    if logits.ndim != 3 or true_labels.ndim != 2:
        raise ValueError(f'Expect logits shape (T,N,C) and true_labels (T,N), got {logits.shape} and {true_labels.shape}')

    T, N, C = logits.shape
    if true_labels.shape != (T, N):
        raise ValueError('true_labels shape must be (T, N) matching logits first two dims.')

    # ---------- benign index ----------
    benign_idx_list = [i for i, name in enumerate(labels) if str(name).lower() == 'benign']
    if len(benign_idx_list) != 1:
        raise ValueError("Benign class is not unique or missing, or label order mismatched with logits columns.")
    benign_idx = int(benign_idx_list[0])


    # ---------- iterate episodes ----------
    per_task_auc = []
    per_task_pauc = {alpha: [] for alpha in max_fprs}
    skipped_tasks = 0
    reasons = []

    for t in range(T):
        task_logits = logits[t]      # (N, C)
        task_labels = true_labels[t] # (N,)
        # print(task_logits[0])
        # print(task_labels[0])
        # exit()
        # exit()
        # Binary labels: 1=malicious, 0=benign
        binary_labels = (task_labels != benign_idx).astype(int)

        # 如果任务中只有单一类别，ROC 不可计算，跳过
        n_pos = int(np.sum(binary_labels == 1))
        n_neg = int(np.sum(binary_labels == 0))
        if n_pos == 0 or n_neg == 0:
            skipped_tasks += 1
            reasons.append((t, f"skip: n_pos={n_pos}, n_neg={n_neg}"))
            continue

        # 分数（越大越恶性）
        y_scores = make_scores_from_logits(task_logits, benign_idx, method=score_method)
        # print(y_scores[0])
        # exit()
        # 计算 AUC
        try:
            auc_val = roc_auc_score(binary_labels, y_scores)
            per_task_auc.append(auc_val)
        except Exception as e:
            skipped_tasks += 1
            reasons.append((t, f"AUC error: {e}"))
            continue

        # 计算多个 pAUC
        for alpha in max_fprs:
            try:
                pauc_val = compute_pauc_sklearn(binary_labels, y_scores, max_fpr=alpha)
                per_task_pauc[alpha].append(pauc_val)
            except Exception as e:
                reasons.append((t, f"pAUC error@{alpha}: {e}"))
                # 不中断，继续后面的 alpha

    # ---------- aggregate ----------
    results = {
        'n_tasks_total': int(T),
        'n_tasks_used': int(len(per_task_auc)),
        'auc_mean': 0.0,
        'auc_std': 0.0,
    }

    if len(per_task_auc) > 0:
        auc_mean, auc_std = _mean_std(per_task_auc)
        results['auc_mean'] = auc_mean
        results['auc_std'] = auc_std

    for alpha in max_fprs:
        vals = per_task_pauc[alpha]
        key_mean = f'pauc@{alpha}_mean'
        key_std  = f'pauc@{alpha}_std'
        if len(vals) > 0:
            m, s = _mean_std(vals)
            results[key_mean] = m
            results[key_std]  = s
        else:
            results[key_mean] = None
            results[key_std]  = None

    if return_per_task:
        results['per_task_auc'] = per_task_auc
        for alpha in max_fprs:
            results[f'per_task_pauc@{alpha}'] = per_task_pauc[alpha]
        results['skipped_tasks'] = skipped_tasks
        results['skip_reasons'] = reasons

    # 友好打印
    print(f"[Episodic] Tasks used: {results['n_tasks_used']}/{results['n_tasks_total']}")
    if results['n_tasks_used'] > 0:
        print(f"AUC (mean±std): {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
        for alpha in max_fprs:
            m = results[f'pauc@{alpha}_mean']
            s = results[f'pauc@{alpha}_std']
            if m is not None:
                print(f"pAUC_sklearn@{alpha} (mean±std): {m:.4f} ± {s:.4f}")
            else:
                print(f"pAUC_sklearn@{alpha}: None (no valid tasks)")

    if skipped_tasks > 0:
        print(f"Skipped {skipped_tasks} tasks (see results['skip_reasons'])")

    return results
    

def get_metric(exp_path, data, wanted_metrics, class_pool, folders):    
    metrics = util.logger.get_metric(
        exp_path, wanted_metrics=wanted_metrics, folders=folders, class_pool=class_pool)
    for k, v in metrics.items():
        data[k] = v
    return data  

                           
def get_metric_dataframe(exp_path):
    tmp = []
    
    # Get exp args
    for args_path in tqdm(Path(exp_path).rglob('dict_args.json'), desc='Generating parquet'):
        try:
            data = dict()
            with open(args_path) as f:
                dict_args = json.load(f)
                
            # Extract version number
            version_match = re.search(r'version_(\d+)', str(args_path))
            if version_match:
                data['version'] = int(version_match.group(1))
            else:
                data['version'] = np.nan
            
            # Define path first
            path = str(args_path).replace('dict_args.json', '')
            print(path)
            # Extract pre_mode from folder name instead of dict_args
            for exp_arg in exp_args:
                if exp_arg == 'pre_mode':
                    data[exp_arg] = extract_pre_mode_from_path(path)
                else:
                    data[exp_arg] = dict_args[exp_arg]
            is_fscil = dict_args['is_fscil']
            
            match = re.search(r'_al_cycle_(\d+)', path)
            data['curr_al_cycle']  = match.group(1) if match else np.nan
            
            
            if is_fscil:
                try:
                    with open(f'{path}classes_info.json') as f:
                        classes_info = json.load(f)
                        
                    data['old_classes'] = [int(k) for k in classes_info['old_classes'].keys()]
                    data['new_classes'] = [int(k) for k in classes_info['new_classes'].keys()]
                    class_pool = dict()
                    class_pool['new'] = classes_info['all_classes'][0][len(data['old_classes']):]
                    class_pool['old'] = classes_info['all_classes'][0][:len(data['old_classes'])]
                    class_pool['all'] = None
                            
                    wanted_metrics = default_metrics + fscil_metric
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not process FSCIL data for {path}: {str(e)}")
                    continue
            else:
                wanted_metrics = default_metrics
                class_pool = None
            
            try:
                data['train_time'] = compute_train_time(path)
            except Exception as e:
                print(f"Warning: Could not compute training time for {path}: {str(e)}")
                data['train_time'] = np.nan
            
            try:
                data = get_metric(
                    path, data, wanted_metrics=wanted_metrics, class_pool=class_pool, 
                    folders=['adaptation_data']
                )

                
                # Calculate pAUC
                try:
                    dataset_name = dict_args.get('dataset', 'unknown')
                    pauc_results = compute_binary_pauc(path, dataset_name)
                    data.update(pauc_results)
                except Exception as e:
                    print(f"Warning: Could not compute pAUC for {path}: {str(e)}")
                    # Add default pAUC value
                    data['adaptation data pauc'] = np.nan
                
                tmp.append(pd.DataFrame([data]))
            except Exception as e:
                print(f"Warning: Could not compute metrics for {path}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Warning: Could not process {args_path}: {str(e)}")
            continue

    if not tmp:
        print("Warning: No valid data was collected!")
        return pd.DataFrame()
        
    return pd.concat(tmp, ignore_index=True)


def main():
    parser = ArgumentParser(conflict_handler="resolve", add_help=True)

    parser.add_argument('--exp-path', type=str, default='../experiments')
    parser.add_argument('--save-path', type=str, default='../metrics')
    parser.add_argument('--name-file', type=str, default='al_fscil_metrics')
    
    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)

    try: 
        os.mkdir(args.save_path) 
    except OSError as error: 
        print(error)  
    
    df = get_metric_dataframe(args.exp_path)
    df.to_parquet(f'{args.save_path}/{args.name_file}.parquet')
    df.to_csv(f'{args.save_path}/{args.name_file}.csv')
    print('PARQUET GENERATED!!')
    
if __name__ == '__main__':
    main()