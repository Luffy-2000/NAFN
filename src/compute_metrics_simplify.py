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


def compute_pauc(y_true, y_scores, max_fpr=0.01):
    """
    Calculate pAUC (partial AUC) using sklearn's roc_auc_score
    
    Parameters:
        - y_true: true labels (binary classification: 0=benign, 1=attack)
        - y_scores: prediction scores (higher values indicate positive class)
        - max_fpr: maximum false positive rate, default 0.01 (1%)
    
    Returns:
        - pauc: normalized partial AUC value (range 0~1)
    """
    return roc_auc_score(y_true, y_scores, max_fpr=max_fpr)


def compute_binary_pauc(exp_path, dataset):
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
    print(dataset)
    # print(label_map)
    # print(class_info)

    # Load class names
    old_keys = list(map(int, class_info['old_classes'].keys()))
    new_keys = list(map(int, class_info['new_classes'].keys()))
    inv_label_map = {v: k for k, v in label_map.items()}
    old_labels = [inv_label_map[k] for k in old_keys]
    new_labels = [inv_label_map[k] for k in new_keys]
    labels = old_labels + new_labels
    # print(labels)

    pauc_results = {}
    
    for folder in ['adaptation_data']:
        folder_path = os.path.join(exp_path, folder)
        if not os.path.isdir(folder_path):
            print(f'INFO: {folder_path} does not exist')
            continue

        # Load logits and labels
        try:
            logits_data = np.load(f'{folder_path}/logits.npz')
            labels_data = np.load(f'{folder_path}/labels.npz')

            if 'query_labels' in labels_data:
                true_labels = labels_data['query_labels']
            else:
                true_labels = labels_data['labels']

            logits = logits_data['logits']

            # If 3D -> flatten
            if logits.ndim == 3:
                logits = logits.reshape(-1, logits.shape[-1])
                true_labels = true_labels.reshape(-1)

        except Exception as e:
            print(f"Warning: Could not load data from {folder_path}: {str(e)}")
            continue

        # Find benign class index
        benign_idx = [i for i, name in enumerate(labels) if name.lower() == 'benign']
        print(benign_idx)
        if len(benign_idx) != 1:
            print("Error: benign class is not unique or missing")
            continue
        benign_idx = benign_idx[0]
        # print(benign_idx)
        # Construct binary labels
        print(true_labels)
        print(true_labels.shape)
        print("logits", logits)
        binary_labels = (true_labels != benign_idx).astype(int)
        print(binary_labels)
        print(f"Binary labels count - 0 (benign): {np.sum(binary_labels == 0)}, 1 (malicious): {np.sum(binary_labels == 1)}")
        # Apply softmax to get probabilities
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
        print(f"Probabilities shape: {probs.shape}")
        
        # Method 1: Use benign class probability as score (higher = more likely benign)
        # For pAUC, we want higher scores for positive class (benign)
        y_scores_benign = probs[:, benign_idx]
        
        # Method 2: Use max probability among malicious classes as score (higher = more likely malicious)
        malicious_idx = [i for i in range(probs.shape[1]) if i != benign_idx]
        y_scores_malicious = np.max(probs[:, malicious_idx], axis=1)
        
        # Method 3: Use logit difference (malicious - benign) as score
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        y_scores_diff = (logits_tensor[:, malicious_idx].max(dim=1)[0] - logits_tensor[:, benign_idx]).numpy()
        
        # Method 4: Aggregate all malicious classes (sum of probabilities)
        y_scores_aggregated = np.sum(probs[:, malicious_idx], axis=1)
        
        # Method 5: Aggregate all malicious classes (weighted sum based on class frequency)
        # Calculate class frequencies in the current batch
        unique_labels, label_counts = np.unique(true_labels, return_counts=True)
        label_freq = dict(zip(unique_labels, label_counts))
        malicious_weights = np.array([label_freq.get(idx, 1) for idx in malicious_idx])
        malicious_weights = malicious_weights / np.sum(malicious_weights)  # Normalize
        y_scores_weighted = np.sum(probs[:, malicious_idx] * malicious_weights, axis=1)
        
        print(f"Score ranges:")
        print(f"  Benign prob: {y_scores_benign.min():.4f} to {y_scores_benign.max():.4f}")
        print(f"  Malicious prob: {y_scores_malicious.min():.4f} to {y_scores_malicious.max():.4f}")
        print(f"  Logit diff: {y_scores_diff.min():.4f} to {y_scores_diff.max():.4f}")
        print(f"  Aggregated malicious: {y_scores_aggregated.min():.4f} to {y_scores_aggregated.max():.4f}")
        print(f"  Weighted malicious: {y_scores_weighted.min():.4f} to {y_scores_weighted.max():.4f}")
        
        # Choose the best scoring method for pAUC
        # For pAUC with max_fpr=0.01, we want to detect malicious samples with low false positive rate
        # You can choose different methods by uncommenting the desired line:
        
        # Method 1: Use benign probability (higher = more likely benign)
        # y_scores = y_scores_benign
        
        # Method 2: Use max malicious probability (higher = more likely malicious)
        y_scores = y_scores_malicious
        
        # Method 3: Use logit difference (malicious - benign)
        # y_scores = y_scores_diff
        
        # Method 4: Use aggregated malicious probability (sum of all malicious classes)
        # y_scores = y_scores_weighted
        
        # Method 5: Use weighted aggregated malicious probability (based on class frequency)
        # y_scores = y_scores_weighted
        
        # Calculate pAUC
        pauc = compute_pauc(binary_labels, y_scores)
        print(f"pAUC (max_fpr=0.01): {pauc:.4f}")
        
        # Also calculate regular AUC for comparison
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(binary_labels, y_scores)
        print(f"Regular AUC: {auc:.4f}")
        
        # Save results
        phase_name = folder.replace('_', ' ')
        pauc_results[f'{phase_name}_pauc'] = pauc
        pauc_results[f'{phase_name}_auc'] = auc  # Also save regular AUC
        
        # print(f"[{folder}] Merged benign vs malicious pAUC = {pauc:.4f}")

    return pauc_results


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
                
            for exp_arg in exp_args:
                data[exp_arg] = dict_args[exp_arg]
            is_fscil = dict_args['is_fscil']
                            
            path = str(args_path).replace('dict_args.json', '')
            
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