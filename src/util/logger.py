import os.path
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os

import util.metrics


folder_2_phase = {
        'adaptation_data': 'adaptation',
        'pt_test_data': 'pt_test',
        'test_data': 'test',
        'train_data': 'train',
        'val_data': 'val' 
    }


def plot_confusion_matrix(exp_path):
    """
    This function computes the averaged per-episode confusion matrix for each folder/learning phase 
    in the experiment path and saves it as a PDF heatmap and a CSV file. If a specified folder does 
    not exist in the experiment path, an IOError is raised.

    Parameters:
        - exp_path (``str``): The path to the experiment directory.
    """
            
    for folder in ['adaptation_data', 'pt_test_data', 'test_data']:
        if not os.path.isdir(f'{exp_path}/{folder}'):
            print(f'INFO: {exp_path}/{folder} does not exist')
            continue
        
        os.makedirs(f'{exp_path}/img', exist_ok=True)
            
        avg_cm, avg_cm_raw = util.metrics.compute_confusion_matrix(
            path=f'{exp_path}/{folder}', 
            files=['logits', 'labels']
        )
        
        # Save normalized confusion matrix
        df_cm = pd.DataFrame(avg_cm)
        plt.figure(figsize=(10, 8))
        p = sn.heatmap(df_cm, annot=True, fmt='.2f', square=True)
        plt.savefig(os.path.join(f'{exp_path}/img', 'confusion_matrix.png'))
        plt.close()
        
        # Save unnormalized confusion matrix
        df_cm_raw = pd.DataFrame(avg_cm_raw)
        plt.figure(figsize=(10, 8))
        p = sn.heatmap(df_cm_raw, fmt='d', square=True)  # Use integer format
        plt.savefig(os.path.join(f'{exp_path}/img', 'confusion_matrix_raw.png'))
        plt.close()


def get_metric(exp_path, folders, wanted_metrics, class_pool=None):
    """
    Computes the average of specified metrics based on logits or cluster for each folder in the 
    experiment path.

    Parameters:
        - exp_path (``str``): The path to the experiment directory.
        - folders (``list``): A list of folders containing the exp files.
        - wanted_metrics (``list``, optional): A list of metrics to compute.

    Returns:
        - ``dict``: A dictionary containing the average and standard deviation of the wanted metrics 
          for each folder in the experiment path.
    """
    wanted_cluster_metrics = [e for e in wanted_metrics if e in util.metrics.cluster_based_metrics.keys()]
    wanted_logit_metrics = [e for e in wanted_metrics if e in util.metrics.logit_based_metrics.keys()]
    avg_metrics = dict()
    
    for folder in folders:
        if not os.path.isdir(f'{exp_path}/{folder}'):
            print(f'INFO: {exp_path}/{folder} does not exist')
            continue
  
        # Compute logit based metrics ('f1', 'acc') if required
        if wanted_logit_metrics != []:  
            
            files = ['logits', 'labels']
            
            l_metrics = util.metrics.compute_logit_based_metrics(
                path=f'{exp_path}/{folder}', 
                files=files,
                wanted_metrics=wanted_logit_metrics,
                class_pool=class_pool,
                is_averaged=True
            )
        else:
            l_metrics = dict()
            
        # Compute cluster based metrics ('sc', 'db', 'ch') if required
        if wanted_cluster_metrics != []:
            
            files = ['supports', 'queries', 'labels']
                
            c_metrics = util.metrics.compute_cluster_based_metrics(
                path=f'{exp_path}/{folder}', 
                files=files,
                wanted_metrics=wanted_cluster_metrics,
                is_averaged=True
            )
        else:
            c_metrics = dict()
        metrics = {**l_metrics, **c_metrics}    
        
        for k, v in metrics.items():
            avg_metrics[f'{folder_2_phase[folder]}_{k}_mean'] = float(v[0])
            avg_metrics[f'{folder_2_phase[folder]}_{k}_std'] = float(v[1])
       
    return avg_metrics