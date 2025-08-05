import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import re
import util.metrics 
import util.logger

def compute_train_time(exp_path):
    """Calculate training time (changed to read from version_x)"""
    try:
        version_dirs = sorted(Path(exp_path).glob('lightning_logs/version_*'))
        if not version_dirs:
            print(f"INFO: No version dir found in {exp_path}")
            return np.nan
        version_path = version_dirs[-1]  # Enter version directory
        log_path = version_path / 'early_stopping.log'
        time_per_epoch = []
        with open(log_path) as log:
            for line in log:
                date = " ".join(line.split(' ')[:2])
                date_format = datetime.strptime(date, "%Y-%m-%d %H:%M:%S,%f")
                unix_time = datetime.timestamp(date_format)
                time_per_epoch.append(unix_time)
        return time_per_epoch[-1] - time_per_epoch[0]
    except Exception as e:
        print(f"Warning: Could not compute training time for {exp_path}: {str(e)}")
        return np.nan



def get_metric(exp_path, data, wanted_metrics, class_pool, folders):    
    """Read adaptation_data / pt_test_data metrics"""
    try:
        version_dirs = sorted(Path(exp_path).glob('lightning_logs/version_*'))
        if not version_dirs:
            print(f"INFO: No version dir in {exp_path}")
            return data
        version_path = version_dirs[-1]  # Enter version directory
        metrics = util.logger.get_metric(
            str(version_path),
            wanted_metrics=wanted_metrics,
            folders=folders,
            class_pool=class_pool
        )
        for k, v in metrics.items():
            data[k] = v

        return data
    except Exception as e:
        print(f"Warning: Could not get metrics for {exp_path}: {str(e)}")
        print(data)
        exit()
        return data


def extract_experiment_info(folder_path):
    """Extract experiment information from folder name"""
    folder_name = os.path.basename(folder_path)
    parts = folder_name.split('_')

    # version_match = re.search(r'version_(\d+)', str(folder_path))
    # version = int(version_match.group(1)) if version_match else 0

    info = {
        'phase': 'teacher' if 'teacher' in folder_name else 'student',
    }

    for dataset in ['cic2018', 'edge_iiot', 'iot_nid']:
        if dataset in folder_name:
            info['dataset'] = dataset
            break

    for part in parts:
        if 'shot' in part:
            info['shots'] = int(part.replace('shot', ''))
            break

    for mode in ['none', 'recon', 'contrastive', 'hybrid']:
        if mode in folder_name:
            info['pre_mode'] = mode
            break

    info.update({
        'queries': 40,
        'network': 'UNet1D2D',
        'fields': "['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL']",
        'num_pkts': 20,
        'seed': 0
    })

    return info


def get_metric_dataframe(exp_paths):
    """Get metrics data for all experiments"""
    tmp = []
    default_metrics = ['acc', 'sc', 'db', 'f1_all_macro', 'f1_all_micro']
    fscil_metric = ['f1_new_macro', 'f1_new_micro', 'f1_old_macro', 'f1_old_micro']
    
    for exp_path in tqdm(exp_paths, desc="Processing experiment folders"):
        try:
            data = extract_experiment_info(exp_path)

            # Mark as experiment record (teacher or student)
            # data['is_experiment'] = data['is_teacher'] or data['is_student']

            # Training time
            data['train_time'] = compute_train_time(exp_path)

            # Extract all metrics uniformly
            all_metrics = default_metrics + fscil_metric
            # all_test_metrics = ['acc', 'f1_all_macro', 'f1_all_micro']

            # Get adaptation metrics
            data = get_metric(
                exp_path, data,
                wanted_metrics=all_metrics,
                class_pool=None,
                folders=['adaptation_data']
            )

            data = get_metric(
                exp_path, data,
                wanted_metrics=['acc', 'f1_all_macro', 'f1_all_micro'],
                class_pool=None,
                folders=['pt_test_data']
            )

            # Try to read class information uniformly (regardless of whether it's student)
            try:
                with open(f'{exp_path}/classes_info.json') as f:
                    classes_info = json.load(f)
                data['old_classes'] = str([int(k) for k in classes_info['old_classes'].keys()])
                data['new_classes'] = str([int(k) for k in classes_info['new_classes'].keys()])
            except:
                data['old_classes'] = '[]'
                data['new_classes'] = '[]'

            tmp.append(pd.DataFrame([data]))

        except Exception as e:
            print(f"Warning: Could not process {exp_path}: {str(e)}")
            continue

    if not tmp:
        print("Warning: No valid data was collected!")
        return pd.DataFrame()

    return pd.concat(tmp, ignore_index=True)



def main():
    root_dir = '../'
    save_dir = './metrics'
    os.makedirs(save_dir, exist_ok=True)

    experiment_folders = []
    for path in Path(root_dir).rglob('results_rfs_*'):
        if path.is_dir():
            experiment_folders.append(str(path))

    df = get_metric_dataframe(experiment_folders)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'{save_dir}/experiment_metrics_{timestamp}.csv', index=False)
    df.to_parquet(f'{save_dir}/experiment_metrics_{timestamp}.parquet')

    print(f"\nAnalysis completed! Results saved to {save_dir} directory")
    print(f"CSV file: experiment_metrics_{timestamp}.csv")
    print(f"Parquet file: experiment_metrics_{timestamp}.parquet")


if __name__ == '__main__':
    main()
