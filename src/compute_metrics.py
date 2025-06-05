from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import json
import os
import re

import util.metrics 
import util.logger


default_metrics = [ 'acc', 'sc', 'db', 'f1_all_macro', 'f1_all_micro']
fscil_metric = ['f1_new_macro', 'f1_new_micro', 'f1_old_macro', 'f1_old_micro']
exp_args = [
    'shots', 'queries', 'pre_mode','network', 'fields', 'num_pkts', 'seed', 'dataset'
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
                
            # 提取版本号
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
                data = get_metric(
                    path, data, wanted_metrics=['acc', 'f1_all_macro', 'f1_all_micro'], class_pool=None, 
                    folders=['pt_test_data']
                )
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