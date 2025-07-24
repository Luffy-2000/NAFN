import os.path
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
import json
import util.metrics


folder_2_phase = {
        'adaptation_data': 'adaptation',
        'pt_test_data': 'pt_test',
        'test_data': 'test',
        'train_data': 'train',
        'val_data': 'val' 
    }



def load_label_names(classes_info_path, label_conv_path):
    with open(classes_info_path, 'r') as f:
        class_info = json.load(f)
    with open(label_conv_path, 'r') as f:
        label_map = eval(f.read())
    
    # 2. 提取 old_classes 和 new_classes 的 key (字符串转 int)
    old_keys = list(map(int, class_info['old_classes'].keys()))
    new_keys = list(map(int, class_info['new_classes'].keys()))

    # 3. 根据 label_map 的 value 找到对应名称
    # 注意：你的 label_map 是 {"名称": 编码}，我们需要反向映射
    inv_label_map = {v: k for k, v in label_map.items()}

    old_labels = [inv_label_map[k] for k in old_keys]
    new_labels = [inv_label_map[k] for k in new_keys]

    old_labels = [label.replace('_', ' ') for label in old_labels]   # ← 修改
    new_labels = [label.replace('_', ' ') for label in new_labels]   # ← 修改
    # 4. 拼接 old + new 得到完整类别名称
    all_labels = old_labels + new_labels

    return all_labels, len(old_labels)



def plot_confusion_matrix(exp_path, dataset):
    """
    This function computes the averaged per-episode confusion matrix for each folder/learning phase 
    in the experiment path and saves it as a PDF heatmap and a CSV file. If a specified folder does 
    not exist in the experiment path, an IOError is raised.

    Parameters:
        - exp_path (``str``): The path to the experiment directory.
    """
    # 加载类名与旧类数量
    classes_info_path = os.path.join(exp_path, 'classes_info.json')
    label_conv_path = f'../data/{dataset}/classes_map_rename.txt'  # 路径视项目结构可能需要调整
    labels, line_x = load_label_names(classes_info_path, label_conv_path)
    print("labels:", labels)
    print("line_x:", line_x)
    for folder in ['adaptation_data', 'pt_test_data', 'test_data']:
        folder_path = os.path.join(exp_path, folder)
        if not os.path.isdir(folder_path):
            print(f'INFO: {folder_path} does not exist')
            continue
        
        os.makedirs(f'{exp_path}/img', exist_ok=True)
            
        # 调用你已有的 util 函数计算混淆矩阵
        avg_cm, avg_cm_raw = util.metrics.compute_confusion_matrix(
            path=folder_path, 
            files=['logits', 'labels']
        )

        # ========= 保存标准化混淆矩阵 ==========
        df_cm = pd.DataFrame(avg_cm) * 100  # 转为百分比
        df_cm = df_cm.rename_axis('Actual Label').rename(columns=lambda x: x)
        df_cm.index.name = 'Actual Label'

        cm_file = f'{exp_path}/img/cm_{folder}.csv'
        pdf_file = f'{exp_path}/img/cm_{folder}.pdf'

        df_cm.to_csv(cm_file)

        plt.figure(figsize=(8, 6))
        sn.set_theme(font_scale=1.0)
        ax = sn.heatmap(
            df_cm, annot=False, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=100
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')  # ha=right让文字不重叠
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # y轴保持不旋转
        plt.axvline(x=line_x, linewidth=3, color='k')
        plt.axhline(y=line_x, linewidth=3, color='k')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.tight_layout()
        plt.savefig(pdf_file, bbox_inches="tight")
        plt.close()

        # ========= 保存未标准化混淆矩阵 ==========
        df_cm_raw = pd.DataFrame(avg_cm_raw)
        raw_cm_file = f'{exp_path}/img/cm_{folder}_raw.csv'
        raw_pdf_file = f'{exp_path}/img/cm_{folder}_raw.pdf'

        df_cm_raw.to_csv(raw_cm_file)

        plt.figure(figsize=(8, 6))
        sn.set_theme(font_scale=1.0)
        ax = sn.heatmap(
            df_cm_raw, annot=False, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.axvline(x=line_x, linewidth=3, color='k')
        plt.axhline(y=line_x, linewidth=3, color='k')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.tight_layout()
        plt.savefig(raw_pdf_file, bbox_inches="tight")
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