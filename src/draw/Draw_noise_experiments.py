import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import pprint
from matplotlib.ticker import PercentFormatter

def read_results_multi(noise_file_path, denoise_paths_dict):
    noise_results = pd.read_csv(noise_file_path)
    selected_columns = [
        'dataset', 'base_learner', 'noise_ratio',
        'adaptation_f1_all_macro_mean', 'adaptation_f1_all_macro_std',
        'adaptation_f1_new_macro_mean', 'adaptation_f1_new_macro_std',
        'adaptation_f1_old_macro_mean', 'adaptation_f1_old_macro_std'
    ]
    noise_results = noise_results[selected_columns].round(4)

    denoise_results_dict = {}
    for method_name, path in denoise_paths_dict.items():
        df = pd.read_csv(path)
        df = df[selected_columns].round(4)
        denoise_results_dict[method_name] = df
    return noise_results, denoise_results_dict

def map_data_to_dict_multi(noise_results, denoise_results_dict):
    dataset_list = ['cic2018', 'edge_iiot', 'iot_nid']
    result_dict = {clf: {ds: {} for ds in dataset_list} for clf in ['NN','LR']}

    for clf in ['NN', 'LR']:
        clf_key = clf.lower()

        noise_clf = noise_results.groupby('base_learner').get_group(clf_key)
        for ds in dataset_list:
            noise_ds = noise_clf.groupby('dataset').get_group(ds).sort_values(by='noise_ratio')
            result_dict[clf][ds]['noise_ratio'] = noise_ds['noise_ratio'].tolist()
            result_dict[clf][ds]['w/o Denoise'] = {
                'F1-ALL': [noise_ds['adaptation_f1_all_macro_mean'].tolist(),
                           noise_ds['adaptation_f1_all_macro_std'].tolist()],
                'F1-New': [noise_ds['adaptation_f1_new_macro_mean'].tolist(),
                           noise_ds['adaptation_f1_new_macro_std'].tolist()],
                'F1-Old': [noise_ds['adaptation_f1_old_macro_mean'].tolist(),
                           noise_ds['adaptation_f1_old_macro_std'].tolist()],
            }

            # 多个 denoise 方法
            for method_name, df in denoise_results_dict.items():
                method_clf = df.groupby('base_learner').get_group(clf_key)
                method_ds = method_clf.groupby('dataset').get_group(ds).sort_values(by='noise_ratio')
                result_dict[clf][ds][method_name] = {
                    'F1-ALL': [method_ds['adaptation_f1_all_macro_mean'].tolist(),
                               method_ds['adaptation_f1_all_macro_std'].tolist()],
                    'F1-New': [method_ds['adaptation_f1_new_macro_mean'].tolist(),
                               method_ds['adaptation_f1_new_macro_std'].tolist()],
                    'F1-Old': [method_ds['adaptation_f1_old_macro_mean'].tolist(),
                               method_ds['adaptation_f1_old_macro_std'].tolist()],
                }
    return result_dict


def _pretty_name(name):
    mapping = {
        'cic2018': 'CSE-CIC-IDS2018',
        'edge_iiot': 'Edge-IIoT',
        'iot_nid': 'IoT-NID'
    }
    return mapping.get(name, name)

def _metric_keys_for_classifier(clf):
    """Return the metric key names for a classifier in your dict."""
    if clf == 'NN':
        return {
            'F1-All': 'F1-ALL',
            'F1-New': 'F1-New',
            'F1-Old': 'F1-Old'
        }
    else:  # 'LR'
        return {
            'F1-All': 'F1-ALL',
            'F1-New': 'F1-New',
            'F1-Old': 'F1-Old'
        }

def _plot_grid_for_classifier_multi(data, clf, figsize=(14, 12), show_std=False):
    datasets = ['cic2018', 'edge_iiot', 'iot_nid']
    metrics = ['F1-All', 'F1-New', 'F1-Old']
    metric_keys = _metric_keys_for_classifier(clf)

    # ✅ 方法名称对应的 marker 样式
    marker_map = {
        "w/o Denoise": "o",           # 实心圆
        "ProtoMargin": "^",         # 三角形
        "LOF": "s",              # 方形
        "E-CL": "D",              # 菱形
        "IF": "v",              # 倒三角
    }
    # 如果方法名不在字典里，就给个默认 marker
    default_marker = "."

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    # fig.suptitle(f'{clf} — Noise Sensitivity (Multiple Methods)', y=0.92, fontsize=14)

    x_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]

    for r, ds in enumerate(datasets):
        ds_pretty = _pretty_name(ds)
        noise_ratio = data[clf][ds]['noise_ratio']

        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            key = metric_keys[metric]

            # 遍历方法（除了 noise_ratio）
            for method_name, method_data in data[clf][ds].items():
                if method_name == 'noise_ratio':
                    continue
                mean_vals = method_data[key][0]
                std_vals  = method_data[key][1]
                marker = marker_map.get(method_name, default_marker)

                ax.plot(noise_ratio, mean_vals, marker=marker, label=method_name, linewidth=1.2)
                if show_std:
                    ax.fill_between(noise_ratio,
                                    [m-s for m, s in zip(mean_vals, std_vals)],
                                    [m+s for m, s in zip(mean_vals, std_vals)],
                                    alpha=0.15)

            if r == 0:
                ax.set_title(metric)
            if c == 0:
                ax.set_ylabel(ds_pretty)
            ax.set_xlabel('Noise Ratio', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.legend(fontsize=10, loc='lower left', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'./PDF/NoisyLabel Comparison with {clf}.pdf', bbox_inches='tight', facecolor='white', dpi=300)

def plot_nn_lr_3x3(data, show_std=False):
    # """生成两张图：第一张 NN（3x3），第二张 LR（3x3）。"""
    _plot_grid_for_classifier_multi(data, 'NN',  figsize=(15, 6), show_std=show_std)
    _plot_grid_for_classifier_multi(data, 'LR',  figsize=(15, 6), show_std=show_std)

if __name__ == "__main__":
    noise_file_path = "../metrics/bestcombo_student_metrics_noise_new.csv"
    denoise_files = {
        'ProtoMargin': "../metrics/bestcombo_student_metrics_ProtoMargin_denoise.csv",
        'LOF': "../metrics/bestcombo_student_metrics_LOF_denoise_new.csv",
        'E-CL': "../metrics/bestcombo_student_metrics_DCML_denoise.csv",
        'IF': "../metrics/bestcombo_student_metrics_IF_denoise.csv",
    }

    noise_results, denoise_results_dict = read_results_multi(noise_file_path, denoise_files)

    # exit()
    result_dict = map_data_to_dict_multi(noise_results, denoise_results_dict)
    plot_nn_lr_3x3(result_dict, show_std=True)