import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import pprint
from matplotlib.ticker import PercentFormatter
import os
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

def read_results_multi(file_path):
    results = pd.read_csv(file_path)
    selected_columns = [
        'dataset', 'base_learner', 'shots',
        'adaptation_f1_all_macro_mean', 'adaptation_f1_all_macro_std',
        'adaptation_f1_new_macro_mean', 'adaptation_f1_new_macro_std',
        'adaptation_f1_old_macro_mean', 'adaptation_f1_old_macro_std'
    ]
    results = results[selected_columns].round(4)
    return results

def map_data_to_dict(results):
    dataset_list = ['cic2018', 'edge_iiot', 'iot_nid']
    result_dict = {clf: {ds: {} for ds in dataset_list} for clf in ['NN','LR']}

    for clf in ['NN', 'LR']:
        clf_key = clf.lower()
        clf_results = results.groupby('base_learner').get_group(clf_key)
        for ds in dataset_list:
            ds_results = clf_results.groupby('dataset').get_group(ds).sort_values(by='shots')
            result_dict[clf][ds]['shots'] = ds_results['shots'].tolist()
            result_dict[clf][ds]['F1-ALL'] = {
                'mean': ds_results['adaptation_f1_all_macro_mean'].tolist(),
                'std': ds_results['adaptation_f1_all_macro_std'].tolist(),
            }
            result_dict[clf][ds]['F1-New'] = {
                'mean': ds_results['adaptation_f1_new_macro_mean'].tolist(),
                'std': ds_results['adaptation_f1_new_macro_std'].tolist(),
            }
            result_dict[clf][ds]['F1-Old'] = {
                'mean': ds_results['adaptation_f1_old_macro_mean'].tolist(),
                'std': ds_results['adaptation_f1_old_macro_std'].tolist(),
            }   
    return result_dict

def plot_bar_NN_LR(result_dict, save_dir="./PDF"):
    """
    画两张图（NN, LR），每张3个面板（cic2018 / edge_iiot / iot_nid）
    X: shots
    Y: F1-ALL（带误差棒）
    Y轴断轴：低区间 0–5% 压缩显示，高区间 60–100% 正常显示
    """
    datasets = ['cic2018', 'edge_iiot', 'iot_nid']
    pretty_name_map = {
        'cic2018': 'CSE-CIC-IDS2018',
        'edge_iiot': 'Edge-IIoT',
        'iot_nid': 'IoT-NID'
    }

    low_ylim = (0.0, 0.05)
    high_ylim = (0.6, 1.05)

    def percent_fmt(ax):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    def add_break_marks(ax_bottom, ax_top, d=0.01):
        # 在上下轴连接处画“//”断裂标记
        kwargs = dict(color='k', clip_on=False, linewidth=2)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
        ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)

    for clf in ['NN', 'LR']:
        fig = plt.figure(figsize=(16, 2.5))
        # fig.suptitle(f"{clf} — Few-shot Performance (Broken Y-axis)", fontsize=14)
        gs = GridSpec(2, 3, height_ratios=[5, 1], figure=fig, hspace=0.15)  
        # 上面 row 占 3 份（高区间），下面 row 占 1 份（低区间）

        for i, ds in enumerate(datasets):
            shots = result_dict[clf][ds]['shots']
            means = result_dict[clf][ds]['F1-ALL']['mean']
            stds = result_dict[clf][ds]['F1-ALL']['std']
            x_pos = np.arange(len(shots))

            # 高区间轴（上）
            ax_high = fig.add_subplot(gs[0, i])
            bars_high = ax_high.bar(x_pos, means, yerr=stds, capsize=4,
                                    alpha=0.85, edgecolor="black", color="skyblue")
            ax_high.set_ylim(*high_ylim)
            ax_high.set_title(pretty_name_map[ds], fontsize=12)
            percent_fmt(ax_high)
            ax_high.grid(axis='y', linestyle='--', alpha=0.6)
            plt.setp(ax_high.get_xticklabels(), visible=False)
            ax_high.set_ylabel("F1-All[%]")
            # 低区间轴（下）
            ax_low = fig.add_subplot(gs[1, i], sharex=ax_high)
            bars_low = ax_low.bar(x_pos, means, yerr=stds, capsize=4,
                                  alpha=0.85, edgecolor="black", color="skyblue")
            ax_low.set_ylim(*low_ylim)
            ax_low.set_xlabel("K", fontsize=12)
            ax_low.set_xticks(x_pos)
            ax_low.set_xticklabels(shots)
            percent_fmt(ax_low)
            ax_low.grid(axis='y', linestyle='--', alpha=0.6)

            # 去掉连接处脊线
            ax_high.spines['bottom'].set_visible(False)
            ax_low.spines['top'].set_visible(False)

            # 加断裂符号
            add_break_marks(ax_low, ax_high)

            # 标注数值
            for j, m in enumerate(means):
                if m <= low_ylim[1]:
                    ax_low.text(j, m + stds[j] + 0.002, f"{m*100:.2f}%",
                                ha='center', va='bottom', fontsize=10)
                elif m >= high_ylim[0]:
                    ax_high.text(j, m + stds[j] + 0.02, f"{m*100:.2f}%",
                                 ha='center', va='bottom', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(f"{save_dir}/{clf}_shots_sensitive_analysis.pdf", bbox_inches='tight', dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    shot_file_path = "../metrics/bestcombo_student_metrics_Shot.csv"
    results = read_results_multi(shot_file_path)
    print(results.head())
    result_dict = map_data_to_dict(results)
    print(result_dict)
    plot_bar_NN_LR(result_dict, save_dir="./PDF")
    
    
    