import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

df_student = pd.read_csv("../metrics/student_metrics_allpre.csv")
df_teacher = pd.read_csv("../metrics/teacher_metrics_allpre.csv")

print(df_student.head())
print(df_teacher.head())
print(df_student.columns)

# 选择需要的列
selected_columns = ['dataset', 'pre_mode', 'base_learner', 'memory_selector', 'adaptation_f1_all_macro_mean', 
                    'adaptation_f1_all_macro_std', 'adaptation_f1_new_macro_mean', 'adaptation_f1_new_macro_std',
                    'adaptation_f1_old_macro_mean', 'adaptation_f1_old_macro_std']

df_student = df_student[selected_columns]
df_teacher = df_teacher[selected_columns]

# 定义参数列表
dataset_list = ['cic2018', 'edge_iiot', 'iot_nid']
pre_mode_list = ['none', 'recon', 'contrastive', 'hybrid']
base_learner_list = ['nn', 'lr']
memory_selector_list = ['random', 'uncertainty', 'herding']
phase = ['base', 'adaptation']

# 创建嵌套字典存储所有参数组合
param_combinations = {}

for dataset in dataset_list:
    param_combinations[dataset] = {}
    
    for base_learner in base_learner_list:
        param_combinations[dataset][base_learner] = {}
        
        for pre_mode in pre_mode_list:
            param_combinations[dataset][base_learner][pre_mode] = {}
            
            for memory_selector in memory_selector_list:
                param_combinations[dataset][base_learner][pre_mode][memory_selector] = {}
                
                for phase_name in phase:
                    # 根据阶段选择对应的数据框
                    if phase_name == 'base':
                        df_to_use = df_teacher
                    else:  # adaptation
                        df_to_use = df_student
                    
                    # 过滤数据
                    mask = (
                        (df_to_use['dataset'] == dataset) &
                        (df_to_use['pre_mode'] == pre_mode) &
                        (df_to_use['base_learner'] == base_learner) &
                        (df_to_use['memory_selector'] == memory_selector)
                    )
                    
                    filtered_df = df_to_use[mask]
                    
                    if not filtered_df.empty:
                        # 计算六个指标的平均值
                        metrics = {
                            'adaptation_f1_all_macro_mean': f"{filtered_df['adaptation_f1_all_macro_mean'].mean():.2%}",
                            'adaptation_f1_all_macro_std': f"{filtered_df['adaptation_f1_all_macro_std'].mean():.2%}",
                            'adaptation_f1_new_macro_mean': f"{filtered_df['adaptation_f1_new_macro_mean'].mean():.2%}",
                            'adaptation_f1_new_macro_std': f"{filtered_df['adaptation_f1_new_macro_std'].mean():.2%}",
                            'adaptation_f1_old_macro_mean': f"{filtered_df['adaptation_f1_old_macro_mean'].mean():.2%}",
                            'adaptation_f1_old_macro_std': f"{filtered_df['adaptation_f1_old_macro_std'].mean():.2%}"
                        }
                            
                        param_combinations[dataset][base_learner][pre_mode][memory_selector][phase_name] = metrics
                    else:
                        param_combinations[dataset][base_learner][pre_mode][memory_selector][phase_name] = None

# 创建保存图片的文件夹
output_folder = 'heatmap_charts_seaborn'
os.makedirs(output_folder, exist_ok=True)

print("\n" + "=" * 50)
print("Creating heatmap charts (Seaborn style) for F1 All Macro across all combinations...")

for dataset in ['cic2018', 'edge_iiot', 'iot_nid']:
    for base_learner in ['nn', 'lr']:

        pre_modes = ['none', 'recon', 'contrastive', 'hybrid']
        memory_selectors = ['random', 'uncertainty', 'herding']

        f1_new_means, f1_old_means, f1_all_means = [], [], []
        f1_new_stds, f1_old_stds, f1_all_stds = [], [], []

        for pre_mode in pre_modes:
            new_mean_row, old_mean_row, all_mean_row = [], [], []
            new_std_row, old_std_row, all_std_row = [], [], []

            for memory_selector in memory_selectors:
                value = param_combinations[dataset][base_learner][pre_mode][memory_selector]['adaptation']
                if value:
                    new_mean = float(value['adaptation_f1_new_macro_mean'].rstrip('%')) / 100
                    new_std = float(value['adaptation_f1_new_macro_std'].rstrip('%')) / 100
                    old_mean = float(value['adaptation_f1_old_macro_mean'].rstrip('%')) / 100
                    old_std = float(value['adaptation_f1_old_macro_std'].rstrip('%')) / 100
                    all_mean = float(value['adaptation_f1_all_macro_mean'].rstrip('%')) / 100
                    all_std = float(value['adaptation_f1_all_macro_std'].rstrip('%')) / 100
                else:
                    new_mean = old_mean = all_mean = np.nan
                    new_std = old_std = all_std = np.nan

                new_mean_row.append(new_mean)
                old_mean_row.append(old_mean)
                all_mean_row.append(all_mean)
                new_std_row.append(new_std)
                old_std_row.append(old_std)
                all_std_row.append(all_std)

            f1_new_means.append(new_mean_row)
            f1_old_means.append(old_mean_row)
            f1_all_means.append(all_mean_row)
            f1_new_stds.append(new_std_row)
            f1_old_stds.append(old_std_row)
            f1_all_stds.append(all_std_row)

        f1_new_means = np.array(f1_new_means)
        f1_old_means = np.array(f1_old_means)
        f1_all_means = np.array(f1_all_means)
        f1_new_stds = np.array(f1_new_stds)
        f1_old_stds = np.array(f1_old_stds)
        f1_all_stds = np.array(f1_all_stds)

        best_new = np.nanmax(f1_new_means)
        best_old = np.nanmax(f1_old_means)
        best_all = np.nanmax(f1_all_means)

        display_pre_modes = ['None', 'Reconstruction', 'Contrastive', 'Hybrid']
        display_memory_selectors = ['Random', 'Uncertainty', 'Herding']

        # 转置
        f1_new_means_T = f1_new_means.T
        f1_old_means_T = f1_old_means.T
        f1_all_means_T = f1_all_means.T
        f1_new_stds_T = f1_new_stds.T
        f1_old_stds_T = f1_old_stds.T
        f1_all_stds_T = f1_all_stds.T

        # 冷色调配色
        cmap_choice = "PuBu"  # 可以换成 "mako", "Blues", "PuBu"

        # 三个并列图，高度调低 (变扁)
        fig, axes = plt.subplots(1, 3, figsize=(16, 2))
        sns.set_theme(style="whitegrid")
        sns.set(font_scale=1.2)  # 放大文字

        plt.rcParams['font.family'] = 'DejaVu Sans'

        # F1-New
        heatmap_new_df = pd.DataFrame(f1_new_means_T, index=display_memory_selectors, columns=display_pre_modes)
        ax1 = sns.heatmap(
            heatmap_new_df,
            annot=False,
            cmap=cmap_choice,
            cbar=False,
            linewidths=1,
            linecolor='white',
            vmin=0, vmax=1,
            square=False,   # 允许矩形格
            ax=axes[0]
        )
        for i in range(len(display_memory_selectors)):
            for j in range(len(display_pre_modes)):
                if not np.isnan(f1_new_means_T[i, j]):
                    mean_val = f1_new_means_T[i, j]
                    std_val = f1_new_stds_T[i, j]
                    text = f"{mean_val*100:.2f}±{std_val*100:.2f}%"
                    ax1.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                                fontsize=9, fontweight='bold', color='black')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=11, fontweight='bold')  # 横向
        axes[0].set_title(f'F1-New\nBest: {best_new*100:.2f}%', fontsize=13, fontweight='bold', color='darkred')

        # F1-Old
        heatmap_old_df = pd.DataFrame(f1_old_means_T, index=display_memory_selectors, columns=display_pre_modes)
        ax2 = sns.heatmap(
            heatmap_old_df,
            annot=False,
            cmap=cmap_choice,
            cbar=False,
            linewidths=1,
            linecolor='white',
            vmin=0, vmax=1,
            square=False,
            ax=axes[1]
        )
        for i in range(len(display_memory_selectors)):
            for j in range(len(display_pre_modes)):
                if not np.isnan(f1_old_means_T[i, j]):
                    mean_val = f1_old_means_T[i, j]
                    std_val = f1_old_stds_T[i, j]
                    text = f"{mean_val*100:.2f}±{std_val*100:.2f}%"
                    ax2.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                             fontsize=9, fontweight='bold', color='black')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=11, fontweight='bold')
        axes[1].set_title(f'F1-Old\nBest: {best_old*100:.2f}%', fontsize=13, fontweight='bold', color='darkred')

        # F1-All
        heatmap_all_df = pd.DataFrame(f1_all_means_T, index=display_memory_selectors, columns=display_pre_modes)
        ax3 = sns.heatmap(
            heatmap_all_df,
            annot=False,
            cmap=cmap_choice,
            cbar=False,
            linewidths=1,
            linecolor='white',
            vmin=0, vmax=1,
            square=False,
            ax=axes[2]
        )
        for i in range(len(display_memory_selectors)):
            for j in range(len(display_pre_modes)):
                if not np.isnan(f1_all_means_T[i, j]):
                    mean_val = f1_all_means_T[i, j]
                    std_val = f1_all_stds_T[i, j]
                    text = f"{mean_val*100:.2f}±{std_val*100:.2f}%"
                    ax3.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                             fontsize=9, fontweight='bold', color='black')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=11, fontweight='bold')
        ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=11, fontweight='bold')
        axes[2].set_title(f'F1-All\nBest: {best_all*100:.2f}%', fontsize=13, fontweight='bold', color='darkred')

        # 总标题
        fig.suptitle(f'{dataset.upper()} + {base_learner.upper()}', fontsize=18, fontweight='bold', color='darkred')

        # 更紧凑的布局
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)

        # 保存
        filename = f"{dataset}_{base_learner}_three_f1_heatmap_cool.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")

print(f"\nAll seaborn heatmap charts saved in folder: {output_folder}")


# 打印最佳和最差组合
print("\n" + "=" * 50)
print("Best and Worst F1 All Macro Combinations:")

all_f1_values = []
all_combinations = []

for dataset in ['cic2018', 'edge_iiot', 'iot_nid']:
    for base_learner in ['nn', 'lr']:
        for pre_mode in ['none', 'recon', 'contrastive', 'hybrid']:
            for memory_selector in ['random', 'uncertainty', 'herding']:
                value = param_combinations[dataset][base_learner][pre_mode][memory_selector]['adaptation']
                if value:
                    f1_value = float(value['adaptation_f1_all_macro_mean'].rstrip('%')) / 100
                    combination = f"{dataset}_{base_learner}_{pre_mode}_{memory_selector}"
                    all_f1_values.append(f1_value)
                    all_combinations.append(combination)

# 找到最佳和最差组合
best_idx = np.argmax(all_f1_values)
worst_idx = np.argmin(all_f1_values)

print(f"Best: {all_combinations[best_idx]} = {all_f1_values[best_idx]:.3f}")
print(f"Worst: {all_combinations[worst_idx]} = {all_f1_values[worst_idx]:.3f}")
print(f"Average F1: {np.mean(all_f1_values):.3f}")
print(f"Std F1: {np.std(all_f1_values):.3f}") 