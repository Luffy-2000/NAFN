import pandas as pd

df_student = pd.read_csv("../metrics/student_metrics_allpre.csv")
df_teacher = pd.read_csv("../metrics/teacher_metrics_allpre.csv")

print(df_student.head())
print(df_teacher.head())

print(df_student.columns)

selected_columns = ['dataset', 'pre_mode', 'base_learner', 'memory_selector', 'adaptation_f1_all_macro_mean', 
                    'adaptation_f1_all_macro_std', 'adaptation_f1_new_macro_mean', 'adaptation_f1_new_macro_std',
                    'adaptation_f1_old_macro_mean', 'adaptation_f1_old_macro_std']

df_student = df_student[selected_columns]
df_teacher = df_teacher[selected_columns]

dataset_list = ['cic2018', 'edge_iiot', 'iot_nid']
pre_mode_list = ['none', 'recon', 'contrastive', 'hybrid']
base_learner_list = ['nn', 'lr']
memory_selector_list = ['random', 'uncertainty', 'herding']
phase = ['base', 'adaptation']

# Create a nested dictionary to store all parameter combinations
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
                    # Get the corresponding dataframe based on phase
                    if phase_name == 'base':
                        df_to_use = df_teacher
                    else:  # adaptation
                        df_to_use = df_student
                    
                    # Filter dataframe for this combination
                    mask = (
                        (df_to_use['dataset'] == dataset) &
                        (df_to_use['pre_mode'] == pre_mode) &
                        (df_to_use['base_learner'] == base_learner) &
                        (df_to_use['memory_selector'] == memory_selector)
                    )
                    print(mask)
                    # exit()
                    filtered_df = df_to_use[mask]
                    
                    if not filtered_df.empty:
                        # Calculate the six metrics for this combination
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

# Print the nested dictionary structure
print("Nested dictionary structure:")
print("=" * 50)
for dataset in param_combinations:
    print(f"Dataset: {dataset}")
    for base_learner in param_combinations[dataset]:
        print(f"  Base Learner: {base_learner}")
        for pre_mode in param_combinations[dataset][base_learner]:
            print(f"    Pre Mode: {pre_mode}")
            for memory_selector in param_combinations[dataset][base_learner][pre_mode]:
                print(f"      Memory Selector: {memory_selector}")
                for phase_name in param_combinations[dataset][base_learner][pre_mode][memory_selector]:
                    metrics = param_combinations[dataset][base_learner][pre_mode][memory_selector][phase_name]
                    if metrics:
                        print(f"        Phase: {phase_name} - F1 All Macro: {metrics['adaptation_f1_all_macro_mean']}")
                    else:
                        print(f"        Phase: {phase_name} - No data")

# Example of accessing specific combination
print("\n" + "=" * 50)
print("Example access:")
example = param_combinations['cic2018']['nn']['recon']['herding']['adaptation']
if example:
    print(f"CIC2018 + NN + Recon + Herding + Adaptation:")
    print(f"  F1 All Macro Mean: {example['adaptation_f1_all_macro_mean']}")
    print(f"  F1 All Macro Std: {example['adaptation_f1_all_macro_std']}")
    print(f"  F1 New Macro Mean: {example['adaptation_f1_new_macro_mean']}")
    print(f"  F1 New Macro Std: {example['adaptation_f1_new_macro_std']}")
    print(f"  F1 Old Macro Mean: {example['adaptation_f1_old_macro_mean']}")
    print(f"  F1 Old Macro Std: {example['adaptation_f1_old_macro_std']}")
else:
    print("No data for this combination")




# 创建3D柱状图展示所有组合的F1 All Macro
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

print("\n" + "=" * 50)
print("Creating 3D bar charts for F1 All Macro across all combinations...")



# # 创建3D图形子图
# fig = plt.figure(figsize=(24, 16))
# fig.suptitle('3D Bar Charts: F1 All Macro Across All Parameter Combinations', fontsize=16)

# # 为每个dataset和base_learner组合创建3D子图
# for i, dataset in enumerate(['cic2018', 'edge_iiot', 'iot_nid']):
#     for j, base_learner in enumerate(['nn', 'lr']):
#         # 计算子图位置
#         subplot_idx = i * 2 + j + 1
#         ax = fig.add_subplot(3, 2, subplot_idx, projection='3d')
        
#         # 准备数据
#         pre_modes = ['none', 'recon', 'contrastive', 'hybrid']
#         memory_selectors = ['random', 'uncertainty', 'herding']
        
#         # 创建坐标轴
#         x_pos = np.arange(len(pre_modes))
#         y_pos = np.arange(len(memory_selectors))
#         x_mesh, y_mesh = np.meshgrid(x_pos, y_pos)
        
#         # 准备数据矩阵
#         z_values = []
#         for pre_mode in pre_modes:
#             row = []
#             for memory_selector in memory_selectors:
#                 value = param_combinations[dataset][base_learner][pre_mode][memory_selector]['adaptation']
#                 if value:
#                     f1_value = float(value['adaptation_f1_all_macro_mean'].rstrip('%')) / 100
#                     row.append(f1_value)
#                 else:
#                     row.append(0)
#             z_values.append(row)
        
#         z_values = np.array(z_values).T  # 转置以匹配meshgrid
        
#         # 绘制3D柱状图
#         bars = ax.bar3d(x_mesh.flatten(), 
#                         y_mesh.flatten(), 
#                         np.zeros_like(x_mesh.flatten()),
#                         dx=0.8, dy=0.8, dz=z_values.flatten(),
#                         alpha=0.8, color='skyblue', edgecolor='navy')
        
#         # 在柱子上添加数值标签
#         for x_idx, x_val in enumerate(x_pos):
#             for y_idx, y_val in enumerate(y_pos):
#                 f1_value = z_values[y_idx, x_idx]
#                 if f1_value > 0:  # 只对有数据的柱子添加标签
#                     # 计算标签位置（柱子顶部中心）
#                     label_x = x_val
#                     label_y = y_val
#                     label_z = f1_value + 0.02  # 稍微高于柱子顶部
                    
#                     # 添加数值标签（百分数格式）
#                     ax.text(label_x, label_y, label_z, 
#                            f'{f1_value * 100:.2f}', 
#                            ha='center', va='bottom', 
#                            fontsize=8, fontweight='bold',
#                            color='black', bbox=dict(boxstyle="round,pad=0.2", 
#                                                   facecolor='white', 
#                                                   alpha=0.8, 
#                                                   edgecolor='gray'))
        
#         # 设置坐标轴标签
#         ax.set_xlabel('PRE-TRAINING MODE', fontsize=10, labelpad=10)
#         ax.set_ylabel('MEMORY SELECTOR', fontsize=10, labelpad=10)
#         ax.set_zlabel('F1 (%)', fontsize=10, labelpad=5)
        
#         # 设置刻度标签
#         ax.set_xticks(x_pos)
#         # 将recon替换为reconstruction用于显示
#         display_pre_modes = ['None', 'Reconstruction', 'Contrastive', 'Hybrid']
#         ax.set_xticklabels(display_pre_modes, rotation=0)
#         display_memory_selectors = ['Random', 'Uncertainty', 'Herding']
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(display_memory_selectors)
        
#         # 设置标题
#         ax.set_title(f'{dataset} + {base_learner}', fontsize=12, fontweight='bold')
        
#         # 调整视角
#         ax.view_init(elev=20, azim=45)
        
#         # 设置z轴范围
#         ax.set_zlim(0, 1)

# plt.tight_layout()

# # 保存图片
# plt.savefig('f1_all_macro_3d_subplots.png', dpi=300, bbox_inches='tight')
# print("3D bar charts saved as 'f1_all_macro_3d_subplots.png'")

# # 显示图片
# plt.show()

# # 打印最佳和最差组合
# print("\n" + "=" * 50)
# print("Best and Worst F1 All Macro Combinations:")
# 创建3D图形子图
# 创建保存图片的文件夹
import os
output_folder = '3d_charts_individual'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# 为每个dataset和base_learner组合创建单独的3D图
for i, dataset in enumerate(['cic2018', 'edge_iiot', 'iot_nid']):
    for j, base_learner in enumerate(['nn', 'lr']):
        # 创建单独的图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 准备数据
        pre_modes = ['none', 'recon', 'contrastive', 'hybrid']
        memory_selectors = ['random', 'uncertainty', 'herding']
        
        # 创建坐标轴
        x_pos = np.arange(len(pre_modes))
        y_pos = np.arange(len(memory_selectors))
        x_mesh, y_mesh = np.meshgrid(x_pos, y_pos)
        
        # 准备数据矩阵
        z_values = []
        for pre_mode in pre_modes:
            row = []
            for memory_selector in memory_selectors:
                value = param_combinations[dataset][base_learner][pre_mode][memory_selector]['adaptation']
                if value:
                    f1_value = float(value['adaptation_f1_all_macro_mean'].rstrip('%')) / 100
                    row.append(f1_value)
                else:
                    row.append(0)
            z_values.append(row)
        
        z_values = np.array(z_values).T  # 转置以匹配meshgrid
        
        # 设置颜色映射
        from matplotlib import cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('coolwarm')  # 使用 coolwarm 颜色映射，调整颜色风格
        
        # 找到最优组合（F1值最高）
        best_f1_value = np.max(z_values)
        best_indices = np.where(z_values == best_f1_value)
        best_x_idx = best_indices[1][0]  # x方向索引
        best_y_idx = best_indices[0][0]  # y方向索引
        
        # 绘制3D柱状图，颜色与F1值相关
        colors = cmap(norm(z_values.flatten()))
        
        # 为最优组合设置特殊颜色（金色）
        for i, (x_idx, y_idx) in enumerate(zip(x_mesh.flatten(), y_mesh.flatten())):
            if x_idx == best_x_idx and y_idx == best_y_idx:
                colors[i] = np.array([1.0, 0.843, 0.0, 1.0])  # 金色，RGBA格式
        
        bars = ax.bar3d(x_mesh.flatten(), 
                        y_mesh.flatten(), 
                        np.zeros_like(x_mesh.flatten()),
                        dx=0.8, dy=0.8, dz=z_values.flatten(),
                        alpha=0.8, 
                        color=colors,  # 使用修改后的颜色
                        edgecolor='k')  # 细致的边框颜色
        
        # 为柱子加标签
        for x_idx, x_val in enumerate(x_pos):
            for y_idx, y_val in enumerate(y_pos):
                f1_value = z_values[y_idx, x_idx]
                if f1_value > 0:  # 只对有数据的柱子添加标签
                    label_x = x_val
                    label_y = y_val
                    label_z = f1_value + 0.02  # 稍微高于柱子顶部
                    
                    # 为最优组合添加特殊标签样式
                    if x_idx == best_x_idx and y_idx == best_y_idx:
                        # 最优组合用金色标签和边框
                        ax.text(label_x, label_y, label_z, 
                                f'{f1_value * 100:.2f}*',  # 添加星号标记
                                ha='center', va='bottom', 
                                fontsize=12, fontweight='bold',
                                color='darkred', 
                                bbox=dict(boxstyle="round,pad=0.4", 
                                          facecolor='gold', 
                                          alpha=0.9, 
                                          edgecolor='darkred',
                                          linewidth=2))
                    else:
                        # 普通组合用默认样式
                        ax.text(label_x, label_y, label_z, 
                                f'{f1_value * 100:.2f}', 
                                ha='center', va='bottom', 
                                fontsize=10, fontweight='bold',
                                color='black', 
                                bbox=dict(boxstyle="round,pad=0.3", 
                                          facecolor='white', 
                                          alpha=0.8, 
                                          edgecolor='gray'))
        
        # 设置坐标轴标签
        ax.set_xlabel('PRE-TRAINING MODE', fontsize=12, labelpad=15, fontweight='bold', color='darkblue')
        ax.set_ylabel('MEMORY SELECTOR', fontsize=12, labelpad=15, fontweight='bold', color='darkblue')
        # ax.set_zlabel('F1 (%)', fontsize=12, labelpad=2, fontweight='bold', color='darkblue', rotation=180)
        
        # 设置刻度标签
        ax.set_xticks(x_pos)
        display_pre_modes = ['None', 'Reconstruction', 'Contrastive', 'Hybrid']
        ax.set_xticklabels(display_pre_modes, rotation=0)
        display_memory_selectors = ['Random', 'Uncertainty', 'Herding']
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_memory_selectors)
        # ax.text(x=4, y=0, z=100, 
        # s='F1 (%)',
        # rotation=180,  # rotation 在这里会生效
        # fontsize=12, fontweight='bold', color='darkblue')
        # ax.set_zlabel('F1 (%)', fontsize=12, labelpad=10, fontweight='bold', color='darkblue')
        
        # 设置标题，包含最优组合信息
        best_pre_mode = display_pre_modes[best_x_idx]
        best_memory_selector = display_memory_selectors[best_y_idx]
        title = f'{dataset.upper()} + {base_learner.upper()}\nBest: {best_pre_mode} + {best_memory_selector} (F1: {best_f1_value*100:.1f}%)'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20, color='darkred')
        
        # 调整视角
        ax.view_init(elev=20, azim=45)
        
        # 设置z轴范围
        ax.set_zlim(0, 1)
        ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_zticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        # 网格线
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # ax.set_facecolor('#f0f0f0')  # 设置灰色背景，能突出3D效果
        
        # 调整布局，给z轴标签留出更多空间
        plt.subplots_adjust(right=0.80, bottom=0.15, left=0.10, top=0.9)
        
        # 保存单独的图片
        filename = f'{dataset}_{base_learner}_3d_chart.png'
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        
        # 关闭当前图形以释放内存
        plt.close()

print(f"\nAll individual charts saved in folder: {output_folder}")


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