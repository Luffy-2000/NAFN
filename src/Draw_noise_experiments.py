NN_results = {
    'CSE-CIC_IDS2017':{
        'without_noise':{
            'F1-ALL Macro':[[97.13, 97.13, 97.13, 97.13, 97.13],[0.85, 0.85, 0.85, 0.85, 0.85]],
            'F1-New Macro':[[93.51, 93.51, 93.51, 93.51, 93.51],[2.19, 2.19, 2.19, 2.19, 2.19]],
            'F1-Old Macro':[[98.33, 98.33, 98.33, 98.33, 98.33],[0.53, 0.53, 0.53, 0.53, 0.53]],
        },
        'Add_noise':{
            'F1-ALL Macro':[[95.08, 93.57, 92.72, 91.99, 90.02],[2.69, 2.78, 3.20, 3.01, 2.77]],
            'F1-New Macro':[[84.71, 79.81, 74.90, 70.78, 63.87],[10.49, 11.65, 12.83, 11.02, 10.93]],
            'F1-Old Macro':[[98.54, 98.98, 98.66, 98.60, 98.73],[0.48, 0.50, 0.48, 0.47, 0.48]],
        },
        'Denoise_LOF':{
            'F1-ALL Macro':[[95.47, 94.98, 93.52, 93.56, 93.96],[1.39, 1.41, 1.52, 1.78, 2.25]],
            'F1-New Macro':[[89.39, 88.24, 85.82, 83.47, 82.54],[3.44, 3.58, 3.80, 5.79, 8.01]],
            'F1-Old Macro':[[97.50, 97.02, 96.09, 96.49, 97.77],[0.95, 1.08, 1.23, 0.81, 0.91]],
        },

    },
    'Edge-IIoT':{
        'without_noise':{
            'F1-ALL Macro':[[77.93, 77.93, 77.93, 77.93, 77.93],[1.52, 1.52, 1.52, 1.52, 1.52]],
            'F1-New Macro':[[72.27, 72.27, 72.27, 72.27, 72.27],[3.26, 3.26, 3.26, 3.26, 3.26]],
            'F1-Old Macro':[[79.35, 79.35, 79.35, 79.35, 79.35],[1.48, 1.48, 1.48, 1.48, 1.48]],
        },
        'Add_noise':{
            'F1-ALL Macro':[[75.84, 72.31, 70.97, 70.71, 70.52],[4.28, 4.01, 3.54, 3.27, 2.85]],
            'F1-New Macro':[[54.45, 43.62, 36.28, 33.00, 25.30],[19.93, 18.70, 16.90, 15.00, 13.67]],
            'F1-Old Macro':[[81.19, 80.45, 79.64, 80.19, 81.82],[1.20, 1.44, 1.54, 1.20, 1.14]],
        },
        'Denoise_LOF':{
            'F1-ALL Macro':[[76.07, 74.98, 73.52, 74.27, 74.83],[1.85, 2.02, 4.66, 3.36, 3.36]],
            'F1-New Macro':[[72.71, 69.80, 51.25, 58.20, 62.36],[5.91, 6.80, 7.98, 5.55, 5.55]],
            'F1-Old Macro':[[76.91, 76.80, 79.08, 77.94, 79.94],[1.45, 1.37, 1.28, 1.28, 1.28]],
        },
    },
    'IoT-NID':{
        'without_noise':{
            'F1-ALL Macro':[[91.18, 91.18, 91.18, 91.18, 91.18],[1.67, 1.67, 1.67, 1.67, 1.67]],
            'F1-New Macro':[[83.24, 83.24, 83.24, 83.24, 83.24],[3.36, 3.36, 3.36, 3.36, 3.36]],
            'F1-Old Macro':[[94.59, 94.59, 94.59, 94.59, 94.59],[1.26, 1.26, 1.26, 1.26, 1.26]],
        },
        'Add_noise':{
            'F1-ALL Macro':[[89.23, 87.91, 86.38, 83.79, 81.93],[2.66, 2.74, 2.84, 2.99, 3.13]],      
            'F1-New Macro':[[76.65, 62.89, 67.73, 61.90, 53.76],[7.44, 8.42, 8.91, 8.42, 8.91]],
            'F1-Old Macro':[[94.61, 94.57, 94.37, 94.20, 94.01],[1.17, 1.13, 1.57, 1.13, 1.57]],
        },
        'Denoise_LOF':{
            'F1-ALL Macro':[[89.63, 88.45, 86.48, 84.55, 83.30],[2.90, 3.25, 3.59, 3.56, 3.45]],
            'F1-New Macro':[[74.77, 73.33, 69.68, 62.73,56.24],[8.43, 9.39, 10.63, 10.65, 10.70]],
            'F1-Old Macro':[[94.56, 94.12, 93.67, 94.30, 94.90],[1.11, 1.39, 0.99, 1.11, 0.99]],
        },
    }
}

LR_results = {
    'CSE-CIC_IDS2017':{
        'without_noise':{
            'F1-ALL Macro':[[94.72, 94.72, 94.72, 94.72, 94.72],[0.85, 0.85, 0.85, 0.85, 0.85]],
            'F1-New Macro':[[89.29, 89.29, 89.29, 89.29, 89.29],[2.81, 2.81, 2.81, 2.81, 2.81]],
            'F1-Old Macro':[[96.53, 96.53, 96.53, 96.53, 96.53],[0.60, 0.60, 0.60, 0.60, 0.60]],
        },
        'Add_noise':{
            'F1-ALL Macro':[[94.41, 93.17, 91.75, 91.05, 90.68],[1.14, 1.37, 2.16, 1.79, 1.61]],
            'F1-New Macro':[[87.79, 82.34, 78.32, 76.01, 73.66],[3.96, 5.29, 7.72, 6.94, 5.94]],
            'F1-Old Macro':[[96.62, 96.23, 96.35, 96.35, 96.35],[0.64, 0.59, 0.64, 0.44, 0.33]],
        },
        'Denoise_LOF':{
            'F1-ALL Macro':[[94.64, 94.60, 94.58, 94.23, 93.86],[0.92, 0.93, 0.94, 1.28, 1.59]],
            'F1-New Macro':[[89.55, 89.50, 89.45, 87.40, 86.31],[2.99, 3.15, 3.32, 4.87, 5.95]],
            'F1-Old Macro':[[96.33, 96.30, 96.29, 96.34, 96.37],[0.66, 0.64, 0.61, 0.63,0.63]],
        },
    },
    'Edge-IIoT':{
        'without_noise':{
            'F1-ALL Macro':[[79.01, 79.01, 79.01, 79.01, 79.01],[0.94, 0.94, 0.94, 0.94, 0.94]],
            'F1-New Macro':[[82.81, 82.81, 82.81, 82.81, 82.81],[1.15, 1.15, 1.15, 1.15, 1.15]],
            'F1-Old Macro':[[78.05, 78.05, 78.05, 78.05, 78.05],[1.08, 1.08, 1.08, 1.08, 1.08]],
        },
        'Add_noise':{
            'F1-ALL Macro':[[69.66, 69.12, 68.38, 68.39, 68.38,],[1.02, 1.01, 1.27, 1.46, 2.43]],
            'F1-New Macro':[[47.30, 47.44, 47.32, 41.61, 38.89],[1.46, 3.78, 5.91, 8.94, 11.60]],
            'F1-Old Macro':[[75.25, 74.80, 73.65, 74.51, 75.75],[1.16, 1.36, 1.64, 2.02, 2.40]],
        },
        'Denoise_LOF':{
            'F1-ALL Macro':[[72.70, 71.89, 70.43, 70.40, 70.39],[1.38, 2.17, 1.07, 1.36, 1.36]],
            'F1-New Macro':[[53.24, 51.28, 48.34, 47.89, 46.51],[3.32, 4.82, 1.39, 4.82, 4.82]],
            'F1-Old Macro':[[77.57, 76.89, 75.95, 75.91, 76.36],[2.70, 2.01, 1.16, 1.10, 1.51]],
        },
    },
    'IoT-NID':{
        'without_noise':{
            'F1-ALL Macro':[[91.10, 91.10, 91.10, 91.10, 91.10],[1.62, 1.62, 1.62, 1.62, 1.62]],
            'F1-New Macro':[[82.40, 82.40, 82.40, 82.40, 82.40],[3.67, 3.67, 3.67, 3.67, 3.67]],
            'F1-Old Macro':[[94.83, 94.83, 94.83, 94.83, 94.83],[1.02, 1.02, 1.02, 1.02, 1.02]],
        },
        'Add_noise':{
            'F1-ALL Macro':[[90.38, 90.27, 90.15, 88.91, 87.89],[1.60, 2.01, 2.25, 2.89, 3.70]],
            'F1-New Macro':[[83.13, 82.38, 81.47, 76.29, 71.94],[3.66, 4.75, 6.34, 8.22, 10.81]],
            'F1-Old Macro':[[94.21, 94.48, 94.89, 94.82, 94.73],[0.99, 1.00, 0.88, 1.12, 1.08]],
        },
        'Denoise_LOF':{
            'F1-ALL Macro':[[91.38, 91.27, 91.15, 90.18, 88.89],[1.60, 2.00, 2.25, 2.89, 3.70]],
            'F1-New Macro':[[83.13, 82.26, 81.47, 80.38, 79.94],[3.66, 4.19, 5.34, 5.55, 5.81]],
            'F1-Old Macro':[[94.91, 95.31, 94.73, 94.82, 94.73],[0.99, 0.88, 1.08, 1.12, 1.08]],
        },
    }
}

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = '#333333'

# Noise ratios
noise_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

# Colors for different datasets
colors = {
    'CSE-CIC_IDS2017': '#2E86AB',
    'Edge-IIoT': '#A23B72', 
    'IoT-NID': '#F18F01'
}

# Line styles
line_styles = {
    'F1-ALL Macro': '-',
    'F1-New Macro': '--'
}

# def create_dual_axis_plot(results, title, save_name):
#     """Create a dual-axis line plot for F1-ALL Macro and F1-New Macro"""
#     fig, ax1 = plt.subplots(figsize=(12, 8))
    
#     # Create second y-axis
#     ax2 = ax1.twinx()
    
#     # Plot data for each dataset
#     for dataset in ['CSE-CIC_IDS2017', 'Edge-IIoT', 'IoT-NID']:
#         color = colors[dataset]
        
#         # Extract F1-ALL Macro values (mean across 5 runs)
#         f1_all_values = []
#         for noise_ratio in noise_ratios:
#             # Use Add_noise data for noise ratios > 0
#             if noise_ratio == 0.1:
#                 values = results[dataset]['Add_noise']['F1-ALL Macro'][0]
#             elif noise_ratio == 0.2:
#                 values = results[dataset]['Add_noise']['F1-ALL Macro'][0]
#             elif noise_ratio == 0.3:
#                 values = results[dataset]['Add_noise']['F1-ALL Macro'][0]
#             elif noise_ratio == 0.4:
#                 values = results[dataset]['Add_noise']['F1-ALL Macro'][0]
#             elif noise_ratio == 0.5:
#                 values = results[dataset]['Add_noise']['F1-ALL Macro'][0]
#             f1_all_values.append(np.mean(values))
        
#         # Extract F1-New Macro values (mean across 5 runs)
#         f1_new_values = []
#         for noise_ratio in noise_ratios:
#             if noise_ratio == 0.1:
#                 values = results[dataset]['Add_noise']['F1-New Macro'][0]
#             elif noise_ratio == 0.2:
#                 values = results[dataset]['Add_noise']['F1-New Macro'][0]
#             elif noise_ratio == 0.3:
#                 values = results[dataset]['Add_noise']['F1-New Macro'][0]
#             elif noise_ratio == 0.4:
#                 values = results[dataset]['Add_noise']['F1-New Macro'][0]
#             elif noise_ratio == 0.5:
#                 values = results[dataset]['Add_noise']['F1-New Macro'][0]
#             f1_new_values.append(np.mean(values))
        
#         # Plot F1-ALL Macro on left y-axis
#         line1 = ax1.plot(noise_ratios, f1_all_values, 
#                         color=color, linewidth=3, marker='o', markersize=8,
#                         label=f'{dataset} (F1-ALL)', linestyle='-')
        
#         # Plot F1-New Macro on right y-axis
#         line2 = ax2.plot(noise_ratios, f1_new_values, 
#                         color=color, linewidth=3, marker='s', markersize=8,
#                         label=f'{dataset} (F1-New)', linestyle='--', alpha=0.8)
    
#     # Customize the plot
#     ax1.set_xlabel('Noise Ratio', fontsize=14, fontweight='bold')
#     ax1.set_ylabel('F1-ALL Macro Score (%)', fontsize=14, fontweight='bold', color='#2E86AB')
#     ax2.set_ylabel('F1-New Macro Score (%)', fontsize=14, fontweight='bold', color='#A23B72')
    
#     # Set title
#     ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
#     # Customize grid
#     ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
#     ax1.set_axisbelow(True)
    
#     # Set axis limits and ticks
#     ax1.set_xlim(0.05, 0.55)
#     ax1.set_xticks(noise_ratios)
#     ax1.set_xticklabels([f'{r:.1f}' for r in noise_ratios])
    
#     # Set y-axis limits based on data
#     ax1.set_ylim(65, 100)
#     ax2.set_ylim(25, 95)
    
#     # Customize tick colors
#     ax1.tick_params(axis='y', labelcolor='#2E86AB')
#     ax2.tick_params(axis='y', labelcolor='#A23B72')
    
#     # Add legend
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
    
#     # Combine legends
#     all_lines = lines1 + lines2
#     all_labels = labels1 + labels2
    
#     # Create custom legend
#     legend = ax1.legend(all_lines, all_labels, 
#                        loc='center left', bbox_to_anchor=(1.15, 0.5),
#                        fontsize=11, frameon=True, fancybox=True, shadow=True)
    
#     # Add a subtle background color
#     ax1.set_facecolor('#f8f9fa')
#     fig.patch.set_facecolor('white')
    
#     # Adjust layout
#     plt.tight_layout()
    
#     # Save the plot
#     # plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
#     plt.savefig(f'{save_name}.pdf', bbox_inches='tight', facecolor='white', dpi=300)
    
#     plt.show()

# # Create plots
# print("Creating NN plot...")
# create_dual_axis_plot(NN_results, 'Neural Network (NN) Performance vs Noise Ratio', 'NN_noise_performance')

# print("Creating LR plot...")
# create_dual_axis_plot(LR_results, 'Logistic Regression (LR) Performance vs Noise Ratio', 'LR_noise_performance')

# print("Plots saved as NN_noise_performance.png/pdf and LR_noise_performance.png/pdf")

def create_subplot_comparison(results, title, save_name):
    """Create subplot comparison with 3 datasets and 3 methods per subplot"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    datasets = ['CSE-CIC_IDS2017', 'Edge-IIoT', 'IoT-NID']
    methods = ['without_noise', 'Add_noise', 'Denoise_LOF']
    method_labels = ['Without Noise', 'Add Noise', 'Denoise LOF']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    
    # Noise ratios for x-axis (excluding 0.0 for Add_noise and Denoise_LOF)
    noise_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Define y-axis limits for each dataset
    if title == 'Neural Network (NN) Performance Comparison':
        y_limits = {
            'CSE-CIC_IDS2017': (90, 98),
            'Edge-IIoT': (65, 80),
            'IoT-NID': (80, 93)
        }
    else:
        y_limits = {
            'CSE-CIC_IDS2017': (90, 96),
            'Edge-IIoT': (65, 81),
            'IoT-NID': (80, 94)
        }
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        for j, method in enumerate(methods):
            color = colors[j]
            marker = markers[j]
            label = method_labels[j]
            
            # Extract F1-ALL Macro values and standard deviations
            f1_all_values = []
            f1_all_stds = []
            
            if method == 'without_noise':
                # For without noise, use the same value for all noise ratios
                f1_all_mean = np.mean(results[dataset][method]['F1-ALL Macro'][0])
                f1_all_std = np.mean(results[dataset][method]['F1-ALL Macro'][1])
                f1_all_values = [f1_all_mean] * len(noise_ratios)
                f1_all_stds = [f1_all_std] * len(noise_ratios)
            elif method == 'Add_noise':
                # For add noise, use the actual values corresponding to noise ratios
                f1_all_values = results[dataset][method]['F1-ALL Macro'][0]
                f1_all_stds = results[dataset][method]['F1-ALL Macro'][1]
            elif method == 'Denoise_LOF':
                # For denoise LOF, use the actual values corresponding to noise ratios
                f1_all_values = results[dataset][method]['F1-ALL Macro'][0]
                f1_all_stds = results[dataset][method]['F1-ALL Macro'][1]
            
            # Plot F1-ALL Macro with shaded error bands
            ax.plot(noise_ratios, f1_all_values, 
                   color=color, linewidth=3, marker=marker, markersize=8,
                   label=label, linestyle='-')
            
            # Add shaded error band
            ax.fill_between(noise_ratios, 
                           np.array(f1_all_values) - np.array(f1_all_stds),
                           np.array(f1_all_values) + np.array(f1_all_stds),
                           color=color, alpha=0.3)
        
        # Customize subplot
        ax.set_xlabel('Noise Ratio', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-ALL Macro Score (%)', fontsize=12, fontweight='bold')
        # ax.set_title(dataset, fontsize=14, fontweight='bold', pad=15)
        
        # Set grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set axis limits with dataset-specific y-range
        ax.set_xlim(0.05, 0.55)
        y_min, y_max = y_limits[dataset]
        ax.set_ylim(y_min, y_max)
        
        # Set x-ticks
        ax.set_xticks(noise_ratios)
        ax.set_xticklabels([f'{r:.1f}' for r in noise_ratios])
        
        # Add legend
        ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc='lower left')
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
    
    # Adjust layout
    # plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_name}.pdf', bbox_inches='tight', facecolor='white', dpi=300)
    plt.show()

# Create subplot comparison charts
print("Creating NN subplot comparison...")
create_subplot_comparison(NN_results, 'Neural Network (NN) Performance Comparison', 'NN_subplot_comparison')

print("Creating LR subplot comparison...")
create_subplot_comparison(LR_results, 'Logistic Regression (LR) Performance Comparison', 'LR_subplot_comparison')

print("Subplot comparison charts saved as NN_subplot_comparison.pdf and LR_subplot_comparison.pdf")



