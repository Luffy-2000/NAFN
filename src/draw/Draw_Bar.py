import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
datasets = ['CIC2018', 'Edge-IIoT', 'IoT-NID']

# NN results
nn_f1_means_5 = [95.84, 80.51, 91.71]
nn_f1_stds_5 = [1.24, 1.15, 1.96]
nn_f1_means_10 = [97.07, 80.66, 93.38]
nn_f1_stds_10 = [0.76, 0.92, 1.14]

# LR results
lr_f1_means_5 = [92.71, 77.16, 90.38]
lr_f1_stds_5 = [2.01, 1.11, 2.10]
lr_f1_means_10 = [92.98, 80.17, 91.95]
lr_f1_stds_10 = [1.59, 0.94, 1.43]

# Set font sizes
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Function to create a single plot
def create_plot(means_5, stds_5, means_10, stds_10, title, filename):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = plt.bar(x - width/2, means_5, width, yerr=stds_5, capsize=10, label='5-shot', color='skyblue')
    bars2 = plt.bar(x + width/2, means_10, width, yerr=stds_10, capsize=10, label='10-shot', color='lightcoral')

    plt.ylabel('F1-All Macro (%)', fontsize=16)
    # plt.title(title, fontsize=18)
    plt.xticks(x, datasets)
    plt.legend()

    # Add value labels
    for bars, stds in [(bars1, stds_5), (bars2, stds_10)]:
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            text_height = height + std + 0.2
            plt.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, text_height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

    # Set y-axis limits
    min_y = min(min(means_5), min(means_10)) - 2
    plt.ylim(min_y, 100)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Create NN plot
create_plot(nn_f1_means_5, nn_f1_stds_5, nn_f1_means_10, nn_f1_stds_10, 
           'NN Performance', 'sensitive_analysis_NN.pdf')

# Create LR plot
create_plot(lr_f1_means_5, lr_f1_stds_5, lr_f1_means_10, lr_f1_stds_10, 
           'LR Performance', 'sensitive_analysis_LR.pdf') 
