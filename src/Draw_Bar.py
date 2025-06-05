import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
datasets = ['CIC2018', 'Edge-IIoT', 'IoT-NID']
f1_means_5 = [95.84, 80.51, 91.71]
f1_stds_5 = [1.24, 1.15, 1.96]
f1_means_10 = [97.07, 80.66, 93.38]
f1_stds_10 = [0.76, 0.92, 1.14]

x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, f1_means_5, width, yerr=f1_stds_5, capsize=5, label='5-shot', color='skyblue')
bars2 = ax.bar(x + width/2, f1_means_10, width, yerr=f1_stds_10, capsize=5, label='10-shot', color='lightcoral')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1-All Macro (%)')
ax.set_title('Performance across Different Shot Settings')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

# Add value labels with dynamic height based on error bars
for bars, stds in [(bars1, f1_stds_5), (bars2, f1_stds_10)]:
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        text_height = height + std + 0.2
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, text_height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# Adjust y-axis to start closer to min value
min_y = min(min(f1_means_5), min(f1_means_10)) - 2
ax.set_ylim(min_y, 100)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('sensitive_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close() 
