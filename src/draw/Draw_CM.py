import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import json
from pathlib import Path
import re

def load_label_names(classes_info_path, label_conv_path):
    """Load class names and determine old/new class split"""
    with open(classes_info_path, 'r') as f:
        class_info = json.load(f)
    with open(label_conv_path, 'r') as f:
        label_map = eval(f.read())
    
    # Extract keys from old_classes and new_classes (convert string to int)
    old_keys = list(map(int, class_info['old_classes'].keys()))
    new_keys = list(map(int, class_info['new_classes'].keys()))

    # Find corresponding names based on label_map values
    inv_label_map = {v: k for k, v in label_map.items()}

    old_labels = [inv_label_map[k] for k in old_keys]
    new_labels = [inv_label_map[k] for k in new_keys]

    old_labels = [label.replace('_', ' ') for label in old_labels]
    new_labels = [label.replace('_', ' ') for label in new_labels]
    
    # Concatenate old + new to get complete class names
    all_labels = old_labels + new_labels

    return all_labels, len(old_labels)


def plot_confusion_matrix_from_csv(csv_path, output_dir):
    """
    Read confusion matrix from CSV file and plot it as a heatmap.
    
    Parameters:
        - csv_path (str): Path to the cm_adaptation_data_raw.csv file
        - output_dir (str): Directory to save the PDF file
    """
    # Read the confusion matrix CSV
    df_cm_raw = pd.read_csv(csv_path, index_col=0)
    print(f"Confusion matrix shape: {df_cm_raw.shape}")
    print(f"Confusion matrix:\n{df_cm_raw}")
    
    # Extract dataset name from path
    path_parts = csv_path.split(os.sep)
    dataset = None

    # pattern = re.compile(r'cic2018|edge_iiot|iot_nid')
    match = re.search(re.compile(r'cic2018|edge_iiot|iot_nid'), csv_path)
    if match:
        dataset = match.group(0)
    print(f"dataset: {dataset}")
    # exit()
    if dataset is None:
        dataset = 'unknown'
    
    # Create different style labels based on dataset
    n_classes = len(df_cm_raw)
    print("dataset:", dataset)
    if dataset == 'cic2018':
        # CIC2018: use lowercase letters
        labels = [chr(ord('a') + i) for i in range(n_classes)]
    elif dataset == 'edge_iiot':
        # Edge-IIoT: use uppercase letters
        labels = [f'$\\mathscr{{{chr(ord("A") + i)}}}$' for i in range(n_classes)]
    elif dataset == 'iot_nid':
        # IoT-NID: use Greek letters
        greek_letters = ['$\\alpha$', '$\\beta$', '$\\gamma$', '$\\delta$', '$\\epsilon$', '$\\zeta$', 
                        '$\\eta$', '$\\theta$', '$\\iota$', '$\\kappa$', '$\\lambda$', '$\\mu$',
                        '$\\nu$', '$\\xi$', '$\\omicron$', '$\\pi$', '$\\rho$', '$\\sigma$',
                        '$\\tau$', '$\\upsilon$', '$\\phi$', '$\\chi$', '$\\psi$', '$\\omega$']
        labels = [greek_letters[i] if i < len(greek_letters) else f'$\\alpha_{{{i-len(greek_letters)+1}}}$' for i in range(n_classes)]
    else:
        # Default: use lowercase letters
        labels = [chr(ord('a') + i) for i in range(n_classes)]
    
    # Draw line after the last 3 classes (assuming last 3 are new classes)
    line_x = n_classes - 3
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sn.set_theme(font_scale=1.5)  # Increased font scale
    
    ax = sn.heatmap(
        df_cm_raw, 
        annot=False,  # Don't show values in cells
        fmt='.1f',   # Format for annotations
        cmap='jet',
        xticklabels=labels, 
        yticklabels=labels,
        # cbar_kws={'label': 'Count'}
    )
    
    # Rotate x-axis labels with larger font and bold
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=30, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=30, weight='bold')
    
    # Add vertical and horizontal lines to separate old and new classes
    if line_x < n_classes:
        plt.axvline(x=line_x, linewidth=7, color='white', linestyle='-')
        plt.axhline(y=line_x, linewidth=7, color='white', linestyle='-')
    
    plt.xlabel('Predicted Label', fontsize=30, weight='bold')
    plt.ylabel('Actual Label', fontsize=30, weight='bold')
    # plt.title(f'Confusion Matrix - {dataset.upper()}', fontsize=18, weight='bold')
    
    plt.tight_layout()
    
    # Generate output filename
    base_name = os.path.basename(csv_path).replace('.csv', '')
    output_file = os.path.join(output_dir, f"{base_name}_replotted.pdf")
    
    plt.savefig(output_file, bbox_inches="tight", facecolor='white', dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to: {output_file}")
    return output_file

def find_and_plot_all_cm_files(save_files_dir):
    """
    Find all cm_adaptation_data_raw.csv files in save_files directory and plot them.
    
    Parameters:
        - save_files_dir (str): Path to save_files directory
        - output_dir (str): Directory to save the PDF files
    """
    # Find all cm_adaptation_data_raw.csv files
    cm_files = []
    for dirs in os.listdir(save_files_dir):
        cm_files_path = os.path.join(save_files_dir, dirs, 'lightning_logs', 'version_0', 'img')
        cm_files.append(cm_files_path)

    # # Plot each confusion matrix
    for csv_path in cm_files:
        try:
            cm_file_name = os.path.join(csv_path, 'cm_adaptation_data.csv')
            plot_confusion_matrix_from_csv(cm_file_name, csv_path)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")


if __name__ == "__main__":
    # Example usage
    save_files_dir = "../save_files/results_rfs_student_bestcombo_noise_new"
    # output_dir = "./PDF"
    # Create PDF directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)
    
    # Find and plot all confusion matrices
    find_and_plot_all_cm_files(save_files_dir)
    
    print(f"\nAll confusion matrices have been plotted and saved to {save_files_dir}/")