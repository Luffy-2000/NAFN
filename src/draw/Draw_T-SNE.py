import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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


def plot_four_subplots_queries(four_folders, output_dir):
    """
    Plot four subplots for four folders (nn-noise, nn-denoise, lr-noise, lr-denoise).
    
    Parameters:
        - four_folders (list): List of four folder paths in order
        - output_dir (str): Directory to save the PDF file
    """
    # Extract dataset name from first folder
    match = re.search(re.compile(r'cic2018|edge_iiot|iot_nid'), four_folders[0])
    if match:
        dataset = match.group(0)
    else:
        dataset = 'unknown'
    
    # We'll create labels dynamically based on actual data
    labels = None  # Will be created when we know the actual number of classes
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    sn.set_theme(font_scale=1.0)
    
    # Subplot titles
    subplot_titles = ['NN-Noise', 'NN-Denoise', 'LR-Noise', 'LR-Denoise']
    
    # Store all unique labels for global legend
    all_labels = set()
    
    for i, folder_path in enumerate(four_folders):
        print(f"Processing {folder_path}")
        try:
            # Construct file paths
            query_file_supports = os.path.join(folder_path, 'lightning_logs', 'version_0', 'adaptation_data', 'supports.npz')
            query_file_queries = os.path.join(folder_path, 'lightning_logs', 'version_0', 'adaptation_data', 'queries.npz')
            query_file_labels = os.path.join(folder_path, 'lightning_logs', 'version_0', 'adaptation_data', 'labels.npz')
            
            # Check if files exist
            if not all(os.path.exists(f) for f in [query_file_supports, query_file_queries, query_file_labels]):
                print(f"Missing files in {folder_path}")
                axes[i].text(0.5, 0.5, 'Missing Data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(subplot_titles[i], fontsize=12, weight='bold')
                continue
            
            # Read the features from npz files
            df_supports = np.load(query_file_supports)['supports']
            df_support_labels = np.load(query_file_labels)['support_labels']

            # Flatten the data for processing
            supports_flat = df_supports.reshape(-1, df_supports.shape[-1])
            support_labels_flat = df_support_labels.flatten()

            # Create label mapping
            unique_labels = np.unique(support_labels_flat)
            n_classes = len(unique_labels)
            print(f"  Number of classes: {n_classes}")
            # exit()
            # For support plots, only use the last 3 classes (new classes)
            if len(unique_labels) >= 3:
                # Get the last 3 unique labels
                last_3_labels = unique_labels[-3:]
                print(f"  Using last 3 classes: {last_3_labels}")
                # Filter data to only include these 3 classes
                mask = np.isin(support_labels_flat, last_3_labels)
                supports_flat = supports_flat[mask]
                support_labels_flat = support_labels_flat[mask]
                unique_labels = last_3_labels
                n_classes = 3
                print(f"  Filtered data shape: {supports_flat.shape}, labels: {len(support_labels_flat)}")
            else:
                print(f"  Warning: Only {len(unique_labels)} classes found, using all classes")
            
            # Create labels dynamically based on dataset and actual number of classes
            if dataset == 'cic2018':
                # CIC2018: use lowercase letters
                labels = [chr(ord('a') + i) for i in range(n_classes)]
            elif dataset == 'edge_iiot':
                # Edge-IIoT: use script style letters
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
            
            label_map = {label: labels[j] for j, label in enumerate(unique_labels)}
            support_labels_mapped = [label_map[label] for label in support_labels_flat]
            
            # Store labels for global legend
            all_labels.update(support_labels_mapped)
            
            # Apply t-SNE with adjusted perplexity for smaller datasets
            n_samples = len(supports_flat)
            # perplexity = min(30, max(5, n_samples // 4))  # Adjust perplexity based on sample size
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_result = tsne.fit_transform(supports_flat)
            
            # Create DataFrame for plotting
            plot_data = pd.DataFrame({
                'x': tsne_result[:, 0],
                'y': tsne_result[:, 1],
                'label': support_labels_mapped
            })
            
            hue_order = sorted(plot_data['label'].unique().tolist())
            palette = sn.color_palette(None, n_colors=len(hue_order))
            
            # Plot KDE only
            sn.kdeplot(data=plot_data, x="x", y="y", hue="label", fill=True,
                      levels=5, thresh=0.1, alpha=0.5, warn_singular=False, bw_method=0.5,
                      hue_order=hue_order, palette=palette, ax=axes[i], legend=False)
            
            # Set subplot properties
            axes[i].set_title(subplot_titles[i], fontsize=12, weight='bold')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].tick_params(labelsize=8)
            
        except Exception as e:
            print(f"Error processing {folder_path}: {e}")
            axes[i].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(subplot_titles[i], fontsize=12, weight='bold')
    
    # Create global legend at the top
    if all_labels:
        sorted_labels = sorted(list(all_labels))
        n_unique_labels = len(sorted_labels)
        palette = sn.color_palette(None, n_colors=n_unique_labels)
        
        # For KDE, use patches
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(facecolor=palette[i], alpha=0.6, linewidth=0.1)
            for i, _ in enumerate(sorted_labels)
        ]
        
        # Add legend at the top, moved higher and in 2 rows
        fig.legend(legend_handles, sorted_labels, 
                  loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                  ncol=(n_unique_labels + 1) // 2, fontsize=10,  # 2 rows
                  prop={'weight': 'bold'})
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)  # Make more room for legend
    
    # Generate output filename
    base_name = f"supports_kde_four_subplots"
    output_file = os.path.join(output_dir, f"{base_name}_{dataset}.pdf")
    
    plt.savefig(output_file, bbox_inches="tight", facecolor='white', dpi=300)
    plt.close()
    
    print(f"Four subplots plot saved to: {output_file}")
    return output_file


def find_four_folders_for_dataset(dataset_name, base_dirs):
    """
    Find four folders for a dataset in order: nn-noise, nn-denoise, lr-noise, lr-denoise
    
    Parameters:
        - dataset_name (str): Dataset name (cic2018, edge_iiot, iot_nid)
        - base_dirs (list): List of base directories to search in
    
    Returns:
        - list: Four folder paths in the specified order
    """
    four_folders = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Base directory not found: {base_dir}")
            continue
            
        # Find all folders containing the dataset name and 0.5
        matching_folders = []
        for item in os.listdir(base_dir):
            if dataset_name in item and '0.5' in item:
                matching_folders.append(os.path.join(base_dir, item))
        
        # Sort folders to ensure consistent order
        matching_folders.sort()
        
        # Categorize folders by type
        nn_noise_folders = [f for f in matching_folders if 'nn' in f and 'noise' in f and 'denoise' not in f]
        nn_denoise_folders = [f for f in matching_folders if 'nn' in f and 'denoise' in f]
        lr_noise_folders = [f for f in matching_folders if 'lr' in f and 'noise' in f and 'denoise' not in f]
        lr_denoise_folders = [f for f in matching_folders if 'lr' in f and 'denoise' in f]
        
        print(f"  Found {len(nn_noise_folders)} NN-Noise folders:")
        for f in nn_noise_folders:
            print(f"    - {os.path.basename(f)}")
        print(f"  Found {len(nn_denoise_folders)} NN-Denoise folders:")
        for f in nn_denoise_folders:
            print(f"    - {os.path.basename(f)}")
        print(f"  Found {len(lr_noise_folders)} LR-Noise folders:")
        for f in lr_noise_folders:
            print(f"    - {os.path.basename(f)}")
        print(f"  Found {len(lr_denoise_folders)} LR-Denoise folders:")
        for f in lr_denoise_folders:
            print(f"    - {os.path.basename(f)}")
        
        # Add the first matching folder of each type
        if nn_noise_folders:
            four_folders.append(nn_noise_folders[0])
        if nn_denoise_folders:
            four_folders.append(nn_denoise_folders[0])
        if lr_noise_folders:
            four_folders.append(lr_noise_folders[0])
        if lr_denoise_folders:
            four_folders.append(lr_denoise_folders[0])
    
    return four_folders

if __name__ == "__main__":
    # Base directories to search in
    base_dirs = [
        "../save_files/results_rfs_student_bestcombo_noise_new",
        "../save_files/results_rfs_student_bestcombo_ProtoMargin_denoise"
    ]
    
    # Find four folders for each dataset
    datasets = ['cic2018', 'edge_iiot', 'iot_nid']
    all_datasets_folders = {}
    
    for dataset in datasets:
        print(f"Searching for {dataset} folders...")
        four_folders = find_four_folders_for_dataset(dataset, base_dirs)
        all_datasets_folders[dataset] = four_folders
        
        print(f"Found {len(four_folders)} folders for {dataset}:")
        for i, folder in enumerate(four_folders):
            print(f"  {i+1}. {os.path.basename(folder)}")
        print()
    
    # Create output directory
    output_dir = "./PDF"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot four subplots for each dataset
    for dataset_name, four_folders in all_datasets_folders.items():
        if len(four_folders) == 4:
            print(f"Processing {dataset_name}...")
            plot_four_subplots_queries(four_folders, output_dir)
        else:
            print(f"Warning: Found only {len(four_folders)} folders for {dataset_name}, skipping...")

    print(f"\nAll four-subplot visualizations have been plotted and saved to {output_dir}/")