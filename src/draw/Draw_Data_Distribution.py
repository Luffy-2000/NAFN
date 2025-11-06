import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
import numpy as np

# Class name mapping
name = {
    "CSE-CIC-IDS2018":["DoS GoldenEye", "DoS Hulk", "DDoS HOIC", "Bot", "Bruteforce-SSH", "Benign", "DDoS LOIC-HTTP", "DoS Slowloris", "Bruteforce-Web", "DDoS LOIC-UDP", "Bruteforce-XSS", "SQL Injection"],
    "IoT-NID":["Benign", "DoS SYN Flooding", "Mirai ACK Flooding", "Port Scanning", "Mirai HTTP Flooding", "Host Discovery", "Mirai UDP Flooding", "OS Detection", "MITM ARP Spoofing", "Mirai Host Bruteforce"],
    "Edge-IIoT":["Benign", "Password Attack", "DDoS UDP Flooding", "DDoS ICMP Flooding", "DDoS HTTP Flooding", "DDoS TCP SYN Flooding", "Port Scanning", "Uploading Attack", "SQL Injection", "Vulnerability scanner", "XSS", "Backdoor", "Ransomware Attack", "OS Fingerprinting", "MITM Attack"]
}

def is_benign(label):
    """Check if a label represents benign traffic"""
    label = str(label).strip()
    return label == '0,Normal,Normal,Normal' or label.lower() == 'benign'

def read_and_count_labels(file_path, label_column='LABEL'):
    """Read dataset and count label distribution"""
    print(f"\nReading file: {os.path.basename(file_path)}")
    
    try:
        # Read only label column using pyarrow
        table = pq.read_table(file_path, columns=[label_column])
        # Convert to pandas DataFrame first
        df = table.to_pandas()
        # Handle the label column
        if label_column in df.columns:
            # Convert to string and handle any special characters
            df[label_column] = df[label_column].astype(str).str.strip()
            # Count unique labels and sort by count in descending order
            label_counts = df[label_column].value_counts().sort_values(ascending=False)
            
            print("\nLabel distribution:")
            print(label_counts)
            return label_counts
        else:
            print(f"Column {label_column} not found in dataset")
            return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        # Print more detailed error information
        import traceback
        print(traceback.format_exc())
        return None

def get_attack_type(label):
    """Get the type of attack from the label"""
    label = str(label).lower()
    if 'benign' in label or 'normal' in label:
        return 'benign'
    elif 'ddos' in label or 'ackflood' in label or 'httpflood' in label or 'udpflood' in label:
        return 'ddos'
    elif 'dos' in label or 'synflood' in label:
        return 'dos'
    elif 'bruteforce' in label or 'brute' in label:
        return 'bruteforce'
    elif 'bot' in label:
        return 'bot'
    elif 'sql' in label:
        return 'sql'
    elif 'xss' in label:
        return 'xss'
    elif 'scan' in label:
        return 'scan'
    elif 'mitm' in label:
        return 'mitm'
    else:
        print(f"Other type label: {label}")
        return 'other'

# def plot_all_distributions(label_counts_dict):
#     """Plot all dataset distributions in one figure"""
#     # Set style
#     plt.style.use('seaborn')
    
#     # Create figure with white background
#     fig, axes = plt.subplots(3, 1, figsize=(6, 10), facecolor='white')
    
#     # Find the maximum count across all datasets
#     max_count = max([max(counts.values) for counts in label_counts_dict.values()])
    
#     # Define patterns for different attack types
#     patterns = {
#         'ddos': '/',
#         'dos': '\\'
#     }
    
#     # Count attack types across all datasets
#     all_attack_types = {}
#     for label_counts in label_counts_dict.values():
#         if label_counts is None:
#             continue
#         for label in label_counts.index:
#             attack_type = get_attack_type(label)
#             all_attack_types[attack_type] = all_attack_types.get(attack_type, 0) + 1
    
#     print("Attack types across all datasets:", all_attack_types)
    
#     # Plot each dataset
#     for i, ((title, label_counts), ax) in enumerate(zip(label_counts_dict.items(), axes)):
#         if label_counts is None:
#             continue
            
#         # Get friendly names for the dataset
#         friendly_names = name.get(title, [])
        
#         # Create color list based on whether the label is benign or not
#         colors = ['#2ecc71' if is_benign(label) else '#e74c3c' for label in label_counts.index]
        
#         # Create horizontal bar plot with reversed y-axis
#         y_pos = np.arange(len(label_counts))
#         bars = ax.barh(y_pos, label_counts.values, color=colors, height=1.0, edgecolor='white', linewidth=0.5)
        
#         # Add patterns to bars based on attack type
#         for bar, label in zip(bars, label_counts.index):
#             attack_type = get_attack_type(label)
#             # Only add pattern for DoS and DDoS attacks
#             if attack_type in ['dos', 'ddos']:
#                 bar.set_hatch(patterns[attack_type])
        
#         # Set y-axis labels and reverse the order
#         ax.set_yticks(y_pos)
#         # Use friendly names if available, otherwise use original labels
#         y_labels = [friendly_names[i] if i < len(friendly_names) else label for i, label in enumerate(label_counts.index)]
#         ax.set_yticklabels(y_labels, fontsize=12)
#         ax.invert_yaxis()  # This will put the highest count at the top
        
#         # Set x-axis to log scale with same limits for all subplots
#         ax.set_xscale('log')
#         ax.set_xlim(1, max_count * 2.0)  # Add 10% padding
        
#         # Add labels
#         if i == 2:  # Only show x-axis label for the bottom subplot
#             ax.set_xlabel('Number of Biflows [log]', fontsize=14, labelpad=10)
#             ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
#         else:
#             ax.set_xlabel('')
#             ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            
#         ax.set_ylabel(title, fontsize=14, labelpad=10, fontweight='bold')
        
#         # Add grid
#         ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        
#         # Remove top and right spines
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
        
#         # Add value labels on the bars
#         for i, v in enumerate(label_counts.values):
#             ax.text(v, i, f' {v:,}', va='center', fontsize=11)
            
#         # Hide all x-axis ticks and labels
#         # ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    
#     # Adjust layout with minimal spacing between subplots
#     plt.subplots_adjust(hspace=0.05)
    
#     # Save figure with high DPI
#     if not os.path.exists('./PDF'):
#         os.makedirs('./PDF')
#     plt.savefig('./PDF/data_distribution_all.pdf', bbox_inches='tight', facecolor='white')
#     plt.close()

def plot_all_distributions(label_counts_dict):
    """Plot all dataset distributions in one figure"""
    # 保留原来的风格设置
    plt.style.use('seaborn')
    
    # Create figure with white background
    fig, axes = plt.subplots(3, 1, figsize=(6, 10), facecolor='white')
    
    # Find the maximum count across all datasets（容错处理，防止为空）
    valid_counts = [counts for counts in label_counts_dict.values() if counts is not None and len(counts) > 0]
    if not valid_counts:
        print("No valid label distributions to plot.")
        return
    max_count = max([max(counts.values) for counts in valid_counts])
    
    # 取消：按攻击类型统计与添加阴影（原先针对 DoS/DDoS 的逻辑已移除）
    
    # Plot each dataset
    for ax_idx, ((title, label_counts), ax) in enumerate(zip(label_counts_dict.items(), axes)):
        if label_counts is None or len(label_counts) == 0:
            ax.axis('off')
            continue
            
        # Get friendly names for the dataset
        friendly_names = name.get(title, [])
        
        # Create color list based on whether the label is benign or not
        colors = ['#2ecc71' if is_benign(label) else '#e74c3c' for label in label_counts.index]
        
        # Create horizontal bar plot with reversed y-axis
        y_pos = np.arange(len(label_counts))
        bars = ax.barh(y_pos, label_counts.values, color=colors, height=1.0, edgecolor='black', linewidth=1.2)
        
        # ===== 新逻辑：给每个数据集的“最后三个类”加阴影 =====
        n = len(label_counts)
        start_idx = max(0, n - 3)  # 若不足3类，则从0开始
        for j, bar in enumerate(bars):
            if j >= start_idx:
                bar.set_hatch('XX')  # 需要其他样式可改成 '///', 'xx', '++' 等
        
        # Set y-axis labels and reverse the order
        ax.set_yticks(y_pos)
        # Use friendly names if available, otherwise use original labels
        y_labels = [friendly_names[i] if i < len(friendly_names) else label for i, label in enumerate(label_counts.index)]
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.invert_yaxis()  # This will put the highest count at the top
        
        # Set x-axis to log scale with same limits for all subplots
        ax.set_xscale('log')
        ax.set_xlim(1, max_count * 2.0)  # 适当留白
        
        # Add labels
        if ax_idx == 2:  # Only show x-axis label for the bottom subplot
            ax.set_xlabel('Number of Biflows [log]', fontsize=14, labelpad=10)
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            
        ax.set_ylabel(title, fontsize=14, labelpad=10, fontweight='bold')
        
        # Add grid
        # ax.set_axisbelow(False)
        # ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.8, zorder=5)
        # # ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.2, zorder=5)
        ax.set_axisbelow(False)
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=1.0, zorder=5, color='gray')
        ax.grid(False, axis='y')  # hide y-axis gridlines

        # Add value labels on the bars
        for i_row, v in enumerate(label_counts.values):
            ax.text(v, i_row, f' {v:,}', va='center', fontsize=11)
    
    # Adjust layout with minimal spacing between subplots
    plt.subplots_adjust(hspace=0.05)
    
    # Save figure with high DPI
    if not os.path.exists('./PDF'):
        os.makedirs('./PDF')
    plt.savefig('./PDF/data_distribution_all_without_y_axis.pdf', bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    # Dataset paths and their label columns
    datasets = {
        'CSE-CIC-IDS2018': {
            'path': '../data/cic2018/cic2018_dataset_df_no_obf_20pkts_6feats_median_sampled_no_infiltration_clean_330ts.parquet',
            'label_column': 'LABEL_FULL'
        },
        'Edge-IIoT': {
            'path': '../data/edge_iiot/edge-iiot_100pkts_6f_1p-mt100k_benign_class_clean.parquet',
            'label_column': 'LABEL'
        },
        'IoT-NID': {
            'path': '../data/iot_nid/iot-nidd_100pkts_6f_clean.parquet',
            'label_column': 'LABEL'
        }
    }
    
    # Read and count each dataset
    label_counts_dict = {}
    for name, info in datasets.items():
        try:
            label_counts = read_and_count_labels(info['path'], info['label_column'])
            label_counts_dict[name] = label_counts
        except Exception as e:
            print(f"Error processing {name} dataset: {str(e)}")
    
    # Plot all distributions in one figure
    plot_all_distributions(label_counts_dict)

if __name__ == "__main__":
    main()
