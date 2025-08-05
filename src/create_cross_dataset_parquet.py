#!/usr/bin/env python3
"""
Cross-dataset replacement parquet file generation script
Replace test classes from different datasets to generate new parquet files
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from data.dataset_config import dataset_config



cic2018_label_map = {
            '0,Normal,Normal,Normal': 'Benign', 
            '1,Bot,Bot,Bot': 'Bot', 
            '1,BruteForce,SSH,SSH': 'BruteForce-SSH', 
            '1,BruteForce,Web,Web': 'BruteForce-Web',   
            '1,BruteForce,XSS,XSS': 'BruteForce-XSS', 
            '1,DDoS,HOIC,HOIC': 'DDoS HOIC', 
            '1,DDoS,LOIC,HTTP': 'DDoS LOIC-HTTP', 
            '1,DDoS,LOIC,UDP': 'DDoS LOIC-UDP', 
            '1,DoS,GoldenEye,GoldenEye': 'DoS GoldenEye', 
            '1,DoS,Hulk,Hulk': 'DoS Hulk', 
            '1,DoS,Slowloris,Slowloris': 'DoS Slowloris', 
            '1,SQLInjection,SQLInjection,SQLInjection': 'SQL Injection',
}
cic2018_reverse_map = {v: k for k, v in cic2018_label_map.items()}


def get_cross_dataset_config(mix_name):
    """
    Get cross-dataset mix configuration
    
    Parameters:
        - mix_name (str): mix configuration name
    
    Returns:
        - dict: configuration dictionary
    """
    # All possible cross-dataset mix configurations
    # Configure correctly according to classes_map_rename.txt and dataset_config.py
    cross_dataset_configs = {
        "iot_nid_with_cic2018_test": {
            "base_dataset": "iot_nid",
            "replacement_dataset": "cic2018",
            "base_train_classes": [2, 7, 0, 6],  # benign, synflooding, ackflooding, portscanning
            "base_val_classes": [4, 3, 9],       # httpflooding, hostdiscovery, udpflooding
            "base_test_classes": [5, 1, 8],      # osversiondetection, arpspoofing, telnetbruteforce
            "replacement_test_classes": [7, 4, 11],  # DDoS LOIC-UDP, BruteForce-XSS, SQL Injection
            "replacement_mapping": {5: 7, 1: 4, 8: 11}  # iot_nid test -> cic2018 test
        },
        "iot_nid_with_edge_iiot_test": {
            "base_dataset": "iot_nid",
            "replacement_dataset": "edge_iiot",
            "base_train_classes": [2, 7, 0, 6],  # benign, synflooding, ackflooding, portscanning
            "base_val_classes": [4, 3, 9],       # httpflooding, hostdiscovery, udpflooding
            "base_test_classes": [5, 1, 8],      # osversiondetection, arpspoofing, telnetbruteforce
            "replacement_test_classes": [10, 7, 6],  # Ransomware, OS_Fingerprinting, MITM
            "replacement_mapping": {5: 10, 1: 7, 8: 6}  # iot_nid test -> edge_iiot test
        },
        "cic2018_with_iot_nid_test": {
            "base_dataset": "cic2018",
            "replacement_dataset": "iot_nid",
            "base_train_classes": [8, 9, 5, 1, 2, 0],  # Benign, Bot, BruteForce-SSH, BruteForce-Web, BruteForce-XSS, DDoS HOIC
            "base_val_classes": [6, 10, 3],       # DDoS LOIC-HTTP, DoS Slowloris, BruteForce-Web
            "base_test_classes": [7, 4, 11],      # DDoS LOIC-UDP, DoS Hulk, SQL Injection
            "replacement_test_classes": [5, 1, 8],  # osversiondetection, arpspoofing, telnetbruteforce
            "replacement_mapping": {7: 5, 4: 1, 11: 8}  # cic2018 test -> iot_nid test
        },
        "cic2018_with_edge_iiot_test": {
            "base_dataset": "cic2018",
            "replacement_dataset": "edge_iiot",
            "base_train_classes": [8, 9, 5, 1, 2, 0],  # Benign, Bot, BruteForce-SSH, BruteForce-Web, BruteForce-XSS, DDoS HOIC
            "base_val_classes": [6, 10, 3],       # DDoS LOIC-HTTP, DoS Slowloris, BruteForce-Web
            "base_test_classes": [7, 4, 11],      # DDoS LOIC-UDP, DoS Hulk, SQL Injection
            "replacement_test_classes": [10, 7, 6],  # Ransomware, OS_Fingerprinting, MITM
            "replacement_mapping": {7: 10, 4: 7, 11: 6}  # cic2018 test -> edge_iiot test
        },
        "edge_iiot_with_iot_nid_test": {
            "base_dataset": "edge_iiot",
            "replacement_dataset": "iot_nid",
            "base_train_classes": [1, 8, 5, 3, 2, 4, 9, 12, 11],  # Benign, Password, DDoS_UDP_Flood, DDoS_ICMP_Flood, DDoS_HTTP_Flood, DDoS_TCP_SYN_Flood, Port_Scanning, Uploading, SQL_injection
            "base_val_classes": [13, 14, 0],       # Vulnerability_scanner, XSS, Backdoor
            "base_test_classes": [10, 7, 6],      # Ransomware, OS_Fingerprinting, MITM
            "replacement_test_classes": [5, 1, 8],  # osversiondetection, arpspoofing, telnetbruteforce
            "replacement_mapping": {10: 5, 7: 1, 6: 8}  # edge_iiot test -> iot_nid test
        },
        "edge_iiot_with_cic2018_test": {
            "base_dataset": "edge_iiot",
            "replacement_dataset": "cic2018",
            "base_train_classes": [1, 8, 5, 3, 2, 4, 9, 12, 11],  # Benign, Password, DDoS_UDP_Flood, DDoS_ICMP_Flood, DDoS_HTTP_Flood, DDoS_TCP_SYN_Flood, Port_Scanning, Uploading, SQL_injection
            "base_val_classes": [13, 14, 0],       # Vulnerability_scanner, XSS, Backdoor
            "base_test_classes": [10, 7, 6],      # Ransomware, OS_Fingerprinting, MITM
            "replacement_test_classes": [7, 4, 11],  # DDoS LOIC-UDP, BruteForce-XSS, SQL Injection
            "replacement_mapping": {10: 7, 7: 4, 6: 11}  # edge_iiot test -> cic2018 test
        }
    }
    if mix_name not in cross_dataset_configs:
        raise ValueError(f"Unknown mix name: {mix_name}. Available mixes: {list(cross_dataset_configs.keys())}")
    
    return cross_dataset_configs[mix_name]


def get_all_cross_dataset_mixes():
    """
    Get all available cross-dataset mix configuration names
    
    Returns:
        - list: list of mix configuration names
    """
    return [
        "iot_nid_with_cic2018_test",
        "iot_nid_with_edge_iiot_test", 
        "cic2018_with_iot_nid_test",
        "cic2018_with_edge_iiot_test",
        "edge_iiot_with_iot_nid_test",
        "edge_iiot_with_cic2018_test"
    ]


def load_dataset_parquet(dataset_name):
    """
    Load parquet file for specified dataset
    
    Parameters:
        - dataset_name (str): dataset name
    
    Returns:
        - pd.DataFrame: dataset DataFrame
    """
    dataset_path = dataset_config[dataset_name]['path']
    print(f"Loading {dataset_name} from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} samples from {dataset_name}")
    return df


def filter_and_replace_test_classes(df_base, df_replacement, config):
    """
    Filter and replace test classes
    
    Parameters:
        - df_base (pd.DataFrame): base dataset
        - df_replacement (pd.DataFrame): replacement dataset
        - config (dict): configuration dictionary
    
    Returns:
        - pd.DataFrame: dataset after replacement
    """
    base_dataset = config['base_dataset']
    replacement_dataset = config['replacement_dataset']
    base_train_classes = config['base_train_classes']
    base_val_classes = config['base_val_classes']
    base_test_classes = config['base_test_classes']
    replacement_test_classes = config['replacement_test_classes']
    replacement_mapping = config['replacement_mapping']
    
    # Get label column names
    base_label_column = dataset_config[base_dataset].get('label_column', 'LABEL')
    replacement_label_column = dataset_config[replacement_dataset].get('label_column', 'LABEL')
    
    print(f"Using base label column: {base_label_column}")
    print(f"Using replacement label column: {replacement_label_column}")
    # exit()
    # Load class mapping
    def load_class_map(dataset_name):
        class_map_path = os.path.join(os.path.dirname(dataset_config[dataset_name]['path']), 'classes_map_rename.txt')
        with open(class_map_path, 'r') as f:
            return eval(f.read())
    
    base_class_map = load_class_map(base_dataset)
    replacement_class_map = load_class_map(replacement_dataset)
    
    print(f"Base class map: {base_class_map}")
    print(f"Replacement class map: {replacement_class_map}")
    # Create reverse mapping: numeric labels -> string labels
    base_reverse_map = {v: k for k, v in base_class_map.items()}
    replacement_reverse_map = {v: k for k, v in replacement_class_map.items()}
    
    print(f"Base reverse map: {base_reverse_map}")
    print(f"Replacement reverse map: {replacement_reverse_map}")
    
    # Keep train and val classes from base dataset
    all_base_classes = base_train_classes + base_val_classes
    # Convert numeric labels to string labels for filtering
    all_base_classes_str = [base_reverse_map.get(cls, str(cls)) for cls in all_base_classes]
    print(f"Base train+val classes (string): {all_base_classes_str}")
    
    if base_dataset == 'cic2018':
        all_base_classes_str = [cic2018_reverse_map.get(cls, str(cls)) for cls in all_base_classes_str]
    # print(all_base_classes_str)
    df_base_filtered = df_base[df_base[base_label_column].isin(all_base_classes_str)].copy()


    print(f"Filtered {len(df_base_filtered)} samples from base dataset for train+val classes")

    # Get test classes from replacement dataset
    replacement_test_classes_str = [replacement_reverse_map.get(cls, str(cls)) for cls in replacement_test_classes]
    # print(f"Replacement test classes (string): {replacement_test_classes_str}")
    


    # For cic2018 dataset, need to handle label format differences
    if replacement_dataset == 'cic2018':
        # Handle cic2018 label format differences: map actual data format to classes_map_rename.txt format
        # Create reverse mapping: classes_map_rename.txt format -> actual data format
        # cic2018_reverse_map = {v: k for k, v in cic2018_label_map.items()}
        
        # Convert replacement_test_classes_str to actual data format
        actual_test_classes = []
        for class_name in replacement_test_classes_str:
            if class_name in cic2018_reverse_map:
                actual_test_classes.append(cic2018_reverse_map[class_name])
        
        print(f"Actual test classes in cic2018 data: {actual_test_classes}")
        
        # Filter replacement dataset
        df_replacement_filtered = df_replacement[df_replacement[replacement_label_column].isin(actual_test_classes)].copy()
    else:
        # For other datasets, directly use replacement_test_classes_str
        df_replacement_filtered = df_replacement[df_replacement[replacement_label_column].isin(replacement_test_classes_str)].copy()
    
    print(f"Filtered {len(df_replacement_filtered)} samples from replacement dataset for test classes")

    # Base dataset keeps original labels
    df_base_final = df_base_filtered.copy()
    # Task 1: Merge common columns from df_base_final and df_replacement_final
    # For cic2018 dataset, if LABEL column doesn't exist, create it and copy LABEL_FULL
    df_base_final = df_base_filtered.copy()
    df_replacement_final = df_replacement_filtered.copy()
    
    # Handle cic2018 LABEL column
    if replacement_dataset == 'cic2018' and 'LABEL' not in df_replacement_final.columns:
        df_replacement_final['LABEL'] = df_replacement_final['LABEL_FULL']
    if base_dataset == 'cic2018' and 'LABEL' not in df_base_final.columns:
        df_base_final['LABEL'] = df_base_final['LABEL_FULL']

    # Ensure both datasets have the same columns
    base_columns = set(df_base_final.columns)
    replacement_columns = set(df_replacement_final.columns)
    
    # Select common columns for merging
    common_columns = list(base_columns & replacement_columns)
    print(f"Common columns for merging: {common_columns}")

    # Select common columns for merging
    df_base_merged = df_base_final[common_columns]
    df_replacement_merged = df_replacement_final[common_columns]
    
    # Merge datasets
    merged_df = pd.concat([df_base_merged, df_replacement_merged], ignore_index=True)
    print(f"Total merged samples: {len(merged_df)}")

    # Handle numpy.ndarray objects in LABEL column
    def convert_label_to_string(label):
        if isinstance(label, (list, np.ndarray)):
            return ','.join(str(x) for x in label)
        return str(label)
    
    merged_df['LABEL'] = merged_df['LABEL'].apply(convert_label_to_string)
    print("LABEL column after conversion:")
    print(merged_df['LABEL'].head(10))
    
    # Task 2: Create ENC_LABEL column according to Base class map
    # Get all unique label values
    new_class_map = {}
    # new_class_map = base_reverse_map.copy()
    if base_dataset == 'cic2018':
        for k, v in base_reverse_map.items():
            new_class_map[k] = cic2018_reverse_map[v]
    else:
        new_class_map = base_reverse_map.copy()

    for k, v in replacement_mapping.items():
        if replacement_dataset == 'cic2018':
            new_class_map[k] = cic2018_reverse_map[replacement_reverse_map[v]]    
        else:
            new_class_map[k] = replacement_reverse_map[v]
    enc_label_map = {v: k for k, v in new_class_map.items()}

    print(f"enc_label_map: {enc_label_map}")
    # Create ENC_LABEL column, ensure data type is consistent with le.transform()
    merged_df['ENC_LABEL'] = merged_df['LABEL'].map(enc_label_map).astype('int64')
    # Check final label distribution
    label_counts = merged_df['ENC_LABEL'].value_counts().sort_index()
    print(f"Final ENC_LABEL distribution: {label_counts.to_dict()}")

    return merged_df


def create_cross_dataset_parquet(mix_name, output_dir="../data"):
    """
    Create cross-dataset replacement parquet file
    
    Parameters:
        - mix_name (str): mix configuration name
        - output_dir (str): output directory
    """
    print(f"\n{'='*60}")
    print(f"Creating cross dataset parquet for: {mix_name}")
    print(f"{'='*60}")
    
    # Get mix configuration
    config = get_cross_dataset_config(mix_name)
    base_dataset = config['base_dataset']
    replacement_dataset = config['replacement_dataset']
    
    print(f"Base dataset: {base_dataset}")
    print(f"Replacement dataset: {replacement_dataset}")
    print(f"Base train classes: {config['base_train_classes']}")
    print(f"Base val classes: {config['base_val_classes']}")
    print(f"Base test classes: {config['base_test_classes']}")
    print(f"Replacement test classes: {config['replacement_test_classes']}")
    print(f"Replacement mapping: {config['replacement_mapping']}")
    
    # Load base dataset
    df_base = load_dataset_parquet(base_dataset)
    
    # Load replacement dataset
    df_replacement = load_dataset_parquet(replacement_dataset)
    
    # Filter and replace test classes
    merged_df = filter_and_replace_test_classes(df_base, df_replacement, config)
    
    # Create output directory - following _mix_ format
    mix_dir_name = f"{base_dataset}_mix_{replacement_dataset}"
    output_dir = os.path.join(output_dir, mix_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parquet file
    output_path = os.path.join(output_dir, f"{mix_name}_replaced.parquet")
    merged_df.to_parquet(output_path, index=False)
    print(f"Saved replaced dataset to: {output_path}")

    # === Generate new classes_map_rename.txt ===
    # 1. Get class name mapping
    def load_class_map(dataset_name):
        class_map_path = os.path.join(os.path.dirname(dataset_config[dataset_name]['path']), 'classes_map_rename.txt')
        with open(class_map_path, 'r') as f:
            return eval(f.read())
    base_class_map = load_class_map(base_dataset)
    replacement_class_map = load_class_map(replacement_dataset)

    # 2. Build new class order
    new_label_order = config['base_train_classes'] + config['base_val_classes'] + config['base_test_classes']
    new_test_labels = list(config['base_test_classes'])
    replacement_test_labels = list(config['replacement_test_classes'])
    replacement_mapping = config['replacement_mapping']
    # 3. Build new class names
    new_class_map = {}
    for label in config['base_train_classes'] + config['base_val_classes']:
        # train/val classes use base dataset names
        for k, v in base_class_map.items():
            if v == label:
                new_class_map[k] = label
                break
    # test classes use replacement dataset names, labels use base test labels
    for base_label, replacement_label in replacement_mapping.items():
        for k, v in replacement_class_map.items():
            if v == replacement_label:
                new_class_map[k] = base_label
                break
    # 4. Save
    rename_path = os.path.join(output_dir, f"classes_map_rename.txt")
    with open(rename_path, 'w') as f:
        f.write(str(new_class_map))
    print(f"Saved new classes_map_rename.txt to: {rename_path}")

    # Save config information
    config_path = os.path.join(output_dir, f"{mix_name}_config.json")
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")

    return output_path


def create_all_cross_dataset_parquets(output_dir="../data"):
    """
    Create all cross-dataset replacement parquet files
    
    Parameters:
        - output_dir (str): output directory
    """
    print("Creating all cross dataset parquet files...")
    
    mix_names = get_all_cross_dataset_mixes()
    created_files = []
    
    for mix_name in mix_names:
        try:
            output_path = create_cross_dataset_parquet(mix_name, output_dir)
            created_files.append(output_path)
        except Exception as e:
            print(f"Error creating {mix_name}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Successfully created {len(created_files)} cross dataset parquet files:")
    for file_path in created_files:
        print(f"  - {file_path}")
    
    return created_files


def print_dataset_columns():
    print("\n==== Main dataset field names ====")
    for dataset_name in ['iot_nid', 'cic2018', 'edge_iiot']:
        try:
            df = load_dataset_parquet(dataset_name)
            print(f"\n{dataset_name} field names:")
            print(list(df.columns))
        except Exception as e:
            print(f"Cannot read {dataset_name}: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create cross dataset parquet files")
    parser.add_argument('--mix-name', type=str, default=None,
                       help='Specific mix name to create (e.g., iot_nid_with_cic2018_test)')
    parser.add_argument('--output-dir', type=str, default='../data',
                       help='Output directory for parquet files')
    parser.add_argument('--all', action='store_true',
                       help='Create all cross dataset parquet files')
    parser.add_argument('--show-columns', action='store_true',
                       help='Show columns of all main datasets')
    
    args = parser.parse_args()
    
    if args.show_columns:
        print_dataset_columns()
        return
    if args.all:
        create_all_cross_dataset_parquets(args.output_dir)
    elif args.mix_name:
        create_cross_dataset_parquet(args.mix_name, args.output_dir)
    else:
        print("Please specify --mix-name or --all")
        print("Available mix names:")
        for mix_name in get_all_cross_dataset_mixes():
            print(f"  - {mix_name}")


if __name__ == "__main__":
    main() 