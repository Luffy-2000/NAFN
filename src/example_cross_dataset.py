#!/usr/bin/env python3
"""
Cross-dataset replacement usage example
"""

from create_cross_dataset_parquet import create_cross_dataset_parquet, create_all_cross_dataset_parquets
from data.cross_dataset_config import get_all_cross_dataset_mixes


def example_single_mix():
    """Example: Create a single mixed dataset"""
    print("Example: Replace iot_nid's test_classes with cic2018's test_classes")
    
    # Create a single mixed dataset
    mix_name = "iot_nid_with_cic2018_test"
    output_path = create_cross_dataset_parquet(mix_name, output_dir="../data/cross_datasets")
    
    print(f"Creation completed: {output_path}")


def example_all_mixes():
    """Example: Create all mixed datasets"""
    print("Example: Create all cross-dataset mixes")
    
    # Create all mixed datasets
    created_files = create_all_cross_dataset_parquets(output_dir="../data/cross_datasets")
    
    print(f"Total {len(created_files)} mixed datasets created")


def show_available_mixes():
    """Show all available mix configurations"""
    print("Available cross-dataset mix configurations:")
    print("="*50)
    
    mix_names = get_all_cross_dataset_mixes()
    for i, mix_name in enumerate(mix_names, 1):
        print(f"{i}. {mix_name}")
    
    print("\nDescription:")
    print("- iot_nid_with_cic2018_test: Replace iot_nid's test_classes with cic2018's test_classes")
    print("- iot_nid_with_edge_iiot_test: Replace iot_nid's test_classes with edge_iiot's test_classes")
    print("- cic2018_with_iot_nid_test: Replace cic2018's test_classes with iot_nid's test_classes")
    print("- cic2018_with_edge_iiot_test: Replace cic2018's test_classes with edge_iiot's test_classes")
    print("- edge_iiot_with_iot_nid_test: Replace edge_iiot's test_classes with iot_nid's test_classes")
    print("- edge_iiot_with_cic2018_test: Replace edge_iiot's test_classes with cic2018's test_classes")


if __name__ == "__main__":
    print("Cross-dataset replacement usage example")
    print("="*60)
    
    # Show available configurations
    show_available_mixes()
    
    print("\n" + "="*60)
    print("Running examples...")
    
    # Run examples (commented out to avoid actual execution)
    # example_single_mix()
    # example_all_mixes()
    
    print("Example code is ready, uncomment to run") 