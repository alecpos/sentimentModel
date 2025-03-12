#!/usr/bin/env python
"""
Test script for Kaggle Dataset Integration with Fairness Evaluation

This script demonstrates basic functionality of the Kaggle dataset integration
pipeline by initializing it and retrieving dataset configurations. It doesn't
actually download or process any datasets to avoid requiring Kaggle credentials,
but it verifies that the pipeline is set up correctly.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.data_integration import (
    DatasetCategory,
    DatasetConfig,
    KaggleDatasetPipeline
)

def main():
    """Test the Kaggle dataset integration pipeline."""
    
    print("Initializing Kaggle Dataset Pipeline...")
    pipeline = KaggleDatasetPipeline(
        data_dir="data/kaggle",
        cache_dir="data/cache",
        validate_fairness=True
    )
    
    print("\nRetrieving dataset configurations...")
    configs = pipeline.get_dataset_configs()
    
    print(f"\nNumber of pre-configured datasets: {len(configs)}")
    
    print("\nAvailable datasets:")
    for name, config in configs.items():
        print(f"  - {name} ({config.dataset_slug})")
        print(f"    Category: {config.category}")
        print(f"    Target column: {config.target_column}")
        print(f"    Feature columns: {len(config.feature_columns)} columns")
        print(f"    Protected attributes: {config.protected_attributes}")
        print("")
    
    print("\nTo download and process a dataset, run:")
    print("  processed_dataset = pipeline.process_dataset(configs['customer_conversion'])")
    print("\nNote: This requires Kaggle API credentials to be set up.")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 