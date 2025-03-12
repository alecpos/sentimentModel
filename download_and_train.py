#!/usr/bin/env python
"""
Download and Train Script for Sentiment140

This script:
1. Downloads the Sentiment140 dataset using kagglehub
2. Runs the sentiment analysis training script on the downloaded dataset
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function that downloads data and runs training."""
    try:
        # Check if kagglehub is installed
        try:
            import kagglehub
        except ImportError:
            logger.info("Installing kagglehub...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub
        
        # Download the dataset
        logger.info("Downloading Sentiment140 dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
        logger.info(f"Dataset downloaded to: {dataset_path}")
        
        # Find the CSV file in the downloaded dataset
        csv_files = list(Path(dataset_path).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded dataset")
        
        csv_file = str(csv_files[0])
        logger.info(f"Found CSV file: {csv_file}")
        
        # Create the models directory if it doesn't exist
        os.makedirs("models/sentiment140", exist_ok=True)
        
        # Run the training script with a limited sample size for faster training
        logger.info("Running sentiment analysis training on the dataset...")
        training_command = [
            sys.executable, 
            "sentiment140_direct.py",
            "--dataset_path", csv_file,
            "--max_samples", "100000",  # Use a subset for faster training
            "--model_type", "logistic"
        ]
        
        logger.info(f"Running command: {' '.join(training_command)}")
        subprocess.check_call(training_command)
        
        logger.info("Training completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 