"""
Script to generate sample data for testing and demonstration.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def generate_sample_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 15,
    n_redundant: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate sample classification data.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame containing the generated data
    """
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X_scaled, columns=feature_names)
    df["target"] = y
    
    return df

def main():
    """Main function to generate and save sample data."""
    # Generate training data
    train_data = generate_sample_data(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5
    )
    
    # Generate test data
    test_data = generate_sample_data(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5
    )
    
    # Save data
    train_data.to_csv("data/sample_data.csv", index=False)
    test_data.to_csv("data/sample_test_data.csv", index=False)
    
    print("Sample data generated successfully!")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print("\nFeature statistics:")
    print(train_data.describe())

if __name__ == "__main__":
    main() 