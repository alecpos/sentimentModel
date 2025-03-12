"""
Data preprocessing components for the WITHIN ML Prediction System.

This module provides data preprocessing utilities for transforming raw data into
formats suitable for machine learning models. It includes components for cleaning,
normalizing, encoding, and transforming data to ensure optimal model performance.

Key functionality includes:
- Data cleaning and quality validation
- Feature scaling and normalization
- Categorical encoding and embedding
- Feature extraction and transformation
- Missing value handling and imputation
"""

__version__ = "0.1.0"  # Pre-release version

# Package-level constants
DEFAULT_SCALING_METHOD = "standard"
CATEGORICAL_ENCODING_STRATEGIES = ["one-hot", "label", "target", "embedding"]
MISSING_VALUE_STRATEGIES = ["mean", "median", "mode", "constant", "knn"]

# When implementations are added, they will be imported and exported here 