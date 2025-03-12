#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fairness-Aware Hyperparameter Tuning

This script implements hyperparameter tuning that optimizes for both 
model performance and fairness metrics.
"""

import os
import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score, f1_score

from app.models.ml.fairness.evaluator import FairnessEvaluator
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor

def fairness_aware_hyperparameter_tuning(X_train, y_train, protected_attributes,
                                        X_val, y_val, protected_val,
                                        param_grid, fairness_weight=0.5,
                                        fairness_metric='demographic_parity'):
    """
    Perform hyperparameter tuning that considers both model performance and fairness
    
    Args:
        X_train: Training features
        y_train: Training labels
        protected_attributes: Protected attributes for training data
        X_val: Validation features
        y_val: Validation labels
        protected_val: Protected attributes for validation data
        param_grid: Dictionary mapping parameter names to lists of values
        fairness_weight: Weight given to fairness vs. performance (0-1)
        fairness_metric: Fairness metric to optimize ('demographic_parity', 'equal_opportunity')
        
    Returns:
        Best hyperparameters and performance metrics
    """
    print(f"Starting fairness-aware hyperparameter tuning...")
    print(f"Fairness weight: {fairness_weight}, Metric: {fairness_metric}")
    
    best_score = float('-inf')
    best_params = None
    best_metrics = None
    
    # Create fairness evaluator
    evaluator = FairnessEvaluator(
        protected_attributes=list(protected_val.keys()),
        fairness_threshold=0.2,
        metrics=[fairness_metric]
    )
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    results = []
    
    # Iterate through all hyperparameter combinations
    for i, combination in enumerate(param_combinations):
        param_dict = dict(zip(param_names, combination))
        
        print(f"Combination {i+1}/{len(param_combinations)}: {param_dict}")
        
        # Train model with these hyperparameters
        model = AdScorePredictor(model_config=param_dict)
        
        # Convert to numpy array if it's a pandas Series
        y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
        model.fit(X_train, y_train_values, protected_attributes)
        
        # Evaluate on validation set
        predictions = model.predict(X_val)
        
        # Ensure predictions are numpy array
        if isinstance(predictions, dict):
            if 'score' in predictions:
                predictions = predictions['score']
            elif 'prediction' in predictions:
                predictions = predictions['prediction']
            else:
                predictions = list(predictions.values())[0]
        
        # Convert to numpy array if needed
        if not isinstance(predictions, np.ndarray):
            # If it's a single value, create an array with that value for all samples
            if isinstance(predictions, (int, float, bool)):
                predictions = np.full(len(X_val), float(predictions))
            # If it's a list, convert to numpy array
            elif isinstance(predictions, list):
                predictions = np.array(predictions)
        
        # Ensure predictions are not scalar
        if np.isscalar(predictions) or predictions.ndim == 0:
            predictions = np.full(len(X_val), float(predictions))
        
        # Performance metrics (e.g., accuracy, F1)
        binary_preds = np.array(predictions >= 0.5, dtype=int)
        accuracy = accuracy_score(y_val, binary_preds)
        f1 = f1_score(y_val, binary_preds, zero_division=0)
        performance_score = (accuracy + f1) / 2
        
        # Fairness metrics
        fairness_results = evaluator.evaluate(predictions, y_val, protected_val)
        
        # Get the specific fairness metric we're optimizing for
        attr_key = list(protected_val.keys())[0]  # Use first protected attribute
        fairness_key = f"{attr_key}_{fairness_metric}"
        
        if fairness_key in fairness_results['fairness_metrics']:
            fairness_diff = fairness_results['fairness_metrics'][fairness_key]['difference']
            fairness_score = 1.0 - fairness_diff  # Higher is better (1.0 = perfect fairness)
        else:
            print(f"Warning: Fairness metric {fairness_key} not found in results")
            fairness_score = 0.0
        
        # Combined score with weighting
        combined_score = (1 - fairness_weight) * performance_score + fairness_weight * fairness_score
        
        print(f"  Performance: {performance_score:.4f}, Fairness: {fairness_score:.4f}, Combined: {combined_score:.4f}")
        
        # Track all results
        results.append({
            'params': param_dict,
            'accuracy': accuracy,
            'f1': f1,
            'fairness_score': fairness_score,
            'combined_score': combined_score
        })
        
        if combined_score > best_score:
            best_score = combined_score
            best_params = param_dict
            best_metrics = {
                'accuracy': accuracy,
                'f1': f1,
                'fairness_score': fairness_score,
                'combined_score': combined_score
            }
            print(f"  New best configuration found!")
    
    print("\nHyperparameter tuning complete.")
    print(f"Best parameters: {best_params}")
    print(f"Best metrics: Accuracy={best_metrics['accuracy']:.4f}, F1={best_metrics['f1']:.4f}, "
          f"Fairness={best_metrics['fairness_score']:.4f}, Combined={best_metrics['combined_score']:.4f}")
    
    # Create a DataFrame with all results for analysis
    results_df = pd.DataFrame(results)
    
    return best_params, best_metrics, results_df

def main():
    """Example usage of fairness-aware hyperparameter tuning"""
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print("Loading test data...")
    
    # Create synthetic dataset with gender bias
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    
    # Sensitive attribute with bias
    gender = np.random.choice(['male', 'female'], size=n_samples, p=[0.6, 0.4])
    
    # Inject severe gender bias
    gender_bias = np.zeros(n_samples)
    gender_bias[gender == 'male'] = 0.4
    
    # Other features
    education = np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], size=n_samples)
    location = np.random.choice(['urban', 'suburban', 'rural'], size=n_samples)
    
    # Create target with bias
    ad_score = 0.5 * np.random.random(n_samples) + 0.3 * (age / 50) + 0.2 * (income / 100000) + gender_bias
    ad_score = np.clip(ad_score, 0, 1)
    
    # Binary outcome
    high_performing = (ad_score >= 0.7).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'gender': gender,
        'education': education,
        'location': location,
        'ad_score': ad_score,
        'high_performing': high_performing
    })
    
    # Train-test-val split
    X = data.drop(['ad_score', 'high_performing'], axis=1)
    y = data['high_performing']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Extract protected attributes
    protected_train = {'gender': X_train['gender'].values}
    protected_val = {'gender': X_val['gender'].values}
    protected_test = {'gender': X_test['gender'].values}
    
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'hidden_dim': [32, 64, 128],
        'dropout': [0.1, 0.3, 0.5]
    }
    
    # Run hyperparameter tuning
    best_params, best_metrics, results_df = fairness_aware_hyperparameter_tuning(
        X_train, y_train, protected_train,
        X_val, y_val, protected_val,
        param_grid, fairness_weight=0.7,
        fairness_metric='demographic_parity'
    )
    
    # Save results
    os.makedirs('tuning_results', exist_ok=True)
    results_df.to_csv('tuning_results/fairness_tuning_results.csv', index=False)
    
    print("\nTuning results saved to 'tuning_results/fairness_tuning_results.csv'")

if __name__ == "__main__":
    main() 