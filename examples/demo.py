"""
Demonstration script for the enhanced ensemble package.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.ml.prediction import (
    EnhancedBaggingEnsemble,
    EnhancedStackingEnsemble,
    optimize_ensemble_weights,
    visualize_ensemble_performance
)

def load_data():
    """Load and prepare the sample data."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "sample_data.csv"
    )
    
    data = pd.read_csv(data_path)
    X = data.drop("target", axis=1).values
    y = data["target"].values
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def demo_bagging(X_train, X_test, y_train, y_test):
    """Demonstrate the bagging ensemble."""
    print("\n=== Bagging Ensemble Demo ===")
    
    # Initialize and train bagging ensemble
    base_estimator = DecisionTreeClassifier(random_state=42)
    ensemble = EnhancedBaggingEnsemble(
        base_estimator=base_estimator,
        n_estimators=10,
        random_state=42
    )
    
    print("Training bagging ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Visualize performance
    print("Generating performance visualization...")
    fig = visualize_ensemble_performance(ensemble)
    fig.savefig("bagging_performance.png")
    plt.close(fig)
    
    return ensemble

def demo_stacking(X_train, X_test, y_train, y_test):
    """Demonstrate the stacking ensemble."""
    print("\n=== Stacking Ensemble Demo ===")
    
    # Initialize base estimators
    base_estimators = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42)
    ]
    meta_learner = LogisticRegression(random_state=42)
    
    # Create and train stacking ensemble
    ensemble = EnhancedStackingEnsemble(
        base_estimators=base_estimators,
        meta_learner=meta_learner,
        use_proba=True,
        n_splits=5,
        random_state=42
    )
    
    print("Training stacking ensemble...")
    ensemble.fit(X_train, y_train, X_test, y_test)
    
    # Make predictions
    print("Making predictions...")
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Visualize performance
    print("Generating performance visualization...")
    fig = visualize_ensemble_performance(ensemble)
    fig.savefig("stacking_performance.png")
    plt.close(fig)
    
    return ensemble

def demo_weight_optimization(X_train, X_test, y_train, y_test):
    """Demonstrate ensemble weight optimization."""
    print("\n=== Weight Optimization Demo ===")
    
    # Create and train base models
    base_models = [
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        LogisticRegression(random_state=42)
    ]
    
    print("Training base models...")
    for model in base_models:
        model.fit(X_train, y_train)
    
    # Optimize weights
    print("Optimizing ensemble weights...")
    weights = optimize_ensemble_weights(base_models, X_test, y_test)
    
    # Print results
    print("\nOptimized weights:")
    for i, (model, weight) in enumerate(zip(base_models, weights)):
        print(f"Model {i}: {weight:.4f}")
    
    # Calculate weighted ensemble performance
    predictions = []
    for model in base_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_test)[:, 1]
        else:
            pred = model.predict(X_test)
        predictions.append(pred)
    
    weighted_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        weighted_pred += weight * pred
    
    auc = roc_auc_score(y_test, weighted_pred)
    print(f"\nWeighted ensemble AUC: {auc:.4f}")
    
    return weights

def main():
    """Main demonstration function."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Run demonstrations
    bagging_ensemble = demo_bagging(X_train, X_test, y_train, y_test)
    stacking_ensemble = demo_stacking(X_train, X_test, y_train, y_test)
    weights = demo_weight_optimization(X_train, X_test, y_train, y_test)
    
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main() 