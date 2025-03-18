"""
Script to run performance benchmarks on the ensemble model.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from app.models.ml.prediction.ensemble_model import EnsembleSentimentAnalyzer

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data."""
    train_data = pd.read_csv("data/sample_data.csv")
    test_data = pd.read_csv("data/sample_test_data.csv")
    return train_data, test_data

def measure_training_time(model: EnsembleSentimentAnalyzer, X: np.ndarray, y: np.ndarray) -> float:
    """Measure model training time."""
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    return end_time - start_time

def measure_inference_time(model: EnsembleSentimentAnalyzer, X: np.ndarray, n_runs: int = 100) -> Dict[str, float]:
    """Measure model inference time statistics."""
    inference_times = []
    for _ in range(n_runs):
        start_time = time.time()
        model.predict(X)
        end_time = time.time()
        inference_times.append(end_time - start_time)
    
    return {
        "mean": np.mean(inference_times),
        "std": np.std(inference_times),
        "min": np.min(inference_times),
        "max": np.max(inference_times),
        "p95": np.percentile(inference_times, 95),
        "p99": np.percentile(inference_times, 99)
    }

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate model performance metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def run_benchmarks() -> Dict[str, Any]:
    """Run comprehensive benchmarks on the ensemble model."""
    # Load data
    train_data, test_data = load_data()
    X_train = train_data.drop("target", axis=1).values
    y_train = train_data["target"].values
    X_test = test_data.drop("target", axis=1).values
    y_test = test_data["target"].values
    
    # Initialize model
    model = EnsembleSentimentAnalyzer()
    
    # Measure training time
    training_time = measure_training_time(model, X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Measure inference time
    inference_times = measure_inference_time(model, X_test)
    
    # Calculate metrics
    performance_metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Compile results
    results = {
        "training_time": training_time,
        "inference_times": inference_times,
        "performance_metrics": performance_metrics,
        "model_info": {
            "n_estimators": len(model.estimators),
            "feature_dim": X_train.shape[1],
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0]
        }
    }
    
    return results

def main():
    """Main function to run benchmarks and save results."""
    print("Running benchmarks...")
    results = run_benchmarks()
    
    # Save results
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nBenchmark Results:")
    print("=================")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print("\nInference Time Statistics (seconds):")
    for metric, value in results["inference_times"].items():
        print(f"  {metric}: {value:.4f}")
    print("\nPerformance Metrics:")
    for metric, value in results["performance_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print("\nModel Information:")
    for key, value in results["model_info"].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 