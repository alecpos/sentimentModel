"""
Script to visualize benchmark results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def load_results() -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open("results/benchmark_results.json", "r") as f:
        return json.load(f)

def plot_inference_times(results: Dict[str, Any]) -> None:
    """Plot inference time statistics."""
    plt.figure(figsize=(10, 6))
    inference_times = results["inference_times"]
    metrics = ["mean", "std", "min", "max", "p95", "p99"]
    values = [inference_times[m] * 1000 for m in metrics]  # Convert to milliseconds
    
    sns.barplot(x=metrics, y=values)
    plt.title("Inference Time Statistics")
    plt.xlabel("Metric")
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/inference_times.png")
    plt.close()

def plot_performance_metrics(results: Dict[str, Any]) -> None:
    """Plot model performance metrics."""
    plt.figure(figsize=(10, 6))
    metrics = results["performance_metrics"]
    
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title("Model Performance Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/performance_metrics.png")
    plt.close()

def plot_model_info(results: Dict[str, Any]) -> None:
    """Plot model information."""
    plt.figure(figsize=(10, 6))
    info = results["model_info"]
    
    sns.barplot(x=list(info.keys()), y=list(info.values()))
    plt.title("Model Information")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/model_info.png")
    plt.close()

def create_summary_table(results: Dict[str, Any]) -> None:
    """Create a summary table of all metrics."""
    summary = {
        "Training Time (s)": results["training_time"],
        "Mean Inference Time (ms)": results["inference_times"]["mean"] * 1000,
        "P95 Inference Time (ms)": results["inference_times"]["p95"] * 1000,
        "Accuracy": results["performance_metrics"]["accuracy"],
        "F1 Score": results["performance_metrics"]["f1"],
        "ROC AUC": results["performance_metrics"]["roc_auc"],
        "Number of Estimators": results["model_info"]["n_estimators"],
        "Feature Dimension": results["model_info"]["feature_dim"]
    }
    
    # Save summary as JSON
    with open("results/summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=4)

def main():
    """Main function to create visualizations."""
    print("Creating benchmark visualizations...")
    results = load_results()
    
    # Create visualizations
    plot_inference_times(results)
    plot_performance_metrics(results)
    plot_model_info(results)
    create_summary_table(results)
    
    print("Visualizations created successfully!")
    print("Results saved in the 'results' directory:")
    print("  - inference_times.png")
    print("  - performance_metrics.png")
    print("  - model_info.png")
    print("  - summary_metrics.json")

if __name__ == "__main__":
    main() 