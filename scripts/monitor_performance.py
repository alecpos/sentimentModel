"""
Script to monitor model performance in production.
"""

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from app.models.ml.prediction.ensemble_model import EnsembleSentimentAnalyzer

class ModelMonitor:
    """Class to monitor model performance metrics."""
    
    def __init__(self, model: EnsembleSentimentAnalyzer):
        """Initialize the monitor."""
        self.model = model
        self.metrics_history: List[Dict[str, Any]] = []
        self.performance_thresholds = {
            "accuracy": 0.85,
            "f1": 0.80,
            "roc_auc": 0.85,
            "inference_time_ms": 300  # 300ms threshold
        }
    
    def calculate_metrics(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> Tuple[Dict[str, float], float]:
        """Calculate model performance metrics and inference time."""
        # Measure inference time
        start_time = time.time()
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob)
        }
        
        return metrics, inference_time
    
    def check_performance_degradation(
        self, metrics: Dict[str, float], inference_time: float
    ) -> List[str]:
        """Check for performance degradation against thresholds."""
        alerts = []
        
        # Check metric thresholds
        for metric, threshold in self.performance_thresholds.items():
            if metric == "inference_time_ms":
                if inference_time > threshold:
                    alerts.append(
                        f"Inference time ({inference_time:.2f}ms) exceeds threshold "
                        f"({threshold}ms)"
                    )
            elif metrics.get(metric, 0) < threshold:
                alerts.append(
                    f"{metric.upper()} ({metrics[metric]:.3f}) below threshold "
                    f"({threshold})"
                )
        
        return alerts
    
    def monitor_batch(
        self, X: np.ndarray, y_true: np.ndarray, batch_id: str
    ) -> Dict[str, Any]:
        """Monitor model performance on a batch of data."""
        # Calculate metrics
        metrics, inference_time = self.calculate_metrics(X, y_true)
        
        # Check for degradation
        alerts = self.check_performance_degradation(metrics, inference_time)
        
        # Create monitoring record
        record = {
            "timestamp": datetime.now().isoformat(),
            "batch_id": batch_id,
            "metrics": metrics,
            "inference_time_ms": inference_time,
            "alerts": alerts,
            "data_stats": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "class_distribution": np.bincount(y_true).tolist()
            }
        }
        
        # Add to history
        self.metrics_history.append(record)
        
        return record
    
    def save_metrics(self, filepath: str) -> None:
        """Save monitoring metrics to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def plot_metrics_history(self) -> None:
        """Plot metrics history."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert history to DataFrame
        records = []
        for record in self.metrics_history:
            metrics = record["metrics"].copy()
            metrics["inference_time_ms"] = record["inference_time_ms"]
            metrics["timestamp"] = record["timestamp"]
            records.append(metrics)
        
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Plot metrics over time
        plt.figure(figsize=(15, 10))
        
        # Plot classification metrics
        plt.subplot(2, 1, 1)
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            plt.plot(df["timestamp"], df[metric], label=metric)
        
        plt.title("Classification Metrics Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        # Plot inference time
        plt.subplot(2, 1, 2)
        plt.plot(df["timestamp"], df["inference_time_ms"], label="inference_time")
        plt.axhline(
            y=self.performance_thresholds["inference_time_ms"],
            color="r",
            linestyle="--",
            label="threshold"
        )
        
        plt.title("Inference Time Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("results/monitoring_history.png")
        plt.close()

def main():
    """Main function to demonstrate monitoring."""
    # Load test data
    test_data = pd.read_csv("data/sample_test_data.csv")
    X_test = test_data.drop("target", axis=1).values
    y_test = test_data["target"].values
    
    # Initialize model and monitor
    model = EnsembleSentimentAnalyzer()
    monitor = ModelMonitor(model)
    
    # Simulate batch monitoring
    print("Simulating batch monitoring...")
    batch_size = 50
    n_batches = len(X_test) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_X = X_test[start_idx:end_idx]
        batch_y = y_test[start_idx:end_idx]
        
        record = monitor.monitor_batch(
            batch_X,
            batch_y,
            batch_id=f"batch_{i}"
        )
        
        # Print alerts if any
        if record["alerts"]:
            print(f"\nBatch {i} Alerts:")
            for alert in record["alerts"]:
                print(f"  - {alert}")
    
    # Save monitoring results
    monitor.save_metrics("results/monitoring_metrics.json")
    monitor.plot_metrics_history()
    
    print("\nMonitoring completed!")
    print("Results saved:")
    print("  - results/monitoring_metrics.json")
    print("  - results/monitoring_history.png")

if __name__ == "__main__":
    main() 