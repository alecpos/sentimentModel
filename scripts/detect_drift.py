"""
Script to detect model drift.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from app.models.ml.prediction.ensemble_model import EnsembleSentimentAnalyzer

class DriftDetector:
    """Class to detect model drift."""
    
    def __init__(self, model: EnsembleSentimentAnalyzer):
        """Initialize the detector."""
        self.model = model
        self.scaler = StandardScaler()
        self.baseline_stats = None
        self.drift_metrics = {}
    
    def compute_feature_stats(self, X: np.ndarray) -> Dict[str, Any]:
        """Compute statistical measures for features."""
        stats = {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
            "median": np.median(X, axis=0),
            "q1": np.percentile(X, 25, axis=0),
            "q3": np.percentile(X, 75, axis=0)
        }
        return stats
    
    def compute_prediction_stats(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Compute statistical measures for predictions."""
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        stats = {
            "accuracy": np.mean(y_pred == y_true),
            "positive_rate": np.mean(y_pred),
            "prediction_std": np.std(y_prob),
            "prediction_mean": np.mean(y_prob),
            "prediction_median": np.median(y_prob),
            "prediction_q1": np.percentile(y_prob, 25),
            "prediction_q3": np.percentile(y_prob, 75)
        }
        return stats
    
    def detect_feature_drift(
        self,
        baseline_stats: Dict[str, Any],
        current_stats: Dict[str, Any],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detect drift in feature distributions."""
        drift = {}
        
        for metric in ["mean", "std", "median"]:
            drift[metric] = np.abs(
                baseline_stats[metric] - current_stats[metric]
            ) / (baseline_stats["std"] + 1e-6)
        
        # Calculate drift scores
        drift_scores = {
            "mean_drift": np.mean(drift["mean"]),
            "std_drift": np.mean(drift["std"]),
            "median_drift": np.mean(drift["median"]),
            "max_drift": np.max([np.max(drift[m]) for m in drift])
        }
        
        # Identify drifted features
        drifted_features = {
            metric: np.where(drift[metric] > threshold)[0].tolist()
            for metric in drift
        }
        
        return {
            "drift_scores": drift_scores,
            "drifted_features": drifted_features,
            "feature_drift": drift
        }
    
    def detect_prediction_drift(
        self,
        baseline_stats: Dict[str, Any],
        current_stats: Dict[str, Any],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detect drift in prediction distributions."""
        drift = {}
        
        for metric in baseline_stats:
            if metric != "accuracy":
                drift[metric] = np.abs(
                    baseline_stats[metric] - current_stats[metric]
                )
        
        # Calculate drift scores
        drift_scores = {
            f"{metric}_drift": value
            for metric, value in drift.items()
        }
        
        # Identify drifted metrics
        drifted_metrics = [
            metric for metric, value in drift.items()
            if value > threshold
        ]
        
        return {
            "drift_scores": drift_scores,
            "drifted_metrics": drifted_metrics,
            "prediction_drift": drift
        }
    
    def plot_drift_analysis(
        self,
        baseline_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Plot drift analysis visualizations."""
        # Plot feature distributions
        plt.figure(figsize=(15, 10))
        
        # Plot feature means
        plt.subplot(2, 1, 1)
        plt.plot(
            range(len(feature_names)),
            self.baseline_stats["mean"],
            label="Baseline",
            marker="o"
        )
        plt.plot(
            range(len(feature_names)),
            self.compute_feature_stats(current_data)["mean"],
            label="Current",
            marker="o"
        )
        plt.title("Feature Means Comparison")
        plt.xlabel("Feature Index")
        plt.ylabel("Mean Value")
        plt.legend()
        plt.grid(True)
        
        # Plot feature standard deviations
        plt.subplot(2, 1, 2)
        plt.plot(
            range(len(feature_names)),
            self.baseline_stats["std"],
            label="Baseline",
            marker="o"
        )
        plt.plot(
            range(len(feature_names)),
            self.compute_feature_stats(current_data)["std"],
            label="Current",
            marker="o"
        )
        plt.title("Feature Standard Deviations Comparison")
        plt.xlabel("Feature Index")
        plt.ylabel("Standard Deviation")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("results/drift_analysis.png")
        plt.close()
    
    def save_analysis(self, filepath: str) -> None:
        """Save drift analysis results."""
        with open(filepath, "w") as f:
            json.dump(self.drift_metrics, f, indent=4)

def main():
    """Main function to demonstrate drift detection."""
    # Load baseline and current data
    baseline_data = pd.read_csv("data/sample_data.csv")
    current_data = pd.read_csv("data/sample_test_data.csv")
    
    X_baseline = baseline_data.drop("target", axis=1).values
    y_baseline = baseline_data["target"].values
    X_current = current_data.drop("target", axis=1).values
    y_current = current_data["target"].values
    
    feature_names = [f"feature_{i}" for i in range(X_baseline.shape[1])]
    
    # Initialize model and detector
    model = EnsembleSentimentAnalyzer()
    detector = DriftDetector(model)
    
    print("Computing baseline statistics...")
    detector.baseline_stats = detector.compute_feature_stats(X_baseline)
    baseline_pred_stats = detector.compute_prediction_stats(
        X_baseline, y_baseline
    )
    
    print("Computing current statistics...")
    current_feature_stats = detector.compute_feature_stats(X_current)
    current_pred_stats = detector.compute_prediction_stats(
        X_current, y_current
    )
    
    print("Detecting feature drift...")
    feature_drift = detector.detect_feature_drift(
        detector.baseline_stats,
        current_feature_stats
    )
    
    print("Detecting prediction drift...")
    prediction_drift = detector.detect_prediction_drift(
        baseline_pred_stats,
        current_pred_stats
    )
    
    # Compile drift metrics
    detector.drift_metrics = {
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "analysis_metadata": {
            "baseline_samples": len(X_baseline),
            "current_samples": len(X_current),
            "n_features": X_baseline.shape[1],
            "timestamp": datetime.now().isoformat()
        }
    }
    
    print("Generating drift visualizations...")
    detector.plot_drift_analysis(X_baseline, X_current, feature_names)
    
    print("Saving analysis results...")
    detector.save_analysis("results/drift_analysis.json")
    
    print("\nDrift detection completed!")
    print("Results saved:")
    print("  - results/drift_analysis.png")
    print("  - results/drift_analysis.json")

if __name__ == "__main__":
    main() 