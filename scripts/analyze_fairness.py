"""
Script to analyze model fairness metrics.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import confusion_matrix
from app.models.ml.prediction.ensemble_model import EnsembleSentimentAnalyzer

class FairnessAnalyzer:
    """Class to analyze model fairness metrics."""
    
    def __init__(self, model: EnsembleSentimentAnalyzer):
        """Initialize the analyzer."""
        self.model = model
        self.metrics: Dict[str, Any] = {}
    
    def calculate_group_metrics(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        protected_feature: np.ndarray,
        group_name: str
    ) -> Dict[str, Any]:
        """Calculate metrics for a specific demographic group."""
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        metrics = {
            "sample_size": len(y_true),
            "positive_rate": np.mean(y_pred),
            "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "true_negative_rate": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "selection_rate": (tp + fp) / (tp + tn + fp + fn),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        }
        
        return metrics
    
    def analyze_fairness(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        protected_features: Dict[str, np.ndarray]
    ) -> None:
        """Analyze fairness across multiple protected attributes."""
        fairness_metrics = {}
        
        for feature_name, feature_values in protected_features.items():
            group_metrics = {}
            unique_values = np.unique(feature_values)
            
            # Calculate metrics for each group
            for value in unique_values:
                mask = feature_values == value
                group_metrics[str(value)] = self.calculate_group_metrics(
                    X[mask],
                    y_true[mask],
                    feature_values[mask],
                    str(value)
                )
            
            # Calculate disparities between groups
            disparities = self.calculate_disparities(group_metrics)
            
            fairness_metrics[feature_name] = {
                "group_metrics": group_metrics,
                "disparities": disparities
            }
        
        self.metrics = fairness_metrics
    
    def calculate_disparities(
        self, group_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate disparities between groups."""
        metrics_to_compare = [
            "positive_rate",
            "true_positive_rate",
            "false_positive_rate",
            "accuracy",
            "selection_rate"
        ]
        
        disparities = {}
        groups = list(group_metrics.keys())
        
        if len(groups) < 2:
            return disparities
        
        # Calculate max disparity for each metric
        for metric in metrics_to_compare:
            values = [group_metrics[g][metric] for g in groups]
            max_disparity = max(values) - min(values)
            disparities[f"{metric}_disparity"] = max_disparity
        
        return disparities
    
    def plot_fairness_metrics(self) -> None:
        """Plot fairness metrics visualization."""
        for feature_name, feature_metrics in self.metrics.items():
            group_metrics = feature_metrics["group_metrics"]
            
            # Create subplots
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))
            
            # Plot rates comparison
            rates_data = []
            for group, metrics in group_metrics.items():
                rates_data.append({
                    "Group": group,
                    "True Positive Rate": metrics["true_positive_rate"],
                    "False Positive Rate": metrics["false_positive_rate"],
                    "Selection Rate": metrics["selection_rate"]
                })
            
            rates_df = pd.DataFrame(rates_data)
            rates_df_melted = pd.melt(
                rates_df,
                id_vars=["Group"],
                var_name="Metric",
                value_name="Rate"
            )
            
            sns.barplot(
                data=rates_df_melted,
                x="Group",
                y="Rate",
                hue="Metric",
                ax=axes[0]
            )
            axes[0].set_title(f"Fairness Metrics by Group - {feature_name}")
            axes[0].set_ylabel("Rate")
            
            # Plot confusion matrices
            confusion_data = []
            for group, metrics in group_metrics.items():
                cm = metrics["confusion_matrix"]
                for outcome, count in cm.items():
                    confusion_data.append({
                        "Group": group,
                        "Outcome": outcome,
                        "Count": count
                    })
            
            confusion_df = pd.DataFrame(confusion_data)
            sns.barplot(
                data=confusion_df,
                x="Group",
                y="Count",
                hue="Outcome",
                ax=axes[1]
            )
            axes[1].set_title(f"Confusion Matrix by Group - {feature_name}")
            axes[1].set_ylabel("Count")
            
            plt.tight_layout()
            plt.savefig(f"results/fairness_metrics_{feature_name}.png")
            plt.close()
    
    def save_metrics(self, filepath: str) -> None:
        """Save fairness metrics to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=4)

def main():
    """Main function to demonstrate fairness analysis."""
    # Load test data
    test_data = pd.read_csv("data/sample_test_data.csv")
    X_test = test_data.drop("target", axis=1).values
    y_test = test_data["target"].values
    
    # Create synthetic protected attributes for demonstration
    np.random.seed(42)
    n_samples = len(X_test)
    protected_features = {
        "gender": np.random.choice(["male", "female"], size=n_samples),
        "age_group": np.random.choice(["18-25", "26-35", "36+"], size=n_samples),
        "income_level": np.random.choice(["low", "medium", "high"], size=n_samples)
    }
    
    # Initialize model and analyzer
    model = EnsembleSentimentAnalyzer()
    analyzer = FairnessAnalyzer(model)
    
    # Analyze fairness
    print("Analyzing fairness metrics...")
    analyzer.analyze_fairness(X_test, y_test, protected_features)
    
    # Generate visualizations
    print("Generating fairness visualizations...")
    analyzer.plot_fairness_metrics()
    
    # Save metrics
    analyzer.save_metrics("results/fairness_metrics.json")
    
    print("\nFairness analysis completed!")
    print("Results saved:")
    print("  - results/fairness_metrics.json")
    for feature in protected_features.keys():
        print(f"  - results/fairness_metrics_{feature}.png")

if __name__ == "__main__":
    main() 