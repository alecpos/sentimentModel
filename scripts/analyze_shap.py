"""
Script to analyze model explainability using SHAP values.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, Any, List, Tuple
from app.models.ml.prediction.ensemble_model import EnsembleSentimentAnalyzer

class ShapAnalyzer:
    """Class to analyze model explainability using SHAP values."""
    
    def __init__(self, model: EnsembleSentimentAnalyzer):
        """Initialize the analyzer."""
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        feature_names: List[str],
        n_samples: int = 100
    ) -> None:
        """Compute SHAP values for the model."""
        # If data is too large, sample it
        if len(X) > n_samples:
            np.random.seed(42)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Initialize TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)
        self.feature_names = feature_names
        
        if isinstance(self.shap_values, list):
            # For multi-class output, take the values for class 1
            self.shap_values = self.shap_values[1]
    
    def plot_summary(self) -> None:
        """Create SHAP summary plot."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            features=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig("results/shap_summary.png")
        plt.close()
    
    def plot_feature_importance(self) -> None:
        """Create SHAP feature importance plot."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            features=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig("results/shap_importance.png")
        plt.close()
    
    def plot_dependence_plots(self, top_n: int = 5) -> None:
        """Create dependence plots for top features."""
        # Calculate mean absolute SHAP values for each feature
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        top_features = np.argsort(feature_importance)[-top_n:]
        
        for feature_idx in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                features=self.feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(
                f"results/shap_dependence_{self.feature_names[feature_idx]}.png"
            )
            plt.close()
    
    def analyze_feature_interactions(self, top_n: int = 5) -> Dict[str, float]:
        """Analyze feature interactions using SHAP values."""
        interactions = {}
        
        # Calculate mean absolute SHAP values for each feature
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        top_features = np.argsort(feature_importance)[-top_n:]
        
        for i in top_features:
            for j in top_features:
                if i < j:
                    # Calculate interaction strength
                    interaction = np.abs(
                        self.shap_values[:, i] * self.shap_values[:, j]
                    ).mean()
                    key = f"{self.feature_names[i]}_{self.feature_names[j]}"
                    interactions[key] = float(interaction)
        
        return interactions
    
    def save_analysis(self, filepath: str) -> None:
        """Save SHAP analysis results."""
        # Calculate feature importance
        importance = np.abs(self.shap_values).mean(axis=0)
        feature_importance = {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }
        
        # Calculate interactions
        interactions = self.analyze_feature_interactions()
        
        # Compile results
        results = {
            "feature_importance": feature_importance,
            "feature_interactions": interactions,
            "analysis_metadata": {
                "n_samples": len(self.shap_values),
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names
            }
        }
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

def main():
    """Main function to demonstrate SHAP analysis."""
    # Load test data
    test_data = pd.read_csv("data/sample_test_data.csv")
    X_test = test_data.drop("target", axis=1).values
    feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    # Initialize model and analyzer
    model = EnsembleSentimentAnalyzer()
    analyzer = ShapAnalyzer(model)
    
    print("Computing SHAP values...")
    analyzer.compute_shap_values(X_test, feature_names)
    
    print("Generating SHAP visualizations...")
    analyzer.plot_summary()
    analyzer.plot_feature_importance()
    analyzer.plot_dependence_plots()
    
    print("Saving analysis results...")
    analyzer.save_analysis("results/shap_analysis.json")
    
    print("\nSHAP analysis completed!")
    print("Results saved:")
    print("  - results/shap_summary.png")
    print("  - results/shap_importance.png")
    print("  - results/shap_analysis.json")
    print("  - results/shap_dependence_*.png")

if __name__ == "__main__":
    main() 