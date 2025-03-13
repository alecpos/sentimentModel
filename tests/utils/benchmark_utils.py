"""Utilities for benchmark testing.

This module provides shared functionality for benchmark tests,
including data generation, result formatting, and visualization.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkResults:
    """Container for storing and visualizing benchmark results.
    
    This class provides methods for saving, loading, and visualizing
    benchmark results from multiple test runs.
    
    Attributes:
        output_dir: Directory for storing results
        results: Dictionary of loaded results
    """
    
    def __init__(self, output_dir: str = "benchmark_results") -> None:
        """Initialize benchmark results container.
        
        Args:
            output_dir: Directory for storing results
        """
        self.output_dir = output_dir
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def save_results(
        self, 
        model_name: str, 
        metrics: Dict[str, float], 
        targets: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save benchmark results to JSON file.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric values
            targets: Optional dictionary of target values
            metadata: Optional dictionary of additional information
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_benchmark_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create result dictionary
        result = {
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        # Add targets if provided
        if targets:
            result["targets"] = targets
            
            # Calculate comparison data
            comparison = {}
            for metric, value in metrics.items():
                if metric in targets:
                    comparison[metric] = value - targets[metric]
            
            result["comparison"] = comparison
        
        # Add metadata if provided
        if metadata:
            result["metadata"] = metadata
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved benchmark results to {filepath}")
        
        # Store in results dictionary
        self.results[model_name] = result
        
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load benchmark results from JSON file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Dictionary of loaded results
        """
        with open(filepath, 'r') as f:
            result = json.load(f)
        
        # Store in results dictionary
        if "model_name" in result:
            model_name = result["model_name"]
            self.results[model_name] = result
        
        return result
    
    def load_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Load all benchmark results from output directory.
        
        Returns:
            Dictionary of loaded results
        """
        # Find all JSON files in output directory
        json_files = [
            os.path.join(self.output_dir, f) 
            for f in os.listdir(self.output_dir) 
            if f.endswith(".json") and "benchmark" in f
        ]
        
        for filepath in json_files:
            try:
                self.load_results(filepath)
            except Exception as e:
                logger.error(f"Error loading results from {filepath}: {e}")
        
        return self.results
    
    def plot_metric_comparison(
        self, 
        metric: str, 
        output_file: Optional[str] = None
    ) -> None:
        """Plot comparison of metric across models.
        
        Args:
            metric: Name of metric to plot
            output_file: Optional path for saving plot
        """
        # Load results if not already loaded
        if not self.results:
            self.load_all_results()
        
        # Extract metric values and targets
        model_names = []
        values = []
        targets = []
        
        for model_name, result in self.results.items():
            if metric in result.get("metrics", {}):
                model_names.append(model_name)
                values.append(result["metrics"][metric])
                
                if "targets" in result and metric in result["targets"]:
                    targets.append(result["targets"][metric])
                else:
                    targets.append(None)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot metric values
        bars = plt.bar(model_names, values, alpha=0.7)
        
        # Plot targets if available
        for i, (model, target) in enumerate(zip(model_names, targets)):
            if target is not None:
                plt.axhline(
                    y=target, 
                    xmin=i/len(model_names), 
                    xmax=(i+1)/len(model_names),
                    color='red', 
                    linestyle='--'
                )
        
        plt.title(f"{metric} Comparison Across Models")
        plt.ylabel(metric)
        plt.xlabel("Model")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height,
                f'{height:.4f}', 
                ha='center', 
                va='bottom'
            )
        
        plt.tight_layout()
        
        # Save plot if output file provided
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved plot to {output_file}")
        
        plt.show()
    
    def plot_metric_trends(
        self, 
        model_name: str, 
        output_file: Optional[str] = None
    ) -> None:
        """Plot trends of metrics over time for a specific model.
        
        Args:
            model_name: Name of model to plot
            output_file: Optional path for saving plot
        """
        # Load all result directories
        all_dirs = [
            d for d in os.listdir(self.output_dir) 
            if os.path.isdir(os.path.join(self.output_dir, d)) and d.startswith('20')
        ]
        
        # Sort directories by date
        all_dirs.sort()
        
        # Collect metrics over time
        timestamps = []
        metrics_over_time = {}
        
        for dir_name in all_dirs:
            dir_path = os.path.join(self.output_dir, dir_name)
            
            # Find result files for the specified model
            model_files = [
                f for f in os.listdir(dir_path) 
                if f.startswith(f"{model_name}_benchmark") and f.endswith(".json")
            ]
            
            if model_files:
                # Use the first file found
                filepath = os.path.join(dir_path, model_files[0])
                
                try:
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                    
                    timestamp = result.get("timestamp", dir_name)
                    metrics = result.get("metrics", {})
                    
                    timestamps.append(timestamp)
                    
                    for metric, value in metrics.items():
                        if metric not in metrics_over_time:
                            metrics_over_time[metric] = []
                        
                        metrics_over_time[metric].append(value)
                        
                except Exception as e:
                    logger.error(f"Error loading results from {filepath}: {e}")
        
        # Create plot for each metric
        if metrics_over_time:
            metric_names = list(metrics_over_time.keys())
            
            # Create subplots
            fig, axs = plt.subplots(
                len(metric_names), 
                1, 
                figsize=(10, 4 * len(metric_names)),
                sharex=True
            )
            
            # Plot each metric
            for i, metric in enumerate(metric_names):
                ax = axs[i] if len(metric_names) > 1 else axs
                
                ax.plot(timestamps, metrics_over_time[metric], 'o-', label=metric)
                ax.set_title(f"{metric} Over Time")
                ax.set_ylabel(metric)
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set overall title and labels
            fig.suptitle(f"{model_name} Metrics Over Time", fontsize=16)
            plt.xlabel("Timestamp")
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot if output file provided
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Saved plot to {output_file}")
            
            plt.show()
        else:
            logger.warning(f"No metrics found for model {model_name}")


class SyntheticDataGenerator:
    """Generate synthetic data for benchmark testing.
    
    This class provides methods for generating synthetic data
    with configurable properties for testing ML models.
    
    Attributes:
        random_seed: Seed for random number generation
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize synthetic data generator.
        
        Args:
            random_seed: Seed for random number generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_regression_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        noise_level: float = 0.1,
        nonlinearity: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic regression data.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            noise_level: Level of noise to add
            nonlinearity: Strength of nonlinear relationships
            
        Returns:
            Tuple of features DataFrame and target Series
        """
        # Generate feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Generate weights with decreasing importance
        weights = np.array([1.0 / (i + 1) for i in range(n_features)])
        
        # Linear component
        y_linear = X.dot(weights)
        
        # Add nonlinear components
        if nonlinearity > 0:
            # Interaction terms
            for i in range(min(n_features, 3)):
                for j in range(i+1, min(n_features, 4)):
                    y_linear += nonlinearity * weights[i] * weights[j] * X[:, i] * X[:, j]
            
            # Squared terms
            for i in range(min(n_features, 3)):
                y_linear += nonlinearity * weights[i] * X[:, i]**2
        
        # Add noise
        y = y_linear + noise_level * np.random.randn(n_samples)
        
        # Create DataFrame and Series
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        return X_df, y_series
    
    def generate_classification_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        class_sep: float = 1.0,
        noise_level: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic classification data.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            n_classes: Number of classes
            class_sep: Separation between classes
            noise_level: Level of noise to add
            
        Returns:
            Tuple of features DataFrame and target Series
        """
        from sklearn.datasets import make_classification
        
        # Generate classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_features, 5),
            n_redundant=min(n_features // 3, 2),
            n_classes=n_classes,
            class_sep=class_sep,
            random_state=self.random_seed
        )
        
        # Add noise
        if noise_level > 0:
            X += noise_level * np.random.randn(*X.shape)
        
        # Create DataFrame and Series
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        return X_df, y_series
    
    def generate_ad_data(
        self,
        n_samples: int = 1000,
        include_text: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic ad data.
        
        Args:
            n_samples: Number of samples to generate
            include_text: Whether to include text features
            
        Returns:
            Tuple of features DataFrame and target Series
        """
        # Base features
        features = {
            'word_count': np.random.randint(100, 1000, n_samples),
            'sentiment_score': np.random.uniform(0, 1, n_samples),
            'complexity_score': np.random.uniform(0, 1, n_samples),
            'readability_score': np.random.uniform(0, 1, n_samples),
            'engagement_rate': np.random.uniform(0, 1, n_samples),
            'click_through_rate': np.random.uniform(0, 1, n_samples),
            'conversion_rate': np.random.uniform(0, 1, n_samples),
            'content_category': np.random.randint(0, 5, n_samples)
        }
        
        # Add text features if requested
        if include_text:
            features['ad_content'] = [
                f"Synthetic ad content {i} with keywords and description" 
                for i in range(n_samples)
            ]
        
        # Create target with relationship to features
        scores = (
            0.25 * features['sentiment_score'] +
            0.25 * features['readability_score'] +
            0.15 * features['engagement_rate'] +
            0.15 * features['click_through_rate'] +
            0.1 * features['sentiment_score'] * features['readability_score'] +
            0.1 * (1.0 - features['complexity_score'])
        )
        
        # Add noise
        scores += np.random.normal(0, 0.1, n_samples)
        
        # Scale to [0, 1] range
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return pd.DataFrame(features), pd.Series(scores, name='ad_score')
    
    def add_demographic_attributes(
        self,
        X: pd.DataFrame,
        n_demographics: int = 2
    ) -> pd.DataFrame:
        """Add demographic attributes to dataset.
        
        Args:
            X: Feature DataFrame
            n_demographics: Number of demographic attributes to add
            
        Returns:
            DataFrame with added demographic attributes
        """
        n_samples = len(X)
        X_copy = X.copy()
        
        # Define possible demographic attributes
        demographics = [
            ('gender', ['male', 'female']),
            ('age_group', ['18-25', '26-40', '41-60', '60+']),
            ('region', ['NA', 'EU', 'AS', 'AF', 'SA']),
            ('income_level', ['low', 'medium', 'high']),
            ('education', ['high_school', 'undergraduate', 'graduate']),
            ('device', ['mobile', 'desktop', 'tablet'])
        ]
        
        # Add selected demographic attributes
        for i in range(min(n_demographics, len(demographics))):
            attr_name, attr_values = demographics[i]
            X_copy[attr_name] = np.random.choice(attr_values, n_samples)
        
        return X_copy


def generate_synthetic_benchmark_data(
    output_dir: str = "benchmark_data",
    n_samples: int = 1000
) -> Dict[str, str]:
    """Generate and save synthetic benchmark datasets.
    
    Args:
        output_dir: Directory to save datasets
        n_samples: Number of samples to generate
        
    Returns:
        Dictionary of dataset paths
    """
    # Create generator
    generator = SyntheticDataGenerator()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate datasets
    datasets = {}
    
    # Regression data
    X_reg, y_reg = generator.generate_regression_data(n_samples=n_samples)
    reg_path = os.path.join(output_dir, "regression_data.csv")
    pd.concat([X_reg, y_reg], axis=1).to_csv(reg_path, index=False)
    datasets['regression'] = reg_path
    
    # Classification data
    X_cls, y_cls = generator.generate_classification_data(n_samples=n_samples)
    cls_path = os.path.join(output_dir, "classification_data.csv")
    pd.concat([X_cls, y_cls], axis=1).to_csv(cls_path, index=False)
    datasets['classification'] = cls_path
    
    # Ad data
    X_ad, y_ad = generator.generate_ad_data(n_samples=n_samples)
    ad_path = os.path.join(output_dir, "ad_data.csv")
    pd.concat([X_ad, y_ad], axis=1).to_csv(ad_path, index=False)
    datasets['ad'] = ad_path
    
    # Ad data with demographics
    X_ad_demo = generator.add_demographic_attributes(X_ad)
    ad_demo_path = os.path.join(output_dir, "ad_data_with_demographics.csv")
    pd.concat([X_ad_demo, y_ad], axis=1).to_csv(ad_demo_path, index=False)
    datasets['ad_with_demographics'] = ad_demo_path
    
    logger.info(f"Generated synthetic benchmark datasets in {output_dir}")
    
    return datasets


if __name__ == "__main__":
    # Example usage
    
    # Generate synthetic data
    datasets = generate_synthetic_benchmark_data()
    
    # Create benchmark results container
    results = BenchmarkResults()
    
    # Save example results
    results.save_results(
        model_name="TestModel",
        metrics={
            "RMSE": 8.1,
            "R²": 0.77,
            "Spearman": 0.73
        },
        targets={
            "RMSE": 8.2,
            "R²": 0.76,
            "Spearman": 0.72
        },
        metadata={
            "model_type": "hybrid",
            "n_features": 10,
            "training_time": 120
        }
    ) 