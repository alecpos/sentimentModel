#!/usr/bin/env python
"""
Fairness Visualization Module

This module provides visualization tools for fairness metrics and intersectional analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FairnessVisualizer:
    """
    Creates visualizations for fairness metrics.
    
    This class provides methods to visualize:
    - Demographic disparities
    - Intersectional heatmaps
    - Fairness metrics comparisons
    """
    
    def __init__(self, output_dir: str = "fairness_visualizations"):
        """
        Initialize the fairness visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set default style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })
    
    def create_demographic_disparity_plot(self, 
                                       fairness_metrics: Dict[str, Any], 
                                       metric_name: str = "predicted_positive_rate",
                                       protected_attr: Optional[str] = None,
                                       save_path: Optional[str] = None) -> str:
        """
        Create a bar plot showing disparities across demographic groups.
        
        Args:
            fairness_metrics: Dictionary of fairness metrics
            metric_name: Name of the metric to visualize
            protected_attr: Protected attribute to focus on (if None, will use the first one)
            save_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        # Extract univariate metrics
        univariate_metrics = fairness_metrics.get("univariate_metrics", {})
        
        if not univariate_metrics:
            logger.warning("No univariate metrics found in the fairness metrics")
            return None
        
        # If protected_attr not specified, use the first available
        if protected_attr is None:
            protected_attr = list(univariate_metrics.keys())[0]
        elif protected_attr not in univariate_metrics:
            logger.warning(f"Protected attribute '{protected_attr}' not found in metrics")
            protected_attr = list(univariate_metrics.keys())[0]
        
        # Extract groups and metric values
        group_metrics = univariate_metrics[protected_attr]
        groups = [metrics["group"] for metrics in group_metrics]
        values = [metrics.get(metric_name, 0) for metrics in group_metrics]
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "Group": [str(g) for g in groups],
            metric_name: values
        })
        
        # Calculate overall average for reference line
        overall_avg = np.mean(values)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="Group", y=metric_name, data=df, palette="viridis")
        
        # Add horizontal line for overall average
        plt.axhline(y=overall_avg, color='red', linestyle='--', 
                  label=f'Overall Average: {overall_avg:.3f}')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Formatting
        metric_display_name = metric_name.replace('_', ' ').title()
        plt.title(f'{metric_display_name} by {protected_attr.title()}')
        plt.ylabel(metric_display_name)
        plt.xlabel(protected_attr.title())
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / f"{protected_attr}_{metric_name}_disparity.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Demographic disparity plot saved to {save_path}")
        return str(save_path)
    
    def create_intersectional_heatmap(self, 
                                    fairness_metrics: Dict[str, Any],
                                    intersection_key: Optional[str] = None,
                                    metric_name: str = "group_positive_rate",
                                    save_path: Optional[str] = None) -> str:
        """
        Create a heatmap visualization for intersectional fairness metrics.
        
        Args:
            fairness_metrics: Dictionary of fairness metrics
            intersection_key: Key for the intersection to visualize (if None, will use the first one)
            metric_name: Name of the metric to visualize
            save_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        # Extract intersectional metrics
        intersectional_metrics = fairness_metrics.get("intersectional_metrics", {})
        
        if not intersectional_metrics:
            logger.warning("No intersectional metrics found in the fairness metrics")
            return None
        
        # If intersection_key not specified, use the first available
        if intersection_key is None:
            intersection_key = list(intersectional_metrics.keys())[0]
        elif intersection_key not in intersectional_metrics:
            logger.warning(f"Intersection key '{intersection_key}' not found in metrics")
            intersection_key = list(intersectional_metrics.keys())[0]
        
        # Extract attributes and groups
        attr1, attr2 = intersection_key.split('_')
        group_metrics = intersectional_metrics[intersection_key]
        
        # Create a grid for the heatmap
        unique_values1 = set(m[attr1] for m in group_metrics)
        unique_values2 = set(m[attr2] for m in group_metrics)
        
        # Convert to list and sort for consistent ordering
        unique_values1 = sorted(list(unique_values1), key=str)
        unique_values2 = sorted(list(unique_values2), key=str)
        
        # Create empty grid
        grid = np.zeros((len(unique_values1), len(unique_values2)))
        grid.fill(np.nan)  # Fill with NaN to handle missing combinations
        
        # Map values to indices
        value1_to_idx = {val: i for i, val in enumerate(unique_values1)}
        value2_to_idx = {val: i for i, val in enumerate(unique_values2)}
        
        # Fill grid with metric values
        for m in group_metrics:
            i = value1_to_idx[m[attr1]]
            j = value2_to_idx[m[attr2]]
            grid[i, j] = m.get(metric_name, np.nan)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Create mask for NaN values
        mask = np.isnan(grid)
        
        # Plot heatmap
        ax = sns.heatmap(grid, annot=True, fmt='.3f', 
                        xticklabels=[str(x) for x in unique_values2], 
                        yticklabels=[str(x) for x in unique_values1],
                        cmap='viridis', mask=mask)
        
        # Formatting
        metric_display_name = metric_name.replace('_', ' ').title()
        plt.title(f'{metric_display_name} by {attr1.title()} × {attr2.title()}')
        plt.ylabel(attr1.title())
        plt.xlabel(attr2.title())
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / f"{attr1}_{attr2}_{metric_name}_heatmap.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Intersectional heatmap saved to {save_path}")
        return str(save_path)
    
    def create_disparity_comparison(self,
                                   fairness_metrics: Dict[str, Any],
                                   protected_attr: Optional[str] = None,
                                   metrics: List[str] = None,
                                   save_path: Optional[str] = None) -> str:
        """
        Create a grouped bar plot comparing different fairness metrics across groups.
        
        Args:
            fairness_metrics: Dictionary of fairness metrics
            protected_attr: Protected attribute to focus on (if None, will use the first one)
            metrics: List of metrics to compare (defaults to TPR and FPR)
            save_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        # Extract univariate metrics
        univariate_metrics = fairness_metrics.get("univariate_metrics", {})
        
        if not univariate_metrics:
            logger.warning("No univariate metrics found in the fairness metrics")
            return None
        
        # If protected_attr not specified, use the first available
        if protected_attr is None:
            protected_attr = list(univariate_metrics.keys())[0]
        elif protected_attr not in univariate_metrics:
            logger.warning(f"Protected attribute '{protected_attr}' not found in metrics")
            protected_attr = list(univariate_metrics.keys())[0]
        
        # Default metrics to compare
        if metrics is None:
            metrics = ["true_positive_rate", "false_positive_rate"]
        
        # Extract groups and metric values
        group_metrics = univariate_metrics[protected_attr]
        groups = [str(metrics["group"]) for metrics in group_metrics]
        
        # Create DataFrame for plotting
        data = []
        for metric_name in metrics:
            for i, group in enumerate(groups):
                value = group_metrics[i].get(metric_name, 0)
                data.append({
                    "Group": group,
                    "Metric": metric_name.replace('_', ' ').title(),
                    "Value": value
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x="Group", y="Value", hue="Metric", data=df, palette="deep")
        
        # Formatting
        plt.title(f'Fairness Metrics Comparison by {protected_attr.title()}')
        plt.ylabel('Value')
        plt.xlabel(protected_attr.title())
        plt.legend(title='Metric')
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            metrics_str = '_'.join([m.split('_')[0] for m in metrics])
            save_path = self.output_dir / f"{protected_attr}_{metrics_str}_comparison.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Disparity comparison plot saved to {save_path}")
        return str(save_path)
    
    def visualize_all_metrics(self, fairness_metrics: Dict[str, Any], 
                            prefix: str = "") -> List[str]:
        """
        Create a comprehensive set of visualizations for all metrics.
        
        Args:
            fairness_metrics: Dictionary of fairness metrics
            prefix: Prefix for output filenames
            
        Returns:
            List of paths to all generated visualizations
        """
        visualization_paths = []
        
        # Prepare output directory with prefix if needed
        output_dir = self.output_dir
        if prefix:
            output_dir = self.output_dir / prefix
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. For each protected attribute, create demographic disparity plots
        univariate_metrics = fairness_metrics.get("univariate_metrics", {})
        for attr in univariate_metrics.keys():
            # Positive rate plot (demographic parity)
            path = self.create_demographic_disparity_plot(
                fairness_metrics, 
                metric_name="predicted_positive_rate",
                protected_attr=attr,
                save_path=output_dir / f"{attr}_predicted_positive_rate.png"
            )
            if path:
                visualization_paths.append(path)
            
            # Accuracy plot
            path = self.create_demographic_disparity_plot(
                fairness_metrics, 
                metric_name="accuracy",
                protected_attr=attr,
                save_path=output_dir / f"{attr}_accuracy.png"
            )
            if path:
                visualization_paths.append(path)
            
            # TPR plot (equal opportunity)
            path = self.create_demographic_disparity_plot(
                fairness_metrics, 
                metric_name="true_positive_rate",
                protected_attr=attr,
                save_path=output_dir / f"{attr}_tpr.png"
            )
            if path:
                visualization_paths.append(path)
        
        # 2. For each intersection, create heatmaps
        intersectional_metrics = fairness_metrics.get("intersectional_metrics", {})
        for intersection_key in intersectional_metrics.keys():
            # Positive rate heatmap
            path = self.create_intersectional_heatmap(
                fairness_metrics,
                intersection_key=intersection_key,
                metric_name="group_positive_rate",
                save_path=output_dir / f"{intersection_key}_positive_rate_heatmap.png"
            )
            if path:
                visualization_paths.append(path)
            
            # Disparate impact heatmap
            path = self.create_intersectional_heatmap(
                fairness_metrics,
                intersection_key=intersection_key,
                metric_name="disparate_impact",
                save_path=output_dir / f"{intersection_key}_disparate_impact_heatmap.png"
            )
            if path:
                visualization_paths.append(path)
        
        # 3. Metric comparisons for each protected attribute
        for attr in univariate_metrics.keys():
            # TPR vs FPR (equalized odds components)
            path = self.create_disparity_comparison(
                fairness_metrics,
                protected_attr=attr,
                metrics=["true_positive_rate", "false_positive_rate"],
                save_path=output_dir / f"{attr}_tpr_fpr_comparison.png"
            )
            if path:
                visualization_paths.append(path)
        
        logger.info(f"Generated {len(visualization_paths)} visualizations")
        return visualization_paths


# Example usage if run as script
if __name__ == "__main__":
    # Load metrics from file if available, otherwise create synthetic data
    try:
        with open('demo_fairness_metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        # Generate synthetic data
        from enhanced_fairness_metrics import EnhancedFairnessMetrics
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic true labels and predictions
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        
        # Create synthetic protected attributes
        protected_attributes = pd.DataFrame({
            'gender': np.random.randint(0, 2, size=n_samples),
            'age_group': np.random.choice(['18-25', '26-35', '36-50', '51+'], size=n_samples),
            'location': np.random.choice(['urban', 'suburban', 'rural'], size=n_samples)
        })
        
        # Compute metrics
        calculator = EnhancedFairnessMetrics()
        metrics = calculator.compute_metrics(y_true, y_pred, protected_attributes)
    
    # Create visualizations
    visualizer = FairnessVisualizer(output_dir="fairness_viz_demo")
    paths = visualizer.visualize_all_metrics(metrics)
    
    print(f"Created {len(paths)} visualizations in directory 'fairness_viz_demo'") 