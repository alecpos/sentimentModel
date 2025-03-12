"""
Fairness Evaluation Module for WITHIN System

This module provides utilities for evaluating fairness metrics across protected
attributes, generating visualizations, and implementing bias mitigation strategies.
The module is designed to work with the WITHIN Ad Score & Account Health Predictor
system to ensure fair treatment across demographic groups.

Implementation follows WITHIN ML Backend standards with strict type safety,
comprehensive documentation, and regulatory compliance capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import json
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FairnessMetric(str, Enum):
    """Fairness metrics supported by the WITHIN system."""
    
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    DISPARATE_IMPACT = "disparate_impact"
    FALSE_POSITIVE_RATE_PARITY = "false_positive_rate_parity"
    FALSE_NEGATIVE_RATE_PARITY = "false_negative_rate_parity"
    ACCURACY_PARITY = "accuracy_parity"
    TREATMENT_EQUALITY = "treatment_equality"

class FairnessThreshold(float, Enum):
    """Standard fairness thresholds for different regulations."""
    
    STRICT = 0.05     # 5% difference (e.g., EU AI Act high-risk systems)
    STANDARD = 0.10   # 10% difference (e.g., general fairness standards)
    RELAXED = 0.20    # 20% difference (e.g., early development phase)
    DISPARATE_IMPACT_THRESHOLD = 0.8  # 4/5ths rule

@dataclass
class FairnessResults:
    """Results of fairness evaluation.
    
    Attributes:
        protected_attribute: Name of the protected attribute
        overall_metrics: Dictionary of overall fairness metrics
        group_metrics: Dictionary of metrics by group
        passing_metrics: Set of metrics that pass the fairness threshold
        failing_metrics: Set of metrics that fail the fairness threshold
        threshold: Fairness threshold used for evaluation
        visualization_paths: Paths to generated visualizations
    """
    
    protected_attribute: str
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    group_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    passing_metrics: Set[str] = field(default_factory=set)
    failing_metrics: Set[str] = field(default_factory=set)
    threshold: float = FairnessThreshold.STANDARD.value
    visualization_paths: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the results to a dictionary.
        
        Returns:
            Dictionary representation of fairness results
        """
        return {
            "protected_attribute": self.protected_attribute,
            "overall_metrics": self.overall_metrics,
            "group_metrics": self.group_metrics,
            "passing_metrics": list(self.passing_metrics),
            "failing_metrics": list(self.failing_metrics),
            "threshold": self.threshold,
            "visualization_paths": self.visualization_paths
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FairnessResults':
        """Create FairnessResults from a dictionary.
        
        Args:
            data: Dictionary with fairness results
            
        Returns:
            FairnessResults object
        """
        return cls(
            protected_attribute=data["protected_attribute"],
            overall_metrics=data.get("overall_metrics", {}),
            group_metrics=data.get("group_metrics", {}),
            passing_metrics=set(data.get("passing_metrics", [])),
            failing_metrics=set(data.get("failing_metrics", [])),
            threshold=data.get("threshold", FairnessThreshold.STANDARD.value),
            visualization_paths=data.get("visualization_paths", {})
        )
    
    def summary(self) -> str:
        """Generate a human-readable summary of fairness results.
        
        Returns:
            Text summary of fairness evaluation results
        """
        lines = [
            f"Fairness Evaluation for protected attribute: {self.protected_attribute}",
            f"Threshold: {self.threshold}",
            "\nOverall Metrics:",
        ]
        
        for metric, value in self.overall_metrics.items():
            status = "PASS" if metric in self.passing_metrics else "FAIL"
            lines.append(f"  {metric}: {value:.4f} ({status})")
        
        lines.append("\nGroup Metrics:")
        for group, metrics in self.group_metrics.items():
            lines.append(f"  {group}:")
            for metric, value in metrics.items():
                lines.append(f"    {metric}: {value:.4f}")
        
        if self.visualization_paths:
            lines.append("\nVisualizations:")
            for name, path in self.visualization_paths.items():
                lines.append(f"  {name}: {path}")
        
        return "\n".join(lines)

class FairnessEvaluator:
    """Evaluator for measuring fairness metrics across protected attributes."""
    
    def __init__(
        self, 
        output_dir: str = "fairness_results",
        threshold: float = FairnessThreshold.STANDARD.value,
        metrics: Optional[List[str]] = None,
        save_visualizations: bool = True
    ) -> None:
        """Initialize the fairness evaluator.
        
        Args:
            output_dir: Directory to save fairness results and visualizations
            threshold: Fairness threshold for determining pass/fail
            metrics: List of fairness metrics to evaluate
            save_visualizations: Whether to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.save_visualizations = save_visualizations
        
        # Create output directory if it doesn't exist
        if self.save_visualizations:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Default metrics if none specified
        if metrics is None:
            self.metrics = [
                FairnessMetric.DEMOGRAPHIC_PARITY,
                FairnessMetric.EQUAL_OPPORTUNITY,
                FairnessMetric.DISPARATE_IMPACT
            ]
        else:
            self.metrics = [FairnessMetric(m) if isinstance(m, str) else m for m in metrics]
            
        logger.info(f"Initialized fairness evaluator with threshold {threshold}")
    
    def evaluate(
        self,
        df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        prediction_column: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> FairnessResults:
        """Evaluate fairness metrics for a protected attribute.
        
        Args:
            df: DataFrame containing features, targets, and predictions
            protected_attribute: Name of the protected attribute column
            target_column: Name of the target column
            prediction_column: Name of the prediction column (if None, uses target for analysis)
            threshold: Fairness threshold (overrides the default threshold)
            
        Returns:
            FairnessResults object with evaluation results
        """
        if protected_attribute not in df.columns:
            raise ValueError(f"Protected attribute '{protected_attribute}' not in DataFrame")
            
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not in DataFrame")
            
        # If prediction_column not provided, set to target for fairness metric calculation on raw data
        y_true = df[target_column]
        y_pred = df[prediction_column] if prediction_column and prediction_column in df.columns else y_true
        
        # Handle different data types
        if pd.api.types.is_numeric_dtype(y_true):
            # Convert to binary for fairness metrics if not already binary
            if not set(y_true.unique()).issubset({0, 1, True, False}):
                logger.warning(f"Target is not binary, using median threshold for binarization")
                threshold_value = y_true.median()
                y_true_binary = (y_true > threshold_value).astype(int)
            else:
                y_true_binary = y_true.astype(int)
                
            if not set(y_pred.unique()).issubset({0, 1, True, False}):
                logger.warning(f"Predictions are not binary, using median threshold for binarization")
                threshold_value = y_pred.median()
                y_pred_binary = (y_pred > threshold_value).astype(int)
            else:
                y_pred_binary = y_pred.astype(int)
        else:
            # For categorical targets, convert to binary (1 for positive class, 0 for others)
            # This is a simple approach; might need refinement for multi-class cases
            logger.warning(f"Non-numeric target, treating as binary classification")
            
            # For simplicity, convert the most frequent class to 1, others to 0
            positive_class = y_true.value_counts().index[0]
            y_true_binary = (y_true == positive_class).astype(int)
            y_pred_binary = (y_pred == positive_class).astype(int)
        
        # Get unique values of protected attribute
        protected_groups = df[protected_attribute].unique()
        
        # Initialize fairness results
        results = FairnessResults(
            protected_attribute=protected_attribute,
            threshold=threshold if threshold is not None else self.threshold
        )
        
        # Calculate metrics across groups
        group_metrics = {}
        for group in protected_groups:
            group_mask = df[protected_attribute] == group
            
            if group_mask.sum() == 0:
                logger.warning(f"No samples for group {group}, skipping")
                continue
                
            group_metrics[str(group)] = self._calculate_group_metrics(
                y_true_binary[group_mask], 
                y_pred_binary[group_mask]
            )
        
        # Calculate overall fairness metrics
        overall_metrics = self._calculate_overall_fairness_metrics(group_metrics)
        results.overall_metrics = overall_metrics
        results.group_metrics = group_metrics
        
        # Determine passing and failing metrics
        for metric, value in overall_metrics.items():
            if metric == FairnessMetric.DISPARATE_IMPACT.value:
                # For disparate impact, higher is better (should be close to 1.0)
                if value >= FairnessThreshold.DISPARATE_IMPACT_THRESHOLD.value:
                    results.passing_metrics.add(metric)
                else:
                    results.failing_metrics.add(metric)
            else:
                # For other metrics, lower is better (should be close to 0.0)
                if value <= results.threshold:
                    results.passing_metrics.add(metric)
                else:
                    results.failing_metrics.add(metric)
        
        # Generate visualizations if enabled
        if self.save_visualizations:
            visualization_paths = self._generate_visualizations(
                df, 
                protected_attribute, 
                target_column, 
                prediction_column, 
                results
            )
            results.visualization_paths = visualization_paths
        
        return results 

    def _calculate_group_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for a group.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of performance metrics
        """
        # Convert to numpy arrays for metric calculation
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Handle empty arrays
        if len(y_true) == 0 or len(y_pred) == 0:
            logger.warning("Empty arrays provided for metric calculation")
            return {
                "count": 0,
                "positive_rate": 0.0,
                "true_positive_rate": 0.0,
                "false_positive_rate": 0.0,
                "true_negative_rate": 0.0,
                "false_negative_rate": 0.0,
                "accuracy": 0.0
            }
        
        # Calculate confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError as e:
            logger.error(f"Error calculating confusion matrix: {str(e)}")
            return {
                "count": len(y_true),
                "positive_rate": float(np.mean(y_pred)),
                "true_positive_rate": 0.0,
                "false_positive_rate": 0.0,
                "true_negative_rate": 0.0,
                "false_negative_rate": 0.0,
                "accuracy": float(np.mean(y_true == y_pred))
            }
        
        # Calculate metrics
        count = len(y_true)
        positive_rate = float(np.mean(y_pred))
        
        # True positive rate (sensitivity, recall)
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # False positive rate (fall-out)
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # True negative rate (specificity)
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # False negative rate (miss rate)
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Positive predictive value (precision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return {
            "count": count,
            "positive_rate": positive_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "true_negative_rate": true_negative_rate,
            "false_negative_rate": false_negative_rate,
            "accuracy": accuracy,
            "precision": precision
        }
    
    def _calculate_overall_fairness_metrics(self, group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate overall fairness metrics across groups.
        
        Args:
            group_metrics: Dictionary of metrics by group
            
        Returns:
            Dictionary of fairness metrics
        """
        # If we don't have at least two groups, fairness metrics aren't meaningful
        if len(group_metrics) < 2:
            logger.warning("Not enough groups for fairness metrics calculation")
            return {metric.value: 0.0 for metric in self.metrics}
        
        fairness_metrics = {}
        
        # Demographic parity (difference in positive prediction rates)
        if FairnessMetric.DEMOGRAPHIC_PARITY in self.metrics:
            positive_rates = [metrics["positive_rate"] for metrics in group_metrics.values()]
            fairness_metrics[FairnessMetric.DEMOGRAPHIC_PARITY.value] = max(positive_rates) - min(positive_rates)
        
        # Equal opportunity (difference in true positive rates)
        if FairnessMetric.EQUAL_OPPORTUNITY in self.metrics:
            tpr_values = [metrics["true_positive_rate"] for metrics in group_metrics.values()]
            fairness_metrics[FairnessMetric.EQUAL_OPPORTUNITY.value] = max(tpr_values) - min(tpr_values)
        
        # Predictive parity (difference in precision)
        if FairnessMetric.PREDICTIVE_PARITY in self.metrics:
            precision_values = [metrics["precision"] for metrics in group_metrics.values()]
            fairness_metrics[FairnessMetric.PREDICTIVE_PARITY.value] = max(precision_values) - min(precision_values)
        
        # Accuracy parity (difference in accuracy)
        if FairnessMetric.ACCURACY_PARITY in self.metrics:
            accuracy_values = [metrics["accuracy"] for metrics in group_metrics.values()]
            fairness_metrics[FairnessMetric.ACCURACY_PARITY.value] = max(accuracy_values) - min(accuracy_values)
        
        # False positive rate parity (difference in false positive rates)
        if FairnessMetric.FALSE_POSITIVE_RATE_PARITY in self.metrics:
            fpr_values = [metrics["false_positive_rate"] for metrics in group_metrics.values()]
            fairness_metrics[FairnessMetric.FALSE_POSITIVE_RATE_PARITY.value] = max(fpr_values) - min(fpr_values)
        
        # False negative rate parity (difference in false negative rates)
        if FairnessMetric.FALSE_NEGATIVE_RATE_PARITY in self.metrics:
            fnr_values = [metrics["false_negative_rate"] for metrics in group_metrics.values()]
            fairness_metrics[FairnessMetric.FALSE_NEGATIVE_RATE_PARITY.value] = max(fnr_values) - min(fnr_values)
        
        # Treatment equality (ratio of false negatives to false positives across groups)
        if FairnessMetric.TREATMENT_EQUALITY in self.metrics:
            # Calculate treatment equality as the maximum difference in FN/FP ratios
            fn_fp_ratios = []
            for metrics in group_metrics.values():
                fn = metrics["false_negative_rate"] * metrics["count"]
                fp = metrics["false_positive_rate"] * metrics["count"]
                
                if fp > 0:
                    fn_fp_ratios.append(fn / fp)
                else:
                    fn_fp_ratios.append(0.0)
            
            fairness_metrics[FairnessMetric.TREATMENT_EQUALITY.value] = max(fn_fp_ratios) - min(fn_fp_ratios)
        
        # Disparate impact (ratio of positive prediction rates)
        if FairnessMetric.DISPARATE_IMPACT in self.metrics:
            positive_rates = [metrics["positive_rate"] for metrics in group_metrics.values()]
            max_rate = max(positive_rates)
            min_rate = min(positive_rates)
            
            # Avoid division by zero
            if max_rate > 0:
                fairness_metrics[FairnessMetric.DISPARATE_IMPACT.value] = min_rate / max_rate
            else:
                fairness_metrics[FairnessMetric.DISPARATE_IMPACT.value] = 1.0
        
        return fairness_metrics 

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        prediction_column: Optional[str] = None,
        results: FairnessResults = None
    ) -> Dict[str, str]:
        """Generate visualizations for fairness evaluation.
        
        Args:
            df: DataFrame containing features, targets, and predictions
            protected_attribute: Name of the protected attribute column
            target_column: Name of the target column
            prediction_column: Name of the prediction column (if None, uses target for analysis)
            results: FairnessResults object
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualization_paths = {}
        
        # Create directory for this attribute if it doesn't exist
        vis_dir = self.output_dir / protected_attribute
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Create fairness metrics bar chart
        metrics_path = self._plot_fairness_metrics(results, vis_dir)
        if metrics_path:
            visualization_paths["fairness_metrics"] = str(metrics_path)
            
        # 2. Create group metrics visualization
        group_metrics_path = self._plot_group_metrics(results, vis_dir)
        if group_metrics_path:
            visualization_paths["group_metrics"] = str(group_metrics_path)
            
        # 3. Create protected attribute distribution visualization
        if prediction_column:
            distribution_path = self._plot_protected_attribute_distribution(
                df, protected_attribute, target_column, prediction_column, vis_dir
            )
            if distribution_path:
                visualization_paths["distribution"] = str(distribution_path)
        
        # 4. Create intersectional heatmap if there are multiple protected attributes
        # Check if there are other protected attributes with fairness constraints
        if protected_attribute in df.columns:
            protected_cols = [
                col for col in df.columns 
                if col != protected_attribute and col.endswith("_protected") 
                or col in results.group_metrics 
                or "_" in col and col.split("_")[0] in ["gender", "age", "race", "ethnicity", "location"]
            ]
            
            for other_attr in protected_cols:
                if other_attr in df.columns and df[other_attr].nunique() > 1:
                    heatmap_path = self._plot_intersectional_heatmap(
                        df, protected_attribute, other_attr, target_column, prediction_column, vis_dir
                    )
                    if heatmap_path:
                        visualization_paths[f"intersectional_{other_attr}"] = str(heatmap_path)
        
        return visualization_paths
    
    def _plot_fairness_metrics(self, results: FairnessResults, output_dir: Path) -> Optional[Path]:
        """Plot fairness metrics as a horizontal bar chart.
        
        Args:
            results: FairnessResults object
            output_dir: Directory to save the plot
            
        Returns:
            Path to the saved plot, or None if plotting failed
        """
        try:
            metrics = results.overall_metrics
            if not metrics:
                logger.warning("No metrics to plot")
                return None
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Get metrics to plot (exclude disparate impact which has a different scale)
            metrics_to_plot = {k: v for k, v in metrics.items() if k != FairnessMetric.DISPARATE_IMPACT.value}
            
            # Sort metrics by value
            sorted_metrics = sorted(metrics_to_plot.items(), key=lambda x: x[1], reverse=True)
            
            # Plot bars
            metric_names = [m[0] for m in sorted_metrics]
            metric_values = [m[1] for m in sorted_metrics]
            
            # Format metric names for display
            display_names = [name.replace("_", " ").title() for name in metric_names]
            
            # Create bar chart
            bars = plt.barh(display_names, metric_values)
            
            # Color bars based on threshold
            for i, bar in enumerate(bars):
                if metric_values[i] <= results.threshold:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            # Add threshold line
            plt.axvline(x=results.threshold, color='red', linestyle='--', 
                        label=f'Threshold: {results.threshold}')
            
            # Add values to the end of each bar
            for i, v in enumerate(metric_values):
                plt.text(v + 0.01, i, f"{v:.4f}", va='center')
            
            # Customize plot
            plt.xlabel('Disparity (Lower is Better)')
            plt.title(f'Fairness Metrics for {results.protected_attribute.title()}')
            plt.xlim(0, max(max(metric_values) + 0.1, results.threshold + 0.1))
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            output_path = output_dir / f"{results.protected_attribute}_fairness_metrics.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating fairness metrics plot: {str(e)}")
            return None
    
    def _plot_group_metrics(self, results: FairnessResults, output_dir: Path) -> Optional[Path]:
        """Plot performance metrics by group.
        
        Args:
            results: FairnessResults object
            output_dir: Directory to save the plot
            
        Returns:
            Path to the saved plot, or None if plotting failed
        """
        try:
            group_metrics = results.group_metrics
            if not group_metrics:
                logger.warning("No group metrics to plot")
                return None
            
            # Metrics to visualize
            metrics_to_plot = ["positive_rate", "true_positive_rate"]
            group_names = list(group_metrics.keys())
            
            # Create plot with subplots for each metric
            fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4 * len(metrics_to_plot)))
            
            # If only one metric, axes is not a list
            if len(metrics_to_plot) == 1:
                axes = [axes]
            
            # Plot each metric as a separate bar chart
            for i, metric in enumerate(metrics_to_plot):
                # Extract metric values for each group
                values = [group_metrics[group].get(metric, 0) for group in group_names]
                
                # Sort groups by metric value for better visualization
                sorted_indices = np.argsort(values)[::-1]
                sorted_groups = [group_names[i] for i in sorted_indices]
                sorted_values = [values[i] for i in sorted_indices]
                
                # Create bar chart
                ax = axes[i]
                bars = ax.bar(sorted_groups, sorted_values)
                
                # Add count annotations
                for j, group in enumerate(sorted_groups):
                    count = group_metrics[group].get("count", 0)
                    ax.text(j, sorted_values[j] + 0.02, f"n={count}", 
                            ha='center', va='bottom', fontsize=9)
                
                # Customize subplot
                ax.set_ylim(0, 1.0)
                ax.set_title(f"{metric.replace('_', ' ').title()} by {results.protected_attribute.title()}")
                ax.set_ylabel("Rate")
                ax.tick_params(axis='x', rotation=45)
                
                # Add grid lines for better readability
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add title and adjust layout
            plt.suptitle(f"Group Metrics for {results.protected_attribute.title()}", y=1.02)
            plt.tight_layout()
            
            # Save plot
            output_path = output_dir / f"{results.protected_attribute}_group_metrics.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating group metrics plot: {str(e)}")
            return None
            
    def _plot_protected_attribute_distribution(
        self,
        df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        prediction_column: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Plot distribution of outcomes by protected attribute.
        
        Args:
            df: DataFrame containing features, targets, and predictions
            protected_attribute: Name of the protected attribute column
            target_column: Name of the target column
            prediction_column: Name of the prediction column
            output_dir: Directory to save the plot
            
        Returns:
            Path to the saved plot, or None if plotting failed
        """
        try:
            if protected_attribute not in df.columns:
                logger.warning(f"Protected attribute {protected_attribute} not in DataFrame")
                return None
                
            # Convert target and predictions to binary if needed
            y_true = df[target_column]
            y_pred = df[prediction_column] if prediction_column else y_true
            
            if pd.api.types.is_numeric_dtype(y_true) and not set(y_true.unique()).issubset({0, 1}):
                threshold_value = y_true.median()
                y_true = (y_true > threshold_value).astype(int)
                
            if pd.api.types.is_numeric_dtype(y_pred) and not set(y_pred.unique()).issubset({0, 1}):
                threshold_value = y_pred.median()
                y_pred = (y_pred > threshold_value).astype(int)
            
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                "attribute": df[protected_attribute],
                "target": y_true,
                "prediction": y_pred
            })
            
            # Group by attribute and outcome
            grouped = plot_df.groupby("attribute").agg({
                "target": ["count", "mean"],
                "prediction": "mean"
            })
            
            # Reset index for plotting
            grouped.columns = ["count", "target_rate", "prediction_rate"]
            grouped = grouped.reset_index()
            
            # Sort by count for better visualization
            grouped = grouped.sort_values("count", ascending=False)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Set width of bars
            bar_width = 0.35
            index = np.arange(len(grouped))
            
            # Plot bars
            ax.bar(index, grouped["count"], bar_width, label="Sample Count", color="lightgray")
            
            # Create twin axis for rates
            ax2 = ax.twinx()
            ax2.plot(index, grouped["target_rate"], 'o-', color="blue", label="Actual Rate")
            ax2.plot(index, grouped["prediction_rate"], 's--', color="red", label="Predicted Rate")
            ax2.set_ylim(0, 1.0)
            ax2.set_ylabel("Rate")
            
            # Set x-axis ticks and labels
            ax.set_xticks(index)
            ax.set_xticklabels(grouped["attribute"], rotation=45, ha="right")
            
            # Set labels and title
            ax.set_xlabel(protected_attribute.replace("_", " ").title())
            ax.set_ylabel("Sample Count")
            plt.title(f"Distribution and Outcome Rates by {protected_attribute.replace('_', ' ').title()}")
            
            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = output_dir / f"{protected_attribute}_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            return None
            
    def _plot_intersectional_heatmap(
        self,
        df: pd.DataFrame,
        attr1: str,
        attr2: str,
        target_column: str,
        prediction_column: Optional[str] = None,
        output_dir: Path = None
    ) -> Optional[Path]:
        """Plot intersectional heatmap showing metrics across combinations of protected attributes.
        
        Args:
            df: DataFrame containing features, targets, and predictions
            attr1: First protected attribute
            attr2: Second protected attribute
            target_column: Name of the target column
            prediction_column: Name of the prediction column (if None, uses target)
            output_dir: Directory to save the plot
            
        Returns:
            Path to the saved plot, or None if plotting failed
        """
        try:
            if attr1 not in df.columns or attr2 not in df.columns:
                logger.warning(f"Protected attributes {attr1} or {attr2} not in DataFrame")
                return None
                
            # Get predictions
            y_pred = df[prediction_column] if prediction_column and prediction_column in df.columns else df[target_column]
            
            # Convert to binary if needed
            if pd.api.types.is_numeric_dtype(y_pred) and not set(y_pred.unique()).issubset({0, 1}):
                threshold_value = y_pred.median()
                y_pred = (y_pred > threshold_value).astype(int)
            
            # Create contingency table of positive rates
            combined_df = pd.DataFrame({
                "attr1": df[attr1],
                "attr2": df[attr2],
                "outcome": y_pred
            })
            
            # Group by both attributes and calculate positive rate and count
            grouped = combined_df.groupby(["attr1", "attr2"]).agg({
                "outcome": ["mean", "count"]
            })
            
            # Convert to matrix for heatmap
            positive_rates = grouped["outcome", "mean"].unstack()
            counts = grouped["outcome", "count"].unstack()
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Create heatmap of positive rates
            im = plt.imshow(positive_rates, cmap="coolwarm", aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label("Positive Rate")
            
            # Annotate cells with positive rate and count
            for i in range(len(positive_rates.index)):
                for j in range(len(positive_rates.columns)):
                    try:
                        rate = positive_rates.iloc[i, j]
                        count = counts.iloc[i, j]
                        
                        # Skip if NaN (no data for this combination)
                        if pd.isna(rate) or pd.isna(count):
                            continue
                            
                        # Calculate text color based on background color intensity
                        text_color = "white" if rate > 0.5 else "black"
                        
                        plt.text(j, i, f"{rate:.3f}\nn={int(count)}", 
                                ha="center", va="center", color=text_color,
                                fontsize=9)
                    except Exception as e:
                        logger.warning(f"Error annotating cell ({i},{j}): {str(e)}")
            
            # Set ticks and labels
            plt.xticks(np.arange(len(positive_rates.columns)), positive_rates.columns, rotation=45, ha="right")
            plt.yticks(np.arange(len(positive_rates.index)), positive_rates.index)
            
            # Set labels and title
            plt.xlabel(attr2.replace("_", " ").title())
            plt.ylabel(attr1.replace("_", " ").title())
            plt.title(f"Intersectional Analysis: Positive Rates for {attr1.title()} × {attr2.title()}")
            
            # Adjust layout and save
            plt.tight_layout()
            
            if output_dir:
                output_path = output_dir / f"{attr1}_{attr2}_heatmap.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return output_path
            else:
                plt.show()
                plt.close()
                return None
                
        except Exception as e:
            logger.error(f"Error creating intersectional heatmap: {str(e)}")
            return None 

class FairnessMitigation:
    """Base class for fairness mitigation techniques."""
    
    def __init__(self, protected_attribute: str) -> None:
        """Initialize the fairness mitigation technique.
        
        Args:
            protected_attribute: Name of the protected attribute to mitigate bias for
        """
        self.protected_attribute = protected_attribute
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FairnessMitigation':
        """Fit the mitigation technique to the data.
        
        Args:
            X: Feature DataFrame, must contain the protected attribute
            y: Target series
            
        Returns:
            Self, for method chaining
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply the mitigation technique to transform the data.
        
        Args:
            X: Feature DataFrame, must contain the protected attribute
            y: Target series
            
        Returns:
            Tuple of (transformed X, transformed y)
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit and apply the mitigation technique in one step.
        
        Args:
            X: Feature DataFrame, must contain the protected attribute
            y: Target series
            
        Returns:
            Tuple of (transformed X, transformed y)
        """
        return self.fit(X, y).transform(X, y)

class Reweighing(FairnessMitigation):
    """Mitigate bias by reweighing samples to ensure fairness.
    
    This technique assigns weights to training examples to ensure fairness
    with respect to a protected attribute. It calculates weights such that
    the expected values of protected attributes are balanced in the training set.
    """
    
    def __init__(self, protected_attribute: str) -> None:
        """Initialize the reweighing technique.
        
        Args:
            protected_attribute: Name of the protected attribute to mitigate bias for
        """
        super().__init__(protected_attribute)
        self.weights = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Reweighing':
        """Compute instance weights to ensure fairness.
        
        Args:
            X: Feature DataFrame, must contain the protected attribute
            y: Target series
            
        Returns:
            Self, for method chaining
        """
        if self.protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute '{self.protected_attribute}' not in DataFrame")
            
        # Calculate group frequencies
        protected = X[self.protected_attribute]
        
        # Convert y to binary if needed
        if not set(y.unique()).issubset({0, 1, True, False}):
            logger.warning("Target is not binary, using median threshold for binarization")
            threshold_value = y.median()
            y_binary = (y > threshold_value).astype(int)
        else:
            y_binary = y.astype(int)
        
        # Calculate key values for weighting
        n_samples = len(y_binary)
        protected_groups = protected.unique()
        
        # Calculate expectation values
        expected_y_1 = sum(y_binary) / n_samples
        expected_y_0 = 1 - expected_y_1
        
        # Calculate weights for each combination of (protected, y)
        self.weights = {}
        
        for group in protected_groups:
            group_mask = (protected == group)
            group_size = sum(group_mask)
            
            if group_size == 0:
                continue
                
            expected_protected = group_size / n_samples
            
            for y_val in [0, 1]:
                y_mask = (y_binary == y_val)
                
                # Count samples in this (protected, y) group
                count = sum(group_mask & y_mask)
                
                if count == 0:
                    continue
                    
                # Calculate joint probability
                observed_joint = count / n_samples
                
                # Calculate expected joint probability (independence between protected and y)
                expected_joint = expected_protected * (expected_y_1 if y_val == 1 else expected_y_0)
                
                # Calculate weight for this combination
                weight = expected_joint / observed_joint
                
                self.weights[(group, y_val)] = weight
        
        return self
        
    def transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply instance weights to the data (returns original data with sample weights).
        
        Args:
            X: Feature DataFrame, must contain the protected attribute
            y: Target series
            
        Returns:
            Tuple of (original X, original y) - use get_sample_weights() to get weights
        """
        if self.weights is None:
            raise ValueError("Must call fit() before transform()")
            
        return X, y
        
    def get_sample_weights(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Get sample weights for each instance.
        
        Args:
            X: Feature DataFrame, must contain the protected attribute
            y: Target series
            
        Returns:
            Array of sample weights
        """
        if self.weights is None:
            raise ValueError("Must call fit() before get_sample_weights()")
            
        if self.protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute '{self.protected_attribute}' not in DataFrame")
            
        # Get protected attribute values
        protected = X[self.protected_attribute]
        
        # Convert y to binary if needed
        if not set(y.unique()).issubset({0, 1, True, False}):
            logger.warning("Target is not binary, using median threshold for binarization")
            threshold_value = y.median()
            y_binary = (y > threshold_value).astype(int)
        else:
            y_binary = y.astype(int)
        
        # Assign weights
        sample_weights = np.ones(len(X))
        
        for i in range(len(X)):
            group = protected.iloc[i]
            y_val = y_binary.iloc[i]
            
            key = (group, y_val)
            
            if key in self.weights:
                sample_weights[i] = self.weights[key]
        
        return sample_weights

class FairDataTransformer(FairnessMitigation):
    """Transform data to remove bias with respect to a protected attribute.
    
    This technique transforms the feature space to ensure that the protected
    attribute cannot be predicted from the transformed features, while preserving
    as much information as possible for predicting the target variable.
    """
    
    def __init__(
        self, 
        protected_attribute: str, 
        method: str = "decorrelation",
        random_state: int = 42
    ) -> None:
        """Initialize the fair data transformer.
        
        Args:
            protected_attribute: Name of the protected attribute to mitigate bias for
            method: Transformation method ('decorrelation' or 'adversarial')
            random_state: Random seed for reproducibility
        """
        super().__init__(protected_attribute)
        self.method = method
        self.random_state = random_state
        self.transformer = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FairDataTransformer':
        """Fit the fair data transformer.
        
        Args:
            X: Feature DataFrame, must contain the protected attribute
            y: Target series
            
        Returns:
            Self, for method chaining
        """
        if self.protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute '{self.protected_attribute}' not in DataFrame")
            
        # Store feature names (excluding protected attribute)
        self.feature_names = [col for col in X.columns if col != self.protected_attribute]
        
        # Extract protected attribute
        protected = X[self.protected_attribute]
        
        # Extract features (excluding protected attribute)
        X_features = X[self.feature_names]
        
        # Fit transformer based on method
        if self.method == "decorrelation":
            self._fit_decorrelation(X_features, protected, y)
        elif self.method == "adversarial":
            self._fit_adversarial(X_features, protected, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def _fit_decorrelation(self, X: pd.DataFrame, protected: pd.Series, y: pd.Series) -> None:
        """Fit decorrelation transformer.
        
        This method removes correlations between features and the protected attribute
        by projecting the data onto the null space of the protected attribute.
        
        Args:
            X: Feature DataFrame (excluding protected attribute)
            protected: Protected attribute series
            y: Target series
        """
        # Convert categorical protected attribute to one-hot encoding
        if not pd.api.types.is_numeric_dtype(protected):
            protected_dummies = pd.get_dummies(protected, drop_first=True)
            protected_matrix = protected_dummies.values
        else:
            protected_matrix = protected.values.reshape(-1, 1)
            
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate correlation matrix between features and protected attribute
        X_protected = np.column_stack([X_scaled, protected_matrix])
        corr_matrix = np.corrcoef(X_protected, rowvar=False)
        
        # Extract feature-protected correlations
        n_features = X_scaled.shape[1]
        n_protected = protected_matrix.shape[1]
        
        corr_with_protected = corr_matrix[:n_features, n_features:n_features+n_protected]
        
        # Use singular value decomposition to find projection matrix
        U, S, Vh = np.linalg.svd(corr_with_protected, full_matrices=False)
        
        # Create projection matrix that removes protected attribute information
        self.projection_matrix = np.eye(n_features) - U @ U.T
        
        # Store transformer details
        self.transformer = {
            "scaler": self.scaler,
            "projection_matrix": self.projection_matrix
        }
    
    def _fit_adversarial(self, X: pd.DataFrame, protected: pd.Series, y: pd.Series) -> None:
        """Fit adversarial transformer using scikit-learn components.
        
        This is a simplified version that uses decorrelation as a proxy for 
        adversarial learning. A full implementation would require TensorFlow
        or PyTorch for adversarial training.
        
        Args:
            X: Feature DataFrame (excluding protected attribute)
            protected: Protected attribute series
            y: Target series
        """
        # For this implementation, use the same approach as decorrelation
        # A real implementation would use adversarial training
        self._fit_decorrelation(X, protected, y)
        
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Transform data to remove bias.
        
        Args:
            X: Feature DataFrame, must contain the protected attribute
            y: Target series (returned unchanged)
            
        Returns:
            Tuple of (transformed X, original y)
        """
        if self.transformer is None:
            raise ValueError("Must call fit() before transform()")
            
        if self.protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute '{self.protected_attribute}' not in DataFrame")
            
        # Extract features (excluding protected attribute)
        X_features = X[self.feature_names]
        
        # Apply transformation
        X_scaled = self.scaler.transform(X_features)
        X_transformed = X_scaled @ self.projection_matrix
        
        # Convert back to DataFrame
        X_fair = pd.DataFrame(X_transformed, index=X.index, columns=self.feature_names)
        
        # Add protected attribute back (for transparency, not for modeling)
        X_fair[self.protected_attribute] = X[self.protected_attribute]
        
        # Return transformed data and original y
        return X_fair, y if y is not None else None 