#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Fairness Evaluation Script

This script evaluates ML models for fairness across demographic groups using metrics
like demographic parity and equal opportunity. It helps identify and quantify bias
in model predictions and generates fairness reports.

Usage:
    python evaluate_model_fairness.py --model=path/to/model.pkl --data=path/to/data.csv --protected=column_name
    python evaluate_model_fairness.py --module=app.models.ml.prediction.ad_score_predictor --class=AdScorePredictor --data=path/to/data.csv --protected=gender

Examples:
    python evaluate_model_fairness.py --model=models/classifier.pkl --data=data/test_data.csv --protected=gender --target=outcome --threshold=0.5
    python evaluate_model_fairness.py --module=app.models.ml.prediction.ad_score_predictor --class=AdScorePredictor --data=data/users.csv --protected=age_group --target=conversion --report=fairness_report.json
"""

import os
import sys
import json
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional, Tuple, Set
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fairness_evaluator')

# Try to import optional dependencies
try:
    import sklearn
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some functionality may be limited.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available. Install with 'pip install joblib' for model loading.")

try:
    import aif360
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    logger.warning("AIF360 not available. Install with 'pip install aif360' for advanced fairness metrics.")


class FairnessEvaluator:
    """Evaluate ML models for fairness across demographic groups."""
    
    def __init__(self, model_path: Optional[str] = None, model_object: Optional[Any] = None):
        """Initialize the fairness evaluator.
        
        Args:
            model_path: Path to the serialized model file
            model_object: Pre-loaded model object
        """
        self.model = None
        self.fairness_metrics = {}
        self.fairness_summary = {}
        self.fair_threshold = 0.8  # Default threshold for fairness (80% rule)
        
        if model_path:
            self._load_model(model_path)
        elif model_object:
            self.model = model_object
    
    def _load_model(self, model_path: str) -> None:
        """Load a model from a file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            # Determine file type by extension
            path = Path(model_path)
            extension = path.suffix.lower()
            
            if extension in ('.pkl', '.pickle'):
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif extension == '.joblib':
                if not JOBLIB_AVAILABLE:
                    raise ImportError("joblib is required to load .joblib files.")
                self.model = joblib.load(model_path)
            else:
                logger.warning(f"Unknown model format: {extension}. Attempting to load with pickle.")
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate predictions using the loaded model.
        
        Args:
            X: Feature data for prediction
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Cannot make predictions.")
        
        try:
            # Try to use predict_proba if available (for classification)
            if hasattr(self.model, 'predict_proba') and callable(self.model.predict_proba):
                return self.model.predict_proba(X)
            # Otherwise use standard predict
            elif hasattr(self.model, 'predict') and callable(self.model.predict):
                return self.model.predict(X)
            else:
                raise AttributeError("Model doesn't have a predict or predict_proba method")
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def _get_predictions(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions, converting probabilities if needed.
        
        Args:
            X: Feature data
            threshold: Threshold for binary classification
            
        Returns:
            Binary predictions
        """
        preds = self.predict(X)
        
        # Handle different prediction formats
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            # Probability predictions for multi-class
            if preds.shape[1] == 2:
                # Binary classification, take the probability of class 1
                return (preds[:, 1] >= threshold).astype(int)
            else:
                # Multi-class classification, take the argmax
                return np.argmax(preds, axis=1)
        else:
            # Binary predictions or regression values
            if np.issubdtype(preds.dtype, np.floating):
                # Convert to binary using threshold
                return (preds >= threshold).astype(int)
            else:
                # Already binary
                return preds
    
    def evaluate_demographic_parity(self, X: Union[pd.DataFrame, np.ndarray], 
                                   protected_attribute: str,
                                   threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate demographic parity (statistical parity).
        
        Demographic parity is achieved when the percentage of positive outcomes
        is the same across all demographic groups.
        
        Args:
            X: Feature data
            protected_attribute: Name of the protected attribute column
            threshold: Decision threshold for binary classification
            
        Returns:
            Dictionary containing demographic parity metrics
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with column names")
        
        if protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute '{protected_attribute}' not found in data")
        
        # Get all unique values of the protected attribute
        protected_values = X[protected_attribute].unique()
        
        # Get predictions
        y_pred = self._get_predictions(X, threshold)
        
        # Calculate positive prediction rate for each group
        group_rates = {}
        overall_rate = y_pred.mean()
        
        for value in protected_values:
            mask = X[protected_attribute] == value
            group_preds = y_pred[mask]
            positive_rate = group_preds.mean() if len(group_preds) > 0 else 0
            group_rates[str(value)] = positive_rate
        
        # Calculate disparities
        if len(group_rates) < 2:
            logger.warning("Need at least two demographic groups for comparison")
            disparities = {'disparity_ratio': 1.0, 'max_disparity': 0.0}
        else:
            # Find min and max rates
            min_rate = min(group_rates.values())
            max_rate = max(group_rates.values())
            
            # Calculate disparity metrics
            disparity_ratio = min_rate / max_rate if max_rate > 0 else 1.0
            disparities = {
                'disparity_ratio': disparity_ratio,
                'max_disparity': max_rate - min_rate
            }
        
        # Combine all metrics
        metrics = {
            'overall_positive_rate': overall_rate,
            'group_positive_rates': group_rates,
            **disparities
        }
        
        # Save in the fairness metrics dictionary
        self.fairness_metrics['demographic_parity'] = metrics
        
        logger.info(f"Demographic parity disparities: {disparities}")
        return metrics
    
    def evaluate_equal_opportunity(self, X: Union[pd.DataFrame, np.ndarray], 
                                  y_true: Union[pd.Series, np.ndarray],
                                  protected_attribute: str,
                                  threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate equal opportunity.
        
        Equal opportunity is achieved when the true positive rate is the same
        across all demographic groups.
        
        Args:
            X: Feature data
            y_true: True labels
            protected_attribute: Name of the protected attribute column
            threshold: Decision threshold for binary classification
            
        Returns:
            Dictionary containing equal opportunity metrics
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with column names")
        
        if protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute '{protected_attribute}' not found in data")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for confusion matrix calculation")
        
        # Convert y_true to numpy array if it's not already
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        # Get all unique values of the protected attribute
        protected_values = X[protected_attribute].unique()
        
        # Get predictions
        y_pred = self._get_predictions(X, threshold)
        
        # Calculate true positive rate for each group
        group_tpr = {}
        
        # Overall TPR
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm[1, 0] + cm[1, 1] > 0:  # Check if there are any positive examples
            overall_tpr = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        else:
            overall_tpr = 0
        
        for value in protected_values:
            mask = X[protected_attribute] == value
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            # Skip groups with no positive examples
            if np.sum(group_y_true == 1) == 0:
                logger.warning(f"No positive examples for group {value}, skipping TPR calculation")
                continue
                
            # Calculate confusion matrix
            cm = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1])
            
            # True Positive Rate (Recall)
            if cm[1, 0] + cm[1, 1] > 0:  # Check if there are any positive examples
                tpr = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            else:
                tpr = 0
                
            group_tpr[str(value)] = tpr
        
        # Calculate disparities
        if len(group_tpr) < 2:
            logger.warning("Need at least two demographic groups with positive examples for comparison")
            disparities = {'tpr_disparity_ratio': 1.0, 'max_tpr_disparity': 0.0}
        else:
            # Find min and max rates
            min_tpr = min(group_tpr.values())
            max_tpr = max(group_tpr.values())
            
            # Calculate disparity metrics
            tpr_disparity_ratio = min_tpr / max_tpr if max_tpr > 0 else 1.0
            disparities = {
                'tpr_disparity_ratio': tpr_disparity_ratio,
                'max_tpr_disparity': max_tpr - min_tpr
            }
        
        # Combine all metrics
        metrics = {
            'overall_tpr': overall_tpr,
            'group_tpr': group_tpr,
            **disparities
        }
        
        # Save in the fairness metrics dictionary
        self.fairness_metrics['equal_opportunity'] = metrics
        
        logger.info(f"Equal opportunity disparities: {disparities}")
        return metrics
    
    def evaluate_fairness(self, X: Union[pd.DataFrame, np.ndarray], 
                         y_true: Optional[Union[pd.Series, np.ndarray]] = None,
                         protected_attributes: Union[str, List[str]] = None,
                         threshold: float = 0.5,
                         fairness_threshold: float = 0.8) -> Dict[str, Any]:
        """Evaluate multiple fairness metrics.
        
        Args:
            X: Feature data
            y_true: True labels (needed for some metrics)
            protected_attributes: List of protected attribute column names
            threshold: Decision threshold for binary classification
            fairness_threshold: Minimum ratio for considering a model fair (80% rule)
            
        Returns:
            Dictionary containing all fairness metrics
        """
        if protected_attributes is None:
            raise ValueError("At least one protected attribute must be specified")
            
        # Convert to list if a single string is provided
        if isinstance(protected_attributes, str):
            protected_attributes = [protected_attributes]
            
        self.fair_threshold = fairness_threshold
        results = {}
        
        # Evaluate each protected attribute
        for attr in protected_attributes:
            attr_results = {}
            
            # Always evaluate demographic parity
            demographic_parity = self.evaluate_demographic_parity(
                X, protected_attribute=attr, threshold=threshold
            )
            attr_results['demographic_parity'] = demographic_parity
            
            # Evaluate equal opportunity if y_true is available
            if y_true is not None:
                equal_opportunity = self.evaluate_equal_opportunity(
                    X, y_true, protected_attribute=attr, threshold=threshold
                )
                attr_results['equal_opportunity'] = equal_opportunity
            
            results[attr] = attr_results
        
        # Compute overall fairness summary
        self._compute_fairness_summary(results, fairness_threshold)
        
        return results
    
    def _compute_fairness_summary(self, results: Dict[str, Any], 
                                fairness_threshold: float) -> None:
        """Compute an overall fairness summary.
        
        Args:
            results: Dictionary of fairness results by attribute
            fairness_threshold: Threshold ratio for considering a model fair
        """
        # Track metrics for summary
        summary = {
            'fairness_threshold': fairness_threshold,
            'attributes_evaluated': list(results.keys()),
            'demographic_parity_passed': [],
            'equal_opportunity_passed': [],
            'overall_passed': True,
            'worst_case_metrics': {}
        }
        
        # Track worst case metrics
        worst_dp_ratio = 1.0
        worst_eo_ratio = 1.0
        worst_dp_disparity = 0.0
        worst_eo_disparity = 0.0
        
        # Check each attribute
        for attr, attr_results in results.items():
            # Check demographic parity
            if 'demographic_parity' in attr_results:
                dp_ratio = attr_results['demographic_parity']['disparity_ratio']
                dp_disparity = attr_results['demographic_parity']['max_disparity']
                
                dp_passed = dp_ratio >= fairness_threshold
                if dp_passed:
                    summary['demographic_parity_passed'].append(attr)
                else:
                    summary['overall_passed'] = False
                
                # Update worst case
                if dp_ratio < worst_dp_ratio:
                    worst_dp_ratio = dp_ratio
                if dp_disparity > worst_dp_disparity:
                    worst_dp_disparity = dp_disparity
            
            # Check equal opportunity if available
            if 'equal_opportunity' in attr_results:
                eo_ratio = attr_results['equal_opportunity']['tpr_disparity_ratio']
                eo_disparity = attr_results['equal_opportunity']['max_tpr_disparity']
                
                eo_passed = eo_ratio >= fairness_threshold
                if eo_passed:
                    summary['equal_opportunity_passed'].append(attr)
                else:
                    summary['overall_passed'] = False
                
                # Update worst case
                if eo_ratio < worst_eo_ratio:
                    worst_eo_ratio = eo_ratio
                if eo_disparity > worst_eo_disparity:
                    worst_eo_disparity = eo_disparity
        
        # Update worst case metrics
        summary['worst_case_metrics'] = {
            'demographic_parity_ratio': worst_dp_ratio,
            'demographic_parity_disparity': worst_dp_disparity,
            'equal_opportunity_ratio': worst_eo_ratio,
            'equal_opportunity_disparity': worst_eo_disparity
        }
        
        # Set fairness verdict
        if summary['overall_passed']:
            summary['fairness_verdict'] = "FAIR"
        else:
            summary['fairness_verdict'] = "UNFAIR"
        
        self.fairness_summary = summary
    
    def generate_fairness_report(self, output_path: str, 
                               include_plots: bool = True) -> str:
        """Generate a comprehensive fairness report.
        
        Args:
            output_path: Directory to save the report
            include_plots: Whether to include visualizations
            
        Returns:
            Path to the generated report
        """
        if not self.fairness_metrics:
            raise ValueError("No fairness metrics available. Run evaluate_fairness first.")
            
        os.makedirs(output_path, exist_ok=True)
        
        # Create plots directory if needed
        plots_dir = os.path.join(output_path, "plots")
        if include_plots:
            os.makedirs(plots_dir, exist_ok=True)
            plot_files = self._generate_fairness_plots(plots_dir)
        else:
            plot_files = {}
        
        # Create the report file
        report_path = os.path.join(output_path, "fairness_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Model Fairness Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write("## Summary\n\n")
            f.write(f"**Fairness Threshold:** {self.fairness_summary['fairness_threshold']:.2f} (80% rule)\n\n")
            f.write(f"**Overall Verdict:** {self.fairness_summary['fairness_verdict']}\n\n")
            
            # Add summary table
            f.write("| Metric | Worst Case Ratio | Threshold | Status |\n")
            f.write("|--------|-----------------|-----------|--------|\n")
            
            dp_ratio = self.fairness_summary['worst_case_metrics']['demographic_parity_ratio']
            dp_status = "✅ PASS" if dp_ratio >= self.fair_threshold else "❌ FAIL"
            f.write(f"| Demographic Parity | {dp_ratio:.2f} | {self.fair_threshold:.2f} | {dp_status} |\n")
            
            if 'equal_opportunity_ratio' in self.fairness_summary['worst_case_metrics']:
                eo_ratio = self.fairness_summary['worst_case_metrics']['equal_opportunity_ratio']
                eo_status = "✅ PASS" if eo_ratio >= self.fair_threshold else "❌ FAIL"
                f.write(f"| Equal Opportunity | {eo_ratio:.2f} | {self.fair_threshold:.2f} | {eo_status} |\n")
            
            f.write("\n")
            
            # Protected attributes evaluated
            f.write("### Protected Attributes Evaluated\n\n")
            for attr in self.fairness_summary['attributes_evaluated']:
                f.write(f"- {attr}\n")
            f.write("\n")
            
            # Include plots if available
            if include_plots and 'demographic_parity' in plot_files:
                f.write(f"![Demographic Parity](plots/{os.path.basename(plot_files['demographic_parity'])})\n\n")
            
            if include_plots and 'equal_opportunity' in plot_files:
                f.write(f"![Equal Opportunity](plots/{os.path.basename(plot_files['equal_opportunity'])})\n\n")
            
            # Detailed results for each protected attribute
            f.write("## Detailed Results\n\n")
            
            for attr, metrics in self.fairness_metrics.items():
                f.write(f"### {attr.replace('_', ' ').title()}\n\n")
                
                if 'demographic_parity' in metrics:
                    dp = metrics['demographic_parity']
                    f.write("#### Demographic Parity\n\n")
                    f.write("Demographic parity measures whether the model predicts positive outcomes at the same rate across different demographic groups.\n\n")
                    f.write(f"- Overall positive prediction rate: {dp['overall_positive_rate']:.4f}\n")
                    f.write("- Group positive prediction rates:\n")
                    
                    for group, rate in dp['group_positive_rates'].items():
                        f.write(f"  - {group}: {rate:.4f}\n")
                    
                    f.write(f"- Disparity ratio: {dp['disparity_ratio']:.4f}\n")
                    f.write(f"- Maximum disparity: {dp['max_disparity']:.4f}\n\n")
                
                if 'equal_opportunity' in metrics:
                    eo = metrics['equal_opportunity']
                    f.write("#### Equal Opportunity\n\n")
                    f.write("Equal opportunity measures whether the model has the same true positive rate (recall) across different demographic groups.\n\n")
                    f.write(f"- Overall true positive rate: {eo['overall_tpr']:.4f}\n")
                    f.write("- Group true positive rates:\n")
                    
                    for group, rate in eo['group_tpr'].items():
                        f.write(f"  - {group}: {rate:.4f}\n")
                    
                    f.write(f"- TPR disparity ratio: {eo['tpr_disparity_ratio']:.4f}\n")
                    f.write(f"- Maximum TPR disparity: {eo['max_tpr_disparity']:.4f}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if self.fairness_summary['overall_passed']:
                f.write("The model meets the minimum fairness criteria (80% rule) for all protected attributes. However, consider the following recommendations to further improve fairness:\n\n")
            else:
                f.write("The model does not meet the minimum fairness criteria for all protected attributes. Consider the following recommendations to address fairness issues:\n\n")
            
            f.write("1. **Bias Mitigation Techniques**: Consider implementing pre-processing, in-processing, or post-processing bias mitigation techniques.\n")
            f.write("2. **Data Collection**: Review data collection procedures to ensure representative sampling across all demographic groups.\n")
            f.write("3. **Feature Engineering**: Review features that may introduce or amplify biases in the model.\n")
            f.write("4. **Model Selection**: Consider using different model architectures that may be less prone to learning biased patterns.\n")
            f.write("5. **Threshold Adjustment**: Consider group-specific thresholds to balance error rates across groups.\n")
            
            # Fairness definition and metrics explanation
            f.write("\n## Metrics Explanation\n\n")
            f.write("### Demographic Parity (Statistical Parity)\n\n")
            f.write("A model satisfies demographic parity if the probability of receiving a positive outcome is the same for all demographic groups. In practice, we measure the ratio between the lowest and highest positive prediction rates across groups. A value of 1.0 indicates perfect parity, while the minimum acceptable threshold is typically 0.8 (80% rule).\n\n")
            
            f.write("### Equal Opportunity\n\n")
            f.write("A model satisfies equal opportunity if the true positive rate (recall) is the same for all demographic groups. This means that the probability of a qualified individual receiving a positive prediction should not depend on their demographic group. We measure the ratio between the lowest and highest true positive rates across groups.\n\n")
            
            f.write("### Fairness Threshold\n\n")
            f.write("The fairness threshold (typically 0.8 or 80%) comes from the 'four-fifths rule' used in US employment law, which states that the selection rate for any protected group should be at least 80% of the rate for the group with the highest selection rate.\n\n")
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_path, "fairness_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'fairness_metrics': self.fairness_metrics,
                'fairness_summary': self.fairness_summary
            }, f, indent=2)
        
        logger.info(f"Fairness report saved to {report_path}")
        return report_path
    
    def _generate_fairness_plots(self, output_dir: str) -> Dict[str, str]:
        """Generate plots visualizing fairness metrics.
        
        Args:
            output_dir: Directory to save the plots
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        plot_files = {}
        
        # Plot demographic parity if available
        if 'demographic_parity' in self.fairness_metrics:
            plot_path = self._plot_demographic_parity(output_dir)
            plot_files['demographic_parity'] = plot_path
        
        # Plot equal opportunity if available
        if 'equal_opportunity' in self.fairness_metrics:
            plot_path = self._plot_equal_opportunity(output_dir)
            plot_files['equal_opportunity'] = plot_path
        
        return plot_files
    
    def _plot_demographic_parity(self, output_dir: str) -> str:
        """Generate a plot for demographic parity.
        
        Args:
            output_dir: Directory to save the plot
            
        Returns:
            Path to the saved plot
        """
        dp_metrics = self.fairness_metrics['demographic_parity']
        group_rates = dp_metrics['group_positive_rates']
        
        plt.figure(figsize=(10, 6))
        
        # Sort groups by positive rate
        sorted_groups = sorted(group_rates.items(), key=lambda x: x[1])
        groups = [g[0] for g in sorted_groups]
        rates = [g[1] for g in sorted_groups]
        
        # Create bar chart
        bars = plt.bar(groups, rates, color='skyblue')
        
        # Add a line for the overall rate
        plt.axhline(y=dp_metrics['overall_positive_rate'], color='r', linestyle='-', label='Overall Rate')
        
        # Add threshold line at 80% of max rate
        max_rate = max(rates)
        plt.axhline(y=max_rate * self.fair_threshold, color='orange', linestyle='--', 
                   label=f'{int(self.fair_threshold*100)}% of Max Rate')
        
        # Customize the plot
        plt.xlabel('Demographic Group')
        plt.ylabel('Positive Prediction Rate')
        plt.title('Demographic Parity Comparison')
        plt.legend()
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'demographic_parity.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return plot_path
    
    def _plot_equal_opportunity(self, output_dir: str) -> str:
        """Generate a plot for equal opportunity.
        
        Args:
            output_dir: Directory to save the plot
            
        Returns:
            Path to the saved plot
        """
        eo_metrics = self.fairness_metrics['equal_opportunity']
        group_tpr = eo_metrics['group_tpr']
        
        plt.figure(figsize=(10, 6))
        
        # Sort groups by TPR
        sorted_groups = sorted(group_tpr.items(), key=lambda x: x[1])
        groups = [g[0] for g in sorted_groups]
        tprs = [g[1] for g in sorted_groups]
        
        # Create bar chart
        bars = plt.bar(groups, tprs, color='lightgreen')
        
        # Add a line for the overall TPR
        plt.axhline(y=eo_metrics['overall_tpr'], color='r', linestyle='-', label='Overall TPR')
        
        # Add threshold line at 80% of max TPR
        max_tpr = max(tprs)
        plt.axhline(y=max_tpr * self.fair_threshold, color='orange', linestyle='--', 
                   label=f'{int(self.fair_threshold*100)}% of Max TPR')
        
        # Customize the plot
        plt.xlabel('Demographic Group')
        plt.ylabel('True Positive Rate')
        plt.title('Equal Opportunity Comparison')
        plt.legend()
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'equal_opportunity.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return plot_path


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a file.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Loaded data as a DataFrame
    """
    try:
        # Infer file type from extension
        extension = Path(data_path).suffix.lower()
        
        if extension == '.csv':
            data = pd.read_csv(data_path)
        elif extension == '.parquet':
            data = pd.read_parquet(data_path)
        elif extension in ('.xls', '.xlsx'):
            data = pd.read_excel(data_path)
        elif extension == '.json':
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        logger.info(f"Successfully loaded data from {data_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_model_from_module(module_path: str, class_name: str, **kwargs) -> Any:
    """Load a model by importing a module and instantiating a class.
    
    Args:
        module_path: Import path for the module
        class_name: Name of the class to instantiate
        **kwargs: Arguments to pass to the class constructor
        
    Returns:
        Instantiated model object
    """
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model = model_class(**kwargs)
        
        logger.info(f"Successfully loaded {class_name} from {module_path}")
        return model
    
    except ImportError:
        logger.error(f"Could not import module: {module_path}")
        raise
    except AttributeError:
        logger.error(f"Class {class_name} not found in module {module_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from module: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate model fairness across demographic groups")
    
    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                      help="Path to data file containing features and protected attributes")
    
    # Protected attribute arguments
    parser.add_argument("--protected", type=str, required=True,
                      help="Name of the protected attribute column(s), comma-separated for multiple")
    
    # Target arguments
    parser.add_argument("--target", type=str, default=None,
                      help="Name of the target column for equal opportunity evaluation")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="fairness_evaluation",
                      help="Directory to save fairness report")
    parser.add_argument("--report", type=str, default=None,
                      help="Path to save JSON metrics report")
    
    # Model loading arguments (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str,
                         help="Path to serialized model file")
    model_group.add_argument("--module", type=str,
                         help="Python module path containing the model class")
    
    # Additional arguments
    parser.add_argument("--class", dest="class_name", type=str,
                      help="Class name to instantiate (required with --module)")
    parser.add_argument("--threshold", type=float, default=0.5,
                      help="Decision threshold for binary classification")
    parser.add_argument("--fairness-threshold", type=float, default=0.8,
                      help="Minimum ratio for considering a model fair (80% rule)")
    parser.add_argument("--no-plots", action="store_true",
                      help="Disable generation of plots")
    
    args = parser.parse_args()
    
    try:
        # Load data
        data = load_data(args.data)
        logger.info(f"Loaded data with shape {data.shape}")
        
        # Split features, target, and protected attributes
        protected_attributes = [attr.strip() for attr in args.protected.split(',')]
        
        if args.target and args.target in data.columns:
            y_true = data[args.target]
            columns_to_drop = [args.target]
            logger.info(f"Using '{args.target}' as target variable")
        else:
            y_true = None
            columns_to_drop = []
            logger.info("No target variable specified or not found in data")
        
        # Keep protected attributes in X for fairness evaluation
        X = data.copy()
        
        if args.target:
            X = X.drop(columns=columns_to_drop)
        
        # Load model
        if args.model:
            evaluator = FairnessEvaluator(model_path=args.model)
        elif args.module:
            if not args.class_name:
                logger.error("--class is required when using --module")
                return 1
                
            model = load_model_from_module(args.module, args.class_name)
            evaluator = FairnessEvaluator(model_object=model)
        
        # Evaluate fairness
        evaluator.evaluate_fairness(
            X=X, 
            y_true=y_true, 
            protected_attributes=protected_attributes,
            threshold=args.threshold,
            fairness_threshold=args.fairness_threshold
        )
        
        # Generate report
        report_path = evaluator.generate_fairness_report(
            output_path=args.output,
            include_plots=not args.no_plots
        )
        
        # Save metrics to specified JSON file if requested
        if args.report:
            with open(args.report, 'w') as f:
                json.dump({
                    'fairness_metrics': evaluator.fairness_metrics,
                    'fairness_summary': evaluator.fairness_summary
                }, f, indent=2)
            logger.info(f"Fairness metrics saved to {args.report}")
        
        # Print summary to console
        summary = evaluator.fairness_summary
        print("\n" + "="*60)
        print("MODEL FAIRNESS EVALUATION SUMMARY")
        print("="*60)
        print(f"Fairness threshold: {summary['fairness_threshold']:.2f}")
        print(f"Overall verdict: {summary['fairness_verdict']}")
        print("-"*60)
        print(f"Demographic parity ratio (worst case): {summary['worst_case_metrics']['demographic_parity_ratio']:.4f}")
        if 'equal_opportunity_ratio' in summary['worst_case_metrics']:
            print(f"Equal opportunity ratio (worst case): {summary['worst_case_metrics']['equal_opportunity_ratio']:.4f}")
        print("="*60)
        print(f"Detailed report: {report_path}")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error evaluating model fairness: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 