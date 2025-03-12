#!/usr/bin/env python
"""
Enhanced Fairness Metrics Module

This module implements advanced fairness metrics for measuring model fairness
across various demographic groups with more sophisticated intersectional analysis.
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

class EnhancedFairnessMetrics:
    """
    Implements advanced fairness metrics for evaluating model fairness.
    
    This class extends standard fairness metrics with:
    - Advanced intersectional analysis 
    - Aggregate fairness scores
    - Fairness confidence intervals
    """
    
    def __init__(self):
        """Initialize the enhanced fairness metrics calculator."""
        pass
    
    def compute_metrics(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      protected_attributes: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute enhanced fairness metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Dictionary with computed fairness metrics
        """
        # Basic structure for the metrics
        metrics = {
            "univariate_metrics": self._compute_univariate_metrics(y_true, y_pred, protected_attributes),
            "intersectional_metrics": self._compute_intersectional_metrics(y_true, y_pred, protected_attributes),
            "aggregate_scores": self._compute_aggregate_scores(y_true, y_pred, protected_attributes),
            "summary": self._generate_summary(y_true, y_pred, protected_attributes)
        }
        
        return metrics
    
    def _compute_univariate_metrics(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 protected_attributes: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute fairness metrics for each protected attribute individually.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Dictionary mapping attribute names to metrics for each group
        """
        univariate_metrics = {}
        
        # For each protected attribute column
        for col in protected_attributes.columns:
            # Get unique values
            unique_groups = protected_attributes[col].unique()
            group_metrics = []
            
            # Compute metrics for each group
            for group in unique_groups:
                group_mask = protected_attributes[col] == group
                
                # Skip groups with too few samples
                if sum(group_mask) < 10:
                    continue
                
                # Get predictions and ground truth for this group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                # Compute basic metrics
                accuracy = np.mean(group_y_pred == group_y_true)
                
                # Calculate confusion matrix-based metrics
                true_positives = np.sum((group_y_true == 1) & (group_y_pred == 1))
                false_positives = np.sum((group_y_true == 0) & (group_y_pred == 1))
                true_negatives = np.sum((group_y_true == 0) & (group_y_pred == 0))
                false_negatives = np.sum((group_y_true == 1) & (group_y_pred == 0))
                
                # Handle division by zero
                total_positives = true_positives + false_negatives
                total_negatives = true_negatives + false_positives
                
                true_positive_rate = true_positives / total_positives if total_positives > 0 else 0
                true_negative_rate = true_negatives / total_negatives if total_negatives > 0 else 0
                false_positive_rate = false_positives / total_negatives if total_negatives > 0 else 0
                false_negative_rate = false_negatives / total_positives if total_positives > 0 else 0
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                
                # Demographic parity metric
                predicted_positive_rate = np.mean(group_y_pred)
                
                # Group metrics
                group_metrics.append({
                    "group": group,
                    "count": int(sum(group_mask)),
                    "accuracy": float(accuracy),
                    "true_positive_rate": float(true_positive_rate),
                    "true_negative_rate": float(true_negative_rate),
                    "false_positive_rate": float(false_positive_rate),
                    "false_negative_rate": float(false_negative_rate),
                    "precision": float(precision),
                    "predicted_positive_rate": float(predicted_positive_rate),
                    "actual_positive_rate": float(np.mean(group_y_true))
                })
            
            univariate_metrics[col] = group_metrics
            
        return univariate_metrics
    
    def _compute_intersectional_metrics(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     protected_attributes: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute intersectional fairness metrics across pairs of protected attributes.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Dictionary mapping intersectional groups to metrics
        """
        intersectional_metrics = {}
        
        # Get all pairs of protected attributes for intersectional analysis
        attr_cols = protected_attributes.columns
        
        # For each pair of protected attributes
        for i in range(len(attr_cols)):
            for j in range(i+1, len(attr_cols)):
                col1, col2 = attr_cols[i], attr_cols[j]
                intersection_key = f"{col1}_{col2}"
                group_metrics = []
                
                # Get unique combinations
                unique_groups1 = protected_attributes[col1].unique()
                unique_groups2 = protected_attributes[col2].unique()
                
                for group1 in unique_groups1:
                    for group2 in unique_groups2:
                        # Intersection mask
                        intersection_mask = (protected_attributes[col1] == group1) & (protected_attributes[col2] == group2)
                        group_size = sum(intersection_mask)
                        
                        # Skip groups with too few samples
                        if group_size < 10:
                            continue
                        
                        # Get predictions and ground truth for this intersection
                        group_y_true = y_true[intersection_mask]
                        group_y_pred = y_pred[intersection_mask]
                        
                        # Rest of the dataset (complement)
                        rest_mask = ~intersection_mask
                        rest_y_true = y_true[rest_mask]
                        rest_y_pred = y_pred[rest_mask]
                        
                        # Basic metrics
                        accuracy = np.mean(group_y_pred == group_y_true)
                        
                        # Demographic parity metrics
                        group_positive_rate = np.mean(group_y_pred)
                        rest_positive_rate = np.mean(rest_y_pred)
                        
                        # Disparate impact
                        disparate_impact = group_positive_rate / rest_positive_rate if rest_positive_rate > 0 else None
                        
                        # Equalized odds difference metrics
                        # For positive class (true positive rate difference)
                        pos_mask_group = group_y_true == 1
                        pos_mask_rest = rest_y_true == 1
                        
                        if sum(pos_mask_group) > 0 and sum(pos_mask_rest) > 0:
                            group_tpr = np.mean(group_y_pred[pos_mask_group])
                            rest_tpr = np.mean(rest_y_pred[pos_mask_rest])
                            tpr_difference = abs(group_tpr - rest_tpr)
                        else:
                            group_tpr = None
                            rest_tpr = None
                            tpr_difference = None
                        
                        # For negative class (false positive rate difference)
                        neg_mask_group = group_y_true == 0
                        neg_mask_rest = rest_y_true == 0
                        
                        if sum(neg_mask_group) > 0 and sum(neg_mask_rest) > 0:
                            group_fpr = np.mean(group_y_pred[neg_mask_group])
                            rest_fpr = np.mean(rest_y_pred[neg_mask_rest])
                            fpr_difference = abs(group_fpr - rest_fpr)
                        else:
                            group_fpr = None
                            rest_fpr = None
                            fpr_difference = None
                        
                        # Equalized odds combines both TPR and FPR differences
                        equalized_odds_diff = max(tpr_difference or 0, fpr_difference or 0) if tpr_difference is not None and fpr_difference is not None else None
                        
                        # Add to metrics
                        group_metrics.append({
                            col1: group1,
                            col2: group2,
                            "group_size": int(group_size),
                            "group_positive_rate": float(group_positive_rate),
                            "rest_positive_rate": float(rest_positive_rate),
                            "disparate_impact": float(disparate_impact) if disparate_impact is not None else None,
                            "group_tpr": float(group_tpr) if group_tpr is not None else None,
                            "group_fpr": float(group_fpr) if group_fpr is not None else None,
                            "rest_tpr": float(rest_tpr) if rest_tpr is not None else None,
                            "rest_fpr": float(rest_fpr) if rest_fpr is not None else None,
                            "tpr_difference": float(tpr_difference) if tpr_difference is not None else None,
                            "fpr_difference": float(fpr_difference) if fpr_difference is not None else None,
                            "equalized_odds_diff": float(equalized_odds_diff) if equalized_odds_diff is not None else None,
                            "accuracy": float(accuracy)
                        })
                
                intersectional_metrics[intersection_key] = group_metrics
        
        return intersectional_metrics
    
    def _compute_aggregate_scores(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                protected_attributes: pd.DataFrame) -> Dict[str, float]:
        """
        Compute aggregate fairness scores.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Dictionary of aggregate fairness scores
        """
        # This will be expanded in the next increment
        aggregate_scores = {
            "overall_accuracy": float(np.mean(y_pred == y_true))
        }
        
        return aggregate_scores
    
    def _generate_summary(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        protected_attributes: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of fairness metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Dictionary with summary statistics and overall fairness assessment
        """
        # Basic summary stats
        total_samples = len(y_true)
        accuracy = np.mean(y_pred == y_true)
        positive_rate = np.mean(y_pred)
        
        # This will be expanded in the next increment
        summary = {
            "total_samples": int(total_samples),
            "average_positive_rate": float(positive_rate),
            "accuracy": float(accuracy)
        }
        
        return summary
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str) -> None:
        """
        Save fairness metrics to a JSON file.
        
        Args:
            metrics: Dictionary of fairness metrics
            output_file: Path to save the metrics
        """
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Fairness metrics saved to {output_file}")
    
    def load_metrics(self, input_file: str) -> Dict[str, Any]:
        """
        Load fairness metrics from a JSON file.
        
        Args:
            input_file: Path to the metrics file
            
        Returns:
            Dictionary of fairness metrics
        """
        with open(input_file, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"Fairness metrics loaded from {input_file}")
        return metrics


# Example usage if run as script
if __name__ == "__main__":
    # Simple demo
    # Generate synthetic data
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
    
    # Compute and print metrics
    calculator = EnhancedFairnessMetrics()
    metrics = calculator.compute_metrics(y_true, y_pred, protected_attributes)
    
    # Save metrics
    calculator.save_metrics(metrics, 'demo_fairness_metrics.json')
    
    print("Enhanced fairness metrics computed and saved to demo_fairness_metrics.json") 