#!/usr/bin/env python
"""
Fairness Post-Processing Module

This module implements post-processing techniques to mitigate fairness issues
in sentiment analysis predictions after model inference.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """Optimizes classification thresholds to satisfy fairness constraints."""
    
    def __init__(self, fairness_metric: str = "equalized_odds", 
                 grid_size: int = 100):
        """
        Initialize the threshold optimizer.
        
        Args:
            fairness_metric: The fairness metric to optimize for.
                Options: "demographic_parity", "equalized_odds", "equal_opportunity"
            grid_size: Number of threshold values to try during optimization
        """
        self.fairness_metric = fairness_metric
        self.grid_size = grid_size
        self.group_thresholds = {}
        self.base_threshold = 0.5
        
    def compute_confusion_rates(self, y_true: np.ndarray, 
                               y_pred_proba: np.ndarray, 
                               threshold: float) -> Tuple[float, float]:
        """
        Compute true positive rate and false positive rate for given threshold.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Tuple of (true positive rate, false positive rate)
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Handle division by zero
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return tpr, fpr
    
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
           protected_attributes: pd.DataFrame) -> None:
        """
        Learn optimal thresholds for each demographic group.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
        """
        # Initialize base threshold using overall data
        thresholds = np.linspace(0.01, 0.99, self.grid_size)
        best_threshold = 0.5
        best_f1 = 0
        
        # Find best overall threshold with F1 optimization
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate precision and recall
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            false_positives = np.sum((y_true == 0) & (y_pred == 1))
            false_negatives = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.base_threshold = best_threshold
        logger.info(f"Base threshold set to {self.base_threshold:.3f} with F1 score {best_f1:.3f}")
        
        # Process each protected attribute column
        for col in protected_attributes.columns:
            unique_groups = protected_attributes[col].unique()
            
            # If the metric is demographic parity
            if self.fairness_metric == "demographic_parity":
                self._optimize_demographic_parity(y_true, y_pred_proba, 
                                                protected_attributes, col)
            
            # If the metric is equalized odds or equal opportunity
            elif self.fairness_metric in ["equalized_odds", "equal_opportunity"]:
                self._optimize_equalized_odds(y_true, y_pred_proba, 
                                            protected_attributes, col)
    
    def _optimize_demographic_parity(self, y_true, y_pred_proba, 
                                   protected_attributes, col):
        """Optimize thresholds for demographic parity."""
        unique_groups = protected_attributes[col].unique()
        thresholds = {}
        
        # Calculate overall positive rate with base threshold
        overall_pred = (y_pred_proba >= self.base_threshold).astype(int)
        overall_positive_rate = np.mean(overall_pred)
        
        # Find thresholds that give similar positive rates for each group
        for group in unique_groups:
            group_mask = protected_attributes[col] == group
            if sum(group_mask) < 10:  # Skip groups with too few samples
                thresholds[group] = self.base_threshold
                continue
                
            group_proba = y_pred_proba[group_mask]
            
            # Try different thresholds
            best_threshold = self.base_threshold
            min_diff = float('inf')
            
            for threshold in np.linspace(0.01, 0.99, self.grid_size):
                group_pred = (group_proba >= threshold).astype(int)
                group_positive_rate = np.mean(group_pred)
                diff = abs(group_positive_rate - overall_positive_rate)
                
                if diff < min_diff:
                    min_diff = diff
                    best_threshold = threshold
            
            thresholds[group] = best_threshold
        
        # Store thresholds for this protected attribute
        self.group_thresholds[col] = thresholds
        logger.info(f"Optimized demographic parity thresholds for {col}: {thresholds}")
    
    def _optimize_equalized_odds(self, y_true, y_pred_proba, 
                               protected_attributes, col):
        """Optimize thresholds for equalized odds or equal opportunity."""
        unique_groups = protected_attributes[col].unique()
        thresholds = {}
        
        # Calculate overall TPR and FPR with base threshold
        overall_tpr, overall_fpr = self.compute_confusion_rates(
            y_true, y_pred_proba, self.base_threshold
        )
        
        # Find thresholds that give similar TPR and FPR for each group
        for group in unique_groups:
            group_mask = protected_attributes[col] == group
            if sum(group_mask) < 10:  # Skip groups with too few samples
                thresholds[group] = self.base_threshold
                continue
                
            group_y = y_true[group_mask]
            group_proba = y_pred_proba[group_mask]
            
            # Try different thresholds
            best_threshold = self.base_threshold
            min_diff = float('inf')
            
            for threshold in np.linspace(0.01, 0.99, self.grid_size):
                group_tpr, group_fpr = self.compute_confusion_rates(
                    group_y, group_proba, threshold
                )
                
                # For equalized odds, consider both TPR and FPR
                if self.fairness_metric == "equalized_odds":
                    tpr_diff = abs(group_tpr - overall_tpr)
                    fpr_diff = abs(group_fpr - overall_fpr)
                    diff = tpr_diff + fpr_diff
                
                # For equal opportunity, consider only TPR
                else:  # equal_opportunity
                    diff = abs(group_tpr - overall_tpr)
                
                if diff < min_diff:
                    min_diff = diff
                    best_threshold = threshold
            
            thresholds[group] = best_threshold
        
        # Store thresholds for this protected attribute
        self.group_thresholds[col] = thresholds
        logger.info(f"Optimized {self.fairness_metric} thresholds for {col}: {thresholds}")
    
    def adjust(self, y_pred_proba: np.ndarray, 
              protected_attributes: pd.DataFrame) -> np.ndarray:
        """
        Apply group-specific thresholds to predictions.
        
        Args:
            y_pred_proba: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Adjusted binary predictions
        """
        # Start with base threshold predictions
        y_pred = (y_pred_proba >= self.base_threshold).astype(int)
        
        # Apply group-specific thresholds for each protected attribute
        for col, thresholds in self.group_thresholds.items():
            if col not in protected_attributes.columns:
                continue
                
            for group, threshold in thresholds.items():
                group_mask = protected_attributes[col] == group
                y_pred[group_mask] = (y_pred_proba[group_mask] >= threshold).astype(int)
        
        return y_pred


class RejectionOptionClassifier:
    """
    Implements the rejection option technique for fairness.
    
    This classifier identifies instances near the decision boundary and
    applies a fairness-aware decision rule to them, while keeping confident
    predictions unchanged.
    """
    
    def __init__(self, fairness_metric: str = "demographic_parity", 
                 uncertainty_threshold: float = 0.2,
                 critical_region_size: float = 0.3):
        """
        Initialize the rejection option classifier.
        
        Args:
            fairness_metric: The fairness metric to optimize for.
                Options: "demographic_parity", "equalized_odds"
            uncertainty_threshold: Threshold for prediction uncertainty
                (distance from 0.5) to identify critical region
            critical_region_size: Proportion of instances to consider in the
                critical region (0.0 to 1.0)
        """
        self.fairness_metric = fairness_metric
        self.uncertainty_threshold = uncertainty_threshold
        self.critical_region_size = critical_region_size
        self.base_threshold = 0.5
        self.group_stats = {}
        self.favorable_label = 1  # Positive sentiment
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
           protected_attributes: pd.DataFrame) -> None:
        """
        Learn statistics for each demographic group.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
        """
        # Calculate uncertainty for each prediction
        uncertainty = 1 - 2 * abs(y_pred_proba - 0.5)
        
        # Determine critical region based on uncertainty
        if self.critical_region_size < 1.0:
            # Sort by uncertainty and take top critical_region_size proportion
            uncertainty_threshold = np.percentile(
                uncertainty, 100 * (1 - self.critical_region_size)
            )
            self.uncertainty_threshold = max(
                self.uncertainty_threshold, uncertainty_threshold
            )
        
        logger.info(f"Critical region uncertainty threshold: {self.uncertainty_threshold:.3f}")
        
        # Process each protected attribute column
        for col in protected_attributes.columns:
            unique_groups = protected_attributes[col].unique()
            group_stats = {}
            
            # Calculate overall statistics
            overall_pred = (y_pred_proba >= self.base_threshold).astype(int)
            overall_positive_rate = np.mean(overall_pred)
            
            # Calculate statistics for each group
            for group in unique_groups:
                group_mask = protected_attributes[col] == group
                if sum(group_mask) < 10:  # Skip groups with too few samples
                    continue
                    
                group_y = y_true[group_mask]
                group_proba = y_pred_proba[group_mask]
                group_pred = (group_proba >= self.base_threshold).astype(int)
                
                # Calculate group statistics
                group_positive_rate = np.mean(group_pred)
                
                # Calculate true positive and false positive rates if needed
                if self.fairness_metric == "equalized_odds":
                    # For positive instances
                    pos_mask = group_y == 1
                    if sum(pos_mask) > 0:
                        tpr = np.mean(group_pred[pos_mask])
                    else:
                        tpr = 0
                        
                    # For negative instances
                    neg_mask = group_y == 0
                    if sum(neg_mask) > 0:
                        fpr = np.mean(group_pred[neg_mask])
                    else:
                        fpr = 0
                        
                    group_stats[group] = {
                        'positive_rate': group_positive_rate,
                        'tpr': tpr,
                        'fpr': fpr
                    }
                else:
                    # For demographic parity, just store positive rate
                    group_stats[group] = {
                        'positive_rate': group_positive_rate
                    }
            
            # Store statistics for this protected attribute
            self.group_stats[col] = {
                'groups': group_stats,
                'overall_positive_rate': overall_positive_rate
            }
            
            logger.info(f"Computed group statistics for {col}")
    
    def adjust(self, y_pred_proba: np.ndarray, 
              protected_attributes: pd.DataFrame) -> np.ndarray:
        """
        Apply the rejection option technique to predictions.
        
        Args:
            y_pred_proba: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Adjusted binary predictions
        """
        # Start with base threshold predictions
        y_pred = (y_pred_proba >= self.base_threshold).astype(int)
        
        # Calculate uncertainty for each prediction
        uncertainty = 1 - 2 * abs(y_pred_proba - 0.5)
        
        # Identify instances in the critical region
        critical_region = uncertainty >= self.uncertainty_threshold
        
        logger.info(f"Critical region size: {np.sum(critical_region)} out of {len(y_pred)} ({np.mean(critical_region)*100:.1f}%)")
        
        # Apply fairness adjustments only to instances in the critical region
        for col, stats in self.group_stats.items():
            if col not in protected_attributes.columns:
                continue
                
            overall_positive_rate = stats['overall_positive_rate']
            
            for group, group_stats in stats['groups'].items():
                group_mask = (protected_attributes[col] == group) & critical_region
                if sum(group_mask) == 0:
                    continue
                
                # Get current positive rate for this group
                group_positive_rate = group_stats['positive_rate']
                
                # Determine if we need to increase or decrease positive predictions
                if self.fairness_metric == "demographic_parity":
                    # For demographic parity, adjust toward overall positive rate
                    if group_positive_rate < overall_positive_rate:
                        # Need to increase positive predictions
                        # Find negative predictions in critical region for this group
                        candidates = group_mask & (y_pred == 0)
                        if sum(candidates) > 0:
                            # Sort by probability (highest first)
                            candidate_indices = np.where(candidates)[0]
                            candidate_probs = y_pred_proba[candidate_indices]
                            sorted_indices = candidate_indices[np.argsort(-candidate_probs)]
                            
                            # Calculate how many to flip
                            target_positive = int(overall_positive_rate * sum(group_mask))
                            current_positive = sum(y_pred[group_mask])
                            to_flip = min(target_positive - current_positive, len(sorted_indices))
                            
                            if to_flip > 0:
                                y_pred[sorted_indices[:to_flip]] = 1
                                logger.info(f"Flipped {to_flip} predictions from 0 to 1 for group {group} in {col}")
                    
                    elif group_positive_rate > overall_positive_rate:
                        # Need to decrease positive predictions
                        # Find positive predictions in critical region for this group
                        candidates = group_mask & (y_pred == 1)
                        if sum(candidates) > 0:
                            # Sort by probability (lowest first)
                            candidate_indices = np.where(candidates)[0]
                            candidate_probs = y_pred_proba[candidate_indices]
                            sorted_indices = candidate_indices[np.argsort(candidate_probs)]
                            
                            # Calculate how many to flip
                            target_positive = int(overall_positive_rate * sum(group_mask))
                            current_positive = sum(y_pred[group_mask])
                            to_flip = min(current_positive - target_positive, len(sorted_indices))
                            
                            if to_flip > 0:
                                y_pred[sorted_indices[:to_flip]] = 0
                                logger.info(f"Flipped {to_flip} predictions from 1 to 0 for group {group} in {col}")
                
                # For equalized odds, we would need to adjust TPR and FPR separately
                # This requires ground truth labels, which we don't have at prediction time
                # So we approximate using the critical region approach
        
        return y_pred
    
    def get_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           protected_attributes: pd.DataFrame) -> Dict:
        """
        Calculate fairness metrics for the adjusted predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Adjusted binary predictions
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}
        
        for col in protected_attributes.columns:
            unique_groups = protected_attributes[col].unique()
            group_metrics = {}
            
            # Calculate overall metrics
            overall_acc = np.mean(y_pred == y_true)
            
            # Calculate metrics for each group
            for group in unique_groups:
                group_mask = protected_attributes[col] == group
                if sum(group_mask) < 10:  # Skip groups with too few samples
                    continue
                    
                group_y = y_true[group_mask]
                group_pred = y_pred[group_mask]
                
                # Calculate accuracy
                group_acc = np.mean(group_pred == group_y)
                
                # Calculate positive prediction rate
                group_positive_rate = np.mean(group_pred)
                
                # Calculate TPR and FPR
                pos_mask = group_y == 1
                neg_mask = group_y == 0
                
                if sum(pos_mask) > 0:
                    tpr = np.mean(group_pred[pos_mask])
                else:
                    tpr = 0
                    
                if sum(neg_mask) > 0:
                    fpr = np.mean(group_pred[neg_mask])
                else:
                    fpr = 0
                
                group_metrics[group] = {
                    'accuracy': group_acc,
                    'positive_rate': group_positive_rate,
                    'tpr': tpr,
                    'fpr': fpr,
                    'accuracy_disparity': group_acc - overall_acc,
                    'sample_size': sum(group_mask)
                }
            
            # Calculate disparities between groups
            if len(group_metrics) >= 2:
                positive_rates = [stats['positive_rate'] for stats in group_metrics.values()]
                tprs = [stats['tpr'] for stats in group_metrics.values()]
                fprs = [stats['fpr'] for stats in group_metrics.values()]
                
                demographic_parity_disparity = max(positive_rates) - min(positive_rates)
                equalized_odds_disparity = (max(tprs) - min(tprs)) + (max(fprs) - min(fprs))
                
                metrics[col] = {
                    'groups': group_metrics,
                    'demographic_parity_disparity': demographic_parity_disparity,
                    'equalized_odds_disparity': equalized_odds_disparity
                }
        
        return metrics


def generate_fairness_report(metrics: Dict, 
                           output_dir: str = "fairness_reports", 
                           filename_prefix: str = "fairness_report") -> str:
    """
    Generate a fairness report from metrics.
    
    Args:
        metrics: Dictionary of fairness metrics from get_fairness_metrics
        output_dir: Directory to save the report
        filename_prefix: Prefix for the report filename
        
    Returns:
        Path to the generated report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate JSON report
    json_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate Markdown report
    md_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.md")
    
    with open(md_path, 'w') as f:
        f.write("# Fairness Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        f.write("## Summary\n\n")
        
        # Calculate overall disparities
        dp_disparities = []
        eo_disparities = []
        
        for attr, attr_metrics in metrics.items():
            if 'demographic_parity_disparity' in attr_metrics:
                dp_disparities.append(attr_metrics['demographic_parity_disparity'])
            if 'equalized_odds_disparity' in attr_metrics:
                eo_disparities.append(attr_metrics['equalized_odds_disparity'])
        
        if dp_disparities:
            avg_dp = sum(dp_disparities) / len(dp_disparities)
            max_dp = max(dp_disparities)
            f.write(f"- Average Demographic Parity Disparity: {avg_dp:.4f}\n")
            f.write(f"- Maximum Demographic Parity Disparity: {max_dp:.4f}\n")
        
        if eo_disparities:
            avg_eo = sum(eo_disparities) / len(eo_disparities)
            max_eo = max(eo_disparities)
            f.write(f"- Average Equalized Odds Disparity: {avg_eo:.4f}\n")
            f.write(f"- Maximum Equalized Odds Disparity: {max_eo:.4f}\n")
        
        f.write("\n")
        
        # Detailed metrics for each protected attribute
        for attr, attr_metrics in metrics.items():
            f.write(f"## Protected Attribute: {attr}\n\n")
            
            if 'demographic_parity_disparity' in attr_metrics:
                f.write(f"- Demographic Parity Disparity: {attr_metrics['demographic_parity_disparity']:.4f}\n")
            if 'equalized_odds_disparity' in attr_metrics:
                f.write(f"- Equalized Odds Disparity: {attr_metrics['equalized_odds_disparity']:.4f}\n")
            
            f.write("\n### Group-level Metrics\n\n")
            
            # Table header
            f.write("| Group | Sample Size | Accuracy | Pos. Rate | TPR | FPR | Acc. Disparity |\n")
            f.write("|-------|-------------|----------|-----------|-----|-----|---------------|\n")
            
            # Table rows for each group
            for group, group_metrics in attr_metrics['groups'].items():
                f.write(f"| {group} | {group_metrics['sample_size']} | ")
                f.write(f"{group_metrics['accuracy']:.4f} | ")
                f.write(f"{group_metrics['positive_rate']:.4f} | ")
                f.write(f"{group_metrics['tpr']:.4f} | ")
                f.write(f"{group_metrics['fpr']:.4f} | ")
                f.write(f"{group_metrics['accuracy_disparity']:.4f} |\n")
            
            f.write("\n")
            
            # Add recommendations based on disparities
            f.write("### Recommendations\n\n")
            
            if 'demographic_parity_disparity' in attr_metrics:
                dp_disparity = attr_metrics['demographic_parity_disparity']
                if dp_disparity > 0.1:
                    f.write("- **High Demographic Parity Disparity**: Consider applying demographic parity constraints\n")
                elif dp_disparity > 0.05:
                    f.write("- **Moderate Demographic Parity Disparity**: Monitor this attribute for fairness concerns\n")
                else:
                    f.write("- **Low Demographic Parity Disparity**: No immediate action needed\n")
            
            if 'equalized_odds_disparity' in attr_metrics:
                eo_disparity = attr_metrics['equalized_odds_disparity']
                if eo_disparity > 0.2:
                    f.write("- **High Equalized Odds Disparity**: Consider applying equalized odds constraints\n")
                elif eo_disparity > 0.1:
                    f.write("- **Moderate Equalized Odds Disparity**: Monitor this attribute for fairness concerns\n")
                else:
                    f.write("- **Low Equalized Odds Disparity**: No immediate action needed\n")
            
            f.write("\n")
        
        # Final recommendations
        f.write("## Overall Recommendations\n\n")
        
        # Determine overall fairness level
        fairness_level = "Low"
        if (dp_disparities and max(dp_disparities) > 0.1) or (eo_disparities and max(eo_disparities) > 0.2):
            fairness_level = "High"
        elif (dp_disparities and max(dp_disparities) > 0.05) or (eo_disparities and max(eo_disparities) > 0.1):
            fairness_level = "Moderate"
        
        f.write(f"- **Fairness Concern Level**: {fairness_level}\n")
        
        if fairness_level == "High":
            f.write("- **Action Required**: Apply fairness constraints to mitigate disparities\n")
            f.write("- Consider adjusting classification thresholds for affected groups\n")
            f.write("- Investigate potential biases in the training data\n")
        elif fairness_level == "Moderate":
            f.write("- **Action Suggested**: Monitor fairness metrics during model updates\n")
            f.write("- Consider light fairness interventions for the most affected groups\n")
        else:
            f.write("- **No Immediate Action Required**: Continue monitoring fairness metrics\n")
    
    logger.info(f"Fairness report generated at {md_path}")
    return md_path


def compare_fairness_improvements(
    original_metrics: Dict,
    adjusted_metrics: Dict,
    output_dir: str = "fairness_reports",
    filename: str = "fairness_comparison.md"
) -> str:
    """
    Generate a report comparing fairness metrics before and after adjustments.
    
    Args:
        original_metrics: Fairness metrics before adjustments
        adjusted_metrics: Fairness metrics after adjustments
        output_dir: Directory to save the report
        filename: Filename for the comparison report
        
    Returns:
        Path to the generated comparison report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filepath
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write("# Fairness Improvement Comparison\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary of Improvements\n\n")
        
        # Table header for summary
        f.write("| Protected Attribute | Metric | Before | After | Change | % Improvement |\n")
        f.write("|---------------------|--------|--------|-------|--------|---------------|\n")
        
        # Calculate overall improvements for each attribute and metric
        for attr in original_metrics:
            if attr not in adjusted_metrics:
                continue
                
            orig_attr_metrics = original_metrics[attr]
            adj_attr_metrics = adjusted_metrics[attr]
            
            # Demographic parity
            if 'demographic_parity_disparity' in orig_attr_metrics and 'demographic_parity_disparity' in adj_attr_metrics:
                before_dp = orig_attr_metrics['demographic_parity_disparity']
                after_dp = adj_attr_metrics['demographic_parity_disparity']
                change_dp = before_dp - after_dp
                pct_improvement_dp = (change_dp / before_dp) * 100 if before_dp > 0 else 0
                
                f.write(f"| {attr} | Demographic Parity | {before_dp:.4f} | {after_dp:.4f} | {change_dp:.4f} | {pct_improvement_dp:.1f}% |\n")
            
            # Equalized odds
            if 'equalized_odds_disparity' in orig_attr_metrics and 'equalized_odds_disparity' in adj_attr_metrics:
                before_eo = orig_attr_metrics['equalized_odds_disparity']
                after_eo = adj_attr_metrics['equalized_odds_disparity']
                change_eo = before_eo - after_eo
                pct_improvement_eo = (change_eo / before_eo) * 100 if before_eo > 0 else 0
                
                f.write(f"| {attr} | Equalized Odds | {before_eo:.4f} | {after_eo:.4f} | {change_eo:.4f} | {pct_improvement_eo:.1f}% |\n")
        
        f.write("\n")
        
        # Detailed per-group comparisons
        f.write("## Detailed Group-level Changes\n\n")
        
        for attr in original_metrics:
            if attr not in adjusted_metrics:
                continue
                
            orig_attr_metrics = original_metrics[attr]
            adj_attr_metrics = adjusted_metrics[attr]
            
            f.write(f"### {attr}\n\n")
            
            # Table header for groups
            f.write("| Group | Metric | Before | After | Change |\n")
            f.write("|-------|--------|--------|-------|--------|\n")
            
            # Get common groups
            orig_groups = set(orig_attr_metrics['groups'].keys())
            adj_groups = set(adj_attr_metrics['groups'].keys())
            common_groups = orig_groups.intersection(adj_groups)
            
            for group in common_groups:
                orig_group_metrics = orig_attr_metrics['groups'][group]
                adj_group_metrics = adj_attr_metrics['groups'][group]
                
                # Accuracy
                before_acc = orig_group_metrics['accuracy']
                after_acc = adj_group_metrics['accuracy']
                change_acc = after_acc - before_acc
                f.write(f"| {group} | Accuracy | {before_acc:.4f} | {after_acc:.4f} | {change_acc:.4f} |\n")
                
                # Positive rate
                before_pr = orig_group_metrics['positive_rate']
                after_pr = adj_group_metrics['positive_rate']
                change_pr = after_pr - before_pr
                f.write(f"| {group} | Positive Rate | {before_pr:.4f} | {after_pr:.4f} | {change_pr:.4f} |\n")
                
                # TPR
                if 'tpr' in orig_group_metrics and 'tpr' in adj_group_metrics:
                    before_tpr = orig_group_metrics['tpr']
                    after_tpr = adj_group_metrics['tpr']
                    change_tpr = after_tpr - before_tpr
                    f.write(f"| {group} | TPR | {before_tpr:.4f} | {after_tpr:.4f} | {change_tpr:.4f} |\n")
                
                # FPR
                if 'fpr' in orig_group_metrics and 'fpr' in adj_group_metrics:
                    before_fpr = orig_group_metrics['fpr']
                    after_fpr = adj_group_metrics['fpr']
                    change_fpr = after_fpr - before_fpr
                    f.write(f"| {group} | FPR | {before_fpr:.4f} | {after_fpr:.4f} | {change_fpr:.4f} |\n")
            
            f.write("\n")
        
        # Conclusion section
        f.write("## Conclusion\n\n")
        
        # Analyze overall improvements
        all_dp_improvements = []
        all_eo_improvements = []
        
        for attr in original_metrics:
            if attr not in adjusted_metrics:
                continue
                
            orig_attr_metrics = original_metrics[attr]
            adj_attr_metrics = adjusted_metrics[attr]
            
            if 'demographic_parity_disparity' in orig_attr_metrics and 'demographic_parity_disparity' in adj_attr_metrics:
                before_dp = orig_attr_metrics['demographic_parity_disparity']
                after_dp = adj_attr_metrics['demographic_parity_disparity']
                if before_dp > 0:
                    improvement = (before_dp - after_dp) / before_dp
                    all_dp_improvements.append(improvement)
            
            if 'equalized_odds_disparity' in orig_attr_metrics and 'equalized_odds_disparity' in adj_attr_metrics:
                before_eo = orig_attr_metrics['equalized_odds_disparity']
                after_eo = adj_attr_metrics['equalized_odds_disparity']
                if before_eo > 0:
                    improvement = (before_eo - after_eo) / before_eo
                    all_eo_improvements.append(improvement)
        
        # Generate conclusion text
        if all_dp_improvements:
            avg_dp_improvement = sum(all_dp_improvements) / len(all_dp_improvements) * 100
            f.write(f"- Demographic Parity disparity improved by an average of {avg_dp_improvement:.1f}%\n")
        
        if all_eo_improvements:
            avg_eo_improvement = sum(all_eo_improvements) / len(all_eo_improvements) * 100
            f.write(f"- Equalized Odds disparity improved by an average of {avg_eo_improvement:.1f}%\n")
        
        # Overall assessment
        if (all_dp_improvements and avg_dp_improvement > 30) or (all_eo_improvements and avg_eo_improvement > 30):
            f.write("\nThe fairness adjustments have made **significant improvements** to the model's fairness metrics.\n")
        elif (all_dp_improvements and avg_dp_improvement > 10) or (all_eo_improvements and avg_eo_improvement > 10):
            f.write("\nThe fairness adjustments have made **moderate improvements** to the model's fairness metrics.\n")
        else:
            f.write("\nThe fairness adjustments have made **minor improvements** to the model's fairness metrics. "
                   "Consider trying different fairness constraints or adjusting hyperparameters.\n")
    
    logger.info(f"Fairness comparison report generated at {file_path}")
    return file_path


def example_threshold_optimizer():
    """Simple example demonstrating the ThresholdOptimizer."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features and labels
    X = np.random.normal(0, 1, (n_samples, 2))
    
    # Make y depend on X[0] to create a simple classification problem
    y_proba = 1 / (1 + np.exp(-2 * X[:, 0]))
    y = (np.random.random(n_samples) < y_proba).astype(int)
    
    # Create a synthetic protected attribute (gender: 0=male, 1=female)
    gender = np.random.binomial(1, 0.5, n_samples)
    
    # Create a bias in the data: females with X[0] > 0 are more likely to get positive outcome
    biased_proba = y_proba.copy()
    bias_idx = (gender == 1) & (X[:, 0] > 0)
    biased_proba[bias_idx] = biased_proba[bias_idx] * 1.2
    biased_proba = np.clip(biased_proba, 0, 1)
    
    # Create protected attributes DataFrame
    protected_attributes = pd.DataFrame({'gender': gender})
    
    # Create a ThresholdOptimizer for demographic parity
    optimizer = ThresholdOptimizer(fairness_metric="demographic_parity")
    
    # Fit the optimizer
    optimizer.fit(y, biased_proba, protected_attributes)
    
    # Apply the optimized thresholds
    adjusted_predictions = optimizer.adjust(biased_proba, protected_attributes)
    
    # Create a RejectionOptionClassifier for comparison
    rejection_classifier = RejectionOptionClassifier(fairness_metric="demographic_parity")
    rejection_classifier.fit(y, biased_proba, protected_attributes)
    rejection_adjusted = rejection_classifier.adjust(biased_proba, protected_attributes)
    
    # Calculate fairness metrics before adjustment
    baseline_predictions = (biased_proba >= 0.5).astype(int)
    baseline_metrics = rejection_classifier.get_fairness_metrics(
        y, baseline_predictions, protected_attributes
    )
    
    # Calculate fairness metrics after adjustment (both methods)
    threshold_metrics = rejection_classifier.get_fairness_metrics(
        y, adjusted_predictions, protected_attributes
    )
    
    rejection_metrics = rejection_classifier.get_fairness_metrics(
        y, rejection_adjusted, protected_attributes
    )
    
    # Print summary
    print("\n===== Fairness Metrics Before Adjustment =====")
    if 'gender' in baseline_metrics:
        print(f"Demographic Parity Disparity: {baseline_metrics['gender']['demographic_parity_disparity']:.4f}")
        print(f"Equalized Odds Disparity: {baseline_metrics['gender']['equalized_odds_disparity']:.4f}")
    
    print("\n===== After ThresholdOptimizer Adjustment =====")
    if 'gender' in threshold_metrics:
        print(f"Demographic Parity Disparity: {threshold_metrics['gender']['demographic_parity_disparity']:.4f}")
        print(f"Equalized Odds Disparity: {threshold_metrics['gender']['equalized_odds_disparity']:.4f}")
    
    print("\n===== After RejectionOptionClassifier Adjustment =====")
    if 'gender' in rejection_metrics:
        print(f"Demographic Parity Disparity: {rejection_metrics['gender']['demographic_parity_disparity']:.4f}")
        print(f"Equalized Odds Disparity: {rejection_metrics['gender']['equalized_odds_disparity']:.4f}")
    
    return baseline_metrics, threshold_metrics, rejection_metrics


def generate_fairness_visualizations(baseline_metrics, threshold_metrics, rejection_metrics, output_dir="fairness_reports"):
    """
    Generate visualizations comparing fairness metrics before and after adjustments.
    
    Args:
        baseline_metrics: Metrics before adjustment
        threshold_metrics: Metrics after ThresholdOptimizer adjustment
        rejection_metrics: Metrics after RejectionOptionClassifier adjustment
        output_dir: Directory to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("Matplotlib and/or seaborn not available. Skipping visualizations.")
        return
    
    # Helper function to convert NumPy types to Python native types
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert all metrics to use Python native types
    baseline_metrics = convert_numpy_types(baseline_metrics)
    threshold_metrics = convert_numpy_types(threshold_metrics)
    rejection_metrics = convert_numpy_types(rejection_metrics)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate reports
    generate_fairness_report(baseline_metrics, output_dir=output_dir, filename_prefix="baseline")
    generate_fairness_report(threshold_metrics, output_dir=output_dir, filename_prefix="threshold_optimized")
    generate_fairness_report(rejection_metrics, output_dir=output_dir, filename_prefix="rejection_option")
    
    # Generate comparison reports
    compare_fairness_improvements(
        baseline_metrics, threshold_metrics, 
        output_dir=output_dir, filename="threshold_comparison.md"
    )
    compare_fairness_improvements(
        baseline_metrics, rejection_metrics, 
        output_dir=output_dir, filename="rejection_comparison.md"
    )
    
    # Collect metrics for protected attributes
    common_attrs = set(baseline_metrics.keys()) & set(threshold_metrics.keys()) & set(rejection_metrics.keys())
    
    for attr in common_attrs:
        # Create a figure for demographic parity
        plt.figure(figsize=(10, 6))
        
        methods = ['Original', 'ThresholdOptimizer', 'RejectionOption']
        dp_values = [
            baseline_metrics[attr].get('demographic_parity_disparity', 0),
            threshold_metrics[attr].get('demographic_parity_disparity', 0),
            rejection_metrics[attr].get('demographic_parity_disparity', 0)
        ]
        
        # Bar plot for demographic parity
        plt.bar(methods, dp_values, color=['gray', 'blue', 'green'])
        plt.title(f'Demographic Parity Disparity Comparison - {attr}')
        plt.ylabel('Disparity')
        
        # Add values on top of bars
        for i, v in enumerate(dp_values):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        plt.ylim(0, max(dp_values) * 1.2)  # Add some space above the bars
        plt.savefig(os.path.join(output_dir, f"{attr}_demographic_parity_comparison.png"))
        plt.close()
        
        # Create a figure for equalized odds
        plt.figure(figsize=(10, 6))
        
        eo_values = [
            baseline_metrics[attr].get('equalized_odds_disparity', 0),
            threshold_metrics[attr].get('equalized_odds_disparity', 0),
            rejection_metrics[attr].get('equalized_odds_disparity', 0)
        ]
        
        # Bar plot for equalized odds
        plt.bar(methods, eo_values, color=['gray', 'blue', 'green'])
        plt.title(f'Equalized Odds Disparity Comparison - {attr}')
        plt.ylabel('Disparity')
        
        # Add values on top of bars
        for i, v in enumerate(eo_values):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        plt.ylim(0, max(eo_values) * 1.2)  # Add some space above the bars
        plt.savefig(os.path.join(output_dir, f"{attr}_equalized_odds_comparison.png"))
        plt.close()
        
        # Compare positive rates by group
        if 'groups' in baseline_metrics[attr] and 'groups' in threshold_metrics[attr] and 'groups' in rejection_metrics[attr]:
            # Get common groups
            groups = set(baseline_metrics[attr]['groups'].keys()) & set(threshold_metrics[attr]['groups'].keys()) & set(rejection_metrics[attr]['groups'].keys())
            
            # For each group, collect positive rates
            group_labels = list(groups)
            baseline_rates = [baseline_metrics[attr]['groups'][g]['positive_rate'] for g in group_labels]
            threshold_rates = [threshold_metrics[attr]['groups'][g]['positive_rate'] for g in group_labels]
            rejection_rates = [rejection_metrics[attr]['groups'][g]['positive_rate'] for g in group_labels]
            
            # Create a grouped bar plot
            width = 0.25
            x = np.arange(len(group_labels))
            
            plt.figure(figsize=(10, 6))
            plt.bar(x - width, baseline_rates, width, label='Original', color='gray')
            plt.bar(x, threshold_rates, width, label='ThresholdOptimizer', color='blue')
            plt.bar(x + width, rejection_rates, width, label='RejectionOption', color='green')
            
            plt.xlabel('Group')
            plt.ylabel('Positive Rate')
            plt.title(f'Positive Prediction Rate by Group - {attr}')
            plt.xticks(x, [str(g) for g in group_labels])
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, f"{attr}_positive_rate_by_group.png"))
            plt.close()
    
    logger.info(f"Fairness visualizations generated in {output_dir}")
    return output_dir


if __name__ == "__main__":
    print("Fairness Post-Processing Module Demo")
    print("====================================")
    print("\nRunning example with synthetic data...")
    
    # Run the threshold optimizer example
    baseline_metrics, threshold_metrics, rejection_metrics = example_threshold_optimizer()
    
    # Generate and visualize results
    try:
        report_dir = generate_fairness_visualizations(
            baseline_metrics, threshold_metrics, rejection_metrics
        )
        print(f"\nReports and visualizations generated in: {report_dir}")
        print("Check the reports to see fairness improvements.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    print("\nExample usage:")
    print("""
    # Example code for integrating with your project:
    import fairness_postprocessing as fp
    
    # Use ThresholdOptimizer for demographic parity
    optimizer = fp.ThresholdOptimizer(fairness_metric="demographic_parity")
    optimizer.fit(y_train, y_pred_proba_train, protected_attributes_train)
    fair_predictions = optimizer.adjust(y_pred_proba_test, protected_attributes_test)
    
    # Or use RejectionOptionClassifier for equalized odds
    classifier = fp.RejectionOptionClassifier(fairness_metric="equalized_odds")
    classifier.fit(y_train, y_pred_proba_train, protected_attributes_train)
    fair_predictions = classifier.adjust(y_pred_proba_test, protected_attributes_test)
    
    # Evaluate fairness metrics
    metrics = classifier.get_fairness_metrics(y_test, fair_predictions, protected_attributes_test)
    
    # Generate a fairness report
    report_path = fp.generate_fairness_report(metrics)
    print(f"Fairness report generated at: {report_path}")
    """)
    
    print("\nDemo completed.") 