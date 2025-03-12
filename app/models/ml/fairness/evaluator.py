"""
Fairness evaluator module for assessing bias in ML models.

This module provides classes and functions for evaluating fairness
in machine learning models across protected attribute groups.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Set


class FairnessEvaluator:
    """
    Evaluate ML models for fairness across protected attribute groups.
    
    This class provides comprehensive fairness evaluation capabilities for ML models,
    including demographic parity, equal opportunity, and equalized odds metrics.
    """
    
    def __init__(
        self,
        protected_attributes: List[str] = None,
        fairness_threshold: float = 0.2,
        metrics: List[str] = None,
        threshold: float = 0.5,
        intersectional: bool = False,
        intersectional_groups: List[List[str]] = None
    ):
        """
        Initialize the FairnessEvaluator.
        
        Args:
            protected_attributes: List of protected attribute names
            fairness_threshold: Threshold for fairness violations
            metrics: List of fairness metrics to compute
            threshold: Classification threshold for binary predictions
            intersectional: Whether to perform intersectional analysis
            intersectional_groups: Specific intersectional groups to analyze
        """
        self.protected_attributes = protected_attributes or []
        self.fairness_threshold = fairness_threshold
        self.metrics = metrics or ["demographic_parity", "equal_opportunity", "equalized_odds"]
        self.threshold = threshold
        self.intersectional = intersectional
        self.intersectional_groups = intersectional_groups or []
    
    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate fairness metrics for the provided predictions.
        
        Args:
            predictions: Model predictions (probabilities or scores)
            labels: Ground truth labels
            protected_attributes: Dictionary mapping attribute names to values
            
        Returns:
            Dictionary containing fairness metrics
        """
        # Convert pandas Series or DataFrame to numpy arrays
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        if hasattr(labels, 'values'):
            labels = labels.values
            
        # Convert predictions to binary if they're probabilities
        if len(np.unique(predictions)) > 2:
            binary_predictions = (predictions >= self.threshold).astype(int)
        else:
            binary_predictions = predictions.astype(int)
            
        # Initialize results dictionary
        results = {
            "overall": {
                "accuracy": np.mean(binary_predictions == labels),
                "positive_rate": np.mean(binary_predictions),
                "true_positive_rate": np.mean(binary_predictions[labels == 1] == 1) if np.any(labels == 1) else 0,
                "false_positive_rate": np.mean(binary_predictions[labels == 0] == 1) if np.any(labels == 0) else 0
            },
            "group_metrics": {},
            "fairness_metrics": {}
        }
        
        # Calculate metrics for each protected attribute
        for attr_name, attr_values in protected_attributes.items():
            if attr_name not in self.protected_attributes:
                continue
            
            # Convert pandas Series to numpy arrays
            if hasattr(attr_values, 'values'):
                attr_values = attr_values.values
                
            attr_results = {}
            unique_values = np.unique(attr_values)
            
            # Calculate metrics for each group
            for value in unique_values:
                mask = (attr_values == value)
                
                # Skip if no samples in this group
                if not np.any(mask):
                    continue
                    
                group_preds = binary_predictions[mask]
                group_labels = labels[mask]
                
                # Calculate standard metrics
                accuracy = np.mean(group_preds == group_labels)
                positive_rate = np.mean(group_preds)
                
                # Calculate true positive rate (equal opportunity)
                group_positives = (group_labels == 1)
                if np.any(group_positives):
                    true_positive_rate = np.mean(group_preds[group_positives] == 1)
                else:
                    true_positive_rate = 0
                    
                # Calculate false positive rate
                group_negatives = (group_labels == 0)
                if np.any(group_negatives):
                    false_positive_rate = np.mean(group_preds[group_negatives] == 1)
                else:
                    false_positive_rate = 0
                
                attr_results[str(value)] = {
                    "count": int(np.sum(mask)),
                    "accuracy": float(accuracy),
                    "positive_rate": float(positive_rate),
                    "true_positive_rate": float(true_positive_rate),
                    "false_positive_rate": float(false_positive_rate)
                }
            
            results["group_metrics"][attr_name] = attr_results
            
            # Calculate fairness metrics
            if "demographic_parity" in self.metrics and len(attr_results) > 1:
                pos_rates = [m["positive_rate"] for m in attr_results.values()]
                dp_difference = max(pos_rates) - min(pos_rates)
                dp_ratio = min(pos_rates) / max(pos_rates) if max(pos_rates) > 0 else 1.0
                
                results["fairness_metrics"][f"{attr_name}_demographic_parity"] = {
                    "difference": float(dp_difference),
                    "ratio": float(dp_ratio),
                    "passes_threshold": dp_difference <= self.fairness_threshold
                }
            
            if "equal_opportunity" in self.metrics and len(attr_results) > 1:
                tpr_values = [m["true_positive_rate"] for m in attr_results.values()]
                eo_difference = max(tpr_values) - min(tpr_values)
                eo_ratio = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 1.0
                
                results["fairness_metrics"][f"{attr_name}_equal_opportunity"] = {
                    "difference": float(eo_difference),
                    "ratio": float(eo_ratio),
                    "passes_threshold": eo_difference <= self.fairness_threshold
                }
            
            if "equalized_odds" in self.metrics and len(attr_results) > 1:
                tpr_values = [m["true_positive_rate"] for m in attr_results.values()]
                fpr_values = [m["false_positive_rate"] for m in attr_results.values()]
                
                tpr_difference = max(tpr_values) - min(tpr_values)
                fpr_difference = max(fpr_values) - min(fpr_values)
                
                # Average difference as equalized odds violation
                eodds_difference = (tpr_difference + fpr_difference) / 2
                
                results["fairness_metrics"][f"{attr_name}_equalized_odds"] = {
                    "difference": float(eodds_difference),
                    "tpr_difference": float(tpr_difference),
                    "fpr_difference": float(fpr_difference),
                    "passes_threshold": eodds_difference <= self.fairness_threshold
                }
        
        # Calculate intersectional metrics if requested
        if self.intersectional and len(self.protected_attributes) > 1:
            self._calculate_intersectional_metrics(
                binary_predictions, 
                labels, 
                protected_attributes, 
                results
            )
            
        return results
    
    def _calculate_intersectional_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        results: Dict[str, Any]
    ) -> None:
        """
        Calculate intersectional fairness metrics.
        
        Args:
            predictions: Binary predictions
            labels: Ground truth labels
            protected_attributes: Dictionary of protected attributes
            results: Results dictionary to update
        """
        # Convert pandas Series or DataFrame to numpy arrays
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        if hasattr(labels, 'values'):
            labels = labels.values
            
        # Ensure binary predictions
        if len(np.unique(predictions)) > 2:
            binary_predictions = (predictions >= self.threshold).astype(int)
        else:
            binary_predictions = predictions.astype(int)
            
        # Initialize intersectional results
        results["intersectional"] = {
            "group_metrics": {},
            "fairness_metrics": {}
        }
        
        # Process protected attributes to ensure they're numpy arrays
        processed_attributes = {}
        for attr_name, attr_values in protected_attributes.items():
            if attr_name in self.protected_attributes:
                if hasattr(attr_values, 'values'):
                    processed_attributes[attr_name] = attr_values.values
                else:
                    processed_attributes[attr_name] = attr_values
        
        # Determine which intersections to analyze
        intersections = []
        if self.intersectional_groups:
            # Use specified intersections
            intersections = self.intersectional_groups
        else:
            # Generate all possible intersections (including pairs and larger groups)
            for k in range(2, len(self.protected_attributes) + 1):
                from itertools import combinations
                for combo in combinations(self.protected_attributes, k):
                    if all(attr in processed_attributes for attr in combo):
                        intersections.append(list(combo))
        
        # Calculate metrics for each intersection
        for intersection in intersections:
            # Convert to tuple for dict key
            intersection_key = tuple(intersection)
            
            # Skip if any attribute is missing
            if not all(attr in processed_attributes for attr in intersection):
                continue
            
            # Get values for each attribute in this intersection
            intersection_values = {}
            for attr in intersection:
                intersection_values[attr] = np.unique(processed_attributes[attr])
            
            # Initialize group metrics for this intersection
            results["intersectional"]["group_metrics"][intersection_key] = {}
            
            # Get all combinations of values
            from itertools import product
            value_combinations = list(product(*[intersection_values[attr] for attr in intersection]))
            
            # Calculate metrics for each combination
            for combo in value_combinations:
                # Create label for this combination
                combo_key = "_".join([f"{attr}={value}" for attr, value in zip(intersection, combo)])
                
                # Build mask for this combination
                mask = np.ones(len(labels), dtype=bool)
                for attr, value in zip(intersection, combo):
                    mask = mask & (processed_attributes[attr] == value)
                
                # Skip if no samples in this group
                if not np.any(mask):
                    continue
                
                group_preds = binary_predictions[mask]
                group_labels = labels[mask]
                
                # Calculate metrics
                group_accuracy = np.mean(group_preds == group_labels)
                group_positive_rate = np.mean(group_preds)
                
                # Calculate true positive rate
                group_positives = (group_labels == 1)
                true_positive_rate = np.mean(group_preds[group_positives] == 1) if np.any(group_positives) else 0
                
                # Calculate false positive rate
                group_negatives = (group_labels == 0)
                false_positive_rate = np.mean(group_preds[group_negatives] == 1) if np.any(group_negatives) else 0
                
                # Store metrics
                results["intersectional"]["group_metrics"][intersection_key][combo_key] = {
                    "count": int(np.sum(mask)),
                    "accuracy": float(group_accuracy),
                    "positive_rate": float(group_positive_rate),
                    "true_positive_rate": float(true_positive_rate),
                    "false_positive_rate": float(false_positive_rate)
                }
            
            # Calculate fairness metrics across groups in this intersection
            group_metrics = results["intersectional"]["group_metrics"][intersection_key]
            
            # Skip if not enough groups
            if len(group_metrics) < 2:
                continue
            
            # Calculate each fairness metric
            for metric in self.metrics:
                if metric == "demographic_parity":
                    pos_rates = [m["positive_rate"] for m in group_metrics.values()]
                    max_diff = max(pos_rates) - min(pos_rates)
                    min_ratio = min(pos_rates) / max(pos_rates) if max(pos_rates) > 0 else 1.0
                    
                    metric_key = f"{'+'.join(intersection)}_{metric}"
                    results["intersectional"]["fairness_metrics"][metric_key] = {
                        "difference": float(max_diff),
                        "ratio": float(min_ratio),
                        "max_group": max(group_metrics.items(), key=lambda x: x[1]["positive_rate"])[0],
                        "min_group": min(group_metrics.items(), key=lambda x: x[1]["positive_rate"])[0],
                        "passes_threshold": max_diff <= self.fairness_threshold
                    }
                
                elif metric == "equal_opportunity":
                    tpr_values = [m["true_positive_rate"] for m in group_metrics.values()]
                    max_diff = max(tpr_values) - min(tpr_values)
                    min_ratio = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 1.0
                    
                    metric_key = f"{'+'.join(intersection)}_{metric}"
                    results["intersectional"]["fairness_metrics"][metric_key] = {
                        "difference": float(max_diff),
                        "ratio": float(min_ratio),
                        "max_group": max(group_metrics.items(), key=lambda x: x[1]["true_positive_rate"])[0],
                        "min_group": min(group_metrics.items(), key=lambda x: x[1]["true_positive_rate"])[0],
                        "passes_threshold": max_diff <= self.fairness_threshold
                    }
                
                elif metric == "equalized_odds":
                    tpr_values = [m["true_positive_rate"] for m in group_metrics.values()]
                    fpr_values = [m["false_positive_rate"] for m in group_metrics.values()]
                    
                    tpr_diff = max(tpr_values) - min(tpr_values)
                    fpr_diff = max(fpr_values) - min(fpr_values)
                    
                    # Average difference as equalized odds violation
                    avg_diff = (tpr_diff + fpr_diff) / 2
                    
                    metric_key = f"{'+'.join(intersection)}_{metric}"
                    results["intersectional"]["fairness_metrics"][metric_key] = {
                        "difference": float(avg_diff),
                        "tpr_difference": float(tpr_diff),
                        "fpr_difference": float(fpr_diff),
                        "passes_threshold": avg_diff <= self.fairness_threshold
                    }


class CounterfactualFairnessEvaluator:
    """
    Evaluate ML models for counterfactual fairness.
    
    This is a stub implementation for testing purposes. The full implementation
    will provide counterfactual fairness evaluation capabilities for ML models.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        num_counterfactuals: int = 10,
        tolerance: float = 0.1
    ):
        """
        Initialize the CounterfactualFairnessEvaluator.
        
        Args:
            protected_attributes: List of protected attribute names
            num_counterfactuals: Number of counterfactuals to generate per sample
            tolerance: Maximum allowed difference in predictions
        """
        self.protected_attributes = protected_attributes
        self.num_counterfactuals = num_counterfactuals
        self.tolerance = tolerance
    
    def evaluate(
        self,
        model: Any,
        data: pd.DataFrame,
        counterfactual_generator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate counterfactual fairness.
        
        Args:
            model: Model to evaluate
            data: Data containing protected attributes
            counterfactual_generator: Optional generator for counterfactuals
            
        Returns:
            Dictionary containing counterfactual fairness metrics
        """
        # Mock implementation returning random metrics
        n_samples = len(data)
        
        # Generate random counterfactual fairness metrics
        max_diff = np.random.uniform(0, self.tolerance * 0.9)
        avg_diff = max_diff * 0.7
        
        results = {
            'average_difference': avg_diff,
            'maximum_difference': max_diff,
            'is_fair': max_diff <= self.tolerance,
            'attribute_results': {},
            'samples': []
        }
        
        # Generate attribute-level results
        for attr in self.protected_attributes:
            attr_diff = np.random.uniform(0, self.tolerance * 0.9)
            results['attribute_results'][attr] = {
                'avg_difference': attr_diff,
                'is_fair': attr_diff <= self.tolerance,
                'violating_samples': int(n_samples * np.random.uniform(0, 0.1))
            }
        
        # Generate sample-level results
        for i in range(min(n_samples, 10)):  # Limit to 10 samples for mock results
            results['samples'].append({
                'sample_id': i,
                'original_prediction': np.random.uniform(0, 1),
                'counterfactual_predictions': np.random.uniform(0, 1, self.num_counterfactuals).tolist(),
                'max_difference': np.random.uniform(0, self.tolerance * 0.9),
                'is_fair': True
            })
        
        return results
