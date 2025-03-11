"""
Concept drift detection for ML models.

This module provides tools for detecting and monitoring concept drift
in machine learning models, which occurs when the statistical properties
of the target variable change over time.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class ConceptDriftDetector:
    """
    Detector for concept drift in ML model predictions.
    
    This class implements methods to detect when the relationship between
    features and the target variable changes over time, indicating that
    the model may need to be retrained.
    """
    
    def __init__(
        self,
        reference_predictions: Optional[np.ndarray] = None,
        reference_targets: Optional[np.ndarray] = None,
        window_size: int = 100,
        drift_threshold: float = 0.05,
        significance_level: float = 0.01,
        metrics: Optional[List[str]] = None,
        seasonal_patterns: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the ConceptDriftDetector.
        
        Args:
            reference_predictions: Baseline predictions to compare against
            reference_targets: Baseline targets to compare against
            window_size: Size of the window for drift detection
            drift_threshold: Threshold for drift detection
            significance_level: Significance level for statistical tests
            metrics: List of metrics to track
            seasonal_patterns: Dictionary defining seasonal patterns for reference data
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.reference_predictions = reference_predictions
        self.reference_targets = reference_targets
        self.reference_data = None  # Store reference features when available
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.metrics = metrics or ['accuracy', 'f1', 'auc']
        self.drift_type = 'concept_drift'
        self.drift_history = []
        self.seasonal_patterns = seasonal_patterns or {}
        self.seasonal_reference_data = {}
        
        # Store baseline metrics if reference data is provided
        if reference_predictions is not None and reference_targets is not None:
            self.reference_metrics = self._calculate_metrics(reference_predictions, reference_targets)
        else:
            self.reference_metrics = None
        
        # Storage for recent predictions and targets
        self.recent_predictions = []
        self.recent_targets = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ConceptDriftDetector':
        """
        Fit the detector with reference data to establish a baseline.
        
        Args:
            X: Feature data used for predictions
            y: Target/ground truth values
            
        Returns:
            Self for method chaining
        """
        # Store reference data
        if isinstance(X, pd.DataFrame):
            # Store the feature data for later analysis
            self.reference_data = X.copy()
            # Make predictions using a simple model if needed
            # For now we'll use the actual targets as reference predictions
            # In a real implementation, we would train a model here
            self.reference_predictions = y.values if isinstance(y, pd.Series) else np.array(y)
        else:
            # For numpy arrays
            self.reference_data = X.copy() if hasattr(X, 'copy') else X
            self.reference_predictions = y
            
        self.reference_targets = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Calculate reference metrics
        self.reference_metrics = self._calculate_metrics(
            self.reference_predictions,
            self.reference_targets
        )
        
        logger.info(f"Concept drift detector fitted with {len(self.reference_targets)} reference samples")
        
        return self
    
    def _calculate_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate accuracy metric."""
        # Simple binary classification accuracy
        predictions_binary = (predictions > 0.5).astype(int)
        return np.mean(predictions_binary == targets)
    
    def _calculate_auc_robust(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate AUC with robust handling of edge cases.
        
        Args:
            predictions: Model predictions (continuous values)
            targets: Binary ground truth labels
            
        Returns:
            AUC score
        """
        try:
            # Ensure arrays are 1D and have same length
            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()
            
            if len(predictions) != len(targets):
                logger.warning(f"Length mismatch: predictions ({len(predictions)}) vs targets ({len(targets)})")
                # Truncate to common length
                min_len = min(len(predictions), len(targets))
                predictions = predictions[:min_len]
                targets = targets[:min_len]
            
            # Check if we have both classes
            if np.unique(targets).size < 2:
                logger.warning("AUC calculation requires both positive and negative samples")
                return 0.5
                
            # Use sklearn for reliable calculation
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(targets, predictions)
            
        except Exception as e:
            logger.warning(f"Error calculating AUC: {str(e)}")
            return 0.5  # Default value for degenerate case
    
    def _calculate_f1(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate F1 score metric."""
        predictions_binary = (predictions > 0.5).astype(int)
        true_positives = np.sum((predictions_binary == 1) & (targets == 1))
        false_positives = np.sum((predictions_binary == 1) & (targets == 0))
        false_negatives = np.sum((predictions_binary == 0) & (targets == 1))
        
        # Avoid division by zero
        if true_positives == 0:
            return 0.0
            
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate all specified performance metrics.
        
        Args:
            predictions: Model predictions to evaluate
            targets: Actual target values
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Ensure predictions and targets are numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Flatten arrays for consistent processing
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        if "accuracy" in self.metrics:
            metrics["accuracy"] = self._calculate_accuracy(predictions, targets)
            
        if "auc" in self.metrics:
            metrics["auc"] = self._calculate_auc_robust(predictions, targets)
            
        if "f1" in self.metrics:
            metrics["f1"] = self._calculate_f1(predictions, targets)
            
        # Calculate mean squared error for regression
        metrics["mse"] = np.mean((predictions - targets) ** 2)
        
        # Calculate mean absolute error
        metrics["mae"] = np.mean(np.abs(predictions - targets))
        
        # Calculate correlation between predictions and targets
        try:
            metrics["correlation"] = float(np.corrcoef(predictions, targets)[0, 1])
        except:
            metrics["correlation"] = 0.0
            
        return metrics
    
    def _detect_statistical_drift(self, recent_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect if there is a statistically significant drift in the metrics.
        
        Args:
            recent_metrics: Dictionary of metrics calculated on recent data
            
        Returns:
            Dictionary with drift information
        """
        drift_detected = False
        drift_metrics = {}
        
        for metric in self.metrics:
            reference_value = self.reference_metrics.get(metric, 0)
            current_value = recent_metrics.get(metric, 0)
            
            # Calculate absolute and relative change
            abs_change = current_value - reference_value
            rel_change = abs_change / reference_value if reference_value != 0 else float('inf')
            
            # Determine if drift is significant for this metric
            is_significant = abs(rel_change) > self.drift_threshold
            
            drift_metrics[metric] = {
                "reference_value": reference_value,
                "current_value": current_value,
                "absolute_change": abs_change,
                "relative_change": rel_change,
                "is_significant": is_significant
            }
            
            if is_significant:
                drift_detected = True
        
        return {
            "drift_detected": drift_detected,
            "metrics": drift_metrics,
            "timestamp": datetime.now()
        }
    
    def update_with_batch(
        self,
        new_predictions: np.ndarray,
        new_targets: np.ndarray
    ) -> Dict[str, Any]:
        """
        Update the detector with a new batch of predictions and targets.
        
        Args:
            new_predictions: New model predictions
            new_targets: Corresponding actual target values
            
        Returns:
            Dictionary indicating if drift was detected
        """
        # Convert inputs to numpy arrays if needed
        new_predictions = np.array(new_predictions).flatten()
        new_targets = np.array(new_targets).flatten()
        
        # Add new predictions to recent history
        self.recent_predictions.extend(new_predictions)
        self.recent_targets.extend(new_targets)
        
        # Keep only the most recent window_size observations
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions = self.recent_predictions[-self.window_size:]
            self.recent_targets = self.recent_targets[-self.window_size:]
        
        # Check if we have enough data to detect drift
        if len(self.recent_predictions) < self.window_size:
            return {
                "drift_detected": False,
                "message": f"Not enough data for drift detection. Have {len(self.recent_predictions)}, need {self.window_size}.",
                "timestamp": datetime.now()
            }
        
        # Calculate metrics on recent data
        recent_metrics = self._calculate_metrics(
            np.array(self.recent_predictions),
            np.array(self.recent_targets)
        )
        
        # Detect drift
        drift_result = self._detect_statistical_drift(recent_metrics)
        
        # Store drift history
        self.drift_history.append(drift_result)
        
        return drift_result
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical tests to detect concept drift.
        
        Returns:
            Dictionary with statistical test results
        """
        if len(self.recent_predictions) < self.window_size:
            return {
                "test_valid": False,
                "message": f"Not enough data for statistical tests. Have {len(self.recent_predictions)}, need {self.window_size}."
            }
        
        # Perform Kolmogorov-Smirnov test on prediction error distributions
        reference_errors = np.abs(self.reference_predictions - self.reference_targets)
        recent_errors = np.abs(np.array(self.recent_predictions) - np.array(self.recent_targets))
        
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_errors, recent_errors)
        
        # Check prediction-target correlation change
        reference_corr = np.corrcoef(self.reference_predictions, self.reference_targets)[0, 1]
        recent_corr = np.corrcoef(np.array(self.recent_predictions), np.array(self.recent_targets))[0, 1]
        
        # Determine if there's a significant difference in performance
        recent_metrics = self._calculate_metrics(
            np.array(self.recent_predictions),
            np.array(self.recent_targets)
        )
        
        test_results = {
            "test_valid": True,
            "ks_test": {
                "statistic": float(ks_statistic),
                "p_value": float(ks_pvalue),
                "significant": ks_pvalue < self.significance_level
            },
            "correlation_change": {
                "reference_correlation": float(reference_corr),
                "recent_correlation": float(recent_corr),
                "absolute_change": float(abs(reference_corr - recent_corr)),
                "significant": abs(reference_corr - recent_corr) > 0.1
            },
            "performance_metrics": {
                metric: {
                    "reference": self.reference_metrics.get(metric, 0),
                    "recent": recent_metrics.get(metric, 0),
                    "change": recent_metrics.get(metric, 0) - self.reference_metrics.get(metric, 0)
                }
                for metric in self.metrics
            }
        }
        
        # Overall result
        test_results["drift_detected"] = (
            test_results["ks_test"]["significant"] or 
            test_results["correlation_change"]["significant"]
        )
        
        return test_results
    
    def detect_drift_ensemble(
        self,
        model_id: str,
        current_data: pd.DataFrame,
        current_predictions: np.ndarray,
        current_targets: np.ndarray,
        ensemble_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect drift using an ensemble of multiple detection methods.
        
        Args:
            model_id: ID of the model to check
            current_data: Current feature data
            current_predictions: Current model predictions
            current_targets: Current target/ground truth values
            ensemble_weights: Optional weights for different detection methods
            
        Returns:
            Ensemble drift detection results
        """
        # Default weights if not provided
        if ensemble_weights is None:
            ensemble_weights = {
                "statistical_tests": 0.3,
                "performance_metrics": 0.3,
                "distribution_distance": 0.2,
                "feature_correlation": 0.2
            }
        
        # Run individual detection methods
        results = {}
        
        # 1. Statistical tests (KS test, etc.)
        if "statistical_tests" in ensemble_weights:
            try:
                from scipy import stats
                
                # Statistical tests on feature distributions
                test_results = {}
                if current_data is not None and self.drift_detectors[model_id]['data_drift'].reference_data is not None:
                    reference_data = self.drift_detectors[model_id]['data_drift'].reference_data
                    
                    for col in current_data.columns:
                        if col in reference_data.columns and np.issubdtype(current_data[col].dtype, np.number):
                            # Perform KS test
                            ks_stat, ks_pval = stats.ks_2samp(
                                reference_data[col].dropna().values,
                                current_data[col].dropna().values
                            )
                            test_results[col] = {
                                "ks_statistic": float(ks_stat),
                                "ks_pvalue": float(ks_pval),
                                "significant": ks_pval < 0.01
                            }
                
                # Count significant test results
                sig_count = sum(1 for r in test_results.values() if r.get("significant", False))
                
                # Calculate drift score based on proportion of significant tests
                if test_results:
                    stat_drift_score = sig_count / len(test_results)
                else:
                    stat_drift_score = 0.0
                
                results["statistical_tests"] = {
                    "drift_score": stat_drift_score,
                    "drift_detected": stat_drift_score > 0.1,
                    "test_results": test_results
                }
            except Exception as e:
                logger.warning(f"Error in statistical tests drift detection: {str(e)}")
                results["statistical_tests"] = {"error": str(e)}
        
        # 2. Performance metrics
        if "performance_metrics" in ensemble_weights:
            try:
                # Get concept drift detector
                concept_detector = self.drift_detectors[model_id].get('concept_drift')
                if concept_detector and concept_detector.reference_metrics:
                    # Calculate current metrics
                    current_metrics = concept_detector._calculate_metrics(
                        current_predictions, current_targets
                    )
                    
                    # Compare with reference metrics
                    metric_changes = {}
                    for metric, ref_value in concept_detector.reference_metrics.items():
                        if metric in current_metrics:
                            abs_change = current_metrics[metric] - ref_value
                            rel_change = abs_change / ref_value if ref_value != 0 else float('inf')
                            
                            metric_changes[metric] = {
                                "reference": ref_value,
                                "current": current_metrics[metric],
                                "absolute_change": abs_change,
                                "relative_change": rel_change,
                                "significant": abs(rel_change) > concept_detector.drift_threshold
                            }
                    
                    # Count significant metric changes
                    sig_count = sum(1 for m in metric_changes.values() if m.get("significant", False))
                    
                    # Calculate drift score
                    if metric_changes:
                        perf_drift_score = sig_count / len(metric_changes)
                    else:
                        perf_drift_score = 0.0
                    
                    results["performance_metrics"] = {
                        "drift_score": perf_drift_score,
                        "drift_detected": perf_drift_score > 0.0,  # Any significant change counts
                        "metric_changes": metric_changes
                    }
            except Exception as e:
                logger.warning(f"Error in performance metrics drift detection: {str(e)}")
                results["performance_metrics"] = {"error": str(e)}
        
        # 3. Distribution distance
        if "distribution_distance" in ensemble_weights:
            try:
                # Get prediction drift detector
                pred_detector = self.drift_detectors[model_id].get('prediction_drift')
                if pred_detector and hasattr(pred_detector, 'reference_distribution'):
                    # Compare distributions using Jensen-Shannon divergence
                    from scipy.spatial import distance
                    
                    # Create histograms
                    ref_hist, ref_bins = np.histogram(
                        pred_detector.reference_distribution, 
                        bins=20, 
                        range=(0, 1), 
                        density=True
                    )
                    current_hist, _ = np.histogram(
                        current_predictions, 
                        bins=ref_bins, 
                        density=True
                    )
                    
                    # Calculate JS divergence
                    js_div = distance.jensenshannon(ref_hist, current_hist)
                    
                    # Calculate drift score (normalize to 0-1)
                    dist_drift_score = min(1.0, js_div * 5.0)  # Scale factor of 5
                    
                    results["distribution_distance"] = {
                        "drift_score": float(dist_drift_score),
                        "drift_detected": dist_drift_score > 0.2,
                        "distance_metric": "jensen_shannon",
                        "distance_value": float(js_div)
                    }
            except Exception as e:
                logger.warning(f"Error in distribution distance drift detection: {str(e)}")
                results["distribution_distance"] = {"error": str(e)}
        
        # 4. Feature correlation
        if "feature_correlation" in ensemble_weights:
            try:
                if current_data is not None and self.drift_detectors[model_id]['data_drift'].reference_data is not None:
                    reference_data = self.drift_detectors[model_id]['data_drift'].reference_data
                    
                    # Calculate feature correlations
                    num_ref_data = reference_data.select_dtypes(include=[np.number])
                    num_current_data = current_data.select_dtypes(include=[np.number])
                    
                    # Get common columns
                    common_cols = list(set(num_ref_data.columns) & set(num_current_data.columns))
                    
                    if len(common_cols) >= 2:
                        # Calculate correlation matrices
                        ref_corr = num_ref_data[common_cols].corr().fillna(0)
                        current_corr = num_current_data[common_cols].corr().fillna(0)
                        
                        # Calculate Frobenius norm of the difference
                        corr_diff = np.linalg.norm(ref_corr.values - current_corr.values)
                        
                        # Normalize by the number of correlations
                        n_correlations = len(common_cols) * (len(common_cols) - 1) / 2
                        norm_diff = corr_diff / np.sqrt(n_correlations)
                        
                        # Calculate drift score (normalize to 0-1)
                        corr_drift_score = min(1.0, norm_diff)
                        
                        results["feature_correlation"] = {
                            "drift_score": float(corr_drift_score),
                            "drift_detected": corr_drift_score > 0.15,
                            "correlation_difference": float(corr_diff),
                            "normalized_difference": float(norm_diff)
                        }
                    else:
                        results["feature_correlation"] = {
                            "drift_score": 0.0,
                            "drift_detected": False,
                            "message": "Insufficient numeric features for correlation analysis"
                        }
            except Exception as e:
                logger.warning(f"Error in feature correlation drift detection: {str(e)}")
                results["feature_correlation"] = {"error": str(e)}
        
        # Calculate weighted ensemble score
        ensemble_score = 0.0
        total_weight = 0.0
        
        for method, result in results.items():
            if method in ensemble_weights and "drift_score" in result:
                weight = ensemble_weights[method]
                ensemble_score += weight * result["drift_score"]
                total_weight += weight
        
        if total_weight > 0:
            ensemble_score /= total_weight
        
        # Final ensemble result
        ensemble_result = {
            "drift_detected": ensemble_score > 0.15,  # Lower threshold for ensemble
            "drift_score": float(ensemble_score),
            "method_results": results,
            "weights": ensemble_weights,
            "timestamp": datetime.now().isoformat()
        }
        
        return ensemble_result
    
    def detect_drift(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Detect concept drift by comparing new data with reference data.
        
        Args:
            X: Feature data used for predictions 
            y: Target/ground truth values
            
        Returns:
            Dictionary containing drift detection results
        """
        if not hasattr(self, 'reference_metrics') or self.reference_metrics is None:
            raise ValueError("Detector has not been fitted with reference data")
        
        # Convert inputs to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            # In a real implementation, we would make predictions using the model
            # For now, we'll use the actual targets
            current_predictions = y.values if isinstance(y, pd.Series) else np.array(y)
        else:
            current_predictions = np.array(y)
            
        current_targets = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Calculate current performance metrics
        current_metrics = self._calculate_metrics(current_predictions, current_targets)
        
        # Perform basic metric comparison
        metric_drift = self._detect_statistical_drift(current_metrics)
        
        # Perform statistical tests
        current_predictions = current_predictions.flatten()
        current_targets = current_targets.flatten()
        
        # Add new data to recent history
        self.recent_predictions.extend(current_predictions)
        self.recent_targets.extend(current_targets)
        
        # Keep only the most recent window_size observations
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions = self.recent_predictions[-self.window_size:]
            self.recent_targets = self.recent_targets[-self.window_size:]
        
        # Perform statistical tests for distribution changes
        test_results = self.perform_statistical_tests()
        
        # Calculate feature importance changes if possible
        feature_contribution_change = {}
        feature_drift_analysis = {}
        
        if isinstance(X, pd.DataFrame):
            try:
                # Perform advanced feature-level drift analysis
                feature_drift_analysis = self.analyze_feature_level_drift(X)
                
                # Calculate feature importances for reference data
                from sklearn.ensemble import RandomForestRegressor
                ref_model = RandomForestRegressor(n_estimators=50, random_state=42)
                ref_model.fit(self.reference_data, self.reference_targets)
                ref_importances = {col: imp for col, imp in zip(X.columns, ref_model.feature_importances_)}
                
                # Calculate feature importances for current data
                current_model = RandomForestRegressor(n_estimators=50, random_state=42)
                current_model.fit(X, current_targets)
                current_importances = {col: imp for col, imp in zip(X.columns, current_model.feature_importances_)}
                
                # Calculate changes in feature importance
                for col in ref_importances:
                    if col in current_importances:
                        feature_contribution_change[col] = current_importances[col] - ref_importances[col]
            except Exception as e:
                # If feature importance calculation fails, log but continue
                logger.warning(f"Failed to calculate feature contribution changes: {str(e)}")
        
        # Calculate overall drift score
        drift_score = 0.0
        
        # Contribution from metric drift
        if metric_drift["drift_detected"]:
            drift_score += 0.5
            
        # Contribution from statistical tests
        if test_results.get("drift_detected", False):
            drift_score += 0.5
        
        # Normalize to 0-1 range
        drift_score = min(1.0, drift_score)
        
        # Prepare final result
        result = {
            "concept_drift_detected": metric_drift["drift_detected"] or test_results.get("drift_detected", False),
            "drift_score": drift_score,
            "metric_drift": metric_drift,
            "statistical_tests": test_results,
            "feature_drift_analysis": feature_drift_analysis,
            "feature_contribution_change": feature_contribution_change,
            "timestamp": datetime.now()
        }
        
        # Calculate model health score
        health_score = self.calculate_model_health_score(result)
        result["model_health"] = health_score
        
        # Store result in drift history
        self.drift_history.append(result)
        
        # Update the drift tracker
        self.update_drift_tracker(result)
        
        return result
        
    def analyze_feature_level_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze drift at the feature level to identify the most impactful features.
        
        Args:
            current_data: Current feature data to analyze
            
        Returns:
            Dictionary with feature-level drift analysis
        """
        if not isinstance(current_data, pd.DataFrame):
            return {"error": "Current data must be a pandas DataFrame for feature-level analysis"}
            
        if not hasattr(self, 'reference_data') or self.reference_data is None:
            return {"error": "No reference data available for comparison"}
            
        if not isinstance(self.reference_data, pd.DataFrame):
            return {"error": "Reference data must be a pandas DataFrame for feature-level analysis"}
        
        # Initialize results
        feature_results = {}
        significant_drift_features = []
        drift_impact_scores = {}
        
        # Get common features
        common_features = list(set(current_data.columns) & set(self.reference_data.columns))
        
        # Calculate sample sizes
        n_reference = len(self.reference_data)
        n_current = len(current_data)
        
        # Calculate drift for each feature
        for feature in common_features:
            # Skip features with too many missing values
            if (current_data[feature].isna().mean() > 0.5 or 
                self.reference_data[feature].isna().mean() > 0.5):
                feature_results[feature] = {
                    "status": "skipped",
                    "reason": "Too many missing values"
                }
                continue
                
            # Extract feature values
            ref_values = self.reference_data[feature].dropna()
            current_values = current_data[feature].dropna()
            
            # Check if feature is numeric
            is_numeric = np.issubdtype(ref_values.dtype, np.number)
            
            if is_numeric:
                # Calculate basic statistics
                ref_mean = ref_values.mean()
                ref_std = ref_values.std()
                current_mean = current_values.mean()
                current_std = current_values.std()
                
                # Calculate standardized mean difference (SMD)
                # This is Cohen's d effect size for the difference between distributions
                pooled_std = np.sqrt((ref_std**2 + current_std**2) / 2)
                if pooled_std > 0:
                    smd = abs(ref_mean - current_mean) / pooled_std
                else:
                    smd = 0.0
                
                # Perform two-sample Kolmogorov-Smirnov test
                try:
                    from scipy import stats
                    ks_stat, ks_pvalue = stats.ks_2samp(ref_values, current_values)
                except Exception as e:
                    ks_stat, ks_pvalue = 0.0, 1.0
                    logger.warning(f"Error in KS test for {feature}: {str(e)}")
                
                # Check for drift based on statistical test and effect size
                is_drifting = ks_pvalue < 0.01 or smd > 0.5
                
                # Calculate practical significance
                # - Is the drift large enough to matter?
                # - Effect size > 0.8 is considered large
                practical_significance = "high" if smd > 0.8 else "medium" if smd > 0.5 else "low"
                
                # Calculate additional distributional metrics
                # Jensen-Shannon Divergence for probability distributions
                try:
                    from scipy.spatial import distance
                    # Create histograms with the same bins
                    min_val = min(ref_values.min(), current_values.min())
                    max_val = max(ref_values.max(), current_values.max())
                    bins = np.linspace(min_val, max_val, 20)
                    
                    ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)
                    current_hist, _ = np.histogram(current_values, bins=bins, density=True)
                    
                    # Add small epsilon to avoid zeros
                    epsilon = 1e-10
                    ref_hist = ref_hist + epsilon
                    current_hist = current_hist + epsilon
                    
                    # Normalize
                    ref_hist = ref_hist / ref_hist.sum()
                    current_hist = current_hist / current_hist.sum()
                    
                    # Calculate JS divergence
                    js_divergence = distance.jensenshannon(ref_hist, current_hist)
                except Exception as e:
                    js_divergence = 0.0
                    logger.warning(f"Error calculating Jensen-Shannon divergence for {feature}: {str(e)}")
                
                # Calculate potential outliers
                try:
                    ref_q1, ref_q3 = np.percentile(ref_values, [25, 75])
                    ref_iqr = ref_q3 - ref_q1
                    lower_bound = ref_q1 - 1.5 * ref_iqr
                    upper_bound = ref_q3 + 1.5 * ref_iqr
                    
                    # Calculate percentage of current values outside reference IQR bounds
                    outlier_pct = np.mean((current_values < lower_bound) | (current_values > upper_bound))
                except Exception as e:
                    outlier_pct = 0.0
                    logger.warning(f"Error calculating outliers for {feature}: {str(e)}")
                
                # Store results
                feature_results[feature] = {
                    "type": "numeric",
                    "reference_stats": {
                        "mean": float(ref_mean),
                        "std": float(ref_std),
                        "min": float(ref_values.min()),
                        "max": float(ref_values.max()),
                        "median": float(ref_values.median())
                    },
                    "current_stats": {
                        "mean": float(current_mean),
                        "std": float(current_std),
                        "min": float(current_values.min()),
                        "max": float(current_values.max()),
                        "median": float(current_values.median())
                    },
                    "drift_metrics": {
                        "standardized_mean_diff": float(smd),
                        "ks_statistic": float(ks_stat),
                        "ks_pvalue": float(ks_pvalue),
                        "js_divergence": float(js_divergence),
                        "outlier_percentage": float(outlier_pct)
                    },
                    "is_drifting": is_drifting,
                    "drift_magnitude": float(smd),  # Use SMD as drift magnitude
                    "practical_significance": practical_significance
                }
                
                # Calculate drift impact score (combination of statistical and practical significance)
                # Scale from 0-1 where 1 is high impact
                statistical_sig = 1.0 - min(1.0, ks_pvalue * 100)  # Transform p-value
                effect_size_sig = min(1.0, smd / 1.0)  # Normalize SMD
                
                drift_impact = 0.6 * effect_size_sig + 0.3 * statistical_sig + 0.1 * float(js_divergence)
                drift_impact_scores[feature] = drift_impact
                
                if is_drifting:
                    significant_drift_features.append(feature)
                
            else:
                # Categorical feature analysis
                try:
                    # Calculate category distributions
                    ref_counts = ref_values.value_counts(normalize=True)
                    current_counts = current_values.value_counts(normalize=True)
                    
                    # Get all categories
                    all_categories = list(set(ref_counts.index) | set(current_counts.index))
                    
                    # Calculate chi-square test
                    from scipy import stats
                    
                    # Prepare contingency table
                    ref_dist = np.array([ref_counts.get(cat, 0) for cat in all_categories])
                    current_dist = np.array([current_counts.get(cat, 0) for cat in all_categories])
                    
                    # Ensure vectors represent counts not proportions
                    ref_counts_abs = (ref_dist * n_reference).round().astype(int)
                    current_counts_abs = (current_dist * n_current).round().astype(int)
                    
                    # Handle zero counts by adding small pseudocount
                    ref_counts_abs = np.maximum(ref_counts_abs, 1)
                    current_counts_abs = np.maximum(current_counts_abs, 1)
                    
                    # Perform chi-square test
                    try:
                        chi2_stat, chi2_pvalue = stats.chi2_contingency(
                            np.vstack([ref_counts_abs, current_counts_abs])
                        )[0:2]
                    except Exception as e:
                        chi2_stat, chi2_pvalue = 0.0, 1.0
                        logger.warning(f"Error in chi-square test for {feature}: {str(e)}")
                    
                    # Calculate effect size (Cramer's V)
                    n = n_reference + n_current
                    df = len(all_categories) - 1
                    if n > 0 and df > 0:
                        cramer_v = np.sqrt(chi2_stat / (n * df))
                    else:
                        cramer_v = 0.0
                    
                    # Calculate population stability index (PSI)
                    # Add small epsilon to avoid zeros
                    epsilon = 1e-6
                    psi_terms = []
                    
                    for cat in all_categories:
                        ref_prob = ref_counts.get(cat, epsilon)
                        if ref_prob < epsilon:
                            ref_prob = epsilon
                        
                        current_prob = current_counts.get(cat, epsilon)
                        if current_prob < epsilon:
                            current_prob = epsilon
                        
                        psi_terms.append((current_prob - ref_prob) * np.log(current_prob / ref_prob))
                    
                    psi = sum(psi_terms)
                    
                    # Check for new categories
                    new_categories = set(current_counts.index) - set(ref_counts.index)
                    
                    # Is drift significant?
                    is_drifting = chi2_pvalue < 0.01 or cramer_v > 0.3 or psi > 0.2 or len(new_categories) > 0
                    
                    # Calculate practical significance
                    practical_significance = "high" if (cramer_v > 0.5 or psi > 0.25) else \
                                             "medium" if (cramer_v > 0.3 or psi > 0.1) else "low"
                    
                    # Store results
                    feature_results[feature] = {
                        "type": "categorical",
                        "reference_distribution": ref_counts.to_dict(),
                        "current_distribution": current_counts.to_dict(),
                        "drift_metrics": {
                            "chi2_statistic": float(chi2_stat),
                            "chi2_pvalue": float(chi2_pvalue),
                            "cramers_v": float(cramer_v),
                            "psi": float(psi),
                            "new_categories": list(new_categories),
                            "new_category_count": len(new_categories)
                        },
                        "is_drifting": is_drifting,
                        "drift_magnitude": float(cramer_v),  # Use Cramer's V as magnitude
                        "practical_significance": practical_significance
                    }
                    
                    # Calculate drift impact score
                    statistical_sig = 1.0 - min(1.0, chi2_pvalue * 100)  # Transform p-value
                    effect_size_sig = min(1.0, cramer_v / 0.6)  # Normalize Cramer's V
                    new_cat_impact = min(1.0, len(new_categories) / 5)  # Impact of new categories
                    
                    drift_impact = 0.4 * effect_size_sig + 0.3 * statistical_sig + 0.2 * min(1.0, psi) + 0.1 * new_cat_impact
                    drift_impact_scores[feature] = drift_impact
                    
                    if is_drifting:
                        significant_drift_features.append(feature)
                        
                except Exception as e:
                    logger.warning(f"Error analyzing categorical feature {feature}: {str(e)}")
                    feature_results[feature] = {
                        "type": "categorical",
                        "error": str(e),
                        "is_drifting": False
                    }
        
        # Get top drifting features by impact score
        top_drifting_features = sorted(
            [(f, score) for f, score in drift_impact_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 features
        
        # Calculate overall feature drift score
        if drift_impact_scores:
            # Weight by importance if available
            if hasattr(self, 'feature_importances') and self.feature_importances:
                weighted_scores = []
                for feature, score in drift_impact_scores.items():
                    if feature in self.feature_importances:
                        importance = self.feature_importances[feature]
                        weighted_scores.append(score * importance)
                    else:
                        weighted_scores.append(score * 0.01)  # Default low importance
                        
                overall_feature_drift = sum(weighted_scores) / sum(self.feature_importances.values())
            else:
                # Simple average if no importance weights
                overall_feature_drift = sum(drift_impact_scores.values()) / len(drift_impact_scores)
        else:
            overall_feature_drift = 0.0
        
        # Assemble final results
        results = {
            "feature_details": feature_results,
            "significant_drift_features": significant_drift_features,
            "drift_impact_scores": drift_impact_scores,
            "top_drifting_features": top_drifting_features,
            "overall_feature_drift_score": overall_feature_drift,
            "feature_drift_detected": len(significant_drift_features) > 0,
            "feature_drift_summary": self._generate_feature_drift_summary(
                significant_drift_features, feature_results, top_drifting_features
            )
        }
        
        return results
    
    def _generate_feature_drift_summary(
        self, 
        significant_features: List[str],
        feature_results: Dict[str, Dict[str, Any]],
        top_drifting_features: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        Generate a human-readable summary of feature drift.
        
        Args:
            significant_features: List of features with significant drift
            feature_results: Detailed results for each feature
            top_drifting_features: Top drifting features by impact score
            
        Returns:
            Dictionary with summary information
        """
        if not significant_features:
            return {
                "message": "No significant feature drift detected",
                "details": "All features are stable and within expected distributions"
            }
        
        # Count drifting features by type
        numeric_drift = sum(1 for f in significant_features 
                          if f in feature_results and feature_results[f].get("type") == "numeric")
        categorical_drift = sum(1 for f in significant_features 
                              if f in feature_results and feature_results[f].get("type") == "categorical")
        
        # Count by significance
        high_significance = sum(1 for f in significant_features 
                               if f in feature_results and feature_results[f].get("practical_significance") == "high")
        medium_significance = sum(1 for f in significant_features 
                                if f in feature_results and feature_results[f].get("practical_significance") == "medium")
        
        # Generate feature-specific messages
        feature_messages = []
        
        for feature, impact in top_drifting_features:
            if feature not in feature_results:
                continue
                
            result = feature_results[feature]
            feature_type = result.get("type")
            
            if feature_type == "numeric":
                ref_mean = result.get("reference_stats", {}).get("mean", 0)
                current_mean = result.get("current_stats", {}).get("mean", 0)
                
                if current_mean > ref_mean:
                    direction = "increased"
                else:
                    direction = "decreased"
                
                change_pct = abs(current_mean - ref_mean) / (abs(ref_mean) if ref_mean != 0 else 1) * 100
                
                message = f"{feature}: Mean has {direction} by {change_pct:.1f}% " + \
                          f"({ref_mean:.2f} → {current_mean:.2f})"
                          
            elif feature_type == "categorical":
                new_cats = result.get("drift_metrics", {}).get("new_categories", [])
                
                if new_cats:
                    message = f"{feature}: New categories detected ({', '.join(new_cats[:3])})" + \
                              (f" and {len(new_cats)-3} more" if len(new_cats) > 3 else "")
                else:
                    message = f"{feature}: Distribution has changed significantly"
                    
            else:
                message = f"{feature}: Drift detected"
                
            feature_messages.append(message)
        
        # Generate overall summary
        if high_significance > 0:
            severity = "high"
            recommendation = "Urgent investigation recommended as this may significantly impact model performance"
        elif medium_significance > 0 or len(significant_features) > 3:
            severity = "medium"
            recommendation = "Investigation recommended as this may affect model performance"
        else:
            severity = "low"
            recommendation = "Monitor these features in future drift checks"
        
        summary = {
            "message": f"Feature drift detected in {len(significant_features)} features ({numeric_drift} numeric, {categorical_drift} categorical)",
            "severity": severity,
            "top_features": feature_messages,
            "recommendation": recommendation,
            "action_items": [
                "Investigate data sources for these features",
                "Check for upstream data processing changes",
                "Consider retraining model with recent data" if severity != "low" else "Continue monitoring"
            ]
        }
        
        return summary
    
    def calculate_model_health_score(self, drift_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate a comprehensive model health score based on drift metrics.
        
        Args:
            drift_result: Results from drift detection
            
        Returns:
            Dictionary with model health metrics
        """
        # Initialize health scores with default weights
        health_components = {
            "concept_drift": {
                "score": 1.0,  # Higher is better (1.0 = no drift)
                "weight": 0.4
            },
            "performance_stability": {
                "score": 1.0,
                "weight": 0.3
            },
            "feature_stability": {
                "score": 1.0,
                "weight": 0.2
            },
            "historical_stability": {
                "score": 1.0, 
                "weight": 0.1
            }
        }
        
        # 1. Concept Drift Score (penalize for drift)
        drift_detected = drift_result.get("concept_drift_detected", False)
        drift_score = drift_result.get("drift_score", 0.0)
        
        if drift_detected:
            # Penalize based on drift score (1.0 - drift_score)
            health_components["concept_drift"]["score"] = max(0.0, 1.0 - drift_score * 1.2)  # Extra 20% penalty
        
        # 2. Performance Stability (based on metric changes)
        if "metric_drift" in drift_result and "metrics" in drift_result["metric_drift"]:
            metric_changes = []
            for metric, info in drift_result["metric_drift"]["metrics"].items():
                if "relative_change" in info:
                    metric_changes.append(abs(info["relative_change"]))
            
            if metric_changes:
                # Average metric change (penalize larger changes)
                avg_change = sum(metric_changes) / len(metric_changes)
                # Map change to score (0.3 change -> 0.7 score)
                health_components["performance_stability"]["score"] = max(0.0, 1.0 - min(1.0, avg_change))
        
        # 3. Feature Stability (based on feature importance changes)
        feature_changes = drift_result.get("feature_contribution_change", {})
        if feature_changes:
            # Calculate average absolute change in feature importance
            avg_feature_change = sum(abs(change) for change in feature_changes.values()) / len(feature_changes)
            # Map to score (0-1 range, 0.2 is significant change)
            health_components["feature_stability"]["score"] = max(0.0, 1.0 - min(1.0, avg_feature_change / 0.2))
        
        # 4. Historical Stability (based on drift history)
        if hasattr(self, "drift_tracker"):
            # Calculate stability based on recent drift pattern
            recent_pattern = self.drift_tracker.get("recent_drift_pattern", [])
            if recent_pattern:
                # Percentage of recent checks without drift
                stability_rate = 1.0 - (sum(recent_pattern) / len(recent_pattern))
                health_components["historical_stability"]["score"] = stability_rate
        
        # Calculate weighted health score
        overall_score = sum(
            component["score"] * component["weight"] 
            for component in health_components.values()
        )
        
        # Determine health status based on score
        if overall_score >= 0.8:
            health_status = "healthy"
        elif overall_score >= 0.6:
            health_status = "stable"
        elif overall_score >= 0.4:
            health_status = "degraded"
        else:
            health_status = "critical"
        
        # Calculate days since last retraining (placeholder - would need to be tracked externally)
        days_since_training = 0  # Would be calculated based on model metadata
        
        return {
            "overall_score": overall_score,
            "status": health_status,
            "components": health_components,
            "days_since_training": days_since_training,
            "recommended_action": self._get_health_recommendation(overall_score, drift_detected),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_health_recommendation(self, health_score: float, drift_detected: bool) -> str:
        """
        Get a recommendation based on model health score.
        
        Args:
            health_score: Overall health score
            drift_detected: Whether drift was detected
            
        Returns:
            Recommendation string
        """
        if health_score < 0.4:
            return "URGENT: Model retraining required due to critical health"
        elif health_score < 0.6 and drift_detected:
            return "Schedule model retraining due to significant drift and degraded health"
        elif health_score < 0.7:
            return "Closely monitor model performance and prepare for potential retraining"
        elif drift_detected:
            return "Continue monitoring model despite good health due to detected drift"
        else:
            return "Model healthy - continue regular monitoring"
    
    def check_for_drift(self) -> Dict[str, Any]:
        """
        Check if concept drift has been detected.
        
        Returns:
            Dictionary with drift detection summary
        """
        if not self.drift_history:
            return {
                "drift_detected": False,
                "message": "No drift history available.",
                "timestamp": datetime.now()
            }
        
        # Get most recent drift detection result
        latest_result = self.drift_history[-1]
        
        # Perform additional statistical tests
        test_results = self.perform_statistical_tests()
        
        # Combine results
        summary = {
            "drift_detected": latest_result["drift_detected"] or test_results.get("drift_detected", False),
            "metrics_drift": latest_result["metrics"],
            "statistical_tests": test_results,
            "window_size": self.window_size,
            "drift_threshold": self.drift_threshold,
            "significance_level": self.significance_level,
            "timestamp": datetime.now(),
            "drift_tracker_status": self._get_drift_tracker_status()
        }
        
        return summary
    
    def _get_drift_tracker_status(self) -> Dict[str, Any]:
        """
        Get a summary of the drift tracker status.
        
        Returns:
            Dictionary with drift tracking metrics
        """
        if not hasattr(self, "drift_tracker"):
            self.drift_tracker = {
                "detection_count": 0,
                "last_detection_time": None,
                "total_checks": 0,
                "detection_rate": 0.0,
                "severity_history": [],
                "metric_history": {},
                "detection_by_metric": {},
                "consecutive_detections": 0,
                "drift_free_period": 0,
                "recent_drift_pattern": [],
            }
        
        return self.drift_tracker
    
    def update_drift_tracker(self, drift_result: Dict[str, Any]) -> None:
        """
        Update the drift tracker with new drift detection results.
        
        Args:
            drift_result: Result from drift detection
        """
        if not hasattr(self, "drift_tracker"):
            self.drift_tracker = {
                "detection_count": 0,
                "last_detection_time": None,
                "total_checks": 0,
                "detection_rate": 0.0,
                "severity_history": [],
                "metric_history": {},
                "detection_by_metric": {},
                "consecutive_detections": 0,
                "drift_free_period": 0,
                "recent_drift_pattern": [],
            }
        
        # Update basic counters
        self.drift_tracker["total_checks"] += 1
        
        # Track drift detection
        drift_detected = drift_result.get("concept_drift_detected", drift_result.get("drift_detected", False))
        
        # Update pattern tracking (keep last 10 results)
        self.drift_tracker["recent_drift_pattern"].append(1 if drift_detected else 0)
        if len(self.drift_tracker["recent_drift_pattern"]) > 10:
            self.drift_tracker["recent_drift_pattern"] = self.drift_tracker["recent_drift_pattern"][-10:]
        
        if drift_detected:
            self.drift_tracker["detection_count"] += 1
            self.drift_tracker["last_detection_time"] = datetime.now()
            self.drift_tracker["consecutive_detections"] += 1
            self.drift_tracker["drift_free_period"] = 0
            
            # Track severity if available
            if "drift_score" in drift_result:
                self.drift_tracker["severity_history"].append(drift_result["drift_score"])
                if len(self.drift_tracker["severity_history"]) > 100:
                    self.drift_tracker["severity_history"] = self.drift_tracker["severity_history"][-100:]
        else:
            self.drift_tracker["consecutive_detections"] = 0
            self.drift_tracker["drift_free_period"] += 1
        
        # Update detection rate
        self.drift_tracker["detection_rate"] = self.drift_tracker["detection_count"] / self.drift_tracker["total_checks"]
        
        # Track drift by metric type if available
        if "metric_drift" in drift_result and "metrics" in drift_result["metric_drift"]:
            for metric, metric_info in drift_result["metric_drift"]["metrics"].items():
                # Initialize if needed
                if metric not in self.drift_tracker["metric_history"]:
                    self.drift_tracker["metric_history"][metric] = []
                    self.drift_tracker["detection_by_metric"][metric] = 0
                
                # Track metric value
                if "current_value" in metric_info:
                    self.drift_tracker["metric_history"][metric].append(metric_info["current_value"])
                    # Keep only last 100 values
                    if len(self.drift_tracker["metric_history"][metric]) > 100:
                        self.drift_tracker["metric_history"][metric] = self.drift_tracker["metric_history"][metric][-100:]
                
                # Track if this metric indicated drift
                if metric_info.get("is_significant", False):
                    self.drift_tracker["detection_by_metric"][metric] += 1

    def get_drift_tracking_insights(self) -> Dict[str, Any]:
        """
        Get insights from the drift tracking history.
        
        Returns:
            Dictionary with drift tracking insights
        """
        if not hasattr(self, "drift_tracker"):
            return {"message": "No drift tracking data available"}
        
        # Calculate drift frequency and patterns
        drift_pattern_insights = self._analyze_drift_pattern()
        
        # Calculate metric contribution to drift
        metric_contribution = {}
        for metric, detection_count in self.drift_tracker["detection_by_metric"].items():
            if self.drift_tracker["detection_count"] > 0:
                contribution = detection_count / self.drift_tracker["detection_count"]
                metric_contribution[metric] = contribution
        
        # Get top contributing metrics sorted by contribution
        top_contributing_metrics = sorted(
            [(m, c) for m, c in metric_contribution.items() if c > 0], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Calculate mean and max severity if available
        severity_stats = {}
        if self.drift_tracker["severity_history"]:
            severity_stats = {
                "mean_severity": np.mean(self.drift_tracker["severity_history"]),
                "max_severity": np.max(self.drift_tracker["severity_history"]),
                "current_severity": self.drift_tracker["severity_history"][-1] if self.drift_tracker["severity_history"] else 0,
                "severity_trend": "increasing" if len(self.drift_tracker["severity_history"]) > 1 and 
                                 self.drift_tracker["severity_history"][-1] > self.drift_tracker["severity_history"][0] else "decreasing"
            }
        
        # Calculate stability score (higher is more stable)
        stability_score = 1.0 - (self.drift_tracker["detection_rate"] * 0.7 + 
                              (self.drift_tracker["consecutive_detections"] / 10 if self.drift_tracker["consecutive_detections"] < 10 else 1.0) * 0.3)
        
        # Generate drift forecast
        forecast_results = self.forecast_drift()
        
        result = {
            "detection_rate": self.drift_tracker["detection_rate"],
            "consecutive_detections": self.drift_tracker["consecutive_detections"],
            "drift_free_period": self.drift_tracker["drift_free_period"],
            "drift_pattern": drift_pattern_insights,
            "metric_contribution": metric_contribution,
            "top_contributing_metrics": top_contributing_metrics,
            "severity_stats": severity_stats,
            "stability_score": stability_score,
            "drift_forecast": forecast_results,
            "early_warning": forecast_results.get("early_warning", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _analyze_drift_pattern(self) -> Dict[str, Any]:
        """
        Analyze the pattern of drift detection over time.
        
        Returns:
            Dictionary with drift pattern analysis
        """
        if not hasattr(self, "drift_tracker") or not self.drift_tracker["recent_drift_pattern"]:
            return {"pattern_type": "unknown"}
        
        pattern = self.drift_tracker["recent_drift_pattern"]
        
        # Check for consistent drift
        if all(p == 1 for p in pattern):
            return {
                "pattern_type": "persistent",
                "description": "Continuous drift detected across all recent checks",
                "severity": "high",
                "recommendation": "Urgent model retraining recommended"
            }
        
        # Check for no drift
        if all(p == 0 for p in pattern):
            return {
                "pattern_type": "stable",
                "description": "No drift detected in recent checks",
                "severity": "none",
                "recommendation": "Continue regular monitoring"
            }
        
        # Check for alternating pattern (oscillating)
        alternating = True
        for i in range(len(pattern) - 1):
            if pattern[i] == pattern[i+1]:
                alternating = False
                break
                
        if alternating:
            return {
                "pattern_type": "oscillating",
                "description": "Alternating drift detection pattern",
                "severity": "medium",
                "recommendation": "Investigate potential cyclical data patterns"
            }
        
        # Check for increasing drift (more 1s toward the end)
        first_half = pattern[:len(pattern)//2]
        second_half = pattern[len(pattern)//2:]
        
        if sum(second_half) > sum(first_half):
            return {
                "pattern_type": "increasing",
                "description": "Drift frequency is increasing over time",
                "severity": "high",
                "recommendation": "Prepare for model retraining"
            }
        elif sum(second_half) < sum(first_half):
            return {
                "pattern_type": "decreasing",
                "description": "Drift frequency is decreasing over time",
                "severity": "medium",
                "recommendation": "Continue monitoring for stabilization"
            }
        
        # Default case - irregular pattern
        return {
            "pattern_type": "irregular",
            "description": "No clear pattern in drift detection",
            "severity": "medium",
            "recommendation": "Monitor closely and investigate potential causes"
        }
        
    def forecast_drift(self, horizon: int = 5) -> Dict[str, Any]:
        """
        Forecast future drift based on historical patterns.
        
        Args:
            horizon: Number of future periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        if not hasattr(self, "drift_tracker") or not self.drift_tracker["severity_history"]:
            return {
                "forecast_available": False,
                "message": "Insufficient history for drift forecasting"
            }
        
        # Need at least 10 observations for meaningful forecasting
        if len(self.drift_tracker["severity_history"]) < 10:
            return {
                "forecast_available": False,
                "message": f"Need at least 10 observations, but only have {len(self.drift_tracker['severity_history'])}"
            }
        
        # Initialize forecast results
        forecast_results = {
            "forecast_available": True,
            "forecast_horizon": horizon,
            "forecasted_values": [],
            "confidence_intervals": [],
            "drift_probability": 0.0,
            "days_until_drift": None,
            "trend": "stable"
        }
        
        try:
            # Simple exponential smoothing for forecasting
            alpha = 0.3  # Smoothing parameter
            severity_history = self.drift_tracker["severity_history"]
            
            # Initialize with the first value
            level = severity_history[0]
            smoothed_values = [level]
            
            # Apply smoothing
            for i in range(1, len(severity_history)):
                level = alpha * severity_history[i] + (1 - alpha) * level
                smoothed_values.append(level)
            
            # Forecast future values
            forecasted_values = []
            for _ in range(horizon):
                forecasted_values.append(level)  # Simple forecast is just the last level
            
            # Calculate 95% confidence intervals
            # Assuming normally distributed errors
            errors = [severity_history[i] - smoothed_values[i-1] for i in range(1, len(severity_history))]
            error_std = np.std(errors) if len(errors) > 1 else 0.1
            
            confidence_intervals = []
            for _ in range(horizon):
                confidence_intervals.append({
                    "lower": max(0.0, level - 1.96 * error_std),
                    "upper": min(1.0, level + 1.96 * error_std)
                })
            
            # Calculate drift probability
            threshold = self.drift_threshold
            
            # Probability based on forecast distribution
            # Assume normal distribution with mean=level, std=error_std
            from scipy.stats import norm
            if error_std > 0:
                drift_probability = 1.0 - norm.cdf(threshold, loc=level, scale=error_std)
            else:
                drift_probability = 1.0 if level > threshold else 0.0
            
            # Calculate days until drift threshold reached
            days_until_drift = None
            if level < threshold and error_std > 0:
                # Calculate growth rate from recent history (last 5 points)
                recent_history = severity_history[-5:] if len(severity_history) >= 5 else severity_history
                if len(recent_history) >= 2:
                    growth_rate = (recent_history[-1] - recent_history[0]) / len(recent_history)
                    if growth_rate > 0:
                        # Estimate days until threshold
                        days_until_drift = int((threshold - level) / growth_rate)
            
            # Determine trend
            if len(severity_history) >= 3:
                first_half = severity_history[:len(severity_history)//2]
                second_half = severity_history[len(severity_history)//2:]
                first_mean = sum(first_half) / len(first_half)
                second_mean = sum(second_half) / len(second_half)
                
                if second_mean > first_mean * 1.1:
                    trend = "increasing"
                elif second_mean < first_mean * 0.9:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            # Update forecast results
            forecast_results.update({
                "forecasted_values": forecasted_values,
                "confidence_intervals": confidence_intervals,
                "current_level": level,
                "drift_probability": drift_probability,
                "days_until_drift": days_until_drift,
                "trend": trend
            })
            
            # Generate early warning signal
            early_warning = self._generate_early_warning(
                level, drift_probability, days_until_drift, trend
            )
            forecast_results["early_warning"] = early_warning
            
        except Exception as e:
            # Log error but return partial results
            logger.warning(f"Error in drift forecasting: {str(e)}")
            forecast_results["forecasting_error"] = str(e)
        
        return forecast_results
    
    def _generate_early_warning(
        self, 
        current_level: float, 
        drift_probability: float,
        days_until_drift: Optional[int],
        trend: str
    ) -> Dict[str, Any]:
        """
        Generate early warning signals based on forecast data.
        
        Args:
            current_level: Current drift level
            drift_probability: Probability of drift
            days_until_drift: Predicted days until drift threshold
            trend: Identified trend
            
        Returns:
            Early warning information
        """
        # Initialize no warning
        warning = {
            "warning_level": "none",
            "message": "No drift warning",
            "urgent": False,
            "recommendation": "Continue regular monitoring"
        }
        
        # Check for immediate warning conditions
        if current_level >= self.drift_threshold:
            warning = {
                "warning_level": "critical",
                "message": "Current drift level exceeds threshold",
                "urgent": True,
                "recommendation": "Immediate model retraining recommended"
            }
        # Check for high probability warning
        elif drift_probability >= 0.75:
            warning = {
                "warning_level": "high",
                "message": f"High probability ({drift_probability:.1%}) of drift",
                "urgent": True,
                "recommendation": "Schedule model retraining soon"
            }
        # Check for impending drift
        elif days_until_drift is not None and days_until_drift <= 7 and trend == "increasing":
            warning = {
                "warning_level": "medium",
                "message": f"Drift projected in {days_until_drift} days",
                "urgent": False,
                "recommendation": "Prepare for model retraining"
            }
        # Check for elevated risk
        elif drift_probability >= 0.5 or trend == "increasing":
            warning = {
                "warning_level": "low",
                "message": f"Elevated drift risk ({drift_probability:.1%}) or increasing trend",
                "urgent": False,
                "recommendation": "Increase monitoring frequency"
            }
        
        # Add probabilities and metrics to warning
        warning.update({
            "drift_probability": drift_probability,
            "days_until_drift": days_until_drift,
            "trend": trend,
            "current_level": current_level,
            "threshold": self.drift_threshold
        })
        
        return warning

    def initialize_seasonal_reference(
        self, 
        season_id: str, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Initialize seasonal reference data for drift detection.
        
        This method allows storing multiple reference datasets for different
        seasonal patterns (e.g., weekday vs weekend, holiday vs non-holiday).
        
        Args:
            season_id: Identifier for the seasonal pattern
            X: Feature data for this seasonal pattern
            y: Target data for this seasonal pattern
            
        Returns:
            Dictionary with initialization status
        """
        # Store reference data for this season
        if isinstance(X, pd.DataFrame):
            # Store the feature data for later analysis
            self.seasonal_reference_data[season_id] = {
                'features': X.copy(),
                'targets': y.values if isinstance(y, pd.Series) else np.array(y),
                'predictions': None,  # Will be set when a model is available
                'metrics': None,      # Will be calculated when predictions are available
                'timestamp': datetime.now()
            }
        else:
            # For numpy arrays
            self.seasonal_reference_data[season_id] = {
                'features': X.copy() if hasattr(X, 'copy') else X,
                'targets': y.values if isinstance(y, pd.Series) else np.array(y),
                'predictions': None,
                'metrics': None,
                'timestamp': datetime.now()
            }
            
        logger.info(f"Initialized seasonal reference data for '{season_id}' with {len(y)} samples")
        
        return {
            "success": True,
            "season_id": season_id,
            "samples": len(y),
            "timestamp": datetime.now().isoformat()
        }
    
    def update_seasonal_predictions(
        self, 
        season_id: str, 
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Update predictions for a seasonal reference dataset.
        
        Args:
            season_id: Identifier for the seasonal pattern
            predictions: Model predictions for the seasonal reference data
            
        Returns:
            Dictionary with update status
        """
        if season_id not in self.seasonal_reference_data:
            return {
                "success": False,
                "message": f"No reference data found for season '{season_id}'",
                "timestamp": datetime.now().isoformat()
            }
            
        seasonal_data = self.seasonal_reference_data[season_id]
        seasonal_data['predictions'] = predictions
        
        # Calculate reference metrics for this season
        seasonal_data['metrics'] = self._calculate_metrics(
            predictions,
            seasonal_data['targets']
        )
        
        logger.info(f"Updated seasonal predictions for '{season_id}'")
        
        return {
            "success": True,
            "season_id": season_id,
            "metrics": seasonal_data['metrics'],
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_seasonal_drift(
        self, 
        season_id: str,
        current_data: np.ndarray,
        current_predictions: np.ndarray,
        current_targets: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect concept drift against seasonal reference data.
        
        Args:
            season_id: Identifier for the seasonal pattern to compare against
            current_data: Current feature data
            current_predictions: Current model predictions
            current_targets: Current target values
            
        Returns:
            Dictionary with drift detection results
        """
        if season_id not in self.seasonal_reference_data:
            return {
                "success": False,
                "message": f"No reference data found for season '{season_id}'",
                "timestamp": datetime.now().isoformat()
            }
            
        seasonal_data = self.seasonal_reference_data[season_id]
        
        # If no predictions available for this season, we can't detect concept drift
        if seasonal_data['predictions'] is None:
            return {
                "success": False,
                "message": f"No predictions available for season '{season_id}'",
                "timestamp": datetime.now().isoformat()
            }
            
        # Calculate metrics on current data
        current_metrics = self._calculate_metrics(
            current_predictions,
            current_targets
        )
        
        # Compare metrics to seasonal reference
        drift_detected = False
        drift_metrics = {}
        
        for metric in self.metrics:
            reference_value = seasonal_data['metrics'].get(metric, 0)
            current_value = current_metrics.get(metric, 0)
            
            # Calculate absolute and relative change
            abs_change = current_value - reference_value
            rel_change = abs_change / reference_value if reference_value != 0 else float('inf')
            
            # Determine if drift is significant for this metric
            is_significant = abs(rel_change) > self.drift_threshold
            
            drift_metrics[metric] = {
                "reference_value": reference_value,
                "current_value": current_value,
                "absolute_change": abs_change,
                "relative_change": rel_change,
                "is_significant": is_significant
            }
            
            if is_significant:
                drift_detected = True
        
        # Perform statistical tests
        statistical_tests = {}
        
        # Kolmogorov-Smirnov test on prediction error distributions
        reference_errors = np.abs(seasonal_data['predictions'] - seasonal_data['targets'])
        current_errors = np.abs(current_predictions - current_targets)
        
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_errors, current_errors)
        
        statistical_tests["ks_test"] = {
            "statistic": float(ks_statistic),
            "p_value": float(ks_pvalue),
            "significant": ks_pvalue < self.significance_level
        }
        
        if statistical_tests["ks_test"]["significant"]:
            drift_detected = True
        
        # Store result in drift history
        drift_result = {
            "drift_detected": drift_detected,
            "season_id": season_id,
            "metrics": drift_metrics,
            "statistical_tests": statistical_tests,
            "timestamp": datetime.now().isoformat()
        }
        
        self.drift_history.append(drift_result)
        
        return drift_result