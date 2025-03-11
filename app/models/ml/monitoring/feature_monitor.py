"""
Feature monitoring module for ML model features.

This module provides classes and functions for monitoring feature distributions
and detecting distribution shifts in ML systems.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

class FeatureDistributionMonitor:
    """
    Monitor distributions of model features over time.
    
    This class tracks feature distributions across multiple time periods and
    detects shifts that might affect model performance.
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        features_to_monitor: Optional[List[str]] = None,
        drift_threshold: float = 0.05,
        distribution_distance_metric: str = "ks_test",
        window_size: int = 1000
    ):
        """
        Initialize the feature distribution monitor.
        
        Args:
            reference_data: Baseline dataset to compare against
            features_to_monitor: List of feature names to monitor (None = all)
            drift_threshold: Threshold for significant distribution drift
            distribution_distance_metric: Method to measure distribution distance
                ("ks_test", "js_divergence", "wasserstein", "kl_divergence")
            window_size: Size of the sliding window for online monitoring
        """
        self.reference_data = reference_data
        self.features_to_monitor = features_to_monitor
        self.drift_threshold = drift_threshold
        self.distribution_distance_metric = distribution_distance_metric
        self.window_size = window_size
        self.current_window = []
        self.drift_history = []
        self.is_trained = reference_data is not None
        
        # Store reference distribution statistics
        self.reference_stats = {}
        if self.is_trained and reference_data is not None:
            self._compute_reference_statistics()
    
    def _compute_reference_statistics(self):
        """Compute and store statistics of the reference distribution."""
        if self.reference_data is None:
            return
            
        features = self.features_to_monitor or self.reference_data.columns
        
        for feature in features:
            if feature in self.reference_data.columns:
                feature_data = self.reference_data[feature].values
                self.reference_stats[feature] = {
                    "mean": np.mean(feature_data),
                    "std": np.std(feature_data),
                    "min": np.min(feature_data),
                    "max": np.max(feature_data),
                    "median": np.median(feature_data),
                    "histogram": np.histogram(feature_data, bins=20)
                }
    
    def update(self, new_data: Union[pd.DataFrame, Dict[str, Any]]):
        """
        Update the monitor with new data.
        
        Args:
            new_data: New observation(s) to add to the current window
        """
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
            
        # Add to current window
        self.current_window.append(new_data)
        
        # Trim window if needed
        if len(self.current_window) > self.window_size:
            self.current_window = self.current_window[-self.window_size:]
        
        # Set reference data if not already set
        if self.reference_data is None and len(self.current_window) >= self.window_size:
            self.reference_data = pd.concat(self.current_window[:self.window_size//2])
            self._compute_reference_statistics()
            self.is_trained = True
    
    def detect_drift(self, current_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect drift in feature distributions.
        
        Args:
            current_data: Current data to check for drift (uses current window if None)
            
        Returns:
            Dictionary with drift detection results
        """
        if not self.is_trained:
            return {
                "drift_detected": False,
                "message": "Reference distribution not established",
                "timestamp": datetime.now()
            }
            
        if current_data is None:
            if not self.current_window:
                return {
                    "drift_detected": False,
                    "message": "No current data available",
                    "timestamp": datetime.now()
                }
            current_data = pd.concat(self.current_window[-self.window_size//2:])
        
        # Mock results for stub implementation
        drift_detected = False
        feature_drift_scores = {}
        drifted_features = []
        
        features = self.features_to_monitor or current_data.columns
        
        for feature in features:
            if feature in current_data.columns and feature in self.reference_stats:
                # Generate random drift score for mock implementation
                drift_score = np.random.uniform(0, 0.1)
                feature_drift_scores[feature] = drift_score
                
                if drift_score > self.drift_threshold:
                    drift_detected = True
                    drifted_features.append(feature)
        
        result = {
            "drift_detected": drift_detected,
            "drift_scores": feature_drift_scores,
            "drifted_features": drifted_features,
            "threshold": self.drift_threshold,
            "metric": self.distribution_distance_metric,
            "timestamp": datetime.now()
        }
        
        self.drift_history.append(result)
        return result
    
    def get_drift_history(self) -> List[Dict[str, Any]]:
        """Get the history of drift detection results."""
        return self.drift_history
    
    def reset(self):
        """Reset the monitor to its initial state."""
        self.current_window = []
        self.drift_history = []
        self.is_trained = self.reference_data is not None


class FeatureCorrelationMonitor:
    """
    Monitor changes in feature correlations over time.
    
    This class tracks how correlations between features change,
    which can indicate data quality issues or concept drift.
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        correlation_threshold: float = 0.1,
        window_size: int = 1000
    ):
        """
        Initialize the feature correlation monitor.
        
        Args:
            reference_data: Baseline dataset to compare against
            correlation_threshold: Threshold for significant correlation change
            window_size: Size of the sliding window for online monitoring
        """
        self.reference_data = reference_data
        self.correlation_threshold = correlation_threshold
        self.window_size = window_size
        self.current_window = []
        self.correlation_history = []
        self.is_trained = reference_data is not None
        
        # Store reference correlation matrix
        self.reference_correlation = None
        if self.is_trained and reference_data is not None:
            self.reference_correlation = reference_data.corr().fillna(0)
    
    def update(self, new_data: Union[pd.DataFrame, Dict[str, Any]]):
        """
        Update the monitor with new data.
        
        Args:
            new_data: New observation(s) to add to the current window
        """
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
            
        # Add to current window
        self.current_window.append(new_data)
        
        # Trim window if needed
        if len(self.current_window) > self.window_size:
            self.current_window = self.current_window[-self.window_size:]
        
        # Set reference data if not already set
        if self.reference_data is None and len(self.current_window) >= self.window_size:
            self.reference_data = pd.concat(self.current_window[:self.window_size//2])
            self.reference_correlation = self.reference_data.corr().fillna(0)
            self.is_trained = True
    
    def detect_correlation_drift(self, current_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect drift in feature correlations.
        
        Args:
            current_data: Current data to check for correlation drift
            
        Returns:
            Dictionary with correlation drift detection results
        """
        if not self.is_trained:
            return {
                "drift_detected": False,
                "message": "Reference correlation not established",
                "timestamp": datetime.now()
            }
            
        if current_data is None:
            if not self.current_window:
                return {
                    "drift_detected": False,
                    "message": "No current data available",
                    "timestamp": datetime.now()
                }
            current_data = pd.concat(self.current_window[-self.window_size//2:])
        
        # Mock results for stub implementation
        current_correlation = current_data.corr().fillna(0)
        correlation_difference = {}
        max_difference = 0.0
        drift_detected = False
        
        # Mock random differences for stub implementation
        for col1 in current_correlation.columns:
            for col2 in current_correlation.columns:
                if col1 < col2:  # Only consider upper triangle
                    if col1 in self.reference_correlation.columns and col2 in self.reference_correlation.columns:
                        # Generate random difference for mock implementation
                        diff = np.random.uniform(0, 0.05)
                        correlation_difference[f"{col1}_{col2}"] = diff
                        max_difference = max(max_difference, diff)
                        
                        if diff > self.correlation_threshold:
                            drift_detected = True
        
        result = {
            "drift_detected": drift_detected,
            "correlation_differences": correlation_difference,
            "max_difference": max_difference,
            "threshold": self.correlation_threshold,
            "timestamp": datetime.now()
        }
        
        self.correlation_history.append(result)
        return result
    
    def get_correlation_history(self) -> List[Dict[str, Any]]:
        """Get the history of correlation drift detection results."""
        return self.correlation_history
    
    def reset(self):
        """Reset the monitor to its initial state."""
        self.current_window = []
        self.correlation_history = []
        self.is_trained = self.reference_data is not None 