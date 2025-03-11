# Data Drift Monitoring Implementation

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document describes the implementation of data drift monitoring in the WITHIN Ad Score & Account Health Predictor system. Data drift monitoring is a critical component for ensuring model reliability in production by detecting shifts in input data distributions that could impact model performance.

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Implementation Details](#implementation-details)
4. [Statistical Methods](#statistical-methods)
5. [Detection Thresholds](#detection-thresholds)
6. [Alerting Integration](#alerting-integration)
7. [Remediation Actions](#remediation-actions)
8. [Usage Examples](#usage-examples)
9. [Performance Considerations](#performance-considerations)
10. [Best Practices](#best-practices)

## Introduction

Data drift occurs when the statistical properties of model inputs change over time, potentially leading to degraded model performance. The WITHIN system implements comprehensive data drift monitoring to detect such changes early and trigger appropriate responses.

### Types of Data Drift Monitored

The system monitors three primary types of data drift:

1. **Feature Distribution Drift**: Changes in individual feature distributions
2. **Feature Correlation Drift**: Changes in relationships between features
3. **Data Quality Drift**: Changes in data quality metrics (missing values, outliers, etc.)

### Key Components

The data drift monitoring implementation consists of:

- **Reference Distribution Management**: Storage and versioning of reference distributions
- **Statistical Tests**: Mathematical methods for detecting significant changes
- **Monitoring Service**: System for executing regular drift checks
- **Alerting System**: Notification mechanism for detected drift
- **Visualization**: Tools for exploring and understanding drift

More sections will be added in subsequent updates.

## System Architecture

The data drift monitoring system is integrated with the broader ProductionMonitoringService while maintaining its own specialized components.

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                 Data Drift Monitoring System                    │
│                                                                 │
│  ┌────────────────────┐       ┌──────────────────────────────┐ │
│  │ Reference Data     │       │ Current Data                 │ │
│  │ Management         │       │ Collection                   │ │
│  └─────────┬──────────┘       └──────────────┬───────────────┘ │
│            │                                 │                  │
│            ▼                                 ▼                  │
│  ┌────────────────────┐       ┌──────────────────────────────┐ │
│  │ Statistical        │       │ Drift                        │ │
│  │ Tests              │◄─────►│ Detection                    │ │
│  └─────────┬──────────┘       └──────────────┬───────────────┘ │
│            │                                 │                  │
│            └────────────────┬────────────────┘                  │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────┐    ┌──────────────────────────────────┐ │
│  │ Visualization      │◄──►│ Alert                            │ │
│  │                    │    │ Management                       │ │
│  └────────────────────┘    └──────────────────────────────────┘ │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### Reference Data Management
- Stores reference distributions for each feature
- Manages versioning of reference data
- Provides APIs for accessing and updating reference distributions
- Handles data serialization and storage optimization

#### Current Data Collection
- Collects production data for drift analysis
- Applies sampling strategies for efficient processing
- Handles batching for periodic and on-demand checks
- Ensures data schema consistency

#### Statistical Tests
- Implements various statistical methods for drift detection
- Provides configurable sensitivity settings
- Optimizes computation for high-dimensional data
- Supports both univariate and multivariate tests

#### Drift Detection
- Orchestrates the drift detection process
- Manages detection thresholds and configuration
- Aggregates test results across features
- Determines overall drift severity

#### Alert Management
- Generates alerts when drift is detected
- Routes alerts to appropriate channels
- Includes relevant context and severity information
- Tracks alert history and resolution status

#### Visualization
- Provides interactive visualizations of feature distributions
- Shows drift metrics over time
- Highlights features with significant drift
- Supports drill-down analysis for investigation

### Integration Points

The data drift monitoring system integrates with:

1. **ProductionMonitoringService**: For overall monitoring coordination
2. **Model Registry**: To access model metadata and feature information
3. **Data Pipeline**: To obtain reference and current data
4. **Alert System**: To send notifications about detected drift
5. **Visualization System**: To display drift metrics and trends

### Data Flow

1. Reference distributions are established during model training or manual baseline creation
2. Production data is continuously collected and sampled
3. At scheduled intervals or on-demand, drift detection is performed
4. Statistical tests compare current distributions to reference distributions
5. If significant drift is detected, alerts are generated
6. Visualization tools display drift metrics for analysis
7. Optional remediation actions may be triggered automatically

## Statistical Methods

The data drift monitoring system employs various statistical methods to detect drift in feature distributions. The choice of method depends on the feature type, distribution characteristics, and sensitivity requirements.

### Univariate Methods

#### 1. Kolmogorov-Smirnov Test

The Kolmogorov-Smirnov (KS) test measures the maximum difference between two cumulative distribution functions.

```python
from scipy import stats

def ks_test(reference_data, current_data, alpha=0.05):
    """
    Perform Kolmogorov-Smirnov test for distribution comparison.
    
    Args:
        reference_data: Reference distribution data
        current_data: Current distribution data
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    drift_detected = p_value < alpha
    
    return {
        "test": "kolmogorov_smirnov",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": drift_detected,
        "threshold": alpha
    }
```

**Strengths**:
- Distribution-free (non-parametric)
- Sensitive to changes in shape, location, and scale
- Well-established statistical properties

**Limitations**:
- Less sensitive to changes in distribution tails
- May be less powerful for small sample sizes
- Doesn't indicate the nature of the distribution change

#### 2. Population Stability Index (PSI)

The Population Stability Index measures the distribution change between two samples by binning the data and comparing bin proportions.

```python
import numpy as np

def calculate_psi(reference_data, current_data, bins=10, epsilon=1e-6):
    """
    Calculate Population Stability Index.
    
    Args:
        reference_data: Reference distribution data
        current_data: Current distribution data
        bins: Number of bins for distribution comparison
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Dictionary with PSI results
    """
    # Create bins based on reference data
    bin_edges = np.histogram_bin_edges(reference_data, bins=bins)
    
    # Calculate bin proportions for reference data
    ref_hist, _ = np.histogram(reference_data, bins=bin_edges, density=True)
    ref_props = ref_hist / np.sum(ref_hist)
    
    # Calculate bin proportions for current data
    curr_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)
    curr_props = curr_hist / np.sum(curr_hist)
    
    # Add epsilon to avoid division by zero
    ref_props = np.maximum(ref_props, epsilon)
    curr_props = np.maximum(curr_props, epsilon)
    
    # Calculate PSI
    psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
    psi = np.sum(psi_values)
    
    # Determine if drift is detected based on thresholds
    drift_level = "none"
    if psi > 0.25:
        drift_level = "severe"
    elif psi > 0.1:
        drift_level = "moderate"
    elif psi > 0.05:
        drift_level = "minor"
    
    return {
        "test": "psi",
        "statistic": float(psi),
        "drift_detected": psi > 0.1,  # Moderate or severe drift
        "drift_level": drift_level,
        "bin_details": [
            {
                "bin_index": i,
                "reference_proportion": float(ref_props[i]),
                "current_proportion": float(curr_props[i]),
                "contribution": float(psi_values[i])
            }
            for i in range(len(ref_props))
        ]
    }
```

**Strengths**:
- Interpretable scale (< 0.1: minor change, 0.1-0.25: moderate change, > 0.25: major change)
- Provides bin-level contributions to understand where changes occur
- Commonly used in financial and risk applications

**Limitations**:
- Binning strategy can affect results
- Less suitable for multi-modal distributions
- May miss subtle distribution shape changes

#### 3. Wasserstein Distance (Earth Mover's Distance)

The Wasserstein distance measures the minimum "cost" of transforming one distribution into another.

```python
from scipy import stats

def wasserstein_distance(reference_data, current_data, threshold=0.1):
    """
    Calculate Wasserstein distance between distributions.
    
    Args:
        reference_data: Reference distribution data
        current_data: Current distribution data
        threshold: Threshold for drift detection
        
    Returns:
        Dictionary with distance results
    """
    # Normalize data to same scale for comparable distances
    ref_min, ref_max = np.min(reference_data), np.max(reference_data)
    curr_min, curr_max = np.min(current_data), np.max(current_data)
    
    data_range = max(ref_max, curr_max) - min(ref_min, curr_min)
    if data_range == 0:
        data_range = 1.0  # Avoid division by zero
    
    ref_normalized = (reference_data - ref_min) / data_range
    curr_normalized = (current_data - curr_min) / data_range
    
    # Calculate Wasserstein distance
    distance = stats.wasserstein_distance(ref_normalized, curr_normalized)
    
    return {
        "test": "wasserstein_distance",
        "statistic": float(distance),
        "drift_detected": distance > threshold,
        "threshold": threshold
    }
```

**Strengths**:
- Accounts for distribution geometry
- Sensitive to both subtle and significant distribution changes
- Works well with continuous numerical features

**Limitations**:
- Computationally more intensive
- Less interpretable than some other metrics
- Threshold selection is less standardized

### Multivariate Methods

#### 1. Maximum Mean Discrepancy (MMD)

MMD measures the difference between distributions in a reproduced kernel Hilbert space.

```python
def maximum_mean_discrepancy(reference_data, current_data, kernel='rbf', threshold=0.05):
    """
    Calculate Maximum Mean Discrepancy between multivariate distributions.
    
    Args:
        reference_data: Reference distribution data (n_samples x n_features)
        current_data: Current distribution data (m_samples x n_features)
        kernel: Kernel function ('rbf', 'linear', 'polynomial')
        threshold: Threshold for drift detection
        
    Returns:
        Dictionary with MMD results
    """
    from sklearn.metrics.pairwise import pairwise_kernels
    
    # Compute kernel matrices
    K_XX = pairwise_kernels(reference_data, reference_data, metric=kernel)
    K_YY = pairwise_kernels(current_data, current_data, metric=kernel)
    K_XY = pairwise_kernels(reference_data, current_data, metric=kernel)
    
    # Calculate MMD
    n_x = K_XX.shape[0]
    n_y = K_YY.shape[0]
    
    mmd = (np.sum(K_XX) / (n_x * n_x) + 
           np.sum(K_YY) / (n_y * n_y) - 
           2 * np.sum(K_XY) / (n_x * n_y))
    
    return {
        "test": "maximum_mean_discrepancy",
        "kernel": kernel,
        "statistic": float(mmd),
        "drift_detected": mmd > threshold,
        "threshold": threshold
    }
```

**Strengths**:
- Handles high-dimensional data well
- Can detect subtle distribution changes
- No assumptions about distribution shapes

**Limitations**:
- Computationally expensive for large datasets
- Kernel selection affects sensitivity
- Less interpretable than univariate methods

#### 2. Feature-wise Testing with Multiple Hypothesis Correction

This approach applies univariate tests to each feature and corrects for multiple hypothesis testing.

```python
def multivariate_drift_detection(reference_data, current_data, test_func=ks_test, 
                                 correction='fdr_bh', alpha=0.05):
    """
    Detect drift in multivariate data with multiple hypothesis testing correction.
    
    Args:
        reference_data: Reference data (n_samples x n_features)
        current_data: Current data (m_samples x n_features)
        test_func: Function to apply for univariate testing
        correction: Multiple hypothesis correction method
        alpha: Significance level
        
    Returns:
        Dictionary with multivariate drift results
    """
    from statsmodels.stats.multitest import multipletests
    
    n_features = reference_data.shape[1]
    feature_results = []
    p_values = []
    
    # Apply test to each feature
    for i in range(n_features):
        result = test_func(reference_data[:, i], current_data[:, i])
        feature_results.append(result)
        p_values.append(result['p_value'])
    
    # Apply multiple hypothesis correction
    rejected, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method=correction)
    
    # Count drifted features
    drifted_features = np.sum(rejected)
    
    return {
        "test": "multivariate_drift_detection",
        "correction_method": correction,
        "drift_detected": drifted_features > 0,
        "drifted_feature_count": int(drifted_features),
        "drifted_feature_indices": np.where(rejected)[0].tolist(),
        "alpha": alpha,
        "feature_results": [
            {
                "feature_index": i,
                "original_p_value": float(p_values[i]),
                "corrected_p_value": float(corrected_p_values[i]),
                "drift_detected": bool(rejected[i])
            }
            for i in range(n_features)
        ]
    }
```

**Strengths**:
- Provides feature-level drift detection
- Controls for false positives across multiple features
- Can be used with any univariate test

**Limitations**:
- Doesn't account for feature correlations
- May miss complex multivariate distribution changes
- Correction methods may be too conservative

### Method Selection Guidelines

| Feature Type | Recommended Methods | Notes |
|--------------|---------------------|-------|
| Numerical (continuous) | Wasserstein Distance, KS Test | Wasserstein better for capturing subtle changes |
| Numerical (discrete) | PSI, Chi-squared Test | Binning important for discrete variables |
| Categorical | Chi-squared Test, PSI | Appropriate for categorical distributions |
| High-dimensional | MMD, PCA + univariate tests | Dimensionality reduction often helpful |
| Time series | Dynamic Time Warping, Spectral tests | Special tests for temporal data |

### Computational Efficiency

For features with large data volumes, the system implements:

1. **Data Sampling**: Randomly sample from both reference and current data
2. **Approximate Methods**: Fast approximations of statistical tests
3. **Parallel Computation**: Run tests for different features in parallel
4. **Progressive Testing**: Start with fast methods, use more complex ones only if needed

## Implementation Details

The data drift monitoring system is implemented as a standalone component that integrates with the ProductionMonitoringService. This section provides details on the implementation architecture and key classes.

### Core Classes

#### DataDriftMonitor

The `DataDriftMonitor` class is the main entry point for drift detection functionality:

```python
class DataDriftMonitor:
    """
    Monitor for detecting data drift in production features.
    
    This class provides methods to detect, track, and alert on
    data drift in model features.
    """
    
    def __init__(
        self,
        model_id: str,
        reference_data: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        reference_id: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
        detection_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the data drift monitor.
        
        Args:
            model_id: Identifier for the model being monitored
            reference_data: Optional reference data for feature distributions
            reference_id: Optional identifier for stored reference data
            feature_names: Optional list of feature names to monitor
            feature_types: Optional dictionary mapping feature names to types
            detection_config: Optional configuration for drift detection
        """
        self.model_id = model_id
        self.feature_names = feature_names
        self.feature_types = feature_types or {}
        
        # Set up default configuration
        self.config = {
            "test_methods": {
                "numeric": ["ks", "wasserstein"],
                "categorical": ["chi2", "psi"]
            },
            "thresholds": {
                "ks_p_value": 0.05,
                "wasserstein": 0.1,
                "psi": 0.1,
                "chi2_p_value": 0.05
            },
            "sampling": {
                "max_samples": 10000,
                "enabled": True
            },
            "alerting": {
                "min_drift_ratio": 0.1,  # Minimum ratio of drifted features to alert
                "enabled": True
            }
        }
        
        # Override with user configuration
        if detection_config:
            self._update_config(detection_config)
            
        # Initialize reference data
        self.reference_data = None
        self.reference_stats = {}
        self.reference_id = None
        
        if reference_data is not None:
            self.set_reference_data(reference_data)
        elif reference_id is not None:
            self.load_reference_data(reference_id)
            
        # Initialize storage for drift results
        self.drift_history = []
        
        # Initialize detector registry
        self.detectors = self._initialize_detectors()
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize feature-specific drift detectors based on type."""
        detectors = {}
        
        for feature_name in self.feature_names or []:
            feature_type = self.feature_types.get(feature_name, "numeric")
            
            if feature_type == "numeric":
                detectors[feature_name] = {
                    "ks": KolmogorovSmirnovDetector(
                        threshold=self.config["thresholds"]["ks_p_value"]
                    ),
                    "wasserstein": WassersteinDetector(
                        threshold=self.config["thresholds"]["wasserstein"]
                    )
                }
            elif feature_type == "categorical":
                detectors[feature_name] = {
                    "chi2": ChiSquaredDetector(
                        threshold=self.config["thresholds"]["chi2_p_value"]
                    ),
                    "psi": PSIDetector(
                        threshold=self.config["thresholds"]["psi"]
                    )
                }
                
        return detectors
    
    def set_reference_data(self, reference_data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            reference_data: Reference data for feature distributions
        """
        # Store reference data
        self.reference_data = reference_data
        
        # Extract feature names if not provided
        if self.feature_names is None:
            if isinstance(reference_data, pd.DataFrame):
                self.feature_names = reference_data.columns.tolist()
            elif isinstance(reference_data, dict):
                self.feature_names = list(reference_data.keys())
        
        # Calculate reference statistics
        self._calculate_reference_stats()
        
        # Initialize detectors if needed
        if not self.detectors:
            self.detectors = self._initialize_detectors()
            
        # Store reference data in detectors
        self._update_detectors_with_reference()
        
        # Generate reference ID
        self.reference_id = f"{self.model_id}_ref_{int(time.time())}"
        
        logger.info(f"Reference data set for model {self.model_id} with {len(self.feature_names)} features")
    
    def detect_drift(
        self,
        current_data: Union[pd.DataFrame, Dict[str, Any]],
        features: Optional[List[str]] = None,
        methods: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current data to compare with reference
            features: Optional subset of features to check
            methods: Optional override of test methods
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
        
        # Determine features to check
        features_to_check = features or self.feature_names
        
        # Prepare results structure
        results = {
            "model_id": self.model_id,
            "reference_id": self.reference_id,
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drifted_features_count": 0,
            "feature_results": {},
            "overall_drift_score": 0.0
        }
        
        # Check each feature
        for feature in features_to_check:
            # Get current feature data
            current_feature_data = self._extract_feature_data(current_data, feature)
            
            # Get feature type
            feature_type = self.feature_types.get(feature, "numeric")
            
            # Determine methods to use
            feature_methods = methods.get(feature_type, self.config["test_methods"][feature_type]) if methods else self.config["test_methods"][feature_type]
            
            # Check drift for this feature
            feature_result = self._check_feature_drift(feature, current_feature_data, feature_methods)
            
            # Store result
            results["feature_results"][feature] = feature_result
            
            # Update overall drift status
            if feature_result["drift_detected"]:
                results["drifted_features_count"] += 1
        
        # Calculate overall drift statistics
        total_features = len(features_to_check)
        drift_ratio = results["drifted_features_count"] / total_features if total_features > 0 else 0
        
        results["drift_detected"] = drift_ratio >= self.config["alerting"]["min_drift_ratio"]
        results["drift_ratio"] = drift_ratio
        results["overall_drift_score"] = self._calculate_overall_drift_score(results["feature_results"])
        
        # Store in history
        self.drift_history.append(results)
        
        # Send alert if needed
        if results["drift_detected"] and self.config["alerting"]["enabled"]:
            self._send_drift_alert(results)
        
        return results
    
    def _check_feature_drift(
        self,
        feature: str,
        current_data: Any,
        methods: List[str]
    ) -> Dict[str, Any]:
        """
        Check drift for a specific feature using specified methods.
        
        Args:
            feature: Feature name
            current_data: Current feature data
            methods: List of test methods to use
            
        Returns:
            Dictionary with feature drift results
        """
        if feature not in self.detectors:
            raise ValueError(f"No detector configured for feature: {feature}")
            
        # Initialize result
        result = {
            "feature": feature,
            "drift_detected": False,
            "drift_score": 0.0,
            "method_results": {}
        }
        
        # Apply each method
        for method in methods:
            if method not in self.detectors[feature]:
                logger.warning(f"Method {method} not available for feature {feature}")
                continue
                
            # Get detector
            detector = self.detectors[feature][method]
            
            # Run detection
            method_result = detector.detect_drift(current_data)
            
            # Store result
            result["method_results"][method] = method_result
            
            # Update feature drift status
            if method_result["drift_detected"]:
                result["drift_detected"] = True
        
        # Calculate overall drift score for this feature
        result["drift_score"] = self._calculate_feature_drift_score(result["method_results"])
        
        return result
    
    def _calculate_feature_drift_score(self, method_results: Dict[str, Any]) -> float:
        """Calculate aggregate drift score from multiple method results."""
        if not method_results:
            return 0.0
            
        # Different aggregation strategies could be used
        # Here we use maximum normalized score
        max_score = 0.0
        
        for method, result in method_results.items():
            if method == "ks" or method == "chi2":
                # For p-value tests, lower p means higher drift
                score = 1.0 - min(result["p_value"] / self.config["thresholds"][f"{method}_p_value"], 1.0)
            elif method == "wasserstein":
                # For distance metrics, higher distance means higher drift
                score = min(result["statistic"] / self.config["thresholds"]["wasserstein"], 1.0)
            elif method == "psi":
                # For PSI, higher value means higher drift
                score = min(result["statistic"] / self.config["thresholds"]["psi"], 1.0)
            else:
                score = 0.0 if not result["drift_detected"] else 0.5
                
            max_score = max(max_score, score)
            
        return max_score
    
    def _calculate_overall_drift_score(self, feature_results: Dict[str, Any]) -> float:
        """Calculate overall drift score across all features."""
        if not feature_results:
            return 0.0
            
        # Use weighted average of feature drift scores
        # Features with higher importance get higher weights
        total_weight = 0.0
        weighted_sum = 0.0
        
        for feature, result in feature_results.items():
            # Get feature importance (default to 1.0)
            importance = self.feature_importance.get(feature, 1.0) if hasattr(self, "feature_importance") else 1.0
            
            weighted_sum += result["drift_score"] * importance
            total_weight += importance
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _send_drift_alert(self, drift_result: Dict[str, Any]) -> None:
        """Send alert for detected drift."""
        # This would integrate with the alerting system
        # For now, just log the alert
        logger.warning(
            f"Data drift detected for model {self.model_id}: "
            f"{drift_result['drifted_features_count']} features drifted "
            f"(drift score: {drift_result['overall_drift_score']:.2f})"
        )
```

#### DriftDetector

The `DriftDetector` is a base class for specific drift detection algorithms:

```python
class DriftDetector:
    """Base class for drift detectors."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            threshold: Threshold for drift detection
        """
        self.threshold = threshold
        self.reference_data = None
        self.reference_stats = {}
    
    def set_reference(self, reference_data: Any) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            reference_data: Reference data to compare against
        """
        self.reference_data = reference_data
        self._compute_reference_stats()
    
    def _compute_reference_stats(self) -> None:
        """Compute statistics on reference data."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def detect_drift(self, current_data: Any) -> Dict[str, Any]:
        """
        Detect drift in current data compared to reference.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        raise NotImplementedError("Subclasses must implement this method")
```

#### Implementation of Specific Detectors

Each statistical method is implemented as a subclass of `DriftDetector`:

```python
class KolmogorovSmirnovDetector(DriftDetector):
    """Detector using Kolmogorov-Smirnov test."""
    
    def _compute_reference_stats(self) -> None:
        """Compute statistics on reference data."""
        # For KS test, we just need to store the reference data
        # No pre-computation needed
        pass
    
    def detect_drift(self, current_data: Any) -> Dict[str, Any]:
        """Detect drift using Kolmogorov-Smirnov test."""
        from scipy import stats
        
        statistic, p_value = stats.ks_2samp(self.reference_data, current_data)
        drift_detected = p_value < self.threshold
        
        return {
            "test": "kolmogorov_smirnov",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "threshold": self.threshold
        }


class WassersteinDetector(DriftDetector):
    """Detector using Wasserstein distance."""
    
    def _compute_reference_stats(self) -> None:
        """Compute statistics on reference data."""
        # Store min/max for normalization
        self.reference_stats["min"] = np.min(self.reference_data)
        self.reference_stats["max"] = np.max(self.reference_data)
    
    def detect_drift(self, current_data: Any) -> Dict[str, Any]:
        """Detect drift using Wasserstein distance."""
        from scipy import stats
        
        # Normalize data to same scale for comparable distances
        ref_min = self.reference_stats["min"] 
        ref_max = self.reference_stats["max"]
        curr_min, curr_max = np.min(current_data), np.max(current_data)
        
        data_range = max(ref_max, curr_max) - min(ref_min, curr_min)
        if data_range == 0:
            data_range = 1.0  # Avoid division by zero
        
        ref_normalized = (self.reference_data - ref_min) / data_range
        curr_normalized = (current_data - curr_min) / data_range
        
        # Calculate Wasserstein distance
        distance = stats.wasserstein_distance(ref_normalized, curr_normalized)
        
        return {
            "test": "wasserstein_distance",
            "statistic": float(distance),
            "drift_detected": distance > self.threshold,
            "threshold": self.threshold
        }


class PSIDetector(DriftDetector):
    """Detector using Population Stability Index."""
    
    def __init__(self, threshold: float = 0.1, bins: int = 10):
        """
        Initialize PSI detector.
        
        Args:
            threshold: Threshold for drift detection
            bins: Number of bins for PSI calculation
        """
        super().__init__(threshold)
        self.bins = bins
        self.bin_edges = None
    
    def _compute_reference_stats(self) -> None:
        """Compute statistics on reference data."""
        # Create bins based on reference data
        self.bin_edges = np.histogram_bin_edges(self.reference_data, bins=self.bins)
        
        # Calculate reference bin counts
        ref_hist, _ = np.histogram(self.reference_data, bins=self.bin_edges)
        self.reference_stats["hist"] = ref_hist
        self.reference_stats["bin_edges"] = self.bin_edges
    
    def detect_drift(self, current_data: Any) -> Dict[str, Any]:
        """Detect drift using PSI."""
        epsilon = 1e-6  # Small constant to avoid division by zero
        
        # Get reference bin proportions
        ref_hist = self.reference_stats["hist"]
        ref_props = ref_hist / np.sum(ref_hist)
        
        # Calculate bin proportions for current data
        curr_hist, _ = np.histogram(current_data, bins=self.bin_edges)
        curr_props = curr_hist / np.sum(curr_hist)
        
        # Add epsilon to avoid division by zero
        ref_props = np.maximum(ref_props, epsilon)
        curr_props = np.maximum(curr_props, epsilon)
        
        # Calculate PSI
        psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
        psi = np.sum(psi_values)
        
        # Determine drift level
        drift_level = "none"
        if psi > 0.25:
            drift_level = "severe"
        elif psi > 0.1:
            drift_level = "moderate"
        elif psi > 0.05:
            drift_level = "minor"
        
        return {
            "test": "psi",
            "statistic": float(psi),
            "drift_detected": psi > self.threshold,
            "drift_level": drift_level,
            "bin_details": [
                {
                    "bin_index": i,
                    "bin_range": [float(self.bin_edges[i]), float(self.bin_edges[i+1])],
                    "reference_proportion": float(ref_props[i]),
                    "current_proportion": float(curr_props[i]),
                    "contribution": float(psi_values[i])
                }
                for i in range(len(ref_props))
            ]
        }
```

### Integration with ProductionMonitoringService

The DataDriftMonitor integrates with the ProductionMonitoringService through the DriftMonitoringService:

```python
from app.services.monitoring.production_monitoring_service import ProductionMonitoringService, AlertLevel

# Initialize services
prod_monitoring = ProductionMonitoringService()
drift_monitor = DataDriftMonitor(model_id="ad_score_model_v1")

# Set reference data (typically done during model deployment)
reference_data = load_training_data()
drift_monitor.set_reference_data(reference_data)

# In production code, periodically check for drift
def check_for_drift(current_batch_data):
    # Detect drift
    drift_result = drift_monitor.detect_drift(current_batch_data)
    
    # If drift detected, send alert through production monitoring
    if drift_result["drift_detected"]:
        # Determine severity based on drift score
        severity = AlertLevel.WARNING
        if drift_result["overall_drift_score"] > 0.7:
            severity = AlertLevel.CRITICAL
        elif drift_result["overall_drift_score"] > 0.5:
            severity = AlertLevel.ERROR
        
        # Send alert
        prod_monitoring.send_alert(
            severity=severity,
            message=f"Data drift detected in model ad_score_model_v1",
            data={
                "drift_result": drift_result,
                "drifted_features": [
                    feature for feature, result in drift_result["feature_results"].items()
                    if result["drift_detected"]
                ]
            }
        )
        
        # Trigger remediation action if needed
        if drift_result["overall_drift_score"] > 0.6:
            trigger_model_retraining("ad_score_model_v1")
```

### Reference Data Management

Reference data is crucial for drift detection. The system provides several ways to manage reference data:

1. **Model Training Data**: The default reference distribution is based on the training data.

```python
from app.models.ml.training import get_model_training_data

# Get training data used for the model
training_data = get_model_training_data(model_id="ad_score_model_v1")

# Initialize drift monitor with training data as reference
drift_monitor = DataDriftMonitor(
    model_id="ad_score_model_v1",
    reference_data=training_data
)
```

2. **Manual Reference Creation**: Manually specify a reference dataset.

```python
# Create reference from production data
production_sample = sample_production_data(
    start_date="2023-09-01",
    end_date="2023-09-30",
    sample_size=10000
)

# Set as reference
drift_monitor.set_reference_data(production_sample)
```

3. **Versioned References**: Store and manage multiple reference versions.

```python
# Store current reference
reference_id = drift_monitor.store_reference()

# Later, load a specific reference
drift_monitor.load_reference(reference_id)

# List available references
references = drift_monitor.list_references()
```

### Drift Visualization

The system includes visualization tools for understanding detected drift:

```python
from app.services.monitoring.visualization import DriftVisualizer

# Create visualizer
visualizer = DriftVisualizer()

# Generate distribution comparison plots
plots = visualizer.create_distribution_plots(
    reference_data=drift_monitor.reference_data,
    current_data=current_batch_data,
    drifted_features=drift_result["drifted_features"]
)

# Generate drift history plot
history_plot = visualizer.create_drift_history_plot(
    drift_history=drift_monitor.drift_history,
    start_date="2023-09-01",
    end_date="2023-10-15"
)

# Generate feature correlation drift plot
correlation_plot = visualizer.create_correlation_drift_plot(
    reference_data=drift_monitor.reference_data,
    current_data=current_batch_data
)
```

These visualizations are integrated into the monitoring dashboards for easy access by ML engineers and data scientists.

## Monitoring Strategies and Automation

Effective data drift monitoring requires well-defined monitoring strategies and automation. This section describes the approaches used in our system.

### Monitoring Frequencies

Data drift monitoring operates at multiple frequencies to balance computational cost with timely detection:

1. **Real-time monitoring**: Applied to critical features for high-stakes models
2. **Batch monitoring**: Regular checks (daily/weekly) for all production models
3. **On-demand monitoring**: Triggered by specific events or manual requests

The appropriate frequency depends on:
- Model importance and criticality
- Feature volatility
- Computational resources
- Business requirements

For the ad score predictor, batch monitoring runs daily with real-time monitoring for specific high-importance features.

### Sampling Strategies

For large data volumes, sampling is essential for efficient monitoring:

```python
class SamplingStrategy:
    """Base class for sampling strategies."""
    
    def sample(self, data: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Sample data according to strategy."""
        raise NotImplementedError
        
class RandomSampling(SamplingStrategy):
    """Simple random sampling strategy."""
    
    def sample(self, data: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Randomly sample data."""
        if len(data) <= sample_size:
            return data
        return data.sample(sample_size, random_state=42)
        
class StratifiedSampling(SamplingStrategy):
    """Stratified sampling to preserve class distribution."""
    
    def __init__(self, strata_column: str):
        """Initialize with column to stratify on."""
        self.strata_column = strata_column
        
    def sample(self, data: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Sample data preserving strata proportions."""
        if len(data) <= sample_size:
            return data
            
        # Get distribution of strata
        value_counts = data[self.strata_column].value_counts(normalize=True)
        
        # Sample from each stratum
        result = pd.DataFrame()
        for value, proportion in value_counts.items():
            stratum = data[data[self.strata_column] == value]
            stratum_size = int(sample_size * proportion)
            if stratum_size > 0:
                sampled = stratum.sample(min(stratum_size, len(stratum)), random_state=42)
                result = pd.concat([result, sampled])
                
        return result
```

Our system employs adaptive sampling that adjusts sample size based on:
- Historical drift patterns
- Computational resources
- Required statistical power
- Desired confidence level

### Automated Alerting and Response

The drift monitoring system includes automated responses to detected drift:

1. **Alert Tiers**:
   - **Info**: Minimal drift detected, no action required
   - **Warning**: Significant drift detected, investigation recommended
   - **Critical**: Severe drift detected, immediate action required

2. **Response Actions**:
   - **Auto-logging**: Record all drift events with metrics
   - **Notification**: Send alerts to appropriate stakeholders
   - **Investigation triggers**: Launch automated investigation workflows
   - **Remediation**: Execute predefined remediation steps

```python
class DriftResponseHandler:
    """Handle automated responses to drift detection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with response configuration."""
        self.config = {
            "alert_thresholds": {
                "info": 0.1,
                "warning": 0.3,
                "critical": 0.6
            },
            "notification_channels": ["slack", "email"],
            "auto_remediation": {
                "enabled": True,
                "max_drift_score": 0.8,  # Maximum score for auto-remediation
                "actions": ["retraining_trigger", "feature_investigation"]
            }
        }
        
        if config:
            self.config.update(config)
            
        self.notification_service = NotificationService()
        self.remediation_service = RemediationService()
        
    def process_drift_result(self, drift_result: Dict[str, Any]) -> None:
        """Process drift detection result and take appropriate actions."""
        # Determine severity
        overall_score = drift_result["overall_drift_score"]
        severity = self._determine_severity(overall_score)
        
        # Log drift event
        self._log_drift_event(drift_result, severity)
        
        # Send notifications
        if severity != "info":
            self._send_notifications(drift_result, severity)
            
        # Trigger remediation if needed
        if severity == "critical" and self.config["auto_remediation"]["enabled"]:
            if overall_score <= self.config["auto_remediation"]["max_drift_score"]:
                self._trigger_remediation(drift_result)
                
    def _determine_severity(self, drift_score: float) -> str:
        """Determine alert severity based on drift score."""
        if drift_score >= self.config["alert_thresholds"]["critical"]:
            return "critical"
        elif drift_score >= self.config["alert_thresholds"]["warning"]:
            return "warning"
        elif drift_score >= self.config["alert_thresholds"]["info"]:
            return "info"
        return "info"
        
    def _log_drift_event(self, drift_result: Dict[str, Any], severity: str) -> None:
        """Log drift event to monitoring system."""
        # Implementation details omitted
        pass
        
    def _send_notifications(self, drift_result: Dict[str, Any], severity: str) -> None:
        """Send notifications to configured channels."""
        model_id = drift_result["model_id"]
        drifted_features = [
            feature for feature, result in drift_result["feature_results"].items()
            if result["drift_detected"]
        ]
        
        message = f"Data drift detected for model {model_id}\n"
        message += f"Severity: {severity.upper()}\n"
        message += f"Overall drift score: {drift_result['overall_drift_score']:.2f}\n"
        message += f"Drifted features: {', '.join(drifted_features[:5])}"
        if len(drifted_features) > 5:
            message += f" and {len(drifted_features) - 5} more"
            
        for channel in self.config["notification_channels"]:
            self.notification_service.send(
                channel=channel,
                message=message,
                severity=severity,
                data=drift_result
            )
            
    def _trigger_remediation(self, drift_result: Dict[str, Any]) -> None:
        """Trigger automated remediation actions."""
        model_id = drift_result["model_id"]
        
        for action in self.config["auto_remediation"]["actions"]:
            if action == "retraining_trigger":
                self.remediation_service.trigger_retraining(
                    model_id=model_id,
                    drift_result=drift_result
                )
            elif action == "feature_investigation":
                self.remediation_service.investigate_features(
                    model_id=model_id,
                    drift_result=drift_result,
                    features=[
                        feature for feature, result in drift_result["feature_results"].items()
                        if result["drift_detected"] and result["drift_score"] > 0.5
                    ]
                )
```

### Continuous Monitoring Improvements

The monitoring system incorporates feedback mechanisms to improve over time:

1. **Drift Pattern Analysis**: Analyzing historical drift patterns to refine thresholds
2. **False Alarm Reduction**: Learning from past alerts to reduce false positives
3. **Adaptive Thresholds**: Automatically adjusting thresholds based on feature behavior
4. **Performance Impact Correlation**: Correlating drift with model performance metrics

Example implementation of adaptive thresholds:

```python
class AdaptiveThresholdManager:
    """Manage adaptive thresholds for drift detection."""
    
    def __init__(self, model_id: str, initial_thresholds: Dict[str, float]):
        """Initialize with model ID and initial thresholds."""
        self.model_id = model_id
        self.thresholds = initial_thresholds
        self.history = []
        
    def update_thresholds(self, drift_result: Dict[str, Any], 
                          performance_impact: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Update thresholds based on drift result and performance impact.
        
        Args:
            drift_result: Result from drift detection
            performance_impact: Optional metrics showing performance impact
            
        Returns:
            Updated thresholds
        """
        # Store result and impact
        self.history.append({
            "drift_result": drift_result,
            "performance_impact": performance_impact,
            "timestamp": datetime.now().isoformat()
        })
        
        # Skip threshold update if not enough history
        if len(self.history) < 10:
            return self.thresholds
            
        # Update thresholds based on patterns
        for feature, feature_result in drift_result["feature_results"].items():
            # Track drift scores for this feature
            feature_scores = [
                entry["drift_result"]["feature_results"].get(feature, {}).get("drift_score", 0)
                for entry in self.history[-10:]
            ]
            
            # Get drift score volatility
            mean_score = np.mean(feature_scores)
            std_score = np.std(feature_scores)
            
            # If performance impact data available, use it
            if performance_impact and "feature_importance" in performance_impact:
                feature_importance = performance_impact["feature_importance"].get(feature, 0.5)
            else:
                feature_importance = 0.5
                
            # Adjust threshold based on volatility and importance
            if feature in self.thresholds:
                # More important features get lower thresholds (more sensitive)
                importance_factor = 1.0 - (feature_importance * 0.5)
                
                # Higher volatility gets higher thresholds (less sensitive to frequent changes)
                volatility_factor = min(std_score / (mean_score + 1e-6), 1.0)
                
                # Combine factors to adjust threshold
                self.thresholds[feature] *= (1.0 + 0.2 * volatility_factor) * importance_factor
                
                # Ensure threshold is within reasonable bounds
                self.thresholds[feature] = max(0.01, min(0.5, self.thresholds[feature]))
                
        return self.thresholds
```

### Integration Points

The data drift monitoring system integrates with several other components:

1. **Model Performance Monitoring**: Correlates drift with performance degradation
2. **Feature Store**: Accesses historical feature distributions
3. **Model Registry**: Obtains expected feature distributions for deployed models
4. **MLOps Pipeline**: Triggers automatic retraining when significant drift occurs
5. **Monitoring Dashboard**: Visualizes drift metrics for stakeholders

This comprehensive integration ensures that drift detection is part of a complete monitoring ecosystem rather than a standalone process.

## Best Practices and Future Improvements

### Current Best Practices

1. **Statistical method selection**: Choose methods appropriate for feature distributions
2. **Multiple detection methods**: Apply multiple methods to increase confidence
3. **Feature importance weighting**: Prioritize monitoring for important features  
4. **Regular reference updates**: Periodically update reference data as distributions evolve
5. **Comprehensive logging**: Log all drift events with detailed metrics
6. **Contextual alerting**: Provide context in alerts to facilitate investigation

### Future Improvements

1. **Multivariate drift detection**: Enhance detection of correlation changes between features
2. **Unsupervised anomaly detection**: Apply advanced techniques for detecting unusual patterns
3. **Causal analysis**: Implement tools to help identify root causes of drift
4. **Automatic feature transformation**: Apply transformations to make features more stable
5. **Transfer learning techniques**: Leverage transfer learning for rapid model updating
6. **Explainable drift detection**: Provide human-readable explanations of detected drift

Planned implementation for multivariate drift detection:

```python
class MultivariateDetector(DriftDetector):
    """Detector for multivariate distribution changes."""
    
    def __init__(self, threshold: float = 0.05, method: str = "mmd"):
        """
        Initialize multivariate detector.
        
        Args:
            threshold: Threshold for drift detection
            method: Method to use ('mmd', 'energy', or 'classifier')
        """
        super().__init__(threshold)
        self.method = method
        
    def detect_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect multivariate drift between reference and current data.
        
        Args:
            current_data: Current data samples (n_samples, n_features)
            
        Returns:
            Dictionary with drift detection results
        """
        if self.method == "mmd":
            return self._detect_mmd(current_data)
        elif self.method == "energy":
            return self._detect_energy(current_data)
        elif self.method == "classifier":
            return self._detect_classifier(current_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
    def _detect_mmd(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect drift using Maximum Mean Discrepancy."""
        # Implementation uses TensorFlow or PyTorch for GPU acceleration
        # Details omitted for brevity
        pass
        
    def _detect_energy(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect drift using Energy Distance test."""
        # Implementation based on scipy
        # Details omitted for brevity
        pass
        
    def _detect_classifier(self, current_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift using classifier-based approach.
        
        This method trains a classifier to distinguish between reference
        and current datasets. If the classifier achieves high accuracy,
        it indicates significant drift.
        """
        # Implementation based on scikit-learn
        # Details omitted for brevity
        pass
```

These future improvements will further enhance the system's ability to detect and respond to data drift, ultimately improving model reliability in production.

## Usage Examples

This section provides practical examples of how to use the data drift monitoring system in common scenarios.

### Basic Drift Monitoring Setup

The following example demonstrates how to set up basic drift monitoring for a model:

```python
from app.services.monitoring.data_drift import DataDriftMonitor
from app.models.ml.loader import load_model_data
import pandas as pd

# Load reference data (typically training data)
reference_data = load_model_data("ad_score_model_v1", "training")

# Define feature names and types
feature_types = {
    "account_age_days": "numeric",
    "campaign_count": "numeric",
    "avg_ctr": "numeric",
    "industry": "categorical",
    "country": "categorical",
    "platform": "categorical"
}

# Create monitor with custom configuration
monitor = DataDriftMonitor(
    model_id="ad_score_model_v1",
    reference_data=reference_data,
    feature_types=feature_types,
    detection_config={
        "thresholds": {
            "ks_p_value": 0.01,  # Stricter p-value threshold
            "wasserstein": 0.15,
            "psi": 0.15,
            "chi2_p_value": 0.01
        },
        "alerting": {
            "min_drift_ratio": 0.2,  # Alert when 20% of features drift
            "enabled": True
        }
    }
)

# Store the monitor for future use
monitor_id = monitor.save()
print(f"Monitor created with ID: {monitor_id}")
```

### Checking Current Data for Drift

Once a monitor is set up, you can check new data for drift:

```python
from app.services.monitoring.data_drift import load_drift_monitor
from app.data.production_data import get_recent_predictions

# Load existing monitor
monitor = load_drift_monitor("ad_score_model_v1_monitor")

# Get recent production data
recent_data = get_recent_predictions(
    model_id="ad_score_model_v1",
    start_time="2023-10-01T00:00:00Z",
    end_time="2023-10-02T00:00:00Z"
)

# Check for drift
drift_result = monitor.detect_drift(recent_data)

# Print summary
if drift_result["drift_detected"]:
    print(f"Drift detected! Overall score: {drift_result['overall_drift_score']:.2f}")
    print(f"Drifted features: {drift_result['drifted_features_count']} out of {len(drift_result['feature_results'])}")
    
    # Print details for drifted features
    for feature, result in drift_result["feature_results"].items():
        if result["drift_detected"]:
            print(f"- {feature}: Drift score = {result['drift_score']:.2f}")
            
            # Print test details
            for method, method_result in result["method_results"].items():
                print(f"  - {method}: {method_result['statistic']:.4f} (threshold: {method_result['threshold']:.4f})")
else:
    print("No significant drift detected.")
```

### Scheduled Monitoring Job

The following example shows how to set up a scheduled monitoring job:

```python
from app.services.monitoring.data_drift import DataDriftMonitor
from app.services.monitoring.scheduler import MonitoringScheduler
from app.services.monitoring.notification import NotificationConfig
import datetime

# Create scheduler
scheduler = MonitoringScheduler()

# Configure notifications
notifications = NotificationConfig(
    channels=["slack", "email"],
    slack_channel="#ml-monitoring",
    email_recipients=["ml-team@example.com", "product-team@example.com"],
    min_severity="warning"
)

# Schedule daily drift monitoring for multiple models
scheduler.schedule_drift_monitoring(
    model_ids=["ad_score_model_v1", "account_health_model_v2"],
    schedule={
        "frequency": "daily",
        "time": "01:00",  # 1 AM UTC
        "timezone": "UTC",
    },
    data_window={
        "duration": datetime.timedelta(days=1),
        "end_offset": datetime.timedelta(hours=1),  # 1 hour before scheduled time
    },
    notification_config=notifications,
    sampling_config={
        "method": "stratified",
        "strata_column": "country",
        "max_samples": 10000
    }
)

# Schedule weekly full analysis with detailed reports
scheduler.schedule_drift_monitoring(
    model_ids=["ad_score_model_v1", "account_health_model_v2"],
    schedule={
        "frequency": "weekly",
        "day": "Monday",
        "time": "02:00",  # 2 AM UTC
        "timezone": "UTC",
    },
    data_window={
        "duration": datetime.timedelta(days=7),
        "end_offset": datetime.timedelta(hours=2),
    },
    notification_config=notifications,
    report_config={
        "generate_report": True,
        "include_visualizations": True,
        "include_feature_details": True,
        "distribution_plots": True,
        "store_report": True
    }
)

# Activate schedules
scheduler.activate()
```

### Drift Investigation Workflow

When drift is detected, the following workflow can help investigate its causes:

```python
from app.services.monitoring.investigation import DriftInvestigator
from app.services.monitoring.data_drift import load_drift_result
from app.data.feature_store import get_feature_history

# Load drift detection result
drift_result = load_drift_result("drift_result_20231002_ad_score_model_v1")

# Create investigator
investigator = DriftInvestigator(drift_result)

# Get drifted features
drifted_features = investigator.get_drifted_features()
print(f"Investigating {len(drifted_features)} drifted features")

# Check for temporal patterns
temporal_analysis = investigator.analyze_temporal_patterns(
    features=drifted_features,
    lookback_days=30,
    interval="daily"
)

# Print temporal findings
for feature, pattern in temporal_analysis.items():
    print(f"{feature} temporal pattern: {pattern['pattern_type']}")
    if pattern['pattern_type'] == 'sudden_shift':
        print(f"  Shift detected at {pattern['shift_date']}")
        print(f"  Magnitude: {pattern['shift_magnitude']:.2f}")
    elif pattern['pattern_type'] == 'gradual_trend':
        print(f"  Trend direction: {pattern['trend_direction']}")
        print(f"  Slope: {pattern['trend_slope']:.4f} per day")

# Check for segment-specific drift
segment_analysis = investigator.analyze_segments(
    segmentation_features=["country", "platform", "industry"],
    min_segment_size=100
)

# Print segment findings
print("\nSegment-specific drift:")
for segment, result in segment_analysis.items():
    if result["drift_score"] > 0.5:  # Focus on high-drift segments
        print(f"- {segment}: Drift score = {result['drift_score']:.2f}")
        print(f"  Affected features: {', '.join(result['drifted_features'][:3])}")
        if len(result['drifted_features']) > 3:
            print(f"  and {len(result['drifted_features']) - 3} more")

# Generate visualization report
report_path = investigator.generate_report(
    output_format="html",
    include_recommendations=True
)
print(f"Investigation report generated at: {report_path}")
```

### Multivariate Drift Analysis

This example demonstrates advanced multivariate drift analysis:

```python
from app.services.monitoring.data_drift import MultivariateDataDriftAnalyzer
from app.models.ml.loader import load_model_data
import numpy as np

# Load data
reference_data = load_model_data("ad_score_model_v1", "training")
current_data = get_recent_predictions("ad_score_model_v1", limit=5000)

# Select features for multivariate analysis (related features)
feature_groups = [
    ["impression_count", "click_count", "conversion_count", "ctr", "cvr"],
    ["account_age_days", "campaign_count", "ad_group_count"],
    ["bid_amount", "daily_budget", "total_spend"]
]

# Create analyzer
analyzer = MultivariateDataDriftAnalyzer()

# Analyze each feature group
for i, feature_group in enumerate(feature_groups):
    print(f"\nAnalyzing feature group {i+1}: {', '.join(feature_group)}")
    
    # Extract feature data
    ref_features = reference_data[feature_group].values
    current_features = current_data[feature_group].values
    
    # Run multivariate analysis
    result = analyzer.analyze_multivariate_drift(
        reference_data=ref_features,
        current_data=current_features,
        methods=["mmd", "energy", "classifier"],
        feature_names=feature_group
    )
    
    # Print results
    print(f"Multivariate drift detected: {result['drift_detected']}")
    print(f"Overall multivariate drift score: {result['drift_score']:.4f}")
    
    # Print method-specific results
    for method, method_result in result["method_results"].items():
        print(f"- {method}: statistic={method_result['statistic']:.4f}, "
              f"p_value={method_result.get('p_value', 'N/A')}, "
              f"drift={method_result['drift_detected']}")
    
    # Print correlation analysis
    if "correlation_drift" in result:
        print("\nCorrelation changes:")
        for pair, change in result["correlation_drift"]["top_changed_pairs"].items():
            print(f"- {pair[0]} vs {pair[1]}: {change['before']:.2f} → {change['after']:.2f} "
                  f"(Δ={change['difference']:.2f})")

# Generate and display correlation heatmap visualizations
visualization_paths = analyzer.generate_correlation_visualizations(
    reference_data=reference_data,
    current_data=current_data,
    feature_groups=feature_groups,
    output_dir="reports/drift_analysis"
)

print(f"\nGenerated {len(visualization_paths)} visualization files:")
for path in visualization_paths:
    print(f"- {path}")
```

These examples demonstrate the flexibility and power of the data drift monitoring system. From basic monitoring setup to advanced multivariate analysis, the system provides comprehensive tools for detecting, analyzing, and responding to data drift in production ML models.

---

*This document was completed on March 11, 2025. It complies with WITHIN ML documentation standards.* 