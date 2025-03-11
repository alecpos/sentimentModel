# Prediction Drift Detection: Technical Implementation

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document provides a technical deep dive into the prediction drift detection mechanisms implemented in the WITHIN ML system. Prediction drift occurs when the distribution of model outputs changes over time, which may indicate underlying issues with model performance even when input data appears stable.

## Mathematical Foundations

### Definition of Prediction Drift

Prediction drift can be formally defined as a change in the probability distribution of the model outputs $\hat{Y}$ over time. For time points $t_0$ and $t_1$:

$P_{t_0}(\hat{Y}) \neq P_{t_1}(\hat{Y})$

This change can manifest in several ways:
- Shifts in central tendency (mean, median) of predictions
- Changes in prediction variance
- Alteration of prediction quantiles
- Emergence of multimodality in predictions

### Statistical Distance Measures

The PredictionDriftDetector employs multiple statistical measures to quantify prediction drift:

#### Kullback-Leibler (KL) Divergence

For discrete distributions $P$ and $Q$:

$D_{KL}(P || Q) = \sum_{y \in Y} P(y) \log \frac{P(y)}{Q(y)}$

For continuous distributions, this becomes:

$D_{KL}(P || Q) = \int_{-\infty}^{\infty} p(y) \log \frac{p(y)}{q(y)} dy$

#### Wasserstein Distance

The Wasserstein distance (Earth Mover's Distance) measures the minimum "cost" of transforming one distribution into another:

$W(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \mathbb{E}_{(x,y) \sim \gamma} [||x - y||]$

where $\Gamma(P, Q)$ is the set of all joint distributions with marginals $P$ and $Q$.

#### Confidence Distribution Shift

For classification models, we monitor shifts in the confidence distribution:

$D_{conf} = \frac{1}{n} \sum_{i=1}^{n} |conf_{ref}(i) - conf_{current}(i)|$

where $conf$ represents the confidence score for each prediction.

## Algorithmic Implementation

### Prediction Drift Detector Class

The `PredictionDriftDetector` class implements the following key methods:

#### Initialization and Configuration

```python
def __init__(
    self,
    drift_threshold: float = 0.1,
    window_size: int = 1000,
    significance_level: float = 0.01,
    prediction_type: str = "continuous",  # or "categorical", "probability"
    metrics: Optional[List[str]] = None,
    use_kde: bool = True
) -> None:
    """
    Initialize a prediction drift detector.
    
    Args:
        drift_threshold: Threshold for determining significant drift
        window_size: Size of sliding window for recent predictions
        significance_level: Significance level for statistical tests
        prediction_type: Type of predictions to monitor
        metrics: List of metrics to use for drift detection
        use_kde: Whether to use KDE for continuous distributions
    """
    self.drift_threshold = drift_threshold
    self.window_size = window_size
    self.significance_level = significance_level
    self.prediction_type = prediction_type
    self.use_kde = use_kde
    
    # Set default metrics based on prediction type
    if metrics is None:
        if prediction_type == "continuous":
            self.metrics = ["mean", "std", "quantile_25", "quantile_50", "quantile_75"]
        elif prediction_type == "categorical":
            self.metrics = ["class_distribution", "entropy"]
        elif prediction_type == "probability":
            self.metrics = ["mean_confidence", "confidence_distribution"]
    else:
        self.metrics = metrics
    
    # Initialize storage for reference and recent predictions
    self.reference_predictions = None
    self.recent_predictions = []
    
    # Initialize metric storage
    self.reference_metrics = {}
    self.is_fitted = False
    
    # Initialize drift history
    self.drift_history = []
```

#### Fitting Reference Data

```python
def fit(self, reference_predictions: np.ndarray) -> 'PredictionDriftDetector':
    """
    Fit the detector with reference prediction data.
    
    Args:
        reference_predictions: Array of reference predictions
        
    Returns:
        Self for method chaining
    """
    # Convert to numpy array if needed
    if isinstance(reference_predictions, list):
        reference_predictions = np.array(reference_predictions)
    elif isinstance(reference_predictions, pd.Series):
        reference_predictions = reference_predictions.to_numpy()
    
    # Store reference predictions
    self.reference_predictions = reference_predictions
    
    # Calculate reference metrics
    self.reference_metrics = self._calculate_metrics(reference_predictions)
    
    # Initialize KDE for continuous predictions
    if self.prediction_type == "continuous" and self.use_kde:
        # Reshape for scikit-learn API
        data = reference_predictions.reshape(-1, 1)
        
        # Estimate bandwidth using Scott's rule
        bandwidth = 1.06 * np.std(data) * (len(data) ** (-1/5))
        
        # Fit KDE model
        self.reference_kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    
    # For categorical data, calculate reference class distribution
    if self.prediction_type == "categorical":
        self.reference_class_distribution = {}
        unique_classes, counts = np.unique(reference_predictions, return_counts=True)
        total = counts.sum()
        
        for cls, count in zip(unique_classes, counts):
            self.reference_class_distribution[cls] = count / total
    
    # For probability distributions, calculate reference confidence statistics
    if self.prediction_type == "probability":
        self.reference_confidence_bins = np.histogram(
            reference_predictions, 
            bins=10, 
            range=(0, 1)
        )[0] / len(reference_predictions)
    
    self.is_fitted = True
    return self
```

#### Calculating Metrics

```python
def _calculate_metrics(self, predictions: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for a set of predictions.
    
    Args:
        predictions: Array of predictions to analyze
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    # Handle different prediction types
    if self.prediction_type == "continuous":
        # Basic statistics
        metrics["mean"] = float(np.mean(predictions))
        metrics["std"] = float(np.std(predictions))
        metrics["min"] = float(np.min(predictions))
        metrics["max"] = float(np.max(predictions))
        
        # Quantiles
        metrics["quantile_25"] = float(np.percentile(predictions, 25))
        metrics["quantile_50"] = float(np.percentile(predictions, 50))
        metrics["quantile_75"] = float(np.percentile(predictions, 75))
        
    elif self.prediction_type == "categorical":
        # Class distribution entropy
        unique_classes, counts = np.unique(predictions, return_counts=True)
        class_distribution = counts / counts.sum()
        
        # Shannon entropy
        entropy = -np.sum(class_distribution * np.log2(class_distribution + 1e-10))
        metrics["entropy"] = float(entropy)
        
        # Number of unique classes
        metrics["n_classes"] = len(unique_classes)
        
        # Percentage of most common class
        metrics["most_common_class_pct"] = float(class_distribution.max())
        
    elif self.prediction_type == "probability":
        # Mean confidence
        metrics["mean_confidence"] = float(np.mean(predictions))
        
        # Confidence distribution
        low_conf = np.mean(predictions < 0.3)
        medium_conf = np.mean((predictions >= 0.3) & (predictions < 0.7))
        high_conf = np.mean(predictions >= 0.7)
        
        metrics["low_confidence_pct"] = float(low_conf)
        metrics["medium_confidence_pct"] = float(medium_conf)
        metrics["high_confidence_pct"] = float(high_conf)
        
        # Calibration metric (if labels are available - placeholder)
        # metrics["calibration_error"] = float(...)
    
    return metrics
```

#### Detecting Drift in New Predictions

```python
def detect_drift(self, new_predictions: np.ndarray) -> Dict[str, Any]:
    """
    Detect drift in new predictions compared to reference data.
    
    Args:
        new_predictions: Array of new predictions to check for drift
        
    Returns:
        Dictionary with drift detection results
    """
    if not self.is_fitted:
        raise ValueError("Detector has not been fitted with reference data")
    
    # Convert to numpy array if needed
    if isinstance(new_predictions, list):
        new_predictions = np.array(new_predictions)
    elif isinstance(new_predictions, pd.Series):
        new_predictions = new_predictions.to_numpy()
    
    # Add new predictions to recent history
    self.recent_predictions.extend(new_predictions)
    
    # Keep only the most recent window_size predictions
    if len(self.recent_predictions) > self.window_size:
        self.recent_predictions = self.recent_predictions[-self.window_size:]
    
    # Convert recent predictions to numpy array
    recent_preds = np.array(self.recent_predictions)
    
    # Calculate metrics for recent predictions
    recent_metrics = self._calculate_metrics(recent_preds)
    
    # Detect metric drift
    metric_drift = self._detect_metric_drift(recent_metrics)
    
    # Perform statistical tests for distribution shift
    dist_drift = self._detect_distribution_drift(recent_preds)
    
    # Combine results
    drift_result = {
        "drift_detected": metric_drift["drift_detected"] or dist_drift["drift_detected"],
        "metric_drift": metric_drift,
        "distribution_drift": dist_drift,
        "timestamp": datetime.now()
    }
    
    # Calculate overall drift score
    drift_result["drift_score"] = 0.5 * (
        metric_drift.get("drift_score", 0) + 
        dist_drift.get("drift_score", 0)
    )
    
    # Add drift result to history
    self.drift_history.append(drift_result)
    
    return drift_result
```

#### Detecting Metric Drift

```python
def _detect_metric_drift(self, recent_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Detect drift in prediction metrics.
    
    Args:
        recent_metrics: Dictionary of metrics for recent predictions
        
    Returns:
        Dictionary with metric drift results
    """
    # Initialize result
    result = {
        "drift_detected": False,
        "metrics": {},
        "drift_score": 0.0
    }
    
    # Count significant drifts
    n_significant = 0
    total_rel_change = 0.0
    
    # Check each metric
    for metric in self.metrics:
        if metric in recent_metrics and metric in self.reference_metrics:
            ref_value = self.reference_metrics[metric]
            current_value = recent_metrics[metric]
            
            # Skip if reference value is zero
            if ref_value == 0:
                continue
            
            # Calculate absolute and relative change
            abs_change = current_value - ref_value
            rel_change = abs(abs_change / ref_value)
            
            # Determine if drift is significant
            is_significant = rel_change > self.drift_threshold
            
            # Add to result
            result["metrics"][metric] = {
                "reference_value": ref_value,
                "current_value": current_value,
                "absolute_change": abs_change,
                "relative_change": rel_change,
                "is_significant": is_significant
            }
            
            # Update significance counter
            if is_significant:
                n_significant += 1
                total_rel_change += rel_change
    
    # Mark drift as detected if any metric has significant drift
    result["drift_detected"] = n_significant > 0
    
    # Calculate overall drift score
    if n_significant > 0:
        result["drift_score"] = min(1.0, total_rel_change / n_significant)
    
    return result
```

#### Detecting Distribution Drift

```python
def _detect_distribution_drift(self, recent_predictions: np.ndarray) -> Dict[str, Any]:
    """
    Detect drift in prediction distributions.
    
    Args:
        recent_predictions: Array of recent predictions
        
    Returns:
        Dictionary with distribution drift results
    """
    # Initialize result
    result = {
        "drift_detected": False,
        "statistical_tests": {},
        "drift_score": 0.0
    }
    
    # Apply Kolmogorov-Smirnov test for continuous predictions
    if self.prediction_type == "continuous":
        # Apply KS test
        ks_statistic, p_value = stats.ks_2samp(
            self.reference_predictions, 
            recent_predictions
        )
        
        # Record test results
        result["statistical_tests"]["ks_test"] = {
            "statistic": float(ks_statistic),
            "p_value": float(p_value),
            "significant": p_value < self.significance_level
        }
        
        # Calculate JS divergence using KDE if available
        if self.use_kde:
            js_div = self._compute_js_divergence(recent_predictions)
            result["statistical_tests"]["js_divergence"] = float(js_div)
            
            # JS divergence contributes to drift score
            result["drift_score"] += min(1.0, js_div * 5)  # Scale up for sensitivity
    
    # Chi-square test for categorical predictions
    elif self.prediction_type == "categorical":
        # Apply Chi-square test
        chi2_test = self._compute_chi2_test(recent_predictions)
        result["statistical_tests"]["chi2_test"] = chi2_test
        
        # Chi-square result contributes to drift score
        if chi2_test.get("significant", False):
            result["drift_score"] += 0.5
        
        # Also compute JS divergence for class distributions
        js_div = self._compute_categorical_js_divergence(recent_predictions)
        result["statistical_tests"]["js_divergence"] = float(js_div)
        
        # JS divergence contributes to drift score
        result["drift_score"] += min(0.5, js_div * 5)  # Scale up for sensitivity
    
    # For probability distributions, compare confidence histograms
    elif self.prediction_type == "probability":
        # Calculate confidence distribution
        current_conf_bins = np.histogram(
            recent_predictions, 
            bins=10, 
            range=(0, 1)
        )[0] / len(recent_predictions)
        
        # Calculate EMD (Earth Mover's Distance)
        emd = self._compute_emd(
            self.reference_confidence_bins,
            current_conf_bins
        )
        
        result["statistical_tests"]["earth_movers_distance"] = float(emd)
        
        # EMD contributes to drift score
        result["drift_score"] += min(1.0, emd * 10)  # Scale up for sensitivity
    
    # Normalize drift score
    result["drift_score"] = min(1.0, result["drift_score"])
    
    # Mark drift as detected if score exceeds threshold
    result["drift_detected"] = result["drift_score"] > self.drift_threshold
    
    return result
```

#### Statistical Tests for Distribution Comparison

```python
def _compute_js_divergence(self, recent_predictions: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence for continuous predictions.
    
    Args:
        recent_predictions: Array of recent predictions
        
    Returns:
        JS divergence value
    """
    # Reshape for scikit-learn API
    recent_data = recent_predictions.reshape(-1, 1)
    
    # Create evaluation grid
    min_val = min(self.reference_predictions.min(), recent_predictions.min())
    max_val = max(self.reference_predictions.max(), recent_predictions.max())
    grid = np.linspace(min_val, max_val, 1000).reshape(-1, 1)
    
    # Fit KDE on recent data
    bandwidth = 1.06 * np.std(recent_data) * (len(recent_data) ** (-1/5))
    recent_kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(recent_data)
    
    # Score samples on grid
    ref_log_density = self.reference_kde.score_samples(grid)
    recent_log_density = recent_kde.score_samples(grid)
    
    # Convert to probabilities
    ref_density = np.exp(ref_log_density)
    recent_density = np.exp(recent_log_density)
    
    # Normalize
    ref_density /= np.sum(ref_density)
    recent_density /= np.sum(recent_density)
    
    # Calculate JS divergence
    m_density = 0.5 * (ref_density + recent_density)
    
    # Calculate KL divergences
    kl_ref = np.sum(ref_density * np.log2(ref_density / m_density + 1e-10))
    kl_recent = np.sum(recent_density * np.log2(recent_density / m_density + 1e-10))
    
    # JS divergence
    js_div = 0.5 * (kl_ref + kl_recent)
    
    return float(js_div)
```

```python
def _compute_categorical_js_divergence(self, recent_predictions: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence for categorical predictions.
    
    Args:
        recent_predictions: Array of recent categorical predictions
        
    Returns:
        JS divergence value
    """
    # Calculate recent class distribution
    unique_classes, counts = np.unique(recent_predictions, return_counts=True)
    recent_dist = {}
    total = counts.sum()
    
    for cls, count in zip(unique_classes, counts):
        recent_dist[cls] = count / total
    
    # Get all unique classes
    all_classes = set(self.reference_class_distribution.keys()) | set(recent_dist.keys())
    
    # Create proper probability vectors with zeros for missing classes
    p = np.array([self.reference_class_distribution.get(cls, 0) for cls in all_classes])
    q = np.array([recent_dist.get(cls, 0) for cls in all_classes])
    
    # Ensure proper normalization
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate JS divergence
    m = 0.5 * (p + q)
    
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-10
    
    # Calculate KL divergences
    kl_p_m = np.sum(p * np.log2(p / (m + epsilon) + epsilon))
    kl_q_m = np.sum(q * np.log2(q / (m + epsilon) + epsilon))
    
    # JS divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)
    
    return float(js_div)
```

```python
def _compute_emd(self, reference_hist: np.ndarray, current_hist: np.ndarray) -> float:
    """
    Compute Earth Mover's Distance (1D Wasserstein) between histograms.
    
    Args:
        reference_hist: Reference histogram
        current_hist: Current histogram
        
    Returns:
        EMD value
    """
    # Ensure proper normalization
    ref_hist = reference_hist / np.sum(reference_hist)
    curr_hist = current_hist / np.sum(current_hist)
    
    # Compute CDF
    ref_cdf = np.cumsum(ref_hist)
    curr_cdf = np.cumsum(curr_hist)
    
    # Return Wasserstein-1 distance
    return float(np.abs(ref_cdf - curr_cdf).sum() / len(ref_cdf))
```

#### Advanced Detection: Time-Series Analysis

For comprehensive monitoring, the system implements time-series analysis of prediction drift:

```python
def analyze_prediction_trend(self, window_size: int = 10) -> Dict[str, Any]:
    """
    Analyze trends in prediction drift over time.
    
    Args:
        window_size: Size of window for moving averages
        
    Returns:
        Dictionary with trend analysis results
    """
    if len(self.drift_history) < window_size * 2:
        return {
            "error": "Insufficient history for trend analysis",
            "trend_detected": False
        }
    
    # Extract drift scores over time
    drift_scores = [entry["drift_score"] for entry in self.drift_history]
    
    # Calculate moving average
    ma = np.convolve(drift_scores, np.ones(window_size)/window_size, mode='valid')
    
    # Check for trend using simple linear regression
    y = ma[-window_size:]
    x = np.arange(len(y))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate metrics
    result = {
        "slope": float(slope),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "trend_detected": p_value < 0.05 and abs(slope) > 0.01,
        "trend_direction": "increasing" if slope > 0 else "decreasing",
        "recent_average": float(np.mean(drift_scores[-window_size:])),
        "overall_average": float(np.mean(drift_scores))
    }
    
    # Detect pattern type (sudden, gradual, seasonal)
    result["pattern_type"] = self._analyze_pattern_type(drift_scores)
    
    return result
```

## Implementation for Different Prediction Types

### Continuous Predictions

For regression models and continuous prediction outputs, the detector focuses on:

1. **Distribution Shifts**:
   - Changes in mean, variance, and quantiles
   - KS test for overall distribution change
   - KDE-based JS divergence for detailed comparison

2. **Anomalous Predictions**:
   - Sudden spikes or drops in prediction values
   - Emergence of bimodality
   - Increased prediction variance

```python
def check_for_bimodality(self, predictions: np.ndarray) -> Dict[str, Any]:
    """Check for bimodality in prediction distribution."""
    # Calculate Hartigan's dip test for unimodality
    try:
        from diptest import diptest
        dip, p_value = diptest(predictions)
        
        # Interpret result
        return {
            "bimodality_detected": p_value < 0.05,
            "dip_statistic": float(dip),
            "p_value": float(p_value)
        }
    except ImportError:
        # Fall back to simpler approach if diptest not available
        kde = stats.gaussian_kde(predictions)
        x = np.linspace(predictions.min(), predictions.max(), 1000)
        y = kde(x)
        
        # Find peaks in density
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(y)
        
        return {
            "bimodality_detected": len(peaks) > 1,
            "num_modes": len(peaks)
        }
```

### Classification Predictions

For classification models, the detector focuses on:

1. **Class Distribution Changes**:
   - Shifts in class frequencies
   - Changes in entropy of class distribution
   - Appearance of previously rare classes

2. **Decision Boundary Shifts**:
   - Changes in confidence for boundary cases
   - Detection of oscillating predictions

```python
def analyze_classification_stability(
    self, 
    recent_predictions: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Analyze stability of classification predictions over time.
    
    Args:
        recent_predictions: Recent predictions to analyze
        threshold: Threshold for significant change
        
    Returns:
        Dictionary with stability analysis
    """
    # Get unique classes
    classes = np.unique(recent_predictions)
    
    # Calculate reference distribution
    ref_dist = {}
    for cls in classes:
        ref_dist[cls] = np.mean(self.reference_predictions == cls)
    
    # Calculate current distribution
    current_dist = {}
    for cls in classes:
        current_dist[cls] = np.mean(recent_predictions == cls)
    
    # Calculate absolute changes
    changes = {}
    for cls in classes:
        ref_val = ref_dist.get(cls, 0)
        curr_val = current_dist.get(cls, 0)
        changes[cls] = abs(curr_val - ref_val)
    
    # Identify unstable classes
    unstable_classes = [cls for cls, change in changes.items() if change > threshold]
    
    return {
        "stability": {cls: 1 - change for cls, change in changes.items()},
        "unstable_classes": unstable_classes,
        "overall_stability": 1 - np.mean(list(changes.values())),
        "is_stable": len(unstable_classes) == 0
    }
```

### Probability Predictions

For probabilistic outputs, the detector focuses on:

1. **Confidence Distribution Shifts**:
   - Changes in mean confidence
   - Shifts in high/medium/low confidence ratios
   - Detection of calibration drift

2. **Uncertainty Patterns**:
   - Changes in prediction entropy
   - Trends in confidence scores

```python
def analyze_calibration_drift(
    self, 
    recent_predictions: np.ndarray, 
    recent_targets: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze drift in model calibration.
    
    Args:
        recent_predictions: Recent probability predictions
        recent_targets: Corresponding ground truth values
        
    Returns:
        Dictionary with calibration drift analysis
    """
    # Function requires target values
    if recent_targets is None:
        return {"error": "Target values required for calibration analysis"}
    
    # Calculate current calibration curve
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(recent_predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate fraction of positives in each bin
    bin_sums = np.bincount(bin_indices, weights=recent_targets, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_counts = np.clip(bin_counts, 1, None)  # Avoid division by zero
    fraction_positives = bin_sums / bin_counts
    
    # Calculate bin centers (mean predicted probability in each bin)
    bin_centers = np.bincount(bin_indices, weights=recent_predictions, minlength=n_bins) / bin_counts
    
    # Calculate expected calibration error
    ece = np.sum(np.abs(bin_centers - fraction_positives) * (bin_counts / bin_counts.sum()))
    
    # Compare with reference calibration error (if available)
    if hasattr(self, 'reference_ece'):
        calibration_drift = abs(ece - self.reference_ece)
        is_significant = calibration_drift > self.drift_threshold
    else:
        calibration_drift = 0.0
        is_significant = False
        self.reference_ece = ece
    
    return {
        "current_ece": float(ece),
        "reference_ece": getattr(self, 'reference_ece', None),
        "calibration_drift": float(calibration_drift),
        "is_significant": is_significant,
        "calibration_curve": {
            "bin_centers": bin_centers.tolist(),
            "fraction_positives": fraction_positives.tolist(),
            "bin_counts": bin_counts.tolist()
        }
    }
```

## Integration with Monitoring Services

The PredictionDriftDetector integrates with monitoring services:

```python
# In ProductionMonitoringService

def monitor_prediction_drift(
    self,
    model_id: str,
    predictions: np.ndarray,
    prediction_type: str = "continuous",
    targets: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Monitor predictions for drift.
    
    Args:
        model_id: ID of the model being monitored
        predictions: Recent model predictions
        prediction_type: Type of predictions ('continuous', 'categorical', 'probability')
        targets: Optional ground truth targets for calibration monitoring
        
    Returns:
        Dictionary with monitoring results
    """
    # Check if detector exists for this model
    detector = self.prediction_drift_detectors.get(model_id)
    
    if detector is None:
        # Initialize new detector if none exists
        detector = PredictionDriftDetector(
            prediction_type=prediction_type,
            drift_threshold=self.config.drift_threshold
        )
        
        # Fit with initial data
        detector.fit(predictions)
        
        # Store detector
        self.prediction_drift_detectors[model_id] = detector
        
        # Return initialization result
        return {
            "model_id": model_id,
            "status": "initialized",
            "message": "Prediction drift detector initialized"
        }
    
    # Detect drift
    drift_result = detector.detect_drift(predictions)
    
    # Add calibration analysis if targets are provided for probability predictions
    if prediction_type == "probability" and targets is not None:
        calibration_result = detector.analyze_calibration_drift(predictions, targets)
        drift_result["calibration"] = calibration_result
    
    # Determine severity
    if drift_result["drift_detected"]:
        if drift_result["drift_score"] > 0.3:
            severity = AlertLevel.HIGH
        elif drift_result["drift_score"] > 0.15:
            severity = AlertLevel.MEDIUM
        else:
            severity = AlertLevel.LOW
    else:
        severity = AlertLevel.INFO
    
    # Add metadata
    result = {
        "model_id": model_id,
        "timestamp": datetime.now(),
        "drift_detected": drift_result["drift_detected"],
        "drift_score": drift_result["drift_score"],
        "alert_level": severity,
        "details": drift_result
    }
    
    # Log monitoring result
    self._log_monitoring_result(model_id, "prediction_drift", result)
    
    # Send alert if needed
    if drift_result["drift_detected"]:
        self.send_alert(
            alert_level=severity,
            message=f"Prediction drift detected for model {model_id}",
            details=result
        )
    
    return result
```

## Root Cause Analysis

When prediction drift is detected, the system performs root cause analysis:

```python
def analyze_drift_root_cause(
    self,
    model_id: str,
    input_data: pd.DataFrame,
    predictions: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze root causes of prediction drift.
    
    Args:
        model_id: ID of the model
        input_data: Input data corresponding to predictions
        predictions: Predictions to analyze
        
    Returns:
        Dictionary with root cause analysis
    """
    # Get the drift detector
    detector = self.prediction_drift_detectors.get(model_id)
    if detector is None:
        return {"error": f"No detector found for model {model_id}"}
    
    # Get the data drift detector
    data_detector = self.data_drift_detectors.get(model_id)
    if data_detector is None:
        return {"error": f"No data drift detector found for model {model_id}"}
    
    # Check prediction drift
    pred_drift = detector.detect_drift(predictions)
    
    # Check data drift
    data_drift = data_detector.detect_drift(input_data)
    
    # Analyze correlation between feature drift and prediction drift
    feature_drifts = []
    for feature, info in data_drift["feature_drifts"].items():
        if info.get("drift_detected", False):
            feature_drifts.append({
                "feature": feature,
                "drift_score": info.get("drift_score", 0)
            })
    
    # Sort features by drift score
    feature_drifts.sort(key=lambda x: x["drift_score"], reverse=True)
    
    # Analyze impact of drifted features on predictions
    feature_impact = self._analyze_feature_impact(model_id, input_data, predictions, 
                                                [f["feature"] for f in feature_drifts[:5]])
    
    # Determine if data drift is the primary cause
    data_driven = data_drift["drift_detected"] and data_drift["drift_score"] > 0.1
    
    # Determine if model behavior change is the primary cause
    model_behavior_change = pred_drift["drift_detected"] and not data_driven
    
    # Return combined analysis
    return {
        "prediction_drift": pred_drift["drift_detected"],
        "data_drift": data_drift["drift_detected"],
        "primary_cause": "data_drift" if data_driven else 
                        "model_behavior" if model_behavior_change else "unknown",
        "drifted_features": feature_drifts[:5],  # Top 5 drifted features
        "feature_impact": feature_impact,
        "recommended_action": "retrain" if data_driven else 
                            "investigate" if model_behavior_change else "monitor"
    }
```

## Visualization and Reporting

The system provides methods to visualize prediction drift:

```python
def generate_prediction_drift_visualization(
    self,
    model_id: str,
    plot_type: str = "distribution"
) -> Dict[str, Any]:
    """
    Generate visualization data for prediction drift.
    
    Args:
        model_id: ID of the model to visualize
        plot_type: Type of plot ('distribution', 'time_series', 'calibration')
        
    Returns:
        Dictionary with visualization data
    """
    # Get the detector
    detector = self.prediction_drift_detectors.get(model_id)
    if detector is None:
        return {"error": f"No detector found for model {model_id}"}
    
    # Generate appropriate visualization
    if plot_type == "distribution":
        return self._generate_distribution_comparison(detector)
    elif plot_type == "time_series":
        return self._generate_drift_time_series(detector)
    elif plot_type == "calibration":
        return self._generate_calibration_plot(detector)
    else:
        return {"error": f"Unsupported plot type: {plot_type}"}
```

## Best Practices and Usage Guidelines

### Monitoring Frequency

| Application | Update Frequency | Recommended Settings |
|-------------|------------------|----------------------|
| Critical financial systems | Real-time | Low threshold (0.05), granular alerts |
| Recommendation systems | Daily | Medium threshold (0.1), batch alerts |
| Long-term forecasting | Weekly | Higher threshold (0.15), summary alerts |

### Alert Level Configuration

```python
def configure_prediction_drift_alerts(
    self,
    model_id: str,
    thresholds: Dict[str, float]
) -> None:
    """
    Configure alert thresholds for prediction drift.
    
    Args:
        model_id: ID of the model
        thresholds: Dictionary of threshold values
    """
    # Validate model exists
    if model_id not in self.prediction_drift_detectors:
        raise ValueError(f"No detector found for model {model_id}")
    
    # Update thresholds
    detector = self.prediction_drift_detectors[model_id]
    
    # Set drift threshold
    if "drift_threshold" in thresholds:
        detector.drift_threshold = thresholds["drift_threshold"]
    
    # Set significance level
    if "significance_level" in thresholds:
        detector.significance_level = thresholds["significance_level"]
    
    # Store configuration
    self.drift_alert_config[model_id] = {
        "thresholds": thresholds,
        "updated_at": datetime.now()
    }
```

### Response Actions by Severity

| Severity | Action | Description |
|----------|--------|-------------|
| LOW | Monitor | Continue monitoring, increase frequency |
| MEDIUM | Investigate | Investigate root causes, prepare for retraining |
| HIGH | Mitigate | Implement mitigation strategies, consider model rollback |
| CRITICAL | Replace | Immediately replace or retrain model |

## References

1. Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. ICML.
2. Ovadia, Y., et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS.
3. Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. NeurIPS.
4. Webb, G. I., Hyde, R., Cao, H., Nguyen, H. L., & Petitjean, F. (2016). Characterizing concept drift. Data Mining and Knowledge Discovery, 30(4), 964-994.
5. Sugiyama, M., Lawrence, N. D., & Schwaighofer, A. (2017). Dataset shift in machine learning. The MIT Press. 