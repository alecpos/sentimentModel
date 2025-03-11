# Data Drift Detection: Technical Implementation

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document provides a technical deep dive into the data drift detection mechanisms implemented in the WITHIN ML system. Data drift occurs when the statistical properties of model input data change over time, which can lead to degraded model performance even when the underlying relationship between features and targets remains stable.

## Mathematical Foundations

### Definition of Data Drift

Formally, data drift can be defined as a change in the probability distribution of the input features $X$ over time. For time points $t_0$ and $t_1$:

$P_{t_0}(X) \neq P_{t_1}(X)$

This change can affect:
- Individual feature distributions
- Joint distributions across multiple features
- Covariate relationships between features

### Statistical Distance Measures

The DriftDetector class employs multiple distance measures to quantify the degree of drift:

#### Kullback-Leibler (KL) Divergence

For discrete distributions $P$ and $Q$:

$D_{KL}(P || Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}$

For continuous distributions, this becomes:

$D_{KL}(P || Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx$

#### Jensen-Shannon Divergence

A symmetrized version of KL divergence:

$JSD(P || Q) = \frac{1}{2}D_{KL}(P || M) + \frac{1}{2}D_{KL}(Q || M)$

where $M = \frac{1}{2}(P + Q)$

#### Wasserstein Distance (Earth Mover's Distance)

For one-dimensional distributions:

$W(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| dx$

where $F_P$ and $F_Q$ are the cumulative distribution functions.

#### Population Stability Index (PSI)

For binned distributions:

$PSI = \sum_{i=1}^{n} (P_i - Q_i) \ln \frac{P_i}{Q_i}$

where $P_i$ and $Q_i$ are the proportions of observations in bin $i$.

## Algorithmic Implementation

### Data Drift Detector Class

The `DriftDetector` class implements the following key methods:

#### Fitting Reference Data

```python
def fit(self, reference_data: pd.DataFrame) -> 'DriftDetector':
    """
    Fit the detector with reference data to establish baseline distributions.
    
    Args:
        reference_data: DataFrame containing reference data
        
    Returns:
        Self for method chaining
    """
    if not isinstance(reference_data, pd.DataFrame):
        raise TypeError("Reference data must be a pandas DataFrame")
        
    # Store original reference data
    self.reference_data = reference_data.copy()
    
    # Extract and store features
    if self.numerical_features is None and self.categorical_features is None:
        # Auto-detect feature types if not specified
        self.numerical_features = reference_data.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_features = reference_data.select_dtypes(
            include=['object', 'category', 'bool']).columns.tolist()
    
    # Compute statistics for numerical features
    self.num_statistics = {}
    for feature in self.numerical_features:
        if feature in reference_data.columns:
            self.num_statistics[feature] = {
                'mean': reference_data[feature].mean(),
                'std': reference_data[feature].std(),
                'min': reference_data[feature].min(),
                'max': reference_data[feature].max(),
                'quantiles': reference_data[feature].quantile([0.25, 0.5, 0.75]).to_dict()
            }
    
    # Compute value counts for categorical features
    self.cat_statistics = {}
    for feature in self.categorical_features:
        if feature in reference_data.columns:
            # Store value frequencies
            value_counts = reference_data[feature].value_counts(normalize=True)
            self.cat_statistics[feature] = value_counts.to_dict()
    
    # Initialize KDE for continuous features if specified
    if self.use_kde:
        self.kde_models = {}
        for feature in self.numerical_features:
            if feature in reference_data.columns:
                # Drop NaN values for KDE fitting
                feature_data = reference_data[feature].dropna().values.reshape(-1, 1)
                if len(feature_data) > 0:
                    self.kde_models[feature] = KernelDensity(
                        kernel='gaussian', 
                        bandwidth=self._estimate_bandwidth(feature_data)
                    ).fit(feature_data)
    
    self.is_fitted = True
    self.fit_timestamp = datetime.now()
    
    return self
```

#### Detecting Drift

```python
def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect drift between reference data and current data.
    
    Args:
        current_data: DataFrame containing current data to check for drift
        
    Returns:
        Dictionary with drift detection results
    """
    if not self.is_fitted:
        raise ValueError("Detector has not been fitted with reference data")
        
    if not isinstance(current_data, pd.DataFrame):
        raise TypeError("Current data must be a pandas DataFrame")
        
    # Initialize results
    results = {
        "drift_detected": False,
        "drift_score": 0.0,
        "feature_drifts": {},
        "timestamp": datetime.now()
    }
    
    # Check numerical features
    num_drift_scores = []
    for feature in self.numerical_features:
        if feature in current_data.columns and feature in self.num_statistics:
            # Compute statistical distances
            feature_drift = self._compute_numerical_drift(feature, current_data)
            results["feature_drifts"][feature] = feature_drift
            
            # Track feature drift score
            drift_score = feature_drift.get("drift_score", 0)
            if drift_score > 0:
                num_drift_scores.append(drift_score)
                
            # Mark drift detected if above threshold
            if feature_drift.get("drift_detected", False):
                results["drift_detected"] = True
    
    # Check categorical features
    cat_drift_scores = []
    for feature in self.categorical_features:
        if feature in current_data.columns and feature in self.cat_statistics:
            # Compute statistical distances
            feature_drift = self._compute_categorical_drift(feature, current_data)
            results["feature_drifts"][feature] = feature_drift
            
            # Track feature drift score
            drift_score = feature_drift.get("drift_score", 0)
            if drift_score > 0:
                cat_drift_scores.append(drift_score)
                
            # Mark drift detected if above threshold
            if feature_drift.get("drift_detected", False):
                results["drift_detected"] = True
    
    # Compute overall drift score (weighted average of all feature scores)
    if num_drift_scores or cat_drift_scores:
        results["drift_score"] = np.mean(num_drift_scores + cat_drift_scores)
    
    return results
```

#### Feature-Level Drift Computation

```python
def _compute_numerical_drift(self, feature: str, current_data: pd.DataFrame) -> Dict[str, Any]:
    """Compute drift for a numerical feature."""
    ref_stats = self.num_statistics[feature]
    
    # Check if feature exists in current data
    if feature not in current_data.columns:
        return {"error": f"Feature {feature} not found in current data"}
    
    # Get current feature data
    current_feature = current_data[feature].dropna()
    
    # Calculate current statistics
    current_stats = {
        'mean': current_feature.mean(),
        'std': current_feature.std(),
        'min': current_feature.min(),
        'max': current_feature.max(),
        'quantiles': current_feature.quantile([0.25, 0.5, 0.75]).to_dict()
    }
    
    # Initialize result dictionary
    result = {
        "feature_type": "numerical",
        "statistics": {
            "reference": ref_stats,
            "current": current_stats,
            "diff": {
                "mean": abs(current_stats['mean'] - ref_stats['mean']),
                "std": abs(current_stats['std'] - ref_stats['std']),
                "min": abs(current_stats['min'] - ref_stats['min']),
                "max": abs(current_stats['max'] - ref_stats['max'])
            }
        },
        "drift_detected": False,
        "drift_score": 0.0
    }
    
    # Use KS test to determine if distributions are different
    ks_statistic, p_value = stats.ks_2samp(
        self.reference_data[feature].dropna(), 
        current_feature
    )
    
    result["statistical_tests"] = {
        "ks_test": {
            "statistic": float(ks_statistic),
            "p_value": float(p_value),
            "significant": p_value < self.significance_level
        }
    }
    
    # If KDE models are available, use Jensen-Shannon divergence
    if self.use_kde and feature in self.kde_models:
        js_divergence = self._compute_js_divergence(feature, current_feature)
        result["statistical_tests"]["js_divergence"] = js_divergence
        
        # Add to drift score calculation
        if js_divergence > 0:
            result["drift_score"] += js_divergence
    
    # Calculate PSI if requested
    if self.use_psi:
        psi_value = self._compute_psi(feature, current_feature)
        result["statistical_tests"]["psi"] = psi_value
        
        # Add to drift score calculation
        if psi_value > 0:
            result["drift_score"] += psi_value * 0.5  # Weight PSI less than JS divergence
    
    # Normalize drift score to 0-1 range
    result["drift_score"] = min(1.0, result["drift_score"] / 2.0)
    
    # Determine if drift is significant
    result["drift_detected"] = (
        result["statistical_tests"]["ks_test"]["significant"] or 
        result["drift_score"] > self.drift_threshold
    )
    
    return result
```

#### Categorical Feature Drift Analysis

```python
def _compute_categorical_drift(self, feature: str, current_data: pd.DataFrame) -> Dict[str, Any]:
    """Compute drift for a categorical feature."""
    ref_freqs = self.cat_statistics[feature]
    
    # Check if feature exists in current data
    if feature not in current_data.columns:
        return {"error": f"Feature {feature} not found in current data"}
    
    # Get current feature frequencies
    current_freqs = current_data[feature].value_counts(normalize=True).to_dict()
    
    # Find all unique categories in both datasets
    all_categories = set(ref_freqs.keys()) | set(current_freqs.keys())
    
    # Initialize result dictionary
    result = {
        "feature_type": "categorical",
        "statistics": {
            "reference": {cat: ref_freqs.get(cat, 0) for cat in all_categories},
            "current": {cat: current_freqs.get(cat, 0) for cat in all_categories},
            "new_categories": list(set(current_freqs.keys()) - set(ref_freqs.keys())),
            "missing_categories": list(set(ref_freqs.keys()) - set(current_freqs.keys()))
        },
        "drift_detected": False,
        "drift_score": 0.0
    }
    
    # Calculate chi-square test for independence
    chi2_result = self._compute_chi2_test(feature, current_data)
    result["statistical_tests"] = {
        "chi2_test": chi2_result
    }
    
    # Calculate JS Divergence for categorical distributions
    js_divergence = self._compute_categorical_js_divergence(ref_freqs, current_freqs)
    result["statistical_tests"]["js_divergence"] = js_divergence
    
    # Calculate drift score
    result["drift_score"] = js_divergence
    
    # Determine if drift is significant
    result["drift_detected"] = (
        js_divergence > self.drift_threshold or
        chi2_result.get("significant", False) or
        len(result["statistics"]["new_categories"]) > 0
    )
    
    return result
```

#### Statistical Testing Functions

```python
def _compute_js_divergence(self, feature: str, current_feature: pd.Series) -> float:
    """Compute Jensen-Shannon divergence for numerical feature."""
    # Get reference KDE model
    kde_model = self.kde_models[feature]
    
    # Prepare data for evaluation
    x_ref = self.reference_data[feature].dropna().values.reshape(-1, 1)
    x_curr = current_feature.values.reshape(-1, 1)
    
    # Create a common evaluation grid spanning both datasets
    x_min = min(x_ref.min(), x_curr.min())
    x_max = max(x_ref.max(), x_curr.max())
    x_grid = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    
    # Evaluate KDE of reference data on grid
    ref_log_density = kde_model.score_samples(x_grid)
    ref_density = np.exp(ref_log_density)
    ref_density /= ref_density.sum()  # Normalize
    
    # Fit KDE on current data and evaluate on same grid
    if len(x_curr) > 10:  # Ensure enough data points for KDE
        kde_curr = KernelDensity(
            kernel='gaussian', 
            bandwidth=self._estimate_bandwidth(x_curr)
        ).fit(x_curr)
        
        curr_log_density = kde_curr.score_samples(x_grid)
        curr_density = np.exp(curr_log_density)
        curr_density /= curr_density.sum()  # Normalize
        
        # Compute JS divergence
        m_density = 0.5 * (ref_density + curr_density)
        js_div = 0.5 * (
            np.sum(ref_density * np.log(ref_density / m_density)) +
            np.sum(curr_density * np.log(curr_density / m_density))
        )
        
        return float(js_div)
    else:
        return 0.0  # Not enough data points
```

#### Data Quality Assessment

```python
def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality issues that may indicate drift.
    
    Args:
        data: DataFrame to check for quality issues
        
    Returns:
        Dictionary with quality issue information
    """
    if not self.is_fitted:
        raise ValueError("Detector has not been fitted with reference data")
        
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    # Initialize results
    results = {
        "quality_issues_detected": False,
        "missing_values": {},
        "outliers": {},
        "out_of_range_values": {},
        "new_categories": {}
    }
    
    # Check for missing values
    missing_pcts = data.isnull().mean()
    for col, pct in missing_pcts.items():
        if pct > 0:
            results["missing_values"][col] = pct
            if pct > self.missing_threshold:
                results["quality_issues_detected"] = True
    
    # Check for outliers in numerical features
    for feature in self.numerical_features:
        if feature in data.columns and feature in self.num_statistics:
            # Use Z-score method to identify outliers
            mean = self.num_statistics[feature]['mean']
            std = self.num_statistics[feature]['std']
            
            if std > 0:
                z_scores = np.abs((data[feature] - mean) / std)
                outliers = data[z_scores > 3].index.tolist()
                
                if outliers:
                    results["outliers"][feature] = {
                        "count": len(outliers),
                        "percentage": len(outliers) / len(data),
                        "example_indices": outliers[:5]  # Show first 5 examples
                    }
                    
                    if len(outliers) / len(data) > self.outlier_threshold:
                        results["quality_issues_detected"] = True
            
            # Check for out-of-range values
            ref_min = self.num_statistics[feature]['min']
            ref_max = self.num_statistics[feature]['max']
            
            # Add a small buffer to avoid flagging minor variations
            buffer = (ref_max - ref_min) * 0.05
            lower_bound = ref_min - buffer
            upper_bound = ref_max + buffer
            
            out_of_range = data[(data[feature] < lower_bound) | 
                               (data[feature] > upper_bound)].index.tolist()
            
            if out_of_range:
                results["out_of_range_values"][feature] = {
                    "count": len(out_of_range),
                    "percentage": len(out_of_range) / len(data),
                    "example_indices": out_of_range[:5]  # Show first 5 examples
                }
                
                if len(out_of_range) / len(data) > self.out_of_range_threshold:
                    results["quality_issues_detected"] = True
    
    # Check for new categories in categorical features
    for feature in self.categorical_features:
        if feature in data.columns and feature in self.cat_statistics:
            new_cats = set(data[feature].unique()) - set(self.cat_statistics[feature].keys())
            if new_cats:
                results["new_categories"][feature] = {
                    "count": len(new_cats),
                    "categories": list(new_cats)
                }
                
                # Consider it an issue if new categories represent significant portion
                new_cat_pct = data[data[feature].isin(new_cats)].shape[0] / data.shape[0]
                if new_cat_pct > self.new_category_threshold:
                    results["quality_issues_detected"] = True
    
    return results
```

## Statistical Tests for Data Drift

### Kolmogorov-Smirnov (KS) Test

The KS test compares the empirical cumulative distribution functions (ECDFs) of two samples to determine if they come from the same distribution:

```python
def _apply_ks_test(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
    """Apply Kolmogorov-Smirnov test to detect distributional differences."""
    # Ensure sufficient data points
    if len(reference_data) < 5 or len(current_data) < 5:
        return {
            "error": "Insufficient data for KS test",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False
        }
    
    # Apply KS test
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < self.significance_level
    }
```

### Population Stability Index (PSI)

The PSI measures the stability of a feature distribution over time:

```python
def _compute_psi(self, feature: str, current_feature: pd.Series) -> float:
    """
    Compute Population Stability Index (PSI) for a feature.
    
    PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))
    
    Args:
        feature: Feature name
        current_feature: Current feature values
        
    Returns:
        PSI value (0 = identical, <0.1 = no change, <0.2 = minor change,
                  >0.2 = significant change)
    """
    # Create bins based on reference data quantiles
    if len(self.reference_data[feature].dropna()) < 10:
        return 0.0  # Not enough data
        
    bins = 10
    
    # Calculate bin edges based on reference data
    ref_data = self.reference_data[feature].dropna()
    bin_edges = np.percentile(ref_data, np.linspace(0, 100, bins+1))
    
    # Fix potential issues with bin edges
    if len(np.unique(bin_edges)) < len(bin_edges):
        # If we have duplicate edges, use linear spacing instead
        bin_edges = np.linspace(ref_data.min(), ref_data.max(), bins+1)
    
    # Count observations in each bin
    ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
    current_counts, _ = np.histogram(current_feature.dropna(), bins=bin_edges)
    
    # Convert to percentages
    ref_pcts = ref_counts / ref_counts.sum()
    current_pcts = current_counts / current_counts.sum()
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    ref_pcts = np.array([max(p, epsilon) for p in ref_pcts])
    current_pcts = np.array([max(p, epsilon) for p in current_pcts])
    
    # Calculate PSI
    psi_values = (current_pcts - ref_pcts) * np.log(current_pcts / ref_pcts)
    psi = np.sum(psi_values)
    
    return float(psi)
```

### Chi-Square Test for Categorical Features

The Chi-Square test evaluates whether distributions of categorical variables differ:

```python
def _compute_chi2_test(self, feature: str, current_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Chi-square test for categorical drift detection.
    
    Args:
        feature: Feature name
        current_data: Current data
        
    Returns:
        Dictionary with test results
    """
    # Get reference counts
    ref_counts = self.reference_data[feature].value_counts()
    current_counts = current_data[feature].value_counts()
    
    # Ensure all categories are present in both
    all_categories = list(set(ref_counts.index) | set(current_counts.index))
    
    # Create contingency table
    contingency = np.zeros((2, len(all_categories)))
    
    for i, cat in enumerate(all_categories):
        contingency[0, i] = ref_counts.get(cat, 0)
        contingency[1, i] = current_counts.get(cat, 0)
    
    # Check if we have enough data
    if contingency.sum() < 10 or (contingency < 5).sum() > len(all_categories):
        return {
            "error": "Insufficient data for chi-square test",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False
        }
    
    # Perform chi-square test
    statistic, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "significant": p_value < self.significance_level
    }
```

## Feature Importance for Drift Detection

For multivariate drift detection, the system evaluates feature importance in contribution to drift:

```python
def compute_feature_drift_importance(self, current_data: pd.DataFrame) -> Dict[str, float]:
    """
    Compute importance of each feature in contributing to overall drift.
    
    Args:
        current_data: Current data to analyze
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Detect drift in each feature
    feature_results = {}
    
    # Process numerical features
    for feature in self.numerical_features:
        if feature in current_data.columns and feature in self.num_statistics:
            feature_results[feature] = self._compute_numerical_drift(feature, current_data)
    
    # Process categorical features
    for feature in self.categorical_features:
        if feature in current_data.columns and feature in self.cat_statistics:
            feature_results[feature] = self._compute_categorical_drift(feature, current_data)
    
    # Extract drift scores
    drift_scores = {
        feature: results.get("drift_score", 0.0) 
        for feature, results in feature_results.items()
    }
    
    # Normalize to sum to 1.0
    total_drift = sum(drift_scores.values())
    
    if total_drift > 0:
        return {
            feature: score / total_drift 
            for feature, score in drift_scores.items()
        }
    else:
        # Equal importance if no drift detected
        n_features = len(drift_scores)
        return {
            feature: 1.0 / n_features if n_features > 0 else 0.0
            for feature in drift_scores
        }
```

## Multivariate Drift Detection

The system can also detect multivariate drift, which captures changes in joint distributions:

```python
def detect_multivariate_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect multivariate drift between reference and current data.
    
    Args:
        current_data: Current data to check
        
    Returns:
        Dictionary with multivariate drift results
    """
    if not self.is_fitted:
        raise ValueError("Detector has not been fitted with reference data")
    
    # Select common features between reference data and current data
    common_num_features = [f for f in self.numerical_features 
                          if f in current_data.columns]
    
    # Check if we have enough features for multivariate analysis
    if len(common_num_features) < 2:
        return {
            "error": "Insufficient numerical features for multivariate analysis",
            "drift_detected": False
        }
    
    # Extract feature data
    ref_data = self.reference_data[common_num_features].dropna()
    current_data_filtered = current_data[common_num_features].dropna()
    
    # Check if we have enough data points
    if len(ref_data) < 10 or len(current_data_filtered) < 10:
        return {
            "error": "Insufficient data points for multivariate analysis",
            "drift_detected": False
        }
    
    # Train a domain classifier to distinguish between datasets
    # If datasets are very different, classifier will perform well
    X = np.vstack([
        ref_data.values,
        current_data_filtered.values
    ])
    
    # Create domain labels: 0 for reference, 1 for current
    y = np.concatenate([
        np.zeros(len(ref_data)),
        np.ones(len(current_data_filtered))
    ])
    
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Split into train/test
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train a classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate performance
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    
    # Compute feature importances
    feature_importances = None
    try:
        feature_importances = {
            feature: abs(coef) for feature, coef in zip(
                common_num_features, 
                clf.coef_[0]
            )
        }
        
        # Normalize importances
        total_importance = sum(feature_importances.values())
        if total_importance > 0:
            feature_importances = {
                f: v / total_importance 
                for f, v in feature_importances.items()
            }
    except:
        # Some models might not support feature importance
        pass
    
    # Calculate drift score: How well the classifier can distinguish datasets
    # Scale: 0 (no drift) to 1 (complete drift)
    # Accuracy of 0.5 means no detectable drift (random guessing)
    # Subtract 0.5 and scale to 0-1 range
    drift_score = max(0, (test_accuracy - 0.5) * 2)
    
    return {
        "drift_detected": drift_score > self.drift_threshold,
        "drift_score": drift_score,
        "classifier_accuracy": test_accuracy,
        "feature_importances": feature_importances
    }
```

## Implementation Considerations

### Computational Efficiency

For efficiency in large-scale deployments, the implementation:

1. **Uses approximations**: Histogram-based approximations instead of exact KDE for large datasets
2. **Applies sampling**: Random sampling for large datasets to reduce computation time
3. **Implements caching**: Caches intermediate computations for reuse
4. **Optimizes calculations**: Uses vectorized operations for statistical distances

```python
def _apply_sampling(self, data: pd.DataFrame, max_samples: int = 10000) -> pd.DataFrame:
    """Sample data for more efficient computation with large datasets."""
    if len(data) <= max_samples:
        return data
    
    # Stratified sampling for categorical features if possible
    if self.categorical_features and self.categorical_features[0] in data.columns:
        strat_feature = self.categorical_features[0]
        try:
            # Try stratified sampling
            return data.groupby(strat_feature, group_keys=False).apply(
                lambda x: x.sample(
                    min(
                        max(1, int(max_samples * len(x) / len(data))),
                        len(x)
                    )
                )
            )
        except:
            # Fall back to random sampling
            return data.sample(max_samples)
    else:
        # Random sampling
        return data.sample(max_samples)
```

### Hyperparameter Tuning

Key parameters and their recommended settings:

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|------------------|
| `drift_threshold` | Threshold for drift detection | 0.1 | 0.05-0.15 |
| `significance_level` | p-value threshold for statistical tests | 0.01 | 0.001-0.05 |
| `missing_threshold` | Threshold for missing value alerts | 0.1 | 0.05-0.2 |
| `outlier_threshold` | Threshold for outlier alerts | 0.05 | 0.01-0.1 |
| `use_kde` | Whether to use KDE for continuous distributions | True | - |
| `use_psi` | Whether to use PSI for binned distributions | True | - |

## Integration with Monitoring Services

The DriftDetector integrates with the DriftMonitoringService:

```python
# In DriftMonitoringService

def check_for_drift(
    self, 
    model_id: str, 
    current_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Check if there is drift in the current data compared to reference data.
    
    Args:
        model_id: Identifier for the model to check
        current_data: Current production data to check for drift
        
    Returns:
        Dictionary with drift information
    """
    # Validate model exists
    if model_id not in self.model_configs:
        raise ValueError(f"Model {model_id} not registered")
        
    # Get model config
    config = self.model_configs[model_id]
    
    # Get detector for this model
    detector = self.detectors.get(model_id)
    if detector is None or not detector.is_fitted:
        raise ValueError(f"No fitted detector found for model {model_id}")
    
    # Check for drift
    drift_result = detector.detect_drift(current_data)
    
    # Calculate drift severity
    if drift_result["drift_detected"]:
        if drift_result["drift_score"] > 0.3:
            severity = DriftSeverity.HIGH
        elif drift_result["drift_score"] > 0.2:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
    else:
        severity = DriftSeverity.NONE
    
    # Add severity to result
    drift_result["severity"] = severity
    
    # Store drift result
    self._store_drift_result(model_id, drift_result)
    
    # Return drift information
    return {
        "model_id": model_id,
        "drift_detected": drift_result["drift_detected"],
        "drift_score": drift_result["drift_score"],
        "severity": severity,
        "timestamp": datetime.now(),
        "details": drift_result
    }
```

## Best Practices and Usage Guidelines

### Recommended Monitoring Frequency

| Data Volume | Update Frequency | Recommended Checks |
|------------|------------------|------------------|
| High (>1M rows/day) | Hourly | Statistical tests on samples |
| Medium (100k-1M rows/day) | Daily | Full drift detection |
| Low (<100k rows/day) | Weekly | Full drift + multivariate analysis |

### Drift Severity Levels

The DriftMonitoringService defines drift severity levels:

```python
class DriftSeverity(Enum):
    NONE = "none"             # No drift detected
    LOW = "low"               # Minor drift, monitor closely
    MEDIUM = "medium"         # Significant drift, investigate
    HIGH = "high"             # Critical drift, immediate action required
```

### Response Actions by Severity

1. **NONE**: 
   - Continue regular monitoring
   - Log drift measurements

2. **LOW**:
   - Increase monitoring frequency
   - Flag potentially drifting features

3. **MEDIUM**:
   - Trigger investigation
   - Prepare for model retraining
   - Adjust prediction confidence

4. **HIGH**:
   - Immediate alert to stakeholders
   - Consider model rollback
   - Initiate emergency retraining
   - Apply drift compensation

## Visualization Strategies

The implementation includes methods to visualize drift:

```python
def generate_drift_visualization(
    self, 
    feature: str, 
    current_data: pd.DataFrame,
    plot_type: str = 'distribution'
) -> Dict[str, Any]:
    """
    Generate visualization data for a feature's drift.
    
    Args:
        feature: Feature to visualize
        current_data: Current data
        plot_type: Type of plot ('distribution', 'qq', 'time_series')
        
    Returns:
        Dictionary with visualization data
    """
    if not self.is_fitted:
        raise ValueError("Detector has not been fitted with reference data")
    
    if feature not in self.reference_data.columns or feature not in current_data.columns:
        raise ValueError(f"Feature {feature} not found in data")
    
    # Handle different plot types
    if plot_type == 'distribution':
        if feature in self.numerical_features:
            return self._generate_numerical_distribution_plot(feature, current_data)
        elif feature in self.categorical_features:
            return self._generate_categorical_distribution_plot(feature, current_data)
    elif plot_type == 'qq':
        if feature in self.numerical_features:
            return self._generate_qq_plot(feature, current_data)
        else:
            raise ValueError("QQ plots only available for numerical features")
    elif plot_type == 'time_series':
        # Requires timestamp column
        if 'timestamp' not in current_data.columns:
            raise ValueError("Time series plots require a 'timestamp' column")
        return self._generate_time_series_plot(feature, current_data)
    
    raise ValueError(f"Invalid plot type: {plot_type}")
```

## References

1. Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. NeurIPS.
2. Webb, G. I., Hyde, R., Cao, H., Nguyen, H. L., & Petitjean, F. (2016). Characterizing concept drift. Data Mining and Knowledge Discovery, 30(4), 964-994.
3. Baena-García, M., del Campo-Ávila, J., Fidalgo, R., & Bifet, A. (2006). Early drift detection method. ECML PKDD Workshop on Knowledge Discovery from Data Streams.
4. Goldenberg, I., & Webb, G. I. (2019). Survey of distance measures for quantifying concept drift and shift in numeric data. Knowledge and Information Systems, 60(2), 591-615.
5. Dasu, T., Krishnan, S., Venkatasubramanian, S., & Yi, K. (2006). An information-theoretic approach to detecting changes in multi-dimensional data streams. Proc. Symp. on the Interface of Statistics, Computing Science, and Applications. 