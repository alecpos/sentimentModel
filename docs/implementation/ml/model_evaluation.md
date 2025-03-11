# Model Evaluation

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


## Overview

This document details the comprehensive evaluation methodology for machine learning models in the WITHIN Ad Score & Account Health Predictor system. It outlines the metrics, validation approaches, fairness assessments, and monitoring strategies used to ensure model quality.

## Table of Contents

1. [Evaluation Framework](#evaluation-framework)
2. [Offline Evaluation](#offline-evaluation)
3. [Online Evaluation](#online-evaluation)
4. [Fairness Assessment](#fairness-assessment)
5. [Model Explainability](#model-explainability)
6. [Performance Monitoring](#performance-monitoring)
7. [A/B Testing](#ab-testing)
8. [Benchmarking](#benchmarking)
9. [Quality Gates](#quality-gates)
10. [Evaluation Reports](#evaluation-reports)

## Evaluation Framework

The WITHIN model evaluation framework provides a standardized approach to comprehensively assess model quality across multiple dimensions.

### Core Principles

1. **Multi-faceted Assessment**: Models are evaluated across multiple dimensions including performance, fairness, explainability, and robustness
2. **Staged Evaluation**: Evaluation occurs in multiple stages: development, pre-release, and continuous monitoring
3. **Automated + Human Review**: Automated metrics are supplemented with human expert review
4. **Comparative Analysis**: New models are benchmarked against existing baseline models
5. **Contextual Evaluation**: Metrics are interpreted within the business context and requirements

### Evaluation Workflow

The model evaluation follows a structured workflow:

```
┌────────────────────────────────────────────────────────────┐
│                       Model Training                       │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│                    Offline Evaluation                      │
│  ┌─────────────────┐ ┌────────────────┐ ┌──────────────┐   │
│  │ Holdout Testing │ │ Cross-Validate │ │ Back-testing │   │
│  └─────────────────┘ └────────────────┘ └──────────────┘   │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│                    Fairness Assessment                     │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│                     Explainability                         │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│                  Quality Gate Decision                     │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│                     A/B Testing                            │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│                Performance Monitoring                      │
└────────────────────────────────────────────────────────────┘
```

### Implementation Structure

The evaluation framework is implemented in the `app/models/ml/evaluation` directory:

```
app/models/ml/evaluation/
├── __init__.py
├── evaluator.py                 # Main evaluation orchestrator
├── metrics/
│   ├── __init__.py
│   ├── classification.py        # Classification metrics
│   ├── regression.py            # Regression metrics
│   └── ranking.py               # Ranking metrics
├── fairness/
│   ├── __init__.py
│   ├── demographic_parity.py    # Demographic parity implementation
│   ├── equal_opportunity.py     # Equal opportunity implementation
│   └── disparate_impact.py      # Disparate impact analysis
├── explainers/
│   ├── __init__.py
│   ├── feature_importance.py    # Feature importance analysis
│   ├── shap_explainer.py        # SHAP-based explanations
│   └── lime_explainer.py        # LIME-based explanations
├── monitoring/
│   ├── __init__.py
│   ├── drift_detector.py        # Data and concept drift detection
│   └── performance_tracker.py   # Performance tracking over time
└── reporting/
    ├── __init__.py
    ├── visualization.py         # Visualization utilities
    └── report_generator.py      # Evaluation report generation
``` 

## Offline Evaluation

Offline evaluation assesses model performance using historical data before deployment to production.

### 1. Performance Metrics

Performance metrics vary by model type:

#### For Regression Models (Ad Score Prediction)

```python
def evaluate_regression_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    prediction_type: str = "continuous"
) -> Dict[str, float]:
    """Evaluate regression model performance.
    
    Args:
        model: Trained regression model
        X_test: Test features
        y_test: Ground truth values
        prediction_type: "continuous" or "bounded" (e.g., 0-1 range)
        
    Returns:
        Dictionary of performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
        "max_error": max_error(y_test, y_pred)
    }
    
    # Add bounded regression metrics if applicable
    if prediction_type == "bounded":
        # Calculate adjusted R² for bounded values
        metrics["adjusted_r2"] = 1 - (1 - metrics["r2"]) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        
        # Calibration error
        metrics["calibration_error"] = np.mean(np.abs(y_pred - y_test))
    
    return metrics
```

#### For Classification Models (Account Health Prediction)

```python
def evaluate_classification_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate classification model performance.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Ground truth labels
        class_names: Optional list of class names
        threshold: Classification threshold for binary models
        
    Returns:
        Dictionary of performance metrics and plots
    """
    # Get predicted probabilities for each class
    try:
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba[:, 1] >= threshold).astype(int) if y_proba.shape[1] == 2 else np.argmax(y_proba, axis=1)
    except:
        # Model doesn't support predict_proba
        y_pred = model.predict(X_test)
        y_proba = None
    
    # Basic classification metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm
    
    # ROC and AUC for binary classification
    if y_proba is not None and y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        metrics["roc_auc"] = auc(fpr, tpr)
        metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        metrics["pr_auc"] = auc(recall, precision)
        metrics["pr_curve"] = {"precision": precision, "recall": recall}
        
        # Log loss
        metrics["log_loss"] = log_loss(y_test, y_proba)
        
        # Calibration curve (reliability diagram)
        prob_true, prob_pred = calibration_curve(y_test, y_proba[:, 1], n_bins=10)
        metrics["calibration_curve"] = {"prob_true": prob_true, "prob_pred": prob_pred}
        
        # Brier score
        metrics["brier_score"] = brier_score_loss(y_test, y_proba[:, 1])
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    metrics["classification_report"] = report
    
    return metrics
```

### 2. Cross-Validation

Cross-validation ensures robust performance assessment:

```python
def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_strategy: str = "kfold",
    n_splits: int = 5,
    problem_type: str = "regression",
    metrics: List[str] = ["rmse"],
    random_state: int = 42
) -> Dict[str, List[float]]:
    """Perform cross-validation with specified strategy.
    
    Args:
        model: Model class or instance
        X: Feature DataFrame
        y: Target Series
        cv_strategy: "kfold", "stratified", or "timeseries"
        n_splits: Number of CV splits
        problem_type: "regression" or "classification"
        metrics: List of metrics to compute
        random_state: Random seed
        
    Returns:
        Dictionary with CV results for each metric
    """
    results = {metric: [] for metric in metrics}
    
    # Define cross-validation strategy
    if cv_strategy == "kfold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_strategy == "stratified":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_strategy == "timeseries":
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError(f"Unsupported CV strategy: {cv_strategy}")
    
    # Perform cross-validation
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Clone the model if it's already fitted
        if hasattr(model, "fit"):
            estimator = clone(model)
        else:
            # Assume it's a model class that needs to be instantiated
            estimator = model()
        
        # Train the model
        estimator.fit(X_train, y_train)
        
        # Evaluate based on problem type
        if problem_type == "regression":
            y_pred = estimator.predict(X_test)
            
            # Compute requested metrics
            for metric in metrics:
                if metric == "rmse":
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                elif metric == "mae":
                    score = mean_absolute_error(y_test, y_pred)
                elif metric == "r2":
                    score = r2_score(y_test, y_pred)
                else:
                    raise ValueError(f"Unsupported regression metric: {metric}")
                
                results[metric].append(score)
        else:
            # Classification
            if hasattr(estimator, "predict_proba"):
                y_proba = estimator.predict_proba(X_test)
                y_pred = (y_proba[:, 1] >= 0.5).astype(int) if y_proba.shape[1] == 2 else np.argmax(y_proba, axis=1)
            else:
                y_pred = estimator.predict(X_test)
            
            # Compute requested metrics
            for metric in metrics:
                if metric == "accuracy":
                    score = accuracy_score(y_test, y_pred)
                elif metric == "precision":
                    score = precision_score(y_test, y_pred, average="weighted")
                elif metric == "recall":
                    score = recall_score(y_test, y_pred, average="weighted")
                elif metric == "f1":
                    score = f1_score(y_test, y_pred, average="weighted")
                elif metric == "auc" and hasattr(estimator, "predict_proba"):
                    score = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    raise ValueError(f"Unsupported classification metric: {metric}")
                
                results[metric].append(score)
    
    # Compute summary statistics
    summary = {}
    for metric, scores in results.items():
        summary[f"{metric}_mean"] = np.mean(scores)
        summary[f"{metric}_std"] = np.std(scores)
        summary[f"{metric}_min"] = np.min(scores)
        summary[f"{metric}_max"] = np.max(scores)
        summary[f"{metric}_values"] = scores
    
    return summary
```

### 3. Residual Analysis

For regression models, residual analysis helps identify systematic prediction errors:

```python
def analyze_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    X_test: pd.DataFrame = None
) -> Dict[str, Any]:
    """Perform residual analysis for regression models.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        X_test: Test features for analyzing residuals by feature
        
    Returns:
        Dictionary with residual analysis results
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Basic residual statistics
    stats = {
        "mean": np.mean(residuals),
        "median": np.median(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals),
        "q1": np.percentile(residuals, 25),
        "q3": np.percentile(residuals, 75)
    }
    
    # Normality test (Shapiro-Wilk)
    shapiro_test = shapiro(residuals)
    stats["shapiro_test"] = {
        "statistic": shapiro_test.statistic,
        "p_value": shapiro_test.pvalue,
        "is_normal": shapiro_test.pvalue > 0.05
    }
    
    # Residual patterns
    if X_test is not None:
        # Check correlation between residuals and features
        residual_correlations = {}
        for col in X_test.columns:
            if pd.api.types.is_numeric_dtype(X_test[col]):
                correlation = np.corrcoef(X_test[col], residuals)[0, 1]
                residual_correlations[col] = correlation
        
        stats["feature_correlations"] = residual_correlations
        
        # Identify features with highest correlation to residuals
        sorted_correlations = sorted(
            residual_correlations.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        stats["top_correlated_features"] = sorted_correlations[:5]
    
    # Check for heteroscedasticity (Breusch-Pagan test)
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_test = het_breuschpagan(residuals, X_test.values if X_test is not None else np.ones((len(residuals), 1)))
        stats["heteroscedasticity_test"] = {
            "lm_statistic": bp_test[0],
            "lm_pvalue": bp_test[1],
            "f_statistic": bp_test[2],
            "f_pvalue": bp_test[3],
            "has_heteroscedasticity": bp_test[1] < 0.05
        }
    except:
        # Statsmodels may not be available
        pass
    
    return stats
```

### 4. Backtesting

For time-sensitive models, temporal validation evaluates performance across different time periods:

```python
def backtest_model(
    model: Any,
    data: pd.DataFrame,
    target_col: str,
    date_col: str,
    feature_cols: List[str],
    start_date: str,
    end_date: str,
    window_size: str = "1M",
    step_size: str = "1W",
    metric: str = "rmse"
) -> pd.DataFrame:
    """Backtest model on historical data with sliding window.
    
    Args:
        model: Trained model
        data: Historical data with timestamps
        target_col: Target column name
        date_col: Date column name
        feature_cols: Feature column names
        start_date: Start date for backtest
        end_date: End date for backtest
        window_size: Training window size (pandas offset string)
        step_size: Step size for sliding window (pandas offset string)
        metric: Evaluation metric
        
    Returns:
        DataFrame with backtest results
    """
    # Ensure date column is datetime
    data = data.copy()
    if not pd.api.types.is_datetime64_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Sort by date
    data = data.sort_values(date_col)
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate evaluation windows
    eval_dates = pd.date_range(start=start, end=end, freq=step_size)
    results = []
    
    for eval_date in eval_dates:
        # Define training period
        train_start = eval_date - pd.Timedelta(window_size)
        train_end = eval_date
        
        # Define test period (next step)
        test_start = eval_date
        test_end = eval_date + pd.Timedelta(step_size)
        
        # Filter data
        train_mask = (data[date_col] >= train_start) & (data[date_col] < train_end)
        test_mask = (data[date_col] >= test_start) & (data[date_col] < test_end)
        
        train_data = data[train_mask]
        test_data = data[test_mask]
        
        # Skip if insufficient data
        if len(train_data) < 10 or len(test_data) < 5:
            continue
        
        # Train model
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Clone model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model_clone.predict(X_test)
        
        # Calculate metric
        if metric == "rmse":
            score = np.sqrt(mean_squared_error(y_test, y_pred))
        elif metric == "mae":
            score = mean_absolute_error(y_test, y_pred)
        elif metric == "r2":
            score = r2_score(y_test, y_pred)
        elif metric == "accuracy":
            score = accuracy_score(y_test, y_pred)
        elif metric == "f1":
            score = f1_score(y_test, y_pred)
        
        # Store result
        results.append({
            "eval_date": eval_date,
            "train_size": len(train_data),
            "test_size": len(test_data),
            metric: score
        })
    
    return pd.DataFrame(results)
``` 

## Online Evaluation

Online evaluation assesses model performance in production environments on live data.

### 1. Shadow Deployment

Before full production deployment, models undergo shadow testing where predictions are generated but not used for decision-making:

```python
def shadow_deploy_model(
    model_id: str,
    production_model_id: str,
    feature_service: FeatureService,
    logging_service: LoggingService,
    sample_rate: float = 1.0,
    duration_days: int = 7
):
    """Deploy model in shadow mode alongside production model.
    
    Args:
        model_id: ID of model to shadow deploy
        production_model_id: ID of current production model
        feature_service: Service to retrieve feature values
        logging_service: Service to log predictions
        sample_rate: Fraction of requests to shadow (0-1)
        duration_days: Duration of shadow deployment in days
    """
    # Configure shadow deployment
    config = {
        "model_id": model_id,
        "production_model_id": production_model_id,
        "start_time": datetime.now().isoformat(),
        "end_time": (datetime.now() + timedelta(days=duration_days)).isoformat(),
        "sample_rate": sample_rate,
        "feature_service_endpoint": feature_service.endpoint,
        "logging_service_endpoint": logging_service.endpoint
    }
    
    # Register shadow deployment
    deployment_id = register_shadow_deployment(config)
    
    # Set up monitoring dashboard
    setup_shadow_monitoring_dashboard(
        deployment_id=deployment_id,
        model_id=model_id,
        production_model_id=production_model_id
    )
    
    return deployment_id
```

### 2. A/B Testing Framework

The A/B testing framework evaluates model variants on randomized user segments:

```python
def setup_ab_test(
    experiment_name: str,
    model_variants: List[str],
    traffic_split: List[float],
    metrics: List[str],
    segment_id: Optional[str] = None,
    duration_days: int = 14,
    min_sample_size: int = 5000
) -> Dict[str, Any]:
    """Set up A/B test for model variants.
    
    Args:
        experiment_name: Unique name for the experiment
        model_variants: List of model IDs to test
        traffic_split: Traffic percentage for each variant (must sum to 1.0)
        metrics: List of metrics to evaluate
        segment_id: Optional user segment to test on
        duration_days: Test duration in days
        min_sample_size: Minimum sample size required
        
    Returns:
        Experiment configuration details
    """
    # Validate inputs
    if len(model_variants) != len(traffic_split):
        raise ValueError("Must provide traffic split for each model variant")
    
    if sum(traffic_split) != 1.0:
        raise ValueError("Traffic split must sum to 1.0")
    
    # Calculate required sample size for statistical significance
    power_analysis = {
        "min_detectable_effect": 0.05,  # 5% improvement
        "statistical_power": 0.8,       # 80% power
        "significance_level": 0.05,     # 5% significance
        "required_samples": min_sample_size
    }
    
    # Configure experiment
    experiment = {
        "name": experiment_name,
        "start_time": datetime.now().isoformat(),
        "end_time": (datetime.now() + timedelta(days=duration_days)).isoformat(),
        "variants": [
            {
                "model_id": model_id,
                "traffic_percentage": split,
                "is_control": idx == 0  # First model is control
            }
            for idx, (model_id, split) in enumerate(zip(model_variants, traffic_split))
        ],
        "metrics": metrics,
        "segment_id": segment_id,
        "power_analysis": power_analysis
    }
    
    # Register experiment
    experiment_id = register_ab_experiment(experiment)
    
    # Set up real-time monitoring
    setup_experiment_monitoring(experiment_id)
    
    return {"experiment_id": experiment_id, "config": experiment}
```

### 3. Metrics Collection

Structured logging of predictions and outcomes enables evaluation of online performance:

```python
class OnlineMetricsCollector:
    """Collect and analyze online metrics for model performance."""
    
    def __init__(
        self,
        model_id: str,
        database_connector: DatabaseConnector,
        metrics: List[str]
    ):
        """Initialize metrics collector.
        
        Args:
            model_id: ID of the model to collect metrics for
            database_connector: Connection to metrics database
            metrics: List of metrics to collect
        """
        self.model_id = model_id
        self.db = database_connector
        self.metrics = metrics
        
    def log_prediction(
        self,
        prediction_id: str,
        features: Dict[str, Any],
        prediction: float,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a model prediction for later analysis.
        
        Args:
            prediction_id: Unique ID for this prediction
            features: Feature values used for prediction
            prediction: Model prediction value
            confidence: Model confidence/probability (optional)
            metadata: Additional contextual information
        """
        log_entry = {
            "prediction_id": prediction_id,
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "features": json.dumps(features),
            "prediction": prediction,
            "confidence": confidence,
            "metadata": json.dumps(metadata) if metadata else None
        }
        
        self.db.insert("prediction_logs", log_entry)
    
    def log_outcome(
        self,
        prediction_id: str,
        actual_value: Any,
        outcome_timestamp: Optional[datetime] = None
    ):
        """Log the actual outcome for a previous prediction.
        
        Args:
            prediction_id: ID of the prediction this outcome relates to
            actual_value: The ground truth value
            outcome_timestamp: When the outcome was observed
        """
        outcome_entry = {
            "prediction_id": prediction_id,
            "actual_value": actual_value,
            "outcome_timestamp": outcome_timestamp or datetime.now().isoformat()
        }
        
        self.db.insert("outcome_logs", outcome_entry)
    
    def calculate_online_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        segment_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Calculate online metrics for a specified time period.
        
        Args:
            start_time: Start of evaluation period
            end_time: End of evaluation period
            segment_filter: Optional filter for specific segments
            
        Returns:
            Dictionary of metric names and values
        """
        # Query for predictions and outcomes in time period
        query = """
        SELECT 
            p.prediction_id, 
            p.prediction, 
            p.confidence,
            p.metadata,
            o.actual_value
        FROM 
            prediction_logs p
        JOIN 
            outcome_logs o
        ON 
            p.prediction_id = o.prediction_id
        WHERE 
            p.model_id = ? AND
            p.timestamp BETWEEN ? AND ?
        """
        
        params = [self.model_id, start_time.isoformat(), end_time.isoformat()]
        
        # Add segment filtering if specified
        if segment_filter:
            # Parse metadata JSON and apply filter
            # Implementation depends on database specifics
            pass
        
        results = self.db.query(query, params)
        
        # Calculate metrics based on results
        predictions = [r["prediction"] for r in results]
        actuals = [r["actual_value"] for r in results]
        
        metrics_results = {}
        
        for metric in self.metrics:
            if metric == "rmse":
                metrics_results[metric] = np.sqrt(mean_squared_error(actuals, predictions))
            elif metric == "mae":
                metrics_results[metric] = mean_absolute_error(actuals, predictions)
            elif metric == "r2":
                metrics_results[metric] = r2_score(actuals, predictions)
            elif metric == "accuracy":
                metrics_results[metric] = accuracy_score(actuals, predictions)
            elif metric == "auc":
                # Requires confidence scores for binary classification
                confidences = [r["confidence"] for r in results]
                metrics_results[metric] = roc_auc_score(actuals, confidences)
        
        # Additional meta-metrics
        metrics_results["sample_size"] = len(results)
        metrics_results["coverage"] = len(results) / self.db.count(
            "prediction_logs", 
            {"model_id": self.model_id, "timestamp": {"between": [start_time, end_time]}}
        )
        
        return metrics_results
```

### 4. Drift Detection

Production models are continuously monitored for data and prediction drift:

```python
def detect_model_drift(
    model_id: str,
    reference_period: Tuple[datetime, datetime],
    current_period: Tuple[datetime, datetime],
    database_connector: DatabaseConnector,
    drift_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Detect drift in model inputs and predictions.
    
    Args:
        model_id: Model to evaluate
        reference_period: (start, end) of reference period
        current_period: (start, end) of current period
        database_connector: Database connection
        drift_threshold: Threshold for drift detection
        
    Returns:
        Drift assessment results
    """
    # Get features and predictions from reference period
    reference_data = database_connector.query(
        "SELECT features, prediction FROM prediction_logs WHERE model_id = ? AND timestamp BETWEEN ? AND ?",
        [model_id, reference_period[0].isoformat(), reference_period[1].isoformat()]
    )
    
    # Get features and predictions from current period
    current_data = database_connector.query(
        "SELECT features, prediction FROM prediction_logs WHERE model_id = ? AND timestamp BETWEEN ? AND ?",
        [model_id, current_period[0].isoformat(), current_period[1].isoformat()]
    )
    
    # Parse features JSON
    reference_features = [json.loads(r["features"]) for r in reference_data]
    current_features = [json.loads(r["features"]) for r in current_data]
    
    # Convert to DataFrames
    ref_df = pd.DataFrame(reference_features)
    cur_df = pd.DataFrame(current_features)
    
    # Feature drift analysis
    feature_drift = {}
    for col in ref_df.columns:
        if pd.api.types.is_numeric_dtype(ref_df[col]):
            # Kolmogorov-Smirnov test for distribution shift
            ks_stat, p_value = ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
            feature_drift[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "has_drift": p_value < 0.01 and ks_stat > drift_threshold
            }
        elif pd.api.types.is_categorical_dtype(ref_df[col]) or pd.api.types.is_object_dtype(ref_df[col]):
            # Chi-square test for categorical features
            ref_counts = ref_df[col].value_counts(normalize=True, dropna=False)
            cur_counts = cur_df[col].value_counts(normalize=True, dropna=False)
            
            # Calculate JS divergence between distributions
            js_divergence = calculate_js_divergence(ref_counts, cur_counts)
            feature_drift[col] = {
                "js_divergence": js_divergence,
                "has_drift": js_divergence > drift_threshold
            }
    
    # Output drift analysis
    reference_predictions = [r["prediction"] for r in reference_data]
    current_predictions = [r["prediction"] for r in current_data]
    
    # Prediction drift
    ks_stat, p_value = ks_2samp(reference_predictions, current_predictions)
    prediction_drift = {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "has_drift": p_value < 0.01 and ks_stat > drift_threshold,
        "reference_mean": np.mean(reference_predictions),
        "current_mean": np.mean(current_predictions),
        "percent_change": (np.mean(current_predictions) - np.mean(reference_predictions)) / np.mean(reference_predictions) * 100
    }
    
    # Summarize drifting features
    drifting_features = [f for f, metrics in feature_drift.items() if metrics["has_drift"]]
    drift_percentage = len(drifting_features) / len(feature_drift) * 100
    
    return {
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "drifting_features": drifting_features,
        "drift_percentage": drift_percentage,
        "requires_attention": prediction_drift["has_drift"] or drift_percentage > 30,
        "reference_period": {
            "start": reference_period[0].isoformat(),
            "end": reference_period[1].isoformat(),
            "sample_size": len(reference_data)
        },
        "current_period": {
            "start": current_period[0].isoformat(),
            "end": current_period[1].isoformat(),
            "sample_size": len(current_data)
        }
    }

def calculate_js_divergence(p: pd.Series, q: pd.Series) -> float:
    """Calculate Jensen-Shannon divergence between two distributions."""
    # Ensure both distributions have the same categories
    all_categories = list(set(p.index) | set(q.index))
    p_full = pd.Series({cat: p.get(cat, 0) for cat in all_categories})
    q_full = pd.Series({cat: q.get(cat, 0) for cat in all_categories})
    
    # Convert to numpy arrays
    p_array = p_full.values
    q_array = q_full.values
    
    # Ensure they sum to 1
    p_array = p_array / np.sum(p_array)
    q_array = q_array / np.sum(q_array)
    
    # Calculate middle point
    m_array = 0.5 * (p_array + q_array)
    
    # Calculate KL divergence parts, handling 0s
    kl_pm = np.sum(p_array * np.log2(p_array / m_array + np.finfo(float).eps))
    kl_qm = np.sum(q_array * np.log2(q_array / m_array + np.finfo(float).eps))
    
    # JS divergence
    return 0.5 * (kl_pm + kl_qm)
``` 

## Fairness Assessment

Fair and unbiased models are critical for ethical AI deployments. Our fairness assessment methods evaluate models for potential biases across different demographic groups and ad account types.

### 1. Fairness Metrics

The following metrics assess various aspects of fairness:

```python
class FairnessEvaluator:
    """Evaluate model fairness across different segments."""
    
    def __init__(
        self,
        model: Any,
        test_data: pd.DataFrame,
        target_column: str,
        protected_attributes: Dict[str, List[Any]],
        prediction_type: str = "regression"
    ):
        """Initialize fairness evaluator.
        
        Args:
            model: Trained model to evaluate
            test_data: Test dataset
            target_column: Target variable column name
            protected_attributes: Dict mapping attribute names to values
                e.g. {"account_size": ["small", "medium", "large"]}
            prediction_type: "regression" or "classification"
        """
        self.model = model
        self.data = test_data.copy()
        self.target = target_column
        self.protected_attrs = protected_attributes
        self.prediction_type = prediction_type
        
        # Generate predictions
        features = self.data.drop(columns=[target_column])
        self.data["prediction"] = model.predict(features)
        
        # For binary classification, get probabilities too
        if prediction_type == "classification":
            if hasattr(model, "predict_proba"):
                self.data["probability"] = model.predict_proba(features)[:, 1]
        
    def calculate_group_metrics(
        self,
        attribute: str,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """Calculate performance metrics for each group.
        
        Args:
            attribute: Protected attribute to analyze
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with metrics for each group
        """
        if attribute not in self.protected_attrs:
            raise ValueError(f"Attribute {attribute} not in protected attributes")
            
        if metrics is None:
            if self.prediction_type == "regression":
                metrics = ["rmse", "mae", "r2"]
            else:
                metrics = ["accuracy", "precision", "recall", "auc"]
        
        results = []
        
        # Calculate metrics for each group
        for group in self.protected_attrs[attribute]:
            group_data = self.data[self.data[attribute] == group]
            
            if len(group_data) < 10:
                continue  # Skip groups with insufficient data
            
            y_true = group_data[self.target]
            y_pred = group_data["prediction"]
            
            group_metrics = {"group": group, "count": len(group_data)}
            
            # Calculate metrics based on prediction type
            if self.prediction_type == "regression":
                for metric in metrics:
                    if metric == "rmse":
                        group_metrics[metric] = np.sqrt(mean_squared_error(y_true, y_pred))
                    elif metric == "mae":
                        group_metrics[metric] = mean_absolute_error(y_true, y_pred)
                    elif metric == "r2":
                        group_metrics[metric] = r2_score(y_true, y_pred)
            else:
                for metric in metrics:
                    if metric == "accuracy":
                        group_metrics[metric] = accuracy_score(y_true, y_pred)
                    elif metric == "precision":
                        group_metrics[metric] = precision_score(y_true, y_pred)
                    elif metric == "recall":
                        group_metrics[metric] = recall_score(y_true, y_pred)
                    elif metric == "auc" and "probability" in group_data.columns:
                        group_metrics[metric] = roc_auc_score(y_true, group_data["probability"])
            
            results.append(group_metrics)
        
        return pd.DataFrame(results)
    
    def demographic_parity(
        self, 
        attribute: str,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Check if prediction rates are similar across groups (classification).
        
        Args:
            attribute: Protected attribute to analyze
            threshold: Classification threshold (for regression)
            
        Returns:
            Dict with demographic parity results
        """
        if self.prediction_type == "regression":
            # Convert regression to binary outcomes using threshold
            self.data["binary_pred"] = (self.data["prediction"] >= threshold).astype(int)
            pred_col = "binary_pred"
        else:
            pred_col = "prediction"
        
        # Calculate prediction rates for each group
        groups = {}
        for group in self.protected_attrs[attribute]:
            group_data = self.data[self.data[attribute] == group]
            if len(group_data) > 0:
                prediction_rate = group_data[pred_col].mean()
                groups[group] = prediction_rate
        
        # Calculate disparities
        if len(groups) < 2:
            return {"error": "Insufficient groups for comparison"}
        
        base_group = list(groups.keys())[0]
        base_rate = groups[base_group]
        
        disparities = {}
        for group, rate in groups.items():
            if group != base_group:
                # Absolute difference in prediction rates
                abs_disparity = abs(rate - base_rate)
                # Relative ratio of prediction rates
                ratio_disparity = rate / base_rate if base_rate > 0 else float('inf')
                
                disparities[f"{base_group}_vs_{group}"] = {
                    "base_rate": base_rate,
                    "group_rate": rate,
                    "absolute_difference": abs_disparity,
                    "ratio": ratio_disparity,
                    "passes_threshold": abs_disparity <= 0.1
                }
        
        # Summary statistics
        rates = list(groups.values())
        summary = {
            "min_rate": min(rates),
            "max_rate": max(rates),
            "max_disparity": max([d["absolute_difference"] for d in disparities.values()]),
            "passes_fairness_test": all([d["passes_threshold"] for d in disparities.values()])
        }
        
        return {
            "groups": groups,
            "disparities": disparities,
            "summary": summary
        }
    
    def equal_opportunity(
        self, 
        attribute: str,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Check if true positive rates are similar across groups (classification).
        
        Args:
            attribute: Protected attribute to analyze
            threshold: Classification threshold
            
        Returns:
            Dict with equal opportunity results
        """
        if self.prediction_type == "regression":
            # Convert regression to binary outcomes
            self.data["binary_pred"] = (self.data["prediction"] >= threshold).astype(int)
            self.data["binary_true"] = (self.data[self.target] >= threshold).astype(int)
            pred_col = "binary_pred"
            true_col = "binary_true"
        else:
            pred_col = "prediction"
            true_col = self.target
        
        # Calculate TPR for each group
        groups = {}
        for group in self.protected_attrs[attribute]:
            group_data = self.data[self.data[attribute] == group]
            
            # Only consider examples where true label is positive
            positive_cases = group_data[group_data[true_col] == 1]
            
            if len(positive_cases) > 10:  # Ensure sufficient positive samples
                tpr = positive_cases[pred_col].mean()
                groups[group] = tpr
        
        # Calculate disparities
        if len(groups) < 2:
            return {"error": "Insufficient groups for comparison"}
        
        base_group = list(groups.keys())[0]
        base_tpr = groups[base_group]
        
        disparities = {}
        for group, tpr in groups.items():
            if group != base_group:
                # Absolute difference in true positive rates
                abs_disparity = abs(tpr - base_tpr)
                # Relative ratio of true positive rates
                ratio_disparity = tpr / base_tpr if base_tpr > 0 else float('inf')
                
                disparities[f"{base_group}_vs_{group}"] = {
                    "base_tpr": base_tpr,
                    "group_tpr": tpr,
                    "absolute_difference": abs_disparity,
                    "ratio": ratio_disparity,
                    "passes_threshold": abs_disparity <= 0.1
                }
        
        # Summary statistics
        tprs = list(groups.values())
        summary = {
            "min_tpr": min(tprs),
            "max_tpr": max(tprs),
            "max_disparity": max([d["absolute_difference"] for d in disparities.values()]),
            "passes_fairness_test": all([d["passes_threshold"] for d in disparities.values()])
        }
        
        return {
            "groups": groups,
            "disparities": disparities,
            "summary": summary
        }
```

### 2. Fairness Assessment Process

Our standard fairness assessment process involves:

1. **Identification of Protected Attributes**:
   - Account size (small, medium, large)
   - Industry vertical
   - Account age
   - Geographic region

2. **Metrics Calculated**:
   - Demographic parity: Equal prediction rates across groups
   - Equal opportunity: Equal true positive rates across groups
   - Performance disparities: Variation in metrics like RMSE or accuracy

3. **Fairness Report Generation**:

```python
def generate_fairness_report(
    model: Any,
    test_data: pd.DataFrame,
    target_column: str,
    protected_attributes: Dict[str, List[Any]],
    prediction_type: str = "regression"
) -> Dict[str, Any]:
    """Generate comprehensive fairness report for model.
    
    Args:
        model: Trained model to evaluate
        test_data: Test dataset
        target_column: Target variable column name
        protected_attributes: Dict mapping attribute names to values
        prediction_type: "regression" or "classification"
        
    Returns:
        Comprehensive fairness report
    """
    evaluator = FairnessEvaluator(
        model=model,
        test_data=test_data,
        target_column=target_column,
        protected_attributes=protected_attributes,
        prediction_type=prediction_type
    )
    
    report = {"timestamp": datetime.now().isoformat()}
    
    # Performance metrics by group
    group_metrics = {}
    for attribute in protected_attributes:
        metrics_df = evaluator.calculate_group_metrics(attribute)
        
        # Convert DataFrame to serializable format
        group_metrics[attribute] = metrics_df.to_dict(orient="records")
        
        # Calculate disparity metrics
        max_disparity = metrics_df["rmse"].max() - metrics_df["rmse"].min() if prediction_type == "regression" else \
                       metrics_df["accuracy"].max() - metrics_df["accuracy"].min()
                       
        group_metrics[f"{attribute}_disparity"] = max_disparity
    
    report["group_metrics"] = group_metrics
    
    # Fairness metrics
    fairness_metrics = {}
    for attribute in protected_attributes:
        # Demographic parity
        dem_parity = evaluator.demographic_parity(attribute)
        fairness_metrics[f"{attribute}_demographic_parity"] = dem_parity
        
        # Equal opportunity (for classification or thresholded regression)
        if prediction_type == "classification" or "threshold" in test_data.columns:
            eq_opp = evaluator.equal_opportunity(attribute)
            fairness_metrics[f"{attribute}_equal_opportunity"] = eq_opp
    
    report["fairness_metrics"] = fairness_metrics
    
    # Overall fairness assessment
    fairness_tests = []
    for metric_name, metric_result in fairness_metrics.items():
        if "summary" in metric_result and "passes_fairness_test" in metric_result["summary"]:
            fairness_tests.append(metric_result["summary"]["passes_fairness_test"])
    
    report["overall_assessment"] = {
        "passes_all_tests": all(fairness_tests),
        "pass_rate": sum(fairness_tests) / len(fairness_tests) if fairness_tests else 0,
        "failed_tests": len(fairness_tests) - sum(fairness_tests) if fairness_tests else 0
    }
    
    # Recommendations based on assessment
    recommendations = []
    for attribute in protected_attributes:
        dem_parity_key = f"{attribute}_demographic_parity"
        if dem_parity_key in fairness_metrics:
            if not fairness_metrics[dem_parity_key]["summary"]["passes_fairness_test"]:
                max_disp_groups = max(
                    fairness_metrics[dem_parity_key]["disparities"].items(),
                    key=lambda x: x[1]["absolute_difference"]
                )
                recommendations.append(
                    f"Prediction rates show significant disparity for {attribute} between " + 
                    f"{max_disp_groups[0]} (difference: {max_disp_groups[1]['absolute_difference']:.2f})"
                )
    
    report["recommendations"] = recommendations
    report["requires_mitigation"] = not report["overall_assessment"]["passes_all_tests"]
    
    return report
```

## Model Explainability

Explainable AI techniques help understand model predictions and build trust with users.

### 1. Feature Importance Analysis

Global feature importance quantifies each feature's contribution to model predictions:

```python
def analyze_feature_importance(
    model: Any,
    feature_names: List[str],
    X_test: pd.DataFrame = None,
    importance_type: str = "native"
) -> Dict[str, Any]:
    """Analyze global feature importance for model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        X_test: Test data (needed for permutation importance)
        importance_type: "native", "permutation", or "shap"
        
    Returns:
        Dictionary with feature importance analysis
    """
    importances = {}
    
    if importance_type == "native":
        # Check if model has native feature importance
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importances = {
                feature: float(importance) 
                for feature, importance in zip(feature_names, model.feature_importances_)
            }
        elif hasattr(model, "coef_"):
            # Linear models
            if model.coef_.ndim == 1:
                importances = {
                    feature: float(abs(coef)) 
                    for feature, coef in zip(feature_names, model.coef_)
                }
            else:
                # For multi-class models, average absolute coefficients
                importances = {
                    feature: float(np.mean(np.abs(model.coef_[:, i]))) 
                    for i, feature in enumerate(feature_names)
                }
    
    elif importance_type == "permutation" and X_test is not None:
        # Permutation importance
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X_test, 
            y=None,  # No need for target as we're just measuring prediction changes
            n_repeats=10,
            random_state=42
        )
        
        importances = {
            feature: float(importance) 
            for feature, importance in zip(feature_names, result.importances_mean)
        }
    
    elif importance_type == "shap" and X_test is not None:
        try:
            import shap
            
            # Choose explainer based on model type
            if hasattr(model, "tree_") or hasattr(model, "estimators_"):
                # Tree-based models
                explainer = shap.TreeExplainer(model)
            else:
                # Kernel explainer for other models
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_test, 100))
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test.iloc[:100])  # Sample for efficiency
            
            # For multi-output models, average absolute SHAP values
            if isinstance(shap_values, list):
                # Classification with multiple classes
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            
            # Global feature importance from SHAP
            importances = {
                feature: float(np.mean(np.abs(shap_values[:, i]))) 
                for i, feature in enumerate(feature_names)
            }
            
        except ImportError:
            return {"error": "SHAP package not installed"}
    
    # Sort features by importance
    sorted_importances = sorted(
        importances.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Calculate cumulative importance
    total_importance = sum(importances.values())
    cumulative = 0
    features_by_cumulative = []
    
    for feature, importance in sorted_importances:
        cumulative += importance
        features_by_cumulative.append({
            "feature": feature,
            "importance": importance,
            "normalized_importance": importance / total_importance if total_importance > 0 else 0,
            "cumulative": cumulative / total_importance if total_importance > 0 else 0
        })
    
    # Find number of features for different thresholds
    thresholds = {
        "90%": next((i + 1 for i, f in enumerate(features_by_cumulative) if f["cumulative"] >= 0.9), len(features_by_cumulative)),
        "95%": next((i + 1 for i, f in enumerate(features_by_cumulative) if f["cumulative"] >= 0.95), len(features_by_cumulative)),
        "99%": next((i + 1 for i, f in enumerate(features_by_cumulative) if f["cumulative"] >= 0.99), len(features_by_cumulative))
    }
    
    # Get top and bottom features
    top_features = features_by_cumulative[:10]
    bottom_features = features_by_cumulative[-10:] if len(features_by_cumulative) > 10 else []
    
    return {
        "method": importance_type,
        "features_by_importance": features_by_cumulative,
        "top_features": top_features,
        "bottom_features": bottom_features,
        "feature_count_thresholds": thresholds,
        "total_features": len(feature_names)
    }
```

### 2. Local Explanations with SHAP

SHAP (SHapley Additive exPlanations) provides local interpretability for individual predictions:

```python
def explain_prediction(
    model: Any,
    instance: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    baseline_instances: Optional[pd.DataFrame] = None,
    method: str = "shap",
    num_features: int = 10
) -> Dict[str, Any]:
    """Generate explanation for individual prediction.
    
    Args:
        model: Trained model
        instance: Single instance to explain
        feature_names: List of feature names
        baseline_instances: Representative data for baseline
        method: Explanation method ("shap" or "lime")
        num_features: Number of top features to include
        
    Returns:
        Dictionary with prediction explanation
    """
    # Ensure instance is DataFrame
    if isinstance(instance, np.ndarray):
        instance = pd.DataFrame([instance], columns=feature_names)
    elif isinstance(instance, pd.Series):
        instance = pd.DataFrame([instance.values], columns=feature_names)
    elif len(instance) > 1:
        instance = instance.iloc[[0]]
    
    # Get prediction
    prediction = model.predict(instance)[0]
    
    # Initialize explanation
    explanation = {
        "prediction": float(prediction),
        "method": method,
        "feature_contributions": {}
    }
    
    if method == "shap":
        try:
            import shap
            
            # Choose explainer based on model type
            if hasattr(model, "tree_") or hasattr(model, "estimators_"):
                # Tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(instance)
                
                # Handle different output shapes
                if isinstance(shap_values, list):
                    # For classification, use the predicted class
                    pred_class = int(prediction)
                    shap_values = shap_values[pred_class]
                
                # Get base value
                if hasattr(explainer, "expected_value"):
                    base_value = explainer.expected_value
                    if isinstance(base_value, list):
                        base_value = base_value[pred_class]
                else:
                    base_value = 0
            else:
                # For other models, use KernelExplainer
                if baseline_instances is None:
                    raise ValueError("Baseline instances required for KernelExplainer")
                
                # Limit baseline to reasonable size
                baseline_sample = shap.sample(baseline_instances, 100)
                
                explainer = shap.KernelExplainer(model.predict, baseline_sample)
                shap_values = explainer.shap_values(instance)
                base_value = explainer.expected_value
            
            # Prepare feature contributions
            contributions = []
            for i, (feature, value) in enumerate(zip(feature_names, instance.values[0])):
                shap_value = float(shap_values[0, i]) if shap_values.ndim > 1 else float(shap_values[i])
                contributions.append({
                    "feature": feature,
                    "value": float(value),
                    "contribution": shap_value,
                    "impact": abs(shap_value)
                })
            
            # Sort by impact
            contributions.sort(key=lambda x: x["impact"], reverse=True)
            
            # Limit to top features
            top_contributions = contributions[:num_features]
            
            explanation["base_value"] = float(base_value)
            explanation["feature_contributions"] = top_contributions
            
        except ImportError:
            explanation["error"] = "SHAP package not installed"
            
    elif method == "lime":
        try:
            from lime.lime_tabular import LimeTabularExplainer
            
            # Need training data for LIME
            if baseline_instances is None:
                raise ValueError("Baseline instances required for LIME")
            
            # Determine categorical features
            categorical_features = [
                i for i, col in enumerate(feature_names) 
                if baseline_instances[col].dtype.name == "category" or 
                baseline_instances[col].dtype.name == "object" or
                len(baseline_instances[col].unique()) < 10
            ]
            
            # Create explainer
            explainer = LimeTabularExplainer(
                baseline_instances.values,
                feature_names=feature_names,
                categorical_features=categorical_features,
                mode="regression" if not hasattr(model, "predict_proba") else "classification"
            )
            
            # Generate explanation
            if hasattr(model, "predict_proba"):
                lime_exp = explainer.explain_instance(
                    instance.values[0],
                    model.predict_proba,
                    num_features=num_features
                )
                
                # For classification, extract predicted class explanation
                pred_class = int(prediction)
                feature_weights = lime_exp.as_list(label=pred_class)
            else:
                lime_exp = explainer.explain_instance(
                    instance.values[0],
                    model.predict,
                    num_features=num_features
                )
                
                # For regression, extract feature weights
                feature_weights = lime_exp.as_list()
            
            # Extract contributions
            contributions = []
            for feature_desc, weight in feature_weights:
                # Extract feature and value from description
                parts = feature_desc.split(" ")
                if len(parts) >= 3 and parts[1] in ["<", "<=", ">", ">=", "="]:
                    # Numerical feature with threshold
                    feature = parts[0]
                    value = float(instance[feature].values[0])
                    relation = parts[1]
                    threshold = float(parts[2])
                    feature_desc = f"{feature} {relation} {threshold}"
                else:
                    # Categorical feature or simple description
                    feature = feature_desc.split(" =")[0]
                    value = instance[feature].values[0] if feature in instance.columns else None
                
                contributions.append({
                    "feature": feature,
                    "value": float(value) if isinstance(value, (int, float)) else str(value),
                    "description": feature_desc,
                    "contribution": float(weight),
                    "impact": abs(float(weight))
                })
            
            # Sort by impact
            contributions.sort(key=lambda x: x["impact"], reverse=True)
            
            explanation["feature_contributions"] = contributions
            explanation["intercept"] = float(lime_exp.intercept[pred_class] if hasattr(model, "predict_proba") else lime_exp.intercept)
            
        except ImportError:
            explanation["error"] = "LIME package not installed"
    
    # Add overall explanation summary
    if "feature_contributions" in explanation and explanation["feature_contributions"]:
        contributions = explanation["feature_contributions"]
        
        # Generate text explanation
        text_explanation = [f"Predicted value: {prediction:.4f}"]
        
        if method == "shap":
            text_explanation.append(f"Base value: {explanation['base_value']:.4f}")
            
            for contrib in contributions[:5]:  # Top 5 for summary
                direction = "increased" if contrib["contribution"] > 0 else "decreased"
                text_explanation.append(
                    f"{contrib['feature']} = {contrib['value']} {direction} prediction by {abs(contrib['contribution']):.4f}"
                )
        elif method == "lime":
            text_explanation.append(f"Base value (intercept): {explanation['intercept']:.4f}")
            
            for contrib in contributions[:5]:  # Top 5 for summary
                direction = "increased" if contrib["contribution"] > 0 else "decreased"
                text_explanation.append(
                    f"{contrib['description']} {direction} prediction by {abs(contrib['contribution']):.4f}"
                )
        
        explanation["summary"] = text_explanation
    
    return explanation
```

### 3. Model-Specific Explanations

For certain model types, specialized explanation methods are available:

```python
def explain_tree_model(
    model: Any,
    instance: pd.DataFrame,
    feature_names: List[str],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate decision path explanation for tree-based models.
    
    Args:
        model: Trained tree-based model
        instance: Instance to explain
        feature_names: Feature names
        class_names: Class names for classification
        
    Returns:
        Dictionary with decision path explanation
    """
    from sklearn.tree import _tree
    
    def get_decision_path(tree, node_id, feature_names, path):
        """Recursively extract decision path from tree."""
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            # Leaf node
            if tree.n_outputs == 1:
                value = float(tree.value[node_id][0, 0])
            else:
                value = {
                    class_names[i] if class_names else str(i): float(v)
                    for i, v in enumerate(tree.value[node_id][0])
                }
            return path + [{"type": "leaf", "value": value}]
        
        # Decision node
        feature = feature_names[tree.feature[node_id]]
        threshold = float(tree.threshold[node_id])
        
        # Determine which path the instance follows
        if instance[feature].values[0] <= threshold:
            path = path + [{
                "type": "decision",
                "feature": feature,
                "threshold": threshold,
                "relation": "<=",
                "decision": "left"
            }]
            return get_decision_path(tree, tree.children_left[node_id], feature_names, path)
        else:
            path = path + [{
                "type": "decision",
                "feature": feature,
                "threshold": threshold,
                "relation": ">",
                "decision": "right"
            }]
            return get_decision_path(tree, tree.children_right[node_id], feature_names, path)
    
    # Determine model type
    if hasattr(model, "estimators_"):
        # Ensemble of trees (e.g., RandomForest, GradientBoosting)
        trees = model.estimators_
        if isinstance(trees, list):
            # RandomForest
            trees = [trees[0]] if len(trees) > 0 else []  # Just explain first tree
        elif hasattr(trees, "shape"):
            # GradientBoosting with multiple outputs
            trees = [trees[0, 0]] if trees.shape[0] > 0 else []
    else:
        # Single tree
        trees = [model]
    
    # Generate explanation for each tree
    tree_explanations = []
    
    for i, tree_model in enumerate(trees):
        if hasattr(tree_model, "tree_"):
            tree = tree_model.tree_
            path = get_decision_path(tree, 0, feature_names, [])
            
            # Convert path to readable format
            readable_path = ["Root"]
            for step in path:
                if step["type"] == "decision":
                    readable_path.append(
                        f"{step['feature']} {step['relation']} {step['threshold']:.4f} → {step['decision']}"
                    )
                else:
                    readable_path.append(f"Leaf: {step['value']}")
            
            tree_explanations.append({
                "tree_index": i,
                "decision_path": path,
                "readable_path": readable_path
            })
    
    prediction = model.predict(instance)[0]
    
    return {
        "model_type": model.__class__.__name__,
        "prediction": float(prediction) if not isinstance(prediction, (list, np.ndarray)) else prediction.tolist(),
        "tree_explanations": tree_explanations,
        "trees_analyzed": len(tree_explanations),
        "total_trees": len(model.estimators_) if hasattr(model, "estimators_") else 1
    }
``` 

## A/B Testing

A/B testing enables controlled experimentation with model variants to determine which performs best in production.

### 1. Experiment Design

```python
def design_model_experiment(
    experiment_name: str,
    model_variants: Dict[str, Dict[str, Any]],
    evaluation_metrics: List[str],
    primary_metric: str,
    expected_effect_size: float,
    statistical_power: float = 0.8,
    significance_level: float = 0.05,
    stratification_variables: List[str] = None
) -> Dict[str, Any]:
    """Design A/B test experiment for model comparison.
    
    Args:
        experiment_name: Name of the experiment
        model_variants: Dictionary of model variants with metadata
        evaluation_metrics: List of metrics to evaluate
        primary_metric: Primary metric for decision making
        expected_effect_size: Expected minimum detectable effect
        statistical_power: Target statistical power (1-β)
        significance_level: Significance level (α)
        stratification_variables: Optional variables for stratified assignment
        
    Returns:
        Complete experiment design
    """
    # Ensure at least one control and one treatment
    if len(model_variants) < 2:
        raise ValueError("Need at least two model variants (control and treatment)")
    
    # Determine which is control model
    control_model = next((k for k, v in model_variants.items() if v.get("is_control", False)), None)
    
    if not control_model:
        # If no model is explicitly marked as control, use the first one
        control_model = list(model_variants.keys())[0]
        model_variants[control_model]["is_control"] = True
    
    # Calculate required sample size
    sample_size = calculate_required_sample_size(
        effect_size=expected_effect_size,
        power=statistical_power,
        alpha=significance_level,
        test_type="two_tailed"
    )
    
    # Equal allocation by default
    variant_count = len(model_variants)
    default_allocation = 1.0 / variant_count
    
    # Set default allocation if not provided
    for variant_id, variant_config in model_variants.items():
        if "traffic_allocation" not in variant_config:
            variant_config["traffic_allocation"] = default_allocation
    
    # Ensure allocations sum to 1.0
    total_allocation = sum(v["traffic_allocation"] for v in model_variants.values())
    if abs(total_allocation - 1.0) > 0.001:
        # Normalize allocations
        for variant_config in model_variants.values():
            variant_config["traffic_allocation"] = variant_config["traffic_allocation"] / total_allocation
    
    # Generate full experiment design
    experiment_design = {
        "name": experiment_name,
        "variants": model_variants,
        "control_variant": control_model,
        "metrics": {
            "primary": primary_metric,
            "secondary": [m for m in evaluation_metrics if m != primary_metric]
        },
        "statistical_design": {
            "expected_effect_size": expected_effect_size,
            "power": statistical_power,
            "significance_level": significance_level,
            "required_sample_size": {
                "total": sample_size,
                "per_variant": {
                    variant_id: int(sample_size * config["traffic_allocation"])
                    for variant_id, config in model_variants.items()
                }
            }
        },
        "duration_estimate": estimate_experiment_duration(
            sample_size=sample_size,
            daily_traffic_estimate=5000  # Replace with actual estimate
        ),
        "stratification": stratification_variables
    }
    
    return experiment_design

def calculate_required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = "two_tailed"
) -> int:
    """Calculate required sample size for experiment.
    
    Args:
        effect_size: Minimum detectable effect size
        power: Statistical power (1-β)
        alpha: Significance level (α)
        test_type: "one_tailed" or "two_tailed"
        
    Returns:
        Required sample size per variant
    """
    # Z-scores for alpha and beta
    z_alpha = stats.norm.ppf(1 - alpha/2 if test_type == "two_tailed" else 1 - alpha)
    z_beta = stats.norm.ppf(power)
    
    # Basic sample size calculation for comparison of proportions
    # Based on the formula: n = 2 * (z_alpha + z_beta)^2 * p * (1-p) / effect_size^2
    # Where p is the baseline conversion rate (assuming 0.5 for most conservative estimate)
    p = 0.5
    n = 2 * (z_alpha + z_beta)**2 * p * (1-p) / (effect_size**2)
    
    # Add 10% buffer for potential data loss
    n = n * 1.1
    
    return math.ceil(n)

def estimate_experiment_duration(
    sample_size: int,
    daily_traffic_estimate: int,
    traffic_allocation_to_experiment: float = 1.0
) -> Dict[str, Any]:
    """Estimate experiment duration based on traffic.
    
    Args:
        sample_size: Required sample size
        daily_traffic_estimate: Estimated daily traffic
        traffic_allocation_to_experiment: Portion of traffic allocated to experiment
        
    Returns:
        Duration estimate in days and samples per day
    """
    # Calculate daily samples
    daily_samples = daily_traffic_estimate * traffic_allocation_to_experiment
    
    # Calculate days needed
    days_needed = math.ceil(sample_size / daily_samples)
    
    return {
        "daily_samples_estimate": daily_samples,
        "days_needed": days_needed,
        "recommended_duration_days": max(14, days_needed)  # At least 2 weeks
    }
```

### 2. Traffic Allocation

The A/B testing system uses a consistent hashing mechanism to allocate traffic to variants:

```python
class ExperimentAssigner:
    """Assign users/requests to experiment variants."""
    
    def __init__(
        self,
        experiment_config: Dict[str, Any],
        experiment_store: ExperimentStore
    ):
        """Initialize experiment assigner.
        
        Args:
            experiment_config: Experiment configuration
            experiment_store: Store for experiment state
        """
        self.experiment_id = experiment_config["id"]
        self.variants = experiment_config["variants"]
        self.stratification = experiment_config.get("stratification")
        self.store = experiment_store
        
        # Parse variant allocations
        self.allocations = [
            (variant_id, config["traffic_allocation"])
            for variant_id, config in self.variants.items()
        ]
        
        # Create cumulative allocation thresholds
        self.thresholds = []
        cumulative = 0.0
        for variant_id, allocation in self.allocations:
            cumulative += allocation
            self.thresholds.append((variant_id, cumulative))
    
    def assign_variant(
        self,
        entity_id: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Assign entity to experiment variant.
        
        Args:
            entity_id: ID of entity to assign (user ID, session ID, etc.)
            context: Optional context for stratification
            
        Returns:
            Assigned variant ID
        """
        # Check if already assigned
        existing = self.store.get_assignment(self.experiment_id, entity_id)
        if existing:
            return existing
        
        # Handle stratification if needed
        if self.stratification and context:
            variant = self._assign_with_stratification(entity_id, context)
        else:
            variant = self._assign_random(entity_id)
        
        # Store assignment
        self.store.save_assignment(
            experiment_id=self.experiment_id,
            entity_id=entity_id,
            variant_id=variant,
            context=context
        )
        
        return variant
    
    def _assign_random(self, entity_id: str) -> str:
        """Assign variant using random hash of entity ID.
        
        Args:
            entity_id: Entity ID to assign
            
        Returns:
            Assigned variant ID
        """
        # Create hash value from experiment ID and entity ID for stability
        hash_input = f"{self.experiment_id}:{entity_id}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        
        # Convert hash to float in [0, 1)
        hash_float = int(hash_value, 16) / 2**128
        
        # Find variant based on thresholds
        for variant_id, threshold in self.thresholds:
            if hash_float < threshold:
                return variant_id
        
        # Fallback to last variant
        return self.thresholds[-1][0]
    
    def _assign_with_stratification(
        self,
        entity_id: str,
        context: Dict[str, Any]
    ) -> str:
        """Assign variant with stratification.
        
        Args:
            entity_id: Entity ID to assign
            context: Context for stratification
            
        Returns:
            Assigned variant ID
        """
        # Extract stratification variables
        strata_values = []
        for var in self.stratification:
            if var in context:
                strata_values.append(str(context[var]))
            else:
                strata_values.append("unknown")
        
        # Create stratum ID
        stratum_id = ":".join(strata_values)
        
        # Count existing assignments per variant in this stratum
        counts = self.store.get_stratum_counts(
            experiment_id=self.experiment_id,
            stratum_id=stratum_id
        )
        
        # Calculate target counts based on allocations
        total = sum(counts.values())
        targets = {
            variant_id: allocation * (total + 1)
            for variant_id, allocation in self.allocations
        }
        
        # Find variant with largest deficit
        deficits = {
            variant_id: targets.get(variant_id, 0) - counts.get(variant_id, 0)
            for variant_id in self.variants
        }
        
        # Assign to variant with largest deficit
        return max(deficits.items(), key=lambda x: x[1])[0]
```

### 3. Statistical Analysis

The experiment analysis module provides robust statistical inference:

```python
def analyze_experiment(
    experiment_id: str,
    results_data: pd.DataFrame,
    metrics: List[str],
    primary_metric: str = None,
    treatment_column: str = "variant",
    control_label: str = "control",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """Analyze experiment results.
    
    Args:
        experiment_id: Experiment identifier
        results_data: DataFrame with experiment results
        metrics: List of metrics to analyze
        primary_metric: Primary evaluation metric
        treatment_column: Column for variant labels
        control_label: Label for control group
        confidence_level: Statistical confidence level
        
    Returns:
        Complete analysis with metrics and recommendations
    """
    if primary_metric is None and metrics:
        primary_metric = metrics[0]
    
    # Identify treatments vs control
    treatment_variants = [
        v for v in results_data[treatment_column].unique() 
        if v != control_label
    ]
    
    # Overall results
    results = {
        "experiment_id": experiment_id,
        "analysis_timestamp": datetime.now().isoformat(),
        "sample_sizes": {
            variant: int(results_data[results_data[treatment_column] == variant].shape[0])
            for variant in results_data[treatment_column].unique()
        },
        "metrics": {},
        "significance": {}
    }
    
    # Analyze each metric
    for metric in metrics:
        metric_results = analyze_metric(
            results_data,
            metric=metric,
            treatment_column=treatment_column,
            control_label=control_label,
            confidence_level=confidence_level
        )
        
        results["metrics"][metric] = metric_results
        
        # Flag significance for primary metric
        if metric == primary_metric:
            for variant, stats in metric_results["comparisons"].items():
                is_significant = not (
                    stats["confidence_interval"][0] <= 0 <= stats["confidence_interval"][1]
                )
                direction = "increase" if stats["absolute_difference"] > 0 else "decrease"
                
                results["significance"][variant] = {
                    "is_significant": is_significant,
                    "direction": direction if is_significant else "no change",
                    "p_value": stats["p_value"],
                    "relative_change": stats["relative_difference"]
                }
    
    # Determine winner
    if primary_metric in results["metrics"]:
        winner = determine_winner(
            results["metrics"][primary_metric],
            results["sample_sizes"],
            min_confidence=confidence_level
        )
        results["winner"] = winner
    
    # Generate recommendations
    results["recommendations"] = generate_recommendations(results)
    
    return results

def analyze_metric(
    results_data: pd.DataFrame,
    metric: str,
    treatment_column: str = "variant",
    control_label: str = "control",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """Analyze a specific metric from experiment data.
    
    Args:
        results_data: DataFrame with experiment results
        metric: Metric to analyze
        treatment_column: Column for variant labels
        control_label: Label for control group
        confidence_level: Statistical confidence level
        
    Returns:
        Analysis for this metric
    """
    # Get control data
    control_data = results_data[results_data[treatment_column] == control_label][metric]
    
    if len(control_data) == 0:
        return {"error": "No control data found"}
    
    # Calculate control statistics
    control_mean = control_data.mean()
    control_std = control_data.std()
    control_se = control_std / math.sqrt(len(control_data))
    control_sample_size = len(control_data)
    
    # Overall statistics for the metric
    result = {
        "control_statistics": {
            "mean": float(control_mean),
            "std": float(control_std),
            "sample_size": control_sample_size,
            "standard_error": float(control_se)
        },
        "comparisons": {},
        "variant_statistics": {}
    }
    
    # Calculate for each treatment variant
    for variant in results_data[treatment_column].unique():
        if variant == control_label:
            continue
        
        # Treatment data
        treatment_data = results_data[results_data[treatment_column] == variant][metric]
        
        if len(treatment_data) == 0:
            continue
        
        # Treatment statistics
        treatment_mean = treatment_data.mean()
        treatment_std = treatment_data.std()
        treatment_se = treatment_std / math.sqrt(len(treatment_data))
        treatment_sample_size = len(treatment_data)
        
        result["variant_statistics"][variant] = {
            "mean": float(treatment_mean),
            "std": float(treatment_std),
            "sample_size": treatment_sample_size,
            "standard_error": float(treatment_se)
        }
        
        # Calculate differences
        absolute_diff = treatment_mean - control_mean
        relative_diff = absolute_diff / control_mean if control_mean != 0 else float('inf')
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            treatment_data, 
            control_data,
            equal_var=False  # Welch's t-test
        )
        
        # Calculate confidence interval for the difference
        pooled_se = math.sqrt(treatment_se**2 + control_se**2)
        z_value = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_value * pooled_se
        
        ci_lower = absolute_diff - margin_of_error
        ci_upper = absolute_diff + margin_of_error
        
        result["comparisons"][variant] = {
            "absolute_difference": float(absolute_diff),
            "relative_difference": float(relative_diff),
            "p_value": float(p_value),
            "confidence_interval": [float(ci_lower), float(ci_upper)],
            "significant_difference": p_value < (1 - confidence_level)
        }
    
    return result

def determine_winner(
    metric_results: Dict[str, Any],
    sample_sizes: Dict[str, int],
    min_confidence: float = 0.95,
    min_relative_improvement: float = 0.01  # 1% minimum practical improvement
) -> Dict[str, Any]:
    """Determine winning variant based on metric results.
    
    Args:
        metric_results: Results for a specific metric
        sample_sizes: Sample sizes for each variant
        min_confidence: Minimum confidence level to declare winner
        min_relative_improvement: Minimum relative improvement to consider
        
    Returns:
        Winner determination results
    """
    if "comparisons" not in metric_results:
        return {"status": "no_data"}
    
    # Find variants with significant positive improvement
    significant_winners = []
    
    for variant, comparison in metric_results["comparisons"].items():
        # Check if significant and positive
        is_significant = comparison["p_value"] < (1 - min_confidence)
        is_improvement = comparison["relative_difference"] > min_relative_improvement
        has_enough_samples = sample_sizes.get(variant, 0) >= 100  # Minimum sample threshold
        
        if is_significant and is_improvement and has_enough_samples:
            significant_winners.append({
                "variant": variant,
                "relative_improvement": comparison["relative_difference"],
                "p_value": comparison["p_value"],
                "sample_size": sample_sizes.get(variant, 0)
            })
    
    # Sort winners by improvement
    if significant_winners:
        sorted_winners = sorted(
            significant_winners,
            key=lambda x: x["relative_improvement"],
            reverse=True
        )
        
        return {
            "status": "winner_found",
            "winner": sorted_winners[0]["variant"],
            "improvement": sorted_winners[0]["relative_improvement"],
            "p_value": sorted_winners[0]["p_value"],
            "all_winners": sorted_winners
        }
    else:
        # Check if we have enough samples to detect the minimum effect
        control_size = sample_sizes.get("control", 0)
        
        for variant, comparison in metric_results["comparisons"].items():
            variant_size = sample_sizes.get(variant, 0)
            
            # Calculate minimum detectable effect with current sample sizes
            min_detectable_effect = calculate_min_detectable_effect(
                control_size, variant_size, confidence=min_confidence
            )
            
            # If observed effect is below MDE but sample size is adequate
            if (abs(comparison["relative_difference"]) < min_detectable_effect and
                variant_size >= 1000 and control_size >= 1000):
                return {
                    "status": "no_practical_difference",
                    "variant": variant,
                    "observed_difference": comparison["relative_difference"],
                    "min_detectable_effect": min_detectable_effect
                }
        
        # Otherwise inconclusive
        return {
            "status": "inconclusive",
            "reason": "insufficient_data"
        }

def calculate_min_detectable_effect(
    control_size: int,
    variant_size: int,
    confidence: float = 0.95,
    power: float = 0.8
) -> float:
    """Calculate minimum detectable effect based on sample sizes.
    
    Args:
        control_size: Control group sample size
        variant_size: Variant group sample size
        confidence: Statistical confidence level
        power: Statistical power
        
    Returns:
        Minimum detectable effect as a proportion
    """
    # Z-scores for alpha and beta
    z_alpha = stats.norm.ppf(1 - (1 - confidence)/2)
    z_beta = stats.norm.ppf(power)
    
    # Pooled standard error factor
    se_factor = math.sqrt(1/control_size + 1/variant_size)
    
    # Assuming p=0.5 for conservative estimate (maximum variance)
    p = 0.5
    variance = p * (1 - p)
    
    # Minimum detectable effect
    mde = (z_alpha + z_beta) * math.sqrt(2 * variance) * se_factor
    
    return mde

def generate_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate human-readable recommendations based on experiment results.
    
    Args:
        results: Complete experiment results
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if "winner" not in results:
        recommendations.append("Unable to determine a winner. Check experiment configuration.")
        return recommendations
    
    winner_status = results["winner"]["status"]
    
    if winner_status == "winner_found":
        winner = results["winner"]["winner"]
        improvement = results["winner"]["improvement"] * 100  # Convert to percentage
        
        recommendations.append(
            f"Implement variant '{winner}' which showed a {improvement:.2f}% improvement "
            f"with {results['winner']['p_value']:.4f} p-value."
        )
        
        # Additional recommendations for implementation
        recommendations.append(
            "Consider implementing a gradual rollout starting with 10% of traffic, "
            "monitoring performance closely for the first 48 hours."
        )
        
    elif winner_status == "no_practical_difference":
        recommendations.append(
            "No statistically significant or practical difference detected. "
            "Recommend keeping the control variant as it's simpler."
        )
        
        recommendations.append(
            "Consider more ambitious changes in future experiments to achieve "
            "the desired improvement threshold."
        )
        
    elif winner_status == "inconclusive":
        # Additional insights for inconclusive result
        small_samples = [
            variant for variant, size in results["sample_sizes"].items() 
            if size < 500
        ]
        
        if small_samples:
            samples_str = ", ".join(small_samples)
            recommendations.append(
                f"Insufficient sample size for variants: {samples_str}. "
                f"Continue the experiment to gather more data."
            )
        else:
            recommendations.append(
                "Results are inconclusive. Consider extending the experiment duration "
                "or adjusting the minimum detectable effect threshold."
            )
    
    # Check if we need more data overall
    if sum(results["sample_sizes"].values()) < 1000:
        recommendations.append(
            "The overall sample size is small. Consider running the experiment longer "
            "to increase statistical power."
        )
    
    return recommendations
```

### 4. Experiment Management

The experiment lifecycle can be managed through a dedicated service:

```python
class ExperimentService:
    """Service for managing model experiments."""
    
    def __init__(
        self,
        experiment_store: ExperimentStore,
        model_registry: ModelRegistry,
        metrics_service: MetricsService
    ):
        """Initialize experiment service.
        
        Args:
            experiment_store: Storage for experiment configurations
            model_registry: Registry for model variants
            metrics_service: Service for collecting metrics
        """
        self.store = experiment_store
        self.registry = model_registry
        self.metrics = metrics_service
        self.active_experiments = {}
    
    def create_experiment(
        self,
        experiment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create and configure a new experiment.
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            Created experiment details with ID
        """
        # Validate configuration
        self._validate_experiment_config(experiment_config)
        
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Add metadata
        experiment = experiment_config.copy()
        experiment["id"] = experiment_id
        experiment["status"] = "draft"
        experiment["created_at"] = datetime.now().isoformat()
        experiment["updated_at"] = experiment["created_at"]
        
        # Store experiment
        self.store.save_experiment(experiment)
        
        return experiment
    
    def _validate_experiment_config(self, config: Dict[str, Any]):
        """Validate experiment configuration.
        
        Args:
            config: Experiment configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["name", "variants", "metrics"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(config["variants"], dict) or len(config["variants"]) < 2:
            raise ValueError("Experiment must have at least two variants")
        
        if not isinstance(config["metrics"], dict) or "primary" not in config["metrics"]:
            raise ValueError("Experiment must specify a primary metric")
    
    def start_experiment(
        self,
        experiment_id: str,
        start_time: datetime = None
    ) -> Dict[str, Any]:
        """Start an experiment.
        
        Args:
            experiment_id: Experiment identifier
            start_time: Optional explicit start time
            
        Returns:
            Updated experiment details
        """
        # Load experiment
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        if experiment["status"] not in ["draft", "paused"]:
            raise ValueError(f"Cannot start experiment with status: {experiment['status']}")
        
        # Update experiment status
        now = start_time or datetime.now()
        updates = {
            "status": "active",
            "updated_at": now.isoformat()
        }
        
        if experiment["status"] == "draft":
            updates["started_at"] = now.isoformat()
        
        # Apply updates
        self.store.update_experiment(experiment_id, updates)
        
        # Initialize experiment in memory
        updated_experiment = self.store.get_experiment(experiment_id)
        self.active_experiments[experiment_id] = ExperimentAssigner(
            updated_experiment, self.store
        )
        
        # Set up metrics collection
        variant_ids = list(updated_experiment["variants"].keys())
        self.metrics.register_experiment(
            experiment_id=experiment_id,
            variants=variant_ids,
            metrics=self._get_all_metrics(updated_experiment)
        )
        
        return updated_experiment
    
    def _get_all_metrics(self, experiment: Dict[str, Any]) -> List[str]:
        """Extract all metrics from experiment config.
        
        Args:
            experiment: Experiment configuration
            
        Returns:
            List of all metrics
        """
        metrics = []
        if "metrics" in experiment:
            if "primary" in experiment["metrics"]:
                metrics.append(experiment["metrics"]["primary"])
            if "secondary" in experiment["metrics"]:
                metrics.extend(experiment["metrics"]["secondary"])
        return list(set(metrics))  # Deduplicate
    
    def stop_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
        results: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Stop an active experiment.
        
        Args:
            experiment_id: Experiment identifier
            status: Final status ("completed", "aborted", etc.)
            results: Optional final results to store
            
        Returns:
            Updated experiment details
        """
        # Load experiment
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        if experiment["status"] != "active":
            raise ValueError(f"Cannot stop experiment with status: {experiment['status']}")
        
        # Update experiment status
        now = datetime.now()
        updates = {
            "status": status,
            "updated_at": now.isoformat(),
            "ended_at": now.isoformat()
        }
        
        if results:
            updates["results"] = results
        
        # Apply updates
        self.store.update_experiment(experiment_id, updates)
        
        # Remove from active experiments
        if experiment_id in self.active_experiments:
            del self.active_experiments[experiment_id]
        
        # Stop metrics collection
        self.metrics.unregister_experiment(experiment_id)
        
        return self.store.get_experiment(experiment_id)
    
    def assign_variant(
        self,
        experiment_id: str,
        entity_id: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Assign entity to experiment variant.
        
        Args:
            experiment_id: Experiment identifier
            entity_id: Entity to assign (user ID, etc.)
            context: Optional context for stratification
            
        Returns:
            Assigned variant ID
        """
        # Check if experiment is active
        if experiment_id not in self.active_experiments:
            experiment = self.store.get_experiment(experiment_id)
            if not experiment or experiment["status"] != "active":
                raise ValueError(f"Experiment not active: {experiment_id}")
            
            # Initialize experiment
            self.active_experiments[experiment_id] = ExperimentAssigner(
                experiment, self.store
            )
        
        # Assign variant
        assigner = self.active_experiments[experiment_id]
        variant_id = assigner.assign_variant(entity_id, context)
        
        return variant_id
    
    def log_exposure(
        self,
        experiment_id: str,
        entity_id: str,
        variant_id: str,
        timestamp: datetime = None
    ):
        """Log experiment exposure for entity.
        
        Args:
            experiment_id: Experiment identifier
            entity_id: Exposed entity ID
            variant_id: Assigned variant ID
            timestamp: Optional timestamp
        """
        exposure = {
            "experiment_id": experiment_id,
            "entity_id": entity_id,
            "variant_id": variant_id,
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        self.store.log_exposure(exposure)
    
    def log_conversion(
        self,
        experiment_id: str,
        entity_id: str,
        metric: str,
        value: float,
        metadata: Dict[str, Any] = None,
        timestamp: datetime = None
    ):
        """Log conversion event for experiment.
        
        Args:
            experiment_id: Experiment identifier
            entity_id: Entity ID
            metric: Conversion metric name
            value: Metric value
            metadata: Optional conversion metadata
            timestamp: Optional timestamp
        """
        # Get assigned variant
        assignment = self.store.get_assignment(experiment_id, entity_id)
        if not assignment:
            return  # No assignment, cannot attribute conversion
        
        conversion = {
            "experiment_id": experiment_id,
            "entity_id": entity_id,
            "variant_id": assignment,
            "metric": metric,
            "value": value,
            "metadata": metadata,
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        self.store.log_conversion(conversion)
    
    def analyze_experiment_results(
        self,
        experiment_id: str,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Analyze current results for experiment.
        
        Args:
            experiment_id: Experiment identifier
            confidence_level: Statistical confidence level
            
        Returns:
            Analysis of experiment results
        """
        # Get experiment config
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        # Get conversion data
        conversion_data = self.store.get_experiment_conversions(experiment_id)
        
        # Transform to DataFrame for analysis
        df = pd.DataFrame(conversion_data)
        
        if df.empty:
            return {"status": "no_data"}
        
        # Identify metrics for analysis
        metrics = self._get_all_metrics(experiment)
        primary_metric = experiment["metrics"].get("primary")
        
        # Run analysis
        return analyze_experiment(
            experiment_id=experiment_id,
            results_data=df,
            metrics=metrics,
            primary_metric=primary_metric,
            treatment_column="variant_id",
            control_label="control",
            confidence_level=confidence_level
        )
``` 

## Benchmarking

Benchmarking provides standardized comparisons across different model implementations and versions.

### 1. Performance Benchmarks

Standardized benchmarks enable consistent model comparison:

```python
def benchmark_model(
    model: Any,
    benchmark_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    metrics: List[str] = None,
    problem_type: str = "regression",
    include_timing: bool = True
) -> Dict[str, Any]:
    """Run standardized benchmarks on model.
    
    Args:
        model: Trained model to benchmark
        benchmark_datasets: Dict mapping dataset names to (X, y) tuples
        metrics: List of metrics to compute
        problem_type: "regression" or "classification"
        include_timing: Whether to include timing metrics
        
    Returns:
        Dictionary with benchmark results
    """
    # Default metrics based on problem type
    if metrics is None:
        metrics = ["rmse", "mae", "r2"] if problem_type == "regression" else ["accuracy", "f1", "auc"]
    
    results = {
        "model_info": {
            "model_type": type(model).__name__,
            "timestamp": datetime.now().isoformat()
        },
        "datasets": {},
        "summary": {}
    }
    
    all_metric_values = {metric: [] for metric in metrics}
    
    # Benchmark on each dataset
    for dataset_name, (X, y) in benchmark_datasets.items():
        # Run predictions
        start_time = time.time()
        y_pred = model.predict(X)
        prediction_time = time.time() - start_time
        
        # Get prediction probabilities for classification
        y_proba = None
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except:
                pass
        
        # Calculate metrics
        dataset_metrics = {}
        
        for metric in metrics:
            value = calculate_metric(
                y_true=y,
                y_pred=y_pred,
                y_proba=y_proba,
                metric=metric,
                problem_type=problem_type
            )
            dataset_metrics[metric] = value
            all_metric_values[metric].append(value)
        
        # Add timing metrics
        if include_timing:
            # Prediction time per sample
            dataset_metrics["prediction_time_ms"] = (prediction_time * 1000) / len(X)
            
            # Memory usage if possible
            try:
                memory_usage = get_model_memory_usage(model)
                dataset_metrics["memory_usage_mb"] = memory_usage
            except:
                pass
        
        # Store dataset results
        results["datasets"][dataset_name] = {
            "metrics": dataset_metrics,
            "samples": len(X),
            "features": X.shape[1]
        }
    
    # Calculate summary statistics
    for metric, values in all_metric_values.items():
        results["summary"][metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
    
    return results

def calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    metric: str = "rmse",
    problem_type: str = "regression"
) -> float:
    """Calculate a specific evaluation metric.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_proba: Predicted probabilities (for classification)
        metric: Metric name
        problem_type: "regression" or "classification"
        
    Returns:
        Metric value
    """
    if problem_type == "regression":
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        elif metric == "r2":
            return float(r2_score(y_true, y_pred))
        elif metric == "mape":
            return float(mean_absolute_percentage_error(y_true, y_pred))
        elif metric == "max_error":
            return float(max_error(y_true, y_pred))
    else:
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        elif metric == "f1":
            return float(f1_score(y_true, y_pred, average="weighted"))
        elif metric == "precision":
            return float(precision_score(y_true, y_pred, average="weighted"))
        elif metric == "recall":
            return float(recall_score(y_true, y_pred, average="weighted"))
        elif metric == "auc" and y_proba is not None:
            if y_proba.shape[1] == 2:  # Binary classification
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            else:  # Multi-class
                return float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    
    raise ValueError(f"Unsupported metric: {metric} for problem type: {problem_type}")

def get_model_memory_usage(model: Any) -> float:
    """Estimate memory usage of a model in MB.
    
    Args:
        model: Model to analyze
        
    Returns:
        Estimated memory usage in MB
    """
    import sys
    import pickle
    
    # Serialize model
    serialized_model = pickle.dumps(model)
    
    # Get size in MB
    return len(serialized_model) / (1024 * 1024)
```

### 2. Competitor Benchmarks

Benchmark models against industry competitors and baselines:

```python
def create_benchmark_report(
    model_results: Dict[str, Dict[str, Any]],
    baseline_name: str = None,
    benchmark_date: datetime = None
) -> Dict[str, Any]:
    """Create comparative benchmark report across models.
    
    Args:
        model_results: Dict mapping model names to benchmark results
        baseline_name: Optional baseline model for comparison
        benchmark_date: Optional timestamp for benchmark
        
    Returns:
        Benchmark comparison report
    """
    date = benchmark_date or datetime.now()
    baseline = baseline_name or next(iter(model_results.keys()))
    
    report = {
        "timestamp": date.isoformat(),
        "models_compared": list(model_results.keys()),
        "baseline_model": baseline,
        "datasets": {},
        "metrics_compared": [],
        "overall_ranking": {}
    }
    
    # Get common datasets and metrics
    all_datasets = set()
    all_metrics = set()
    
    for model_name, results in model_results.items():
        for dataset in results["datasets"]:
            all_datasets.add(dataset)
        
        for dataset, dataset_results in results["datasets"].items():
            for metric in dataset_results["metrics"]:
                if metric not in ["prediction_time_ms", "memory_usage_mb"]:
                    all_metrics.add(metric)
    
    report["metrics_compared"] = list(all_metrics)
    
    # Compare each dataset
    for dataset in all_datasets:
        dataset_comparison = {
            "models": {},
            "relative_performance": {},
            "ranking": {}
        }
        
        # Get baseline metrics if available
        baseline_metrics = {}
        if baseline in model_results and dataset in model_results[baseline]["datasets"]:
            baseline_metrics = model_results[baseline]["datasets"][dataset]["metrics"]
        
        # Collect metrics for each model
        for model_name, results in model_results.items():
            if dataset in results["datasets"]:
                dataset_comparison["models"][model_name] = results["datasets"][dataset]["metrics"]
        
        # Calculate relative performance and rankings for each metric
        for metric in all_metrics:
            if metric in baseline_metrics:
                relative_perf = {}
                metric_values = []
                
                for model_name, metrics in dataset_comparison["models"].items():
                    if metric in metrics:
                        # Calculate relative improvement over baseline
                        baseline_value = baseline_metrics[metric]
                        model_value = metrics[metric]
                        
                        # Determine if higher is better
                        higher_is_better = metric in ["accuracy", "f1", "precision", "recall", "auc", "r2"]
                        
                        if baseline_value != 0:
                            if higher_is_better:
                                rel_improvement = (model_value - baseline_value) / abs(baseline_value)
                            else:
                                rel_improvement = (baseline_value - model_value) / abs(baseline_value)
                            
                            relative_perf[model_name] = rel_improvement
                        
                        # Store for ranking
                        metric_values.append((model_name, model_value, higher_is_better))
                
                # Calculate rankings
                if higher_is_better:
                    ranked_models = sorted(metric_values, key=lambda x: x[1], reverse=True)
                else:
                    ranked_models = sorted(metric_values, key=lambda x: x[1])
                
                dataset_comparison["ranking"][metric] = [model for model, _, _ in ranked_models]
                dataset_comparison["relative_performance"][metric] = relative_perf
        
        report["datasets"][dataset] = dataset_comparison
    
    # Calculate overall rankings
    model_ranks = {model: 0 for model in model_results.keys()}
    rank_counts = {model: 0 for model in model_results.keys()}
    
    for dataset, dataset_comparison in report["datasets"].items():
        for metric, rankings in dataset_comparison["ranking"].items():
            for rank, model in enumerate(rankings):
                model_ranks[model] += rank
                rank_counts[model] += 1
    
    # Average rank across all datasets and metrics
    for model in model_ranks:
        if rank_counts[model] > 0:
            report["overall_ranking"][model] = model_ranks[model] / rank_counts[model]
    
    # Sort by average rank
    sorted_ranking = sorted(report["overall_ranking"].items(), key=lambda x: x[1])
    report["ranking_summary"] = [{"model": model, "avg_rank": rank} for model, rank in sorted_ranking]
    
    return report
```

### 3. Standardized Test Datasets

The benchmarking system maintains canonical test datasets for consistent comparisons:

```python
class BenchmarkDatasetRegistry:
    """Registry for standardized benchmark datasets."""
    
    def __init__(self, storage_path: str):
        """Initialize dataset registry.
        
        Args:
            storage_path: Path to dataset storage
        """
        self.storage_path = storage_path
        self.dataset_metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata from storage."""
        metadata_path = os.path.join(self.storage_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {"datasets": {}}
    
    def _save_metadata(self):
        """Save dataset metadata to storage."""
        metadata_path = os.path.join(self.storage_path, "metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(self.dataset_metadata, f, indent=2)
    
    def register_dataset(
        self,
        name: str,
        X: pd.DataFrame,
        y: pd.Series,
        description: str = None,
        tags: List[str] = None,
        version: str = "1.0",
        source: str = None
    ):
        """Register a new benchmark dataset.
        
        Args:
            name: Dataset name
            X: Feature DataFrame
            y: Target Series
            description: Dataset description
            tags: List of tags for categorization
            version: Dataset version
            source: Source of the dataset
        """
        dataset_dir = os.path.join(self.storage_path, name, version)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save features and target
        X.to_parquet(os.path.join(dataset_dir, "features.parquet"))
        y.to_parquet(os.path.join(dataset_dir, "target.parquet"))
        
        # Update metadata
        timestamp = datetime.now().isoformat()
        
        if name not in self.dataset_metadata["datasets"]:
            self.dataset_metadata["datasets"][name] = {
                "versions": {},
                "created_at": timestamp,
                "description": description,
                "tags": tags or [],
                "source": source
            }
        
        self.dataset_metadata["datasets"][name]["versions"][version] = {
            "created_at": timestamp,
            "feature_count": X.shape[1],
            "sample_count": len(X),
            "feature_names": list(X.columns),
            "target_type": str(y.dtype),
            "unique_target_values": len(y.unique()) if y.dtype.kind in "iufb" else None
        }
        
        # Save updated metadata
        self._save_metadata()
    
    def get_dataset(
        self,
        name: str,
        version: str = "latest"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Get benchmark dataset by name and version.
        
        Args:
            name: Dataset name
            version: Dataset version or "latest"
            
        Returns:
            Tuple of (features, target)
        """
        if name not in self.dataset_metadata["datasets"]:
            raise ValueError(f"Dataset not found: {name}")
        
        # Determine version to load
        available_versions = list(self.dataset_metadata["datasets"][name]["versions"].keys())
        if not available_versions:
            raise ValueError(f"No versions available for dataset: {name}")
        
        if version == "latest":
            version = sorted(available_versions)[-1]
        elif version not in available_versions:
            raise ValueError(f"Version {version} not found for dataset: {name}")
        
        # Load dataset
        dataset_dir = os.path.join(self.storage_path, name, version)
        X = pd.read_parquet(os.path.join(dataset_dir, "features.parquet"))
        y = pd.read_parquet(os.path.join(dataset_dir, "target.parquet")).squeeze()
        
        return X, y
    
    def list_datasets(self, tags: List[str] = None) -> List[Dict[str, Any]]:
        """List available benchmark datasets.
        
        Args:
            tags: Optional filter by tags
            
        Returns:
            List of dataset metadata
        """
        datasets = []
        
        for name, metadata in self.dataset_metadata["datasets"].items():
            # Filter by tags if specified
            if tags and not all(tag in metadata["tags"] for tag in tags):
                continue
            
            # Get latest version info
            versions = list(metadata["versions"].keys())
            latest_version = sorted(versions)[-1] if versions else None
            
            if latest_version:
                version_info = metadata["versions"][latest_version]
                
                datasets.append({
                    "name": name,
                    "description": metadata.get("description"),
                    "tags": metadata.get("tags", []),
                    "latest_version": latest_version,
                    "created_at": metadata.get("created_at"),
                    "samples": version_info.get("sample_count"),
                    "features": version_info.get("feature_count"),
                    "versions_available": len(versions)
                })
        
        return datasets
``` 

## Quality Gates

Quality gates enforce rigorous standards for models before production deployment.

### 1. Quality Gate Framework

The quality gate framework establishes pass/fail criteria for model releases:

```python
class ModelQualityGate:
    """Quality gate for model evaluation and validation."""
    
    def __init__(
        self,
        name: str,
        criteria: List[Dict[str, Any]],
        description: str = None,
        severity: str = "blocking"
    ):
        """Initialize quality gate.
        
        Args:
            name: Gate name
            criteria: List of criteria dictionaries
            description: Optional description
            severity: "blocking", "warning", or "info"
        """
        self.name = name
        self.criteria = criteria
        self.description = description
        self.severity = severity
    
    def evaluate(
        self,
        metrics: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Evaluate if model passes the quality gate.
        
        Args:
            metrics: Dictionary of model metrics
            metadata: Optional model metadata
            
        Returns:
            Evaluation result
        """
        results = []
        metadata = metadata or {}
        
        # Evaluate each criterion
        for criterion in self.criteria:
            criterion_result = self._evaluate_criterion(criterion, metrics, metadata)
            results.append(criterion_result)
        
        # Determine overall pass/fail
        passed = all(r["passed"] for r in results)
        
        return {
            "gate_name": self.name,
            "passed": passed,
            "criteria_results": results,
            "criteria_passed": sum(1 for r in results if r["passed"]),
            "criteria_total": len(results),
            "severity": self.severity,
            "timestamp": datetime.now().isoformat()
        }
    
    def _evaluate_criterion(
        self,
        criterion: Dict[str, Any],
        metrics: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single criterion.
        
        Args:
            criterion: Criterion definition
            metrics: Model metrics
            metadata: Model metadata
            
        Returns:
            Criterion evaluation result
        """
        criterion_type = criterion.get("type", "metric_threshold")
        criterion_name = criterion.get("name", "unnamed_criterion")
        
        # Default result structure
        result = {
            "name": criterion_name,
            "type": criterion_type,
            "passed": False,
            "details": {}
        }
        
        if criterion_type == "metric_threshold":
            # Check if metric value meets threshold
            metric_name = criterion.get("metric")
            threshold = criterion.get("threshold")
            operator = criterion.get("operator", ">=")
            
            # Use deep path lookup to support nested metrics
            metric_value = self._get_nested_value(metrics, metric_name)
            
            if metric_value is None:
                result["details"] = {
                    "error": f"Metric {metric_name} not found in provided metrics"
                }
                return result
            
            # Perform comparison
            result["details"] = {
                "metric": metric_name,
                "threshold": threshold,
                "operator": operator,
                "actual_value": metric_value
            }
            
            result["passed"] = self._compare_values(metric_value, threshold, operator)
            
        elif criterion_type == "comparison":
            # Compare against baseline or competitor
            metric_name = criterion.get("metric")
            baseline_name = criterion.get("baseline")
            min_improvement = criterion.get("min_improvement", 0)
            
            # Get current and baseline values
            metric_value = self._get_nested_value(metrics, metric_name)
            baseline_value = self._get_nested_value(metrics, f"baselines.{baseline_name}.{metric_name}")
            
            if metric_value is None or baseline_value is None:
                result["details"] = {
                    "error": f"Could not find metric {metric_name} for current model or baseline {baseline_name}"
                }
                return result
            
            # Calculate improvement
            higher_is_better = criterion.get("higher_is_better", True)
            
            if higher_is_better:
                improvement = (metric_value - baseline_value) / abs(baseline_value) if baseline_value != 0 else float('inf')
                result["passed"] = improvement >= min_improvement
            else:
                improvement = (baseline_value - metric_value) / abs(baseline_value) if baseline_value != 0 else float('inf')
                result["passed"] = improvement >= min_improvement
            
            result["details"] = {
                "metric": metric_name,
                "baseline": baseline_name,
                "current_value": metric_value,
                "baseline_value": baseline_value,
                "improvement": improvement,
                "min_improvement_required": min_improvement,
                "higher_is_better": higher_is_better
            }
            
        elif criterion_type == "data_validation":
            # Check data quality
            data_property = criterion.get("property")
            validation_type = criterion.get("validation_type", "schema")
            
            # For simplicity, assume validation results are in metrics
            validation_result = self._get_nested_value(metrics, f"data_validation.{validation_type}.{data_property}")
            
            if validation_result is None:
                result["details"] = {
                    "error": f"Data validation result for {data_property} not found"
                }
                return result
            
            result["passed"] = validation_result.get("passed", False)
            result["details"] = validation_result
            
        elif criterion_type == "custom":
            # Custom evaluation function
            eval_func = criterion.get("function")
            
            if callable(eval_func):
                try:
                    custom_result = eval_func(metrics, metadata)
                    result["passed"] = custom_result.get("passed", False)
                    result["details"] = custom_result
                except Exception as e:
                    result["details"] = {"error": f"Custom evaluation failed: {str(e)}"}
            else:
                result["details"] = {"error": "Custom evaluation function not callable"}
        
        # Add criterion description if available
        if "description" in criterion:
            result["description"] = criterion["description"]
        
        return result
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value from nested dictionary using dot notation.
        
        Args:
            data: Dictionary to search in
            path: Path using dot notation
            
        Returns:
            Value or None if not found
        """
        parts = path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _compare_values(
        self,
        value: float,
        threshold: float,
        operator: str
    ) -> bool:
        """Compare values using the specified operator.
        
        Args:
            value: Value to compare
            threshold: Threshold to compare against
            operator: Comparison operator
            
        Returns:
            Comparison result
        """
        if operator == ">=":
            return value >= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            raise ValueError(f"Unsupported operator: {operator}")
```

### 2. Production Readiness Gates

A comprehensive set of quality gates ensures models are production-ready:

```python
def create_default_quality_gates() -> Dict[str, ModelQualityGate]:
    """Create default set of quality gates for model evaluation.
    
    Returns:
        Dictionary of quality gates
    """
    quality_gates = {}
    
    # Performance gate
    performance_gate = ModelQualityGate(
        name="performance",
        description="Ensures model meets minimum performance thresholds",
        severity="blocking",
        criteria=[
            {
                "name": "min_rmse",
                "type": "metric_threshold",
                "metric": "performance.rmse",
                "threshold": 0.15,
                "operator": "<=",
                "description": "RMSE must be below 0.15"
            },
            {
                "name": "min_r2",
                "type": "metric_threshold",
                "metric": "performance.r2",
                "threshold": 0.7,
                "operator": ">=",
                "description": "R² must be at least 0.7"
            },
            {
                "name": "performance_regression",
                "type": "comparison",
                "metric": "performance.rmse",
                "baseline": "previous_model",
                "min_improvement": -0.05,  # Allow up to 5% regression
                "higher_is_better": False,
                "description": "RMSE should not increase by more than 5% compared to previous model"
            }
        ]
    )
    quality_gates["performance"] = performance_gate
    
    # Data quality gate
    data_quality_gate = ModelQualityGate(
        name="data_quality",
        description="Validates input data meets quality standards",
        severity="blocking",
        criteria=[
            {
                "name": "schema_validation",
                "type": "data_validation",
                "property": "schema",
                "validation_type": "schema",
                "description": "Input data schema must match expected schema"
            },
            {
                "name": "missing_values",
                "type": "metric_threshold",
                "metric": "data_validation.missing_values_ratio",
                "threshold": 0.05,
                "operator": "<=",
                "description": "Missing values must not exceed 5% of total"
            },
            {
                "name": "outliers_ratio",
                "type": "metric_threshold",
                "metric": "data_validation.outliers_ratio",
                "threshold": 0.01,
                "operator": "<=",
                "description": "Outliers must not exceed 1% of total"
            }
        ]
    )
    quality_gates["data_quality"] = data_quality_gate
    
    # Fairness gate
    fairness_gate = ModelQualityGate(
        name="fairness",
        description="Ensures model meets fairness criteria",
        severity="blocking",
        criteria=[
            {
                "name": "demographic_parity",
                "type": "metric_threshold",
                "metric": "fairness.demographic_parity.max_disparity",
                "threshold": 0.1,
                "operator": "<=",
                "description": "Maximum demographic parity disparity must be below 0.1"
            },
            {
                "name": "equal_opportunity",
                "type": "metric_threshold",
                "metric": "fairness.equal_opportunity.max_disparity",
                "threshold": 0.1,
                "operator": "<=",
                "description": "Maximum equal opportunity disparity must be below 0.1"
            }
        ]
    )
    quality_gates["fairness"] = fairness_gate
    
    # Performance budget gate
    perf_budget_gate = ModelQualityGate(
        name="performance_budget",
        description="Ensures model meets computational performance requirements",
        severity="blocking",
        criteria=[
            {
                "name": "inference_time",
                "type": "metric_threshold",
                "metric": "performance_budget.avg_inference_time_ms",
                "threshold": 200,
                "operator": "<=",
                "description": "Average inference time must be below 200ms"
            },
            {
                "name": "memory_usage",
                "type": "metric_threshold",
                "metric": "performance_budget.memory_usage_mb",
                "threshold": 500,
                "operator": "<=",
                "description": "Memory usage must be below 500MB"
            },
            {
                "name": "model_size",
                "type": "metric_threshold",
                "metric": "performance_budget.model_size_mb",
                "threshold": 200,
                "operator": "<=",
                "description": "Model size must be below 200MB"
            }
        ]
    )
    quality_gates["performance_budget"] = perf_budget_gate
    
    # Explainability gate
    explainability_gate = ModelQualityGate(
        name="explainability",
        description="Ensures model meets explainability requirements",
        severity="warning",  # Warning only, not blocking
        criteria=[
            {
                "name": "feature_importance",
                "type": "metric_threshold",
                "metric": "explainability.feature_importance.features_90pct",
                "threshold": 20,
                "operator": "<=",
                "description": "90% of feature importance should be covered by at most 20 features"
            },
            {
                "name": "shap_values",
                "type": "custom",
                "function": lambda metrics, _: {
                    "passed": "explainability.shap_values" in metrics,
                    "details": {
                        "has_shap_values": "explainability.shap_values" in metrics
                    }
                },
                "description": "SHAP values should be available for model explanation"
            }
        ]
    )
    quality_gates["explainability"] = explainability_gate
    
    return quality_gates
```

### 3. Quality Gate Evaluation

The quality gate evaluator applies gates to model evaluation results:

```python
class QualityGateEvaluator:
    """Evaluate models against quality gates."""
    
    def __init__(
        self,
        quality_gates: Dict[str, ModelQualityGate] = None,
        db_connector: Optional[DatabaseConnector] = None
    ):
        """Initialize quality gate evaluator.
        
        Args:
            quality_gates: Dictionary of quality gates
            db_connector: Optional database connector for storing results
        """
        self.quality_gates = quality_gates or create_default_quality_gates()
        self.db = db_connector
    
    def evaluate_model(
        self,
        model_id: str,
        metrics: Dict[str, Any],
        metadata: Dict[str, Any] = None,
        gates_to_evaluate: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model against quality gates.
        
        Args:
            model_id: Model identifier
            metrics: Dictionary of model metrics
            metadata: Optional model metadata
            gates_to_evaluate: Optional list of gates to evaluate
            
        Returns:
            Evaluation results
        """
        # Determine gates to evaluate
        if gates_to_evaluate:
            gates = {k: self.quality_gates[k] for k in gates_to_evaluate if k in self.quality_gates}
        else:
            gates = self.quality_gates
        
        # Evaluate each gate
        results = {}
        for gate_name, gate in gates.items():
            results[gate_name] = gate.evaluate(metrics, metadata)
        
        # Determine overall pass/fail
        blocking_gates = [
            results[gate_name] for gate_name, gate in gates.items()
            if gate.severity == "blocking"
        ]
        
        passed_overall = all(gate["passed"] for gate in blocking_gates)
        
        # Create summary
        summary = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "passed": passed_overall,
            "gates_evaluated": len(results),
            "gates_passed": sum(1 for r in results.values() if r["passed"]),
            "blocking_gates_passed": sum(1 for g in blocking_gates if g["passed"]),
            "blocking_gates_total": len(blocking_gates),
            "results": results
        }
        
        # Store results if database is available
        if self.db:
            self._store_evaluation_results(model_id, summary)
        
        return summary
    
    def _store_evaluation_results(
        self,
        model_id: str,
        results: Dict[str, Any]
    ):
        """Store evaluation results in database.
        
        Args:
            model_id: Model identifier
            results: Evaluation results
        """
        # Flatten results for storage
        flattened = {
            "model_id": model_id,
            "timestamp": results["timestamp"],
            "passed_overall": results["passed"],
            "gates_evaluated": results["gates_evaluated"],
            "gates_passed": results["gates_passed"],
            "blocking_gates_passed": results["blocking_gates_passed"],
            "blocking_gates_total": results["blocking_gates_total"]
        }
        
        # Add pass/fail for each gate
        for gate_name, gate_result in results["results"].items():
            flattened[f"gate_{gate_name}_passed"] = gate_result["passed"]
        
        # Store in database
        self.db.insert("quality_gate_evaluations", flattened)
    
    def get_model_evaluation_history(
        self,
        model_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get history of quality gate evaluations for a model.
        
        Args:
            model_id: Model identifier
            limit: Maximum number of results to return
            
        Returns:
            List of evaluation results
        """
        if not self.db:
            return []
        
        query = """
        SELECT * FROM quality_gate_evaluations 
        WHERE model_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        return self.db.query(query, [model_id, limit])
    
    def compare_evaluations(
        self,
        current_eval: Dict[str, Any],
        previous_eval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare current evaluation to previous evaluation.
        
        Args:
            current_eval: Current evaluation results
            previous_eval: Previous evaluation results
            
        Returns:
            Comparison results
        """
        if not previous_eval:
            return {"status": "no_previous_evaluation"}
        
        comparison = {
            "current_timestamp": current_eval["timestamp"],
            "previous_timestamp": previous_eval["timestamp"],
            "overall_status_changed": current_eval["passed"] != previous_eval["passed"],
            "current_passed": current_eval["passed"],
            "previous_passed": previous_eval["passed"],
            "gates_compared": 0,
            "gates_improved": 0,
            "gates_regressed": 0,
            "gates_unchanged": 0,
            "gate_details": {}
        }
        
        # Compare each gate
        for gate_name in current_eval["results"].keys():
            if gate_name in previous_eval["results"]:
                current_gate = current_eval["results"][gate_name]
                previous_gate = previous_eval["results"][gate_name]
                
                current_passed = current_gate["passed"]
                previous_passed = previous_gate["passed"]
                
                gate_comparison = {
                    "current_passed": current_passed,
                    "previous_passed": previous_passed,
                    "status_changed": current_passed != previous_passed
                }
                
                # Determine if improved or regressed
                if current_passed and not previous_passed:
                    gate_comparison["change"] = "improved"
                    comparison["gates_improved"] += 1
                elif not current_passed and previous_passed:
                    gate_comparison["change"] = "regressed"
                    comparison["gates_regressed"] += 1
                else:
                    gate_comparison["change"] = "unchanged"
                    comparison["gates_unchanged"] += 1
                
                # Compare criteria if available
                if "criteria_results" in current_gate and "criteria_results" in previous_gate:
                    gate_comparison["criteria_comparison"] = self._compare_criteria(
                        current_gate["criteria_results"],
                        previous_gate["criteria_results"]
                    )
                
                comparison["gate_details"][gate_name] = gate_comparison
                comparison["gates_compared"] += 1
        
        return comparison
    
    def _compare_criteria(
        self,
        current_criteria: List[Dict[str, Any]],
        previous_criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare criteria results between evaluations.
        
        Args:
            current_criteria: Current criteria results
            previous_criteria: Previous criteria results
            
        Returns:
            Criteria comparison results
        """
        # Build lookup by name
        current_by_name = {c["name"]: c for c in current_criteria}
        previous_by_name = {c["name"]: c for c in previous_criteria}
        
        # Compare common criteria
        criteria_comparison = {
            "criteria_compared": 0,
            "criteria_improved": 0,
            "criteria_regressed": 0,
            "criteria_unchanged": 0,
            "details": {}
        }
        
        for name, current in current_by_name.items():
            if name in previous_by_name:
                previous = previous_by_name[name]
                
                current_passed = current["passed"]
                previous_passed = previous["passed"]
                
                comparison = {
                    "current_passed": current_passed,
                    "previous_passed": previous_passed,
                    "status_changed": current_passed != previous_passed
                }
                
                # Determine if improved or regressed
                if current_passed and not previous_passed:
                    comparison["change"] = "improved"
                    criteria_comparison["criteria_improved"] += 1
                elif not current_passed and previous_passed:
                    comparison["change"] = "regressed"
                    criteria_comparison["criteria_regressed"] += 1
                else:
                    comparison["change"] = "unchanged"
                    criteria_comparison["criteria_unchanged"] += 1
                
                # Add actual values if available
                if "details" in current and "details" in previous:
                    for key in ["actual_value", "metric_value", "current_value"]:
                        if key in current["details"] and key in previous["details"]:
                            comparison["current_value"] = current["details"][key]
                            comparison["previous_value"] = previous["details"][key]
                            break
                
                criteria_comparison["details"][name] = comparison
                criteria_comparison["criteria_compared"] += 1
        
        return criteria_comparison
``` 

## Evaluation Reports

Structured reports provide comprehensive insights into model performance and quality.

### 1. Report Generation

The report generator creates standardized evaluation reports:

```python
class ModelEvaluationReport:
    """Generate comprehensive evaluation reports for models."""
    
    def __init__(
        self,
        model_id: str,
        model_version: str,
        model_type: str,
        author: str,
        training_date: str,
        evaluation_date: str = None
    ):
        """Initialize model evaluation report.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            model_type: Type of model (classifier, regressor, etc.)
            author: Report author
            training_date: Date model was trained
            evaluation_date: Date of evaluation (defaults to current date)
        """
        self.model_id = model_id
        self.model_version = model_version
        self.model_type = model_type
        self.author = author
        self.training_date = training_date
        self.evaluation_date = evaluation_date or datetime.now().strftime("%Y-%m-%d")
        
        # Initialize report sections
        self.sections = {}
        
        # Add metadata section
        self.add_metadata_section()
    
    def add_metadata_section(self):
        """Add metadata section to report."""
        self.sections["metadata"] = {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "author": self.author,
            "training_date": self.training_date,
            "evaluation_date": self.evaluation_date,
            "report_generated": datetime.now().isoformat()
        }
    
    def add_performance_metrics(
        self,
        metrics: Dict[str, Any],
        dataset_name: str,
        dataset_description: str = None
    ):
        """Add performance metrics section to report.
        
        Args:
            metrics: Dictionary of performance metrics
            dataset_name: Name of evaluation dataset
            dataset_description: Optional description of dataset
        """
        if "performance" not in self.sections:
            self.sections["performance"] = []
        
        performance_entry = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        if dataset_description:
            performance_entry["dataset_description"] = dataset_description
        
        self.sections["performance"].append(performance_entry)
    
    def add_fairness_assessment(
        self,
        fairness_metrics: Dict[str, Any],
        protected_attributes: List[str],
        methodology: str = None
    ):
        """Add fairness assessment section to report.
        
        Args:
            fairness_metrics: Dictionary of fairness metrics
            protected_attributes: List of protected attributes assessed
            methodology: Optional description of assessment methodology
        """
        self.sections["fairness"] = {
            "timestamp": datetime.now().isoformat(),
            "protected_attributes": protected_attributes,
            "metrics": fairness_metrics
        }
        
        if methodology:
            self.sections["fairness"]["methodology"] = methodology
    
    def add_explainability_results(
        self,
        explanation_data: Dict[str, Any],
        explanation_method: str,
        sample_explanations: Optional[List[Dict[str, Any]]] = None
    ):
        """Add explainability results section to report.
        
        Args:
            explanation_data: Explanation metrics and data
            explanation_method: Method used for explanations
            sample_explanations: Optional list of sample explanations
        """
        self.sections["explainability"] = {
            "timestamp": datetime.now().isoformat(),
            "method": explanation_method,
            "data": explanation_data
        }
        
        if sample_explanations:
            self.sections["explainability"]["samples"] = sample_explanations
    
    def add_quality_gate_results(
        self,
        quality_gate_results: Dict[str, Any]
    ):
        """Add quality gate results section to report.
        
        Args:
            quality_gate_results: Results from quality gate evaluation
        """
        self.sections["quality_gates"] = quality_gate_results
    
    def add_benchmark_results(
        self,
        benchmark_results: Dict[str, Any],
        comparison_models: List[Dict[str, str]] = None
    ):
        """Add benchmark results section to report.
        
        Args:
            benchmark_results: Benchmark metrics and data
            comparison_models: Optional list of models compared against
        """
        self.sections["benchmarks"] = {
            "timestamp": datetime.now().isoformat(),
            "results": benchmark_results
        }
        
        if comparison_models:
            self.sections["benchmarks"]["comparison_models"] = comparison_models
    
    def add_data_quality_summary(
        self,
        data_quality_metrics: Dict[str, Any],
        dataset_name: str,
        data_schema: Optional[Dict[str, Any]] = None
    ):
        """Add data quality summary section to report.
        
        Args:
            data_quality_metrics: Data quality metrics
            dataset_name: Name of dataset
            data_schema: Optional schema information
        """
        self.sections["data_quality"] = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "metrics": data_quality_metrics
        }
        
        if data_schema:
            self.sections["data_quality"]["schema"] = data_schema
    
    def add_recommendation(
        self,
        recommended_action: str,
        justification: str,
        confidence: float,
        reviewers: List[str] = None
    ):
        """Add recommendation section to report.
        
        Args:
            recommended_action: Recommended action (approve, reject, etc.)
            justification: Justification for recommendation
            confidence: Confidence in recommendation (0-1)
            reviewers: Optional list of reviewers
        """
        self.sections["recommendation"] = {
            "timestamp": datetime.now().isoformat(),
            "action": recommended_action,
            "justification": justification,
            "confidence": confidence
        }
        
        if reviewers:
            self.sections["recommendation"]["reviewers"] = reviewers
    
    def add_custom_section(
        self,
        section_name: str,
        content: Dict[str, Any]
    ):
        """Add custom section to report.
        
        Args:
            section_name: Name of section
            content: Section content
        """
        self.sections[section_name] = content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.
        
        Returns:
            Report as dictionary
        """
        return {
            "report_id": f"{self.model_id}-{self.model_version}-{self.evaluation_date}",
            "sections": self.sections
        }
    
    def to_json(self, pretty: bool = False) -> str:
        """Convert report to JSON string.
        
        Args:
            pretty: Whether to format JSON with indentation
            
        Returns:
            JSON string representation of report
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(
        self,
        filepath: str,
        format_type: str = "json"
    ):
        """Save report to file.
        
        Args:
            filepath: Path to save report to
            format_type: Format to save as (json, yaml, or markdown)
        """
        if format_type.lower() == "json":
            with open(filepath, "w") as f:
                f.write(self.to_json(pretty=True))
        elif format_type.lower() == "yaml":
            import yaml
            with open(filepath, "w") as f:
                yaml.dump(self.to_dict(), f)
        elif format_type.lower() == "markdown":
            with open(filepath, "w") as f:
                f.write(self.to_markdown())
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def to_markdown(self) -> str:
        """Convert report to Markdown format.
        
        Returns:
            Markdown representation of report
        """
        lines = [
            f"# Model Evaluation Report: {self.model_id} (v{self.model_version})",
            "",
            f"**Evaluation Date:** {self.evaluation_date}",
            f"**Author:** {self.author}",
            f"**Training Date:** {self.training_date}",
            f"**Model Type:** {self.model_type}",
            "",
            "## Contents",
            ""
        ]
        
        # Add table of contents
        for section_name in self.sections.keys():
            section_title = section_name.replace("_", " ").title()
            lines.append(f"- [{section_title}](#{section_name})")
        
        lines.append("")
        
        # Process each section
        for section_name, section_data in self.sections.items():
            section_title = section_name.replace("_", " ").title()
            lines.append(f"## {section_title}")
            lines.append("")
            
            # Handle different section types
            if section_name == "metadata":
                for key, value in section_data.items():
                    lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
            
            elif section_name == "performance":
                for dataset_entry in section_data:
                    lines.append(f"### Dataset: {dataset_entry['dataset']}")
                    
                    if "dataset_description" in dataset_entry:
                        lines.append("")
                        lines.append(dataset_entry["dataset_description"])
                    
                    lines.append("")
                    lines.append("| Metric | Value |")
                    lines.append("|--------|-------|")
                    
                    for metric_name, metric_value in dataset_entry["metrics"].items():
                        # Format numeric values nicely
                        if isinstance(metric_value, (int, float)):
                            if abs(metric_value) < 0.01 or abs(metric_value) >= 1000:
                                formatted_value = f"{metric_value:.4e}"
                            else:
                                formatted_value = f"{metric_value:.4f}"
                            
                            # Remove trailing zeros
                            if "." in formatted_value:
                                formatted_value = formatted_value.rstrip("0").rstrip(".")
                        else:
                            formatted_value = str(metric_value)
                        
                        lines.append(f"| {metric_name} | {formatted_value} |")
            
            elif section_name == "quality_gates":
                lines.append(f"**Overall Status:** {'✅ PASSED' if section_data['passed'] else '❌ FAILED'}")
                lines.append(f"**Gates Passed:** {section_data['gates_passed']}/{section_data['gates_evaluated']}")
                lines.append(f"**Blocking Gates Passed:** {section_data['blocking_gates_passed']}/{section_data['blocking_gates_total']}")
                lines.append("")
                
                # Add gate details
                for gate_name, gate_result in section_data["results"].items():
                    gate_title = gate_name.replace("_", " ").title()
                    status = "✅ PASSED" if gate_result["passed"] else "❌ FAILED"
                    lines.append(f"### {gate_title}: {status}")
                    
                    if "criteria_results" in gate_result:
                        lines.append("")
                        lines.append("| Criterion | Status | Details |")
                        lines.append("|-----------|--------|---------|")
                        
                        for criterion in gate_result["criteria_results"]:
                            criterion_name = criterion["name"]
                            criterion_status = "✅ PASSED" if criterion["passed"] else "❌ FAILED"
                            
                            # Format details
                            details = ""
                            if "details" in criterion:
                                for key, value in criterion["details"].items():
                                    details += f"{key}: {value}, "
                                
                                details = details.rstrip(", ")
                            
                            lines.append(f"| {criterion_name} | {criterion_status} | {details} |")
            
            # Add a blank line after each section
            lines.append("")
            lines.append("")
        
        return "\n".join(lines)
```

### 2. Report Templates

Standardized templates ensure consistent evaluation reporting:

```python
def get_report_template(
    model_type: str,
    purpose: str = "full_evaluation"
) -> Dict[str, Any]:
    """Get standardized report template for model type.
    
    Args:
        model_type: Type of model (e.g. "ad_score", "account_health")
        purpose: Report purpose ("full_evaluation", "quick_check", "drift_analysis")
        
    Returns:
        Report template configuration
    """
    # Base template shared by all reports
    base_template = {
        "required_sections": ["metadata", "performance", "quality_gates"],
        "optional_sections": ["fairness", "explainability", "benchmarks", "data_quality"],
        "formats": ["json", "markdown", "yaml"],
    }
    
    # Model-specific templates
    templates = {
        "ad_score": {
            "metrics": {
                "rmse": {"required": True, "threshold": 0.15},
                "r2": {"required": True, "threshold": 0.7},
                "mae": {"required": True, "threshold": 0.12},
                "explained_variance": {"required": False},
                "max_error": {"required": False}
            },
            "fairness_metrics": {
                "demographic_parity": {"required": True, "threshold": 0.1},
                "equal_opportunity": {"required": True, "threshold": 0.1},
                "disparate_impact": {"required": False}
            },
            "explainability_methods": ["shap", "feature_importance"],
            "benchmarks": ["previous_version", "baseline_linear", "competitor_x"]
        },
        
        "account_health": {
            "metrics": {
                "accuracy": {"required": True, "threshold": 0.85},
                "precision": {"required": True, "threshold": 0.8},
                "recall": {"required": True, "threshold": 0.75},
                "f1": {"required": True, "threshold": 0.8},
                "roc_auc": {"required": True, "threshold": 0.85},
                "pr_auc": {"required": False}
            },
            "fairness_metrics": {
                "demographic_parity": {"required": True, "threshold": 0.1},
                "equal_opportunity": {"required": True, "threshold": 0.1}
            },
            "explainability_methods": ["shap", "lime", "feature_importance"],
            "benchmarks": ["previous_version", "rules_based", "competitor_y"]
        }
    }
    
    # Purpose-specific configurations
    purpose_configs = {
        "full_evaluation": {
            "required_sections": base_template["required_sections"] + ["fairness", "explainability"],
            "default_format": "markdown"
        },
        "quick_check": {
            "required_sections": ["metadata", "performance"],
            "default_format": "json"
        },
        "drift_analysis": {
            "required_sections": ["metadata", "data_quality", "performance"],
            "default_format": "json"
        }
    }
    
    # Combine configurations
    result = copy.deepcopy(base_template)
    
    if model_type in templates:
        result.update(templates[model_type])
    
    if purpose in purpose_configs:
        result.update(purpose_configs[purpose])
    
    return result
```

### 3. Visualizing Results

The visualization module generates standardized charts:

```python
class ModelEvaluationVisualizer:
    """Generate visualizations for model evaluation reports."""
    
    def __init__(
        self,
        output_dir: str = None,
        style: str = "default",
        dpi: int = 300
    ):
        """Initialize model evaluation visualizer.
        
        Args:
            output_dir: Directory to save visualizations to
            style: Visualization style
            dpi: DPI for saved figures
        """
        self.output_dir = output_dir or "evaluation_visualizations"
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Configure plotting style
        import matplotlib.pyplot as plt
        if style == "default":
            plt.style.use("seaborn-whitegrid")
        elif style == "dark":
            plt.style.use("dark_background")
        elif style == "minimal":
            plt.style.use("seaborn-white")
        
        self.plt = plt
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Model Performance Comparison",
        sort_by: str = None,
        ascending: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        filename: str = None
    ) -> "matplotlib.figure.Figure":
        """Plot comparison of metrics across models.
        
        Args:
            metrics_dict: Dictionary mapping model names to metric dictionaries
            title: Plot title
            sort_by: Optional metric to sort by
            ascending: Whether to sort in ascending order
            figsize: Figure size
            filename: Optional filename to save plot to
            
        Returns:
            Matplotlib figure
        """
        import pandas as pd
        import numpy as np
        
        # Extract common metrics present in all models
        common_metrics = set.intersection(
            *[set(metrics.keys()) for metrics in metrics_dict.values()]
        )
        
        # Create DataFrame
        data = {
            model_name: {
                metric: value for metric, value in model_metrics.items()
                if metric in common_metrics
            }
            for model_name, model_metrics in metrics_dict.items()
        }
        df = pd.DataFrame(data)
        
        # Sort if requested
        if sort_by and sort_by in common_metrics:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Create plot
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Get positions for bars
        x = np.arange(len(df.index))
        width = 0.8 / len(df.columns)
        
        # Plot bars
        for i, model_name in enumerate(df.columns):
            offset = i * width - (len(df.columns) - 1) * width / 2
            ax.bar(x + offset, df[model_name].values, width, label=model_name)
        
        # Set labels and title
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha="right")
        ax.legend()
        
        self.plt.tight_layout()
        
        # Save if filename provided
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = True,
        cmap: str = "Blues",
        figsize: Tuple[int, int] = (8, 6),
        filename: str = None
    ) -> "matplotlib.figure.Figure":
        """Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Names of classes
            title: Plot title
            normalize: Whether to normalize values
            cmap: Colormap
            figsize: Figure size
            filename: Optional filename to save plot to
            
        Returns:
            Matplotlib figure
        """
        import numpy as np
        
        # Normalize if requested
        if normalize:
            confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"
        
        # Create plot
        fig, ax = self.plt.subplots(figsize=figsize)
        im = ax.imshow(confusion_matrix, interpolation="nearest", cmap=self.plt.get_cmap(cmap))
        fig.colorbar(im)
        
        # Add labels
        ax.set(
            xticks=np.arange(confusion_matrix.shape[1]),
            yticks=np.arange(confusion_matrix.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True label",
            xlabel="Predicted label"
        )
        
        # Rotate x-axis labels
        self.plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = confusion_matrix.max() / 2.0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(
                    j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black"
                )
        
        self.plt.tight_layout()
        
        # Save if filename provided
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
        
        return fig
    
    def plot_roc_curve(
        self,
        fpr_dict: Dict[str, np.ndarray],
        tpr_dict: Dict[str, np.ndarray],
        auc_dict: Dict[str, float],
        title: str = "ROC Curve",
        figsize: Tuple[int, int] = (8, 6),
        filename: str = None
    ) -> "matplotlib.figure.Figure":
        """Plot ROC curve.
        
        Args:
            fpr_dict: Dictionary mapping names to false positive rates
            tpr_dict: Dictionary mapping names to true positive rates
            auc_dict: Dictionary mapping names to AUC values
            title: Plot title
            figsize: Figure size
            filename: Optional filename to save plot to
            
        Returns:
            Matplotlib figure
        """
        # Create plot
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], "k--", lw=2)
        
        # Plot ROC curves
        for name in fpr_dict:
            ax.plot(
                fpr_dict[name],
                tpr_dict[name],
                lw=2,
                label=f"{name} (AUC = {auc_dict[name]:.3f})"
            )
        
        # Set labels and title
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        self.plt.tight_layout()
        
        # Save if filename provided
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
        
        return fig
```

### 4. Integration with CI/CD

Integrate reports with continuous integration for automated quality assessments:

```python
def integrate_with_ci_pipeline(
    model_id: str,
    model_version: str,
    evaluation_results: Dict[str, Any],
    report_config: Dict[str, Any],
    artifact_dir: str,
    ci_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Integrate evaluation reporting with CI pipeline.
    
    Args:
        model_id: Model identifier
        model_version: Model version
        evaluation_results: Evaluation results from model evaluator
        report_config: Report configuration
        artifact_dir: Directory to store artifacts
        ci_metadata: Optional CI system metadata
        
    Returns:
        Pipeline results including artifact paths
    """
    # Create directories
    artifacts = {}
    report_dir = os.path.join(artifact_dir, "reports")
    viz_dir = os.path.join(artifact_dir, "visualizations")
    
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract CI metadata
    ci_info = {
        "timestamp": datetime.now().isoformat(),
        "build_id": ci_metadata.get("build_id", "unknown"),
        "commit_hash": ci_metadata.get("commit_hash", "unknown"),
        "branch": ci_metadata.get("branch", "unknown"),
        "pipeline_url": ci_metadata.get("pipeline_url", "unknown")
    } if ci_metadata else {}
    
    # Create report
    report = ModelEvaluationReport(
        model_id=model_id,
        model_version=model_version,
        model_type=report_config.get("model_type", "unknown"),
        author=ci_metadata.get("author", "CI/CD Pipeline"),
        training_date=ci_metadata.get("training_date", datetime.now().strftime("%Y-%m-%d"))
    )
    
    # Add CI metadata as custom section
    if ci_info:
        report.add_custom_section("ci_metadata", ci_info)
    
    # Add evaluation results
    if "performance" in evaluation_results:
        report.add_performance_metrics(
            metrics=evaluation_results["performance"],
            dataset_name=evaluation_results.get("dataset_name", "evaluation_dataset")
        )
    
    if "fairness" in evaluation_results:
        report.add_fairness_assessment(
            fairness_metrics=evaluation_results["fairness"],
            protected_attributes=evaluation_results.get("protected_attributes", [])
        )
    
    if "quality_gates" in evaluation_results:
        report.add_quality_gate_results(evaluation_results["quality_gates"])
    
    # Generate reports in different formats
    for format_type in ["json", "markdown", "yaml"]:
        filename = f"model_evaluation_{model_id}_v{model_version}.{format_type}"
        filepath = os.path.join(report_dir, filename)
        
        try:
            report.save(filepath, format_type=format_type)
            artifacts[f"report_{format_type}"] = filepath
        except Exception as e:
            print(f"Error saving report in {format_type} format: {str(e)}")
    
    # Generate visualizations
    try:
        visualizer = ModelEvaluationVisualizer(output_dir=viz_dir)
        
        # Generate visualizations based on available data
        if "confusion_matrix" in evaluation_results and "class_names" in evaluation_results:
            fig = visualizer.plot_confusion_matrix(
                confusion_matrix=evaluation_results["confusion_matrix"],
                class_names=evaluation_results["class_names"],
                title=f"Confusion Matrix - {model_id} v{model_version}",
                filename=f"confusion_matrix_{model_id}_v{model_version}.png"
            )
            artifacts["confusion_matrix_plot"] = os.path.join(viz_dir, f"confusion_matrix_{model_id}_v{model_version}.png")
        
        # Generate ROC curve if available
        if all(k in evaluation_results for k in ["fpr", "tpr", "roc_auc"]):
            fig = visualizer.plot_roc_curve(
                fpr_dict={"model": evaluation_results["fpr"]},
                tpr_dict={"model": evaluation_results["tpr"]},
                auc_dict={"model": evaluation_results["roc_auc"]},
                title=f"ROC Curve - {model_id} v{model_version}",
                filename=f"roc_curve_{model_id}_v{model_version}.png"
            )
            artifacts["roc_curve_plot"] = os.path.join(viz_dir, f"roc_curve_{model_id}_v{model_version}.png")
        
        # Generate metrics comparison if baselines are available
        if "baselines" in evaluation_results and "performance" in evaluation_results:
            metrics_dict = {
                "current": evaluation_results["performance"],
                **evaluation_results["baselines"]
            }
            
            fig = visualizer.plot_metrics_comparison(
                metrics_dict=metrics_dict,
                title=f"Performance Comparison - {model_id} v{model_version}",
                filename=f"metrics_comparison_{model_id}_v{model_version}.png"
            )
            artifacts["metrics_comparison_plot"] = os.path.join(viz_dir, f"metrics_comparison_{model_id}_v{model_version}.png")
    
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    # Determine overall status
    passed = evaluation_results.get("quality_gates", {}).get("passed", False)
    
    # Return results
    return {
        "model_id": model_id,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "artifacts": artifacts,
        "ci_info": ci_info
    }
```

## Related Documentation

- [Feature Engineering](./feature_engineering.md)
- [Ad Score Prediction](./ad_score_prediction.md)
- [Model Training](./model_training.md)