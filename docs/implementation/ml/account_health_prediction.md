# Account Health Prediction Implementation

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


This document describes the implementation details of the Account Health Predictor model in the WITHIN platform.

## Table of Contents

1. [Introduction](#introduction)
2. [Model Architecture](#model-architecture)
3. [Feature Engineering](#feature-engineering)
4. [Training Process](#training-process)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Inference Pipeline](#inference-pipeline)
7. [Integration Points](#integration-points)
8. [Performance Considerations](#performance-considerations)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Introduction

The Account Health Predictor is a machine learning system designed to evaluate the overall health of advertising accounts and predict potential issues before they impact performance. It serves as an early warning system that enables proactive interventions to maintain optimal account performance.

### Purpose and Goals

The primary goals of the Account Health Predictor are:

1. **Early Issue Detection**: Identify potential performance issues 7-14 days before they would be visible in standard performance metrics
2. **Root Cause Analysis**: Provide insights into the underlying factors contributing to account health issues
3. **Actionable Recommendations**: Generate specific, actionable recommendations to address identified issues
4. **Continuous Learning**: Improve predictions over time by learning from outcomes of previous interventions

### Business Context

Account health is a critical factor in maintaining effective advertising campaigns. Poor account health can lead to:

- Decreased return on ad spend (ROAS)
- Reduced conversion rates
- Higher customer acquisition costs
- Increased churn rates

By proactively monitoring and addressing account health issues, the system helps maintain optimal performance and maximize advertiser value.

## Model Architecture

The Account Health Predictor employs a hybrid architecture that combines multiple specialized components to deliver comprehensive health assessments for advertising accounts.

### Overview

The system follows a multi-model approach with four primary components:

1. **Time Series Analysis**: For detecting trends and forecasting future performance
2. **Anomaly Detection**: For identifying unusual patterns and outliers
3. **Classification Models**: For categorizing accounts into health tiers
4. **Recommendation Engine**: For generating action items based on detected issues

These components work together in a Chain of Reasoning pattern, where the output of each stage informs the next, creating an explainable and transparent assessment process.

### Architecture Diagram

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│                    │     │                    │     │                    │
│  Account Data      │────▶│  Feature           │────▶│  Model Ensemble    │
│  Collection        │     │  Engineering       │     │                    │
│                    │     │                    │     │                    │
└────────────────────┘     └────────────────────┘     └──────────┬─────────┘
                                                                 │
                                                                 ▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│                    │     │                    │     │                    │
│  Recommendation    │◀────│  Explanation       │◀────│  Health Score      │
│  Generator         │     │  Generator         │     │  Calculation       │
│                    │     │                    │     │                    │
└────────────────────┘     └────────────────────┘     └────────────────────┘
```

### Model Components

#### 1. Time Series Component

The time series component uses LSTM (Long Short-Term Memory) networks to model temporal patterns in account performance metrics. This component:

- Captures daily, weekly, and seasonal patterns in advertising performance
- Forecasts expected values for key metrics in the upcoming 7-30 day period
- Identifies deviation from expected trends that may indicate emerging issues

```python
class TimeSeriesModel:
    """Time series forecasting model for account metrics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the time series model with configuration."""
        self.config = config or self._get_default_config()
        self.model = self._build_lstm_model()
        
    def _build_lstm_model(self) -> tf.keras.Model:
        """Build and compile LSTM model."""
        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, 
                                input_shape=(self.config["sequence_length"], 
                                            self.config["feature_dim"])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.config["forecast_horizon"])
        ])
        
        # Compile with appropriate loss and optimizer
        model.compile(
            loss=self.config.get("loss", "mse"),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get("learning_rate", 0.001)
            ),
            metrics=['mae']
        )
        
        return model
    
    def forecast(self, historical_data: np.ndarray) -> np.ndarray:
        """Generate forecast for the next forecast_horizon days."""
        # Ensure data is correctly shaped
        if len(historical_data.shape) == 2:
            # Add batch dimension if needed
            input_data = np.expand_dims(historical_data, axis=0)
        else:
            input_data = historical_data
            
        # Generate prediction
        predictions = self.model.predict(input_data)
        
        return predictions
```

#### 2. Anomaly Detection Component

The anomaly detection component identifies unusual patterns across various metrics that may indicate account health issues. The system leverages multiple anomaly detection methods as described in the [Anomaly Detection Methodology](/docs/implementation/ml/technical/anomaly_detection.md) document. Key features include:

- Z-score analysis for univariate anomaly detection
- Isolation Forest for multivariate outlier detection
- Moving window analysis for trend-based anomalies
- Specialized detectors for different metric types (e.g., conversion rates, click-through rates)

```python
from app.models.ml.monitoring.anomaly_detection import ZScoreDetector, IsolationForestDetector

class AnomalyDetectionComponent:
    """Anomaly detection component for account health prediction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the anomaly detection component."""
        self.config = config or {}
        self.detectors = self._initialize_detectors()
        
    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize anomaly detectors based on configuration."""
        detectors = {}
        
        # Configure z-score detector for univariate metrics
        z_score_config = self.config.get("z_score", {})
        detectors["z_score"] = ZScoreDetector(
            sensitivity=z_score_config.get("sensitivity", 2.5),
            window_size=z_score_config.get("window_size", 14)
        )
        
        # Configure isolation forest for multivariate detection
        if_config = self.config.get("isolation_forest", {})
        detectors["isolation_forest"] = IsolationForestDetector(
            contamination=if_config.get("contamination", 0.05),
            n_estimators=if_config.get("n_estimators", 100),
            max_samples=if_config.get("max_samples", "auto")
        )
        
        return detectors
    
    def detect_anomalies(self, account_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies in account metrics.
        
        Args:
            account_data: DataFrame containing account metrics
            
        Returns:
            Dictionary with detected anomalies
        """
        results = {
            "anomalies_detected": False,
            "metric_anomalies": {},
            "multivariate_anomalies": {},
            "anomaly_score": 0.0
        }
        
        # Run univariate detection for each metric
        for metric in self.config.get("metrics", []):
            if metric not in account_data.columns:
                continue
                
            # Get metric data
            metric_data = account_data[metric].values
            
            # Detect anomalies using z-score
            anomaly_result = self.detectors["z_score"].detect(metric_data)
            
            if anomaly_result["anomalies_detected"]:
                results["anomalies_detected"] = True
                results["metric_anomalies"][metric] = anomaly_result
        
        # Run multivariate detection
        if len(account_data) >= self.config.get("min_samples_multivariate", 10):
            # Select features for multivariate detection
            features = [col for col in self.config.get("multivariate_features", []) 
                       if col in account_data.columns]
            
            if len(features) >= 2:  # Need at least 2 features
                # Get feature data
                feature_data = account_data[features].values
                
                # Detect multivariate anomalies
                multivariate_result = self.detectors["isolation_forest"].detect(feature_data)
                
                if multivariate_result["anomalies_detected"]:
                    results["anomalies_detected"] = True
                    results["multivariate_anomalies"] = multivariate_result
        
        # Calculate overall anomaly score
        results["anomaly_score"] = self._calculate_anomaly_score(results)
        
        return results
    
    def _calculate_anomaly_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall anomaly score from detection results."""
        # Start with a base score of 0
        score = 0.0
        
        # Add contribution from metric anomalies
        for metric, anomaly in results.get("metric_anomalies", {}).items():
            metric_weight = self.config.get("metric_weights", {}).get(metric, 1.0)
            score += anomaly.get("score", 0.0) * metric_weight
        
        # Add contribution from multivariate anomalies
        multivariate_anomalies = results.get("multivariate_anomalies", {})
        if multivariate_anomalies:
            multivariate_weight = self.config.get("multivariate_weight", 2.0)
            score += multivariate_anomalies.get("score", 0.0) * multivariate_weight
        
        # Normalize to 0-1 range
        num_metrics = len(results.get("metric_anomalies", {}))
        if num_metrics > 0:
            score /= (num_metrics + (1 if multivariate_anomalies else 0))
        
        return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
```

#### 3. Classification Component

The classification component determines the overall health category of an account based on a wide range of features:

- Uses an ensemble of gradient boosting models (XGBoost and LightGBM)
- Incorporates historical performance data, anomaly scores, and account structure metrics
- Produces calibrated probability estimates for different health tiers
- Identifies key factors contributing to the health assessment

```python
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

class HealthClassifier:
    """Classifier for determining account health status."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the health classifier."""
        self.config = config or {}
        self.model = self._build_ensemble_model()
        self.feature_names = self.config.get("feature_names", [])
        
    def _build_ensemble_model(self) -> Any:
        """Build ensemble classification model."""
        # Configure XGBoost classifier
        xgb_config = self.config.get("xgboost", {})
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_config.get("n_estimators", 100),
            max_depth=xgb_config.get("max_depth", 4),
            learning_rate=xgb_config.get("learning_rate", 0.1),
            subsample=xgb_config.get("subsample", 0.8),
            colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
            objective=xgb_config.get("objective", "multi:softprob"),
            random_state=42
        )
        
        # Configure LightGBM classifier
        lgb_config = self.config.get("lightgbm", {})
        lgb_model = lgb.LGBMClassifier(
            n_estimators=lgb_config.get("n_estimators", 100),
            max_depth=lgb_config.get("max_depth", 4),
            learning_rate=lgb_config.get("learning_rate", 0.1),
            subsample=lgb_config.get("subsample", 0.8),
            colsample_bytree=lgb_config.get("colsample_bytree", 0.8),
            objective=lgb_config.get("objective", "multiclass"),
            random_state=42
        )
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model)
            ],
            voting='soft',
            weights=self.config.get("ensemble_weights", [0.6, 0.4])
        )
        
        # Apply calibration for reliable probability estimates
        if self.config.get("use_calibration", True):
            return CalibratedClassifierCV(
                ensemble, 
                method=self.config.get("calibration_method", "isotonic"),
                cv=self.config.get("calibration_cv", 3)
            )
        else:
            return ensemble
    
    def predict_health(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict account health status.
        
        Args:
            features: DataFrame with account features
            
        Returns:
            Dictionary with health prediction results
        """
        # Ensure features match expected format
        if self.feature_names:
            # Select only required features in the correct order
            input_features = features[self.feature_names]
        else:
            input_features = features
        
        # Get class probabilities
        probabilities = self.model.predict_proba(input_features)
        
        # Get class prediction
        class_idx = np.argmax(probabilities, axis=1)[0]
        predicted_class = self.config.get("class_names", ["critical", "poor", "fair", "good", "excellent"])[class_idx]
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
        
        # Prepare result
        result = {
            "health_class": predicted_class,
            "health_score": float(probabilities[0][class_idx]),
            "class_probabilities": {
                cls: float(prob) for cls, prob in zip(
                    self.config.get("class_names", ["critical", "poor", "fair", "good", "excellent"]),
                    probabilities[0]
                )
            },
            "feature_importance": feature_importance
        }
        
        return result
```

#### 4. Recommendation Engine

The recommendation engine generates actionable insights based on detected issues and account health status. It uses:

- Rule-based systems for well-understood optimization opportunities
- Case-based reasoning to leverage past successful interventions
- Reinforcement learning to improve recommendations over time

```python
class RecommendationEngine:
    """Engine for generating recommendations based on account health."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the recommendation engine."""
        self.config = config or {}
        self.rules = self._load_recommendation_rules()
        
    def _load_recommendation_rules(self) -> List[Dict[str, Any]]:
        """Load recommendation rules from configuration."""
        # Load rules from file if specified
        rules_path = self.config.get("rules_path")
        if rules_path and os.path.exists(rules_path):
            with open(rules_path, 'r') as f:
                return json.load(f)
        
        # Otherwise return default rules
        return [
            {
                "id": "low_ctr",
                "condition": lambda metrics: metrics.get("ctr", 0) < 0.01,
                "recommendation": "Review ad creative and targeting to improve click-through rate",
                "priority": "high",
                "impact": 0.7,
                "factors": ["ad_creative", "targeting"]
            },
            {
                "id": "high_cpa",
                "condition": lambda metrics, benchmarks: (
                    metrics.get("cpa", 0) > 1.5 * benchmarks.get("cpa", metrics.get("cpa", 0))
                ),
                "recommendation": "Optimize bidding strategy to reduce cost per acquisition",
                "priority": "high",
                "impact": 0.8,
                "factors": ["bidding", "conversion_rate"]
            },
            # Additional rules would be defined here
        ]
    
    def generate_recommendations(
        self, 
        account_metrics: Dict[str, float],
        health_prediction: Dict[str, Any],
        anomalies: Dict[str, Any],
        benchmarks: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on account health and metrics.
        
        Args:
            account_metrics: Dictionary of account metrics
            health_prediction: Health prediction results
            anomalies: Detected anomalies
            benchmarks: Optional industry benchmarks
            
        Returns:
            List of recommendation objects
        """
        recommendations = []
        benchmarks = benchmarks or {}
        
        # Apply rule-based recommendations
        for rule in self.rules:
            # Check if condition is met
            condition_met = False
            try:
                if "condition" in rule:
                    if callable(rule["condition"]):
                        condition_met = rule["condition"](account_metrics, benchmarks)
                    else:
                        # Simple string condition
                        condition_met = eval(rule["condition"], 
                                          {"metrics": account_metrics, "benchmarks": benchmarks})
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule.get('id')}: {str(e)}")
                continue
                
            if condition_met:
                recommendations.append({
                    "id": rule.get("id", str(uuid.uuid4())),
                    "text": rule.get("recommendation", ""),
                    "priority": rule.get("priority", "medium"),
                    "impact": rule.get("impact", 0.5),
                    "factors": rule.get("factors", []),
                    "rule_id": rule.get("id")
                })
        
        # Add anomaly-based recommendations
        for metric, anomaly in anomalies.get("metric_anomalies", {}).items():
            if metric in self.config.get("anomaly_recommendations", {}):
                rec_template = self.config["anomaly_recommendations"][metric]
                recommendations.append({
                    "id": f"anomaly_{metric}_{int(time.time())}",
                    "text": rec_template["text"].format(
                        metric=metric, 
                        value=account_metrics.get(metric, 0),
                        threshold=anomaly.get("threshold", 0)
                    ),
                    "priority": rec_template.get("priority", "medium"),
                    "impact": rec_template.get("impact", 0.5),
                    "factors": [metric],
                    "anomaly_id": anomaly.get("id")
                })
        
        # Sort recommendations by priority and impact
        priority_map = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        recommendations.sort(
            key=lambda x: (
                priority_map.get(x["priority"], 0),
                x["impact"]
            ),
            reverse=True
        )
        
        # Limit to top recommendations if configured
        max_recommendations = self.config.get("max_recommendations")
        if max_recommendations and len(recommendations) > max_recommendations:
            recommendations = recommendations[:max_recommendations]
            
        return recommendations
```

The recommendation engine is further detailed in the upcoming [Recommendation Engine](/docs/implementation/ml/technical/recommendation_engine.md) documentation.

### Ensemble Integration

The Account Health Predictor integrates these components through a unified pipeline that:

1. Collects and preprocesses account data
2. Extracts time-series features and detects anomalies
3. Generates health predictions from multiple models
4. Aggregates predictions into a final health score
5. Provides explanations and recommendations

The integration enables the system to leverage different strengths of each component while maintaining explainability throughout the assessment process.

```python
class AccountHealthPredictor:
    """Main account health prediction system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the account health predictor with optional configuration."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.time_series_model = TimeSeriesModel(self.config.get("time_series", {}))
        self.anomaly_detector = AnomalyDetectionComponent(self.config.get("anomaly_detection", {}))
        self.health_classifier = HealthClassifier(self.config.get("classification", {}))
        
        # Initialize recommendation engine
        recommendation_config = self.config.get("recommendation", {})
        self.recommendation_engine = RecommendationEngine(recommendation_config)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Return default configuration
            return {
                "time_series": {
                    "sequence_length": 28,
                    "forecast_horizon": 14,
                    "feature_dim": 10
                },
                "anomaly_detection": {
                    "z_score": {
                        "sensitivity": 2.5,
                        "window_size": 14
                    },
                    "isolation_forest": {
                        "contamination": 0.05
                    },
                    "metrics": ["ctr", "cvr", "cpa", "spend", "impressions", "clicks"]
                },
                "classification": {
                    "class_names": ["critical", "poor", "fair", "good", "excellent"]
                }
            }
        
    def predict(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate health prediction for an account.
        
        Args:
            account_data: Dictionary with account metrics and metadata
            
        Returns:
            Dictionary with health assessment results
        """
        # Extract and preprocess data
        processed_data = self._preprocess_data(account_data)
        
        # Extract features
        features = self._extract_features(processed_data)
        
        # Detect anomalies
        anomaly_results = self.anomaly_detector.detect_anomalies(processed_data)
        
        # Add anomaly features
        for metric, result in anomaly_results.get("metric_anomalies", {}).items():
            features[f"{metric}_anomaly_score"] = result.get("score", 0.0)
        features["overall_anomaly_score"] = anomaly_results.get("anomaly_score", 0.0)
        
        # Generate time series forecast and features
        time_series_features = self._generate_time_series_features(processed_data)
        features.update(time_series_features)
        
        # Predict health status
        health_prediction = self.health_classifier.predict_health(pd.DataFrame([features]))
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            account_metrics=self._extract_current_metrics(processed_data),
            health_prediction=health_prediction,
            anomalies=anomaly_results,
            benchmarks=self._get_industry_benchmarks(account_data)
        )
        
        # Assemble final result
        result = {
            "health_score": health_prediction["health_score"],
            "health_class": health_prediction["health_class"],
            "timestamp": datetime.now().isoformat(),
            "anomalies": anomaly_results,
            "predictions": {
                "class_probabilities": health_prediction["class_probabilities"]
            },
            "recommendations": recommendations,
            "factors": self._extract_key_factors(health_prediction, anomaly_results)
        }
        
        return result
    
    def _preprocess_data(self, account_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess account data into standardized format."""
        # Implementation details omitted for brevity
        return pd.DataFrame()
    
    def _extract_features(self, processed_data: pd.DataFrame) -> Dict[str, float]:
        """Extract features from processed data."""
        # Implementation details omitted for brevity
        return {}
    
    def _generate_time_series_features(self, processed_data: pd.DataFrame) -> Dict[str, float]:
        """Generate features from time series analysis."""
        # Implementation details omitted for brevity
        return {}
        
    def _extract_current_metrics(self, processed_data: pd.DataFrame) -> Dict[str, float]:
        """Extract current metrics from processed data."""
        # Implementation details omitted for brevity
        return {}
        
    def _get_industry_benchmarks(self, account_data: Dict[str, Any]) -> Dict[str, float]:
        """Get industry benchmarks for comparison."""
        # Implementation details omitted for brevity
        return {}
    
    def _extract_key_factors(self, health_prediction: Dict[str, Any],
                            anomaly_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key factors contributing to health assessment."""
        # Implementation details omitted for brevity
        return []
```

### Model Selection Rationale

The hybrid architecture was chosen for several key reasons:

1. **Interpretability**: The modular design allows each component to be interpreted and explained independently
2. **Specialization**: Different algorithms can be optimized for specific tasks (time series, classification, etc.)
3. **Flexibility**: Components can be updated or replaced independently as better algorithms become available
4. **Robustness**: The ensemble approach reduces the impact of individual model failures or weaknesses

The specific models (LSTM, XGBoost, LightGBM) were selected based on:
- Performance on historical advertising data
- Ability to handle the temporal nature of advertising metrics
- Computational efficiency for production deployment
- Interpretability of predictions and feature importance

In the next section, we'll explore the feature engineering process that transforms raw account data into the inputs needed by these model components.

## Feature Engineering

The performance of the Account Health Predictor heavily depends on the quality and relevance of features extracted from raw account data. This section describes the feature engineering pipeline that transforms advertising account data into meaningful features for health prediction.

### Data Sources

The Account Health Predictor leverages data from multiple sources:

1. **Platform Metrics**: Performance data from advertising platforms (Google, Facebook, TikTok, etc.)
2. **Account Structure**: Information about campaign organization, settings, and targeting
3. **Historical Benchmarks**: Industry and vertical-specific performance baselines
4. **Previous Interventions**: Records of past optimizations and their outcomes

### Feature Categories

Features are organized into several key categories:

#### 1. Performance Metrics

These features capture the key advertising performance indicators:

```python
def extract_performance_metrics(account_data: pd.DataFrame) -> Dict[str, float]:
    """Extract performance metric features from account data."""
    # Calculate core metrics
    metrics = {
        # Volume metrics
        "impressions_daily_avg": account_data["impressions"].mean(),
        "clicks_daily_avg": account_data["clicks"].mean(),
        "conversions_daily_avg": account_data["conversions"].mean(),
        "spend_daily_avg": account_data["spend"].mean(),
        
        # Rate metrics
        "ctr": safe_divide(account_data["clicks"].sum(), account_data["impressions"].sum()),
        "cvr": safe_divide(account_data["conversions"].sum(), account_data["clicks"].sum()),
        "cpa": safe_divide(account_data["spend"].sum(), account_data["conversions"].sum()),
        "roas": safe_divide(account_data["revenue"].sum(), account_data["spend"].sum()),
        
        # Derived metrics
        "cpm": safe_divide(account_data["spend"].sum() * 1000, account_data["impressions"].sum()),
        "cpc": safe_divide(account_data["spend"].sum(), account_data["clicks"].sum()),
        "avg_position": account_data.get("average_position", pd.Series([0])).mean()
    }
    
    return metrics

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, returning default value if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator
```

#### 2. Trend Features

These features capture the directional movement of key metrics over time:

```python
def extract_trend_features(account_data: pd.DataFrame, window_sizes: List[int] = [7, 14, 28]) -> Dict[str, float]:
    """Extract trend-based features from account data."""
    trend_features = {}
    
    # Core metrics to analyze
    metrics = ["impressions", "clicks", "conversions", "spend", "ctr", "cvr", "cpa"]
    
    for metric in metrics:
        if metric in account_data.columns:
            # Calculate metric if not directly in data
            if metric == "ctr":
                account_data[metric] = account_data["clicks"] / account_data["impressions"].replace(0, np.nan)
            elif metric == "cvr":
                account_data[metric] = account_data["conversions"] / account_data["clicks"].replace(0, np.nan)
            elif metric == "cpa":
                account_data[metric] = account_data["spend"] / account_data["conversions"].replace(0, np.nan)
            
            # Calculate slope over different windows
            for window in window_sizes:
                if len(account_data) >= window:
                    # Use linear regression to calculate slope
                    y = account_data[metric].values[-window:]
                    x = np.arange(window)
                    slope, _, _, _, _ = stats.linregress(x, y)
                    
                    # Normalize by mean value
                    mean_value = np.mean(y)
                    if mean_value != 0:
                        normalized_slope = slope / mean_value
                    else:
                        normalized_slope = 0.0
                    
                    trend_features[f"{metric}_trend_{window}d"] = normalized_slope
                    
                    # Calculate recent change (percent change from first to last period)
                    first_half = account_data[metric].values[-window:-window//2]
                    second_half = account_data[metric].values[-window//2:]
                    
                    first_mean = np.mean(first_half) if len(first_half) > 0 else 0
                    second_mean = np.mean(second_half) if len(second_half) > 0 else 0
                    
                    if first_mean != 0:
                        percent_change = (second_mean - first_mean) / first_mean
                    else:
                        percent_change = 0.0
                    
                    trend_features[f"{metric}_change_{window}d"] = percent_change
    
    return trend_features
```

#### 3. Anomaly Features

These features capture the presence and magnitude of anomalies in various metrics:

```python
def extract_anomaly_features(account_data: pd.DataFrame, anomaly_config: Dict[str, Any] = None) -> Dict[str, float]:
    """Extract anomaly-based features from account data."""
    # Create default config if not provided
    if anomaly_config is None:
        anomaly_config = {
            "z_score": {
                "window_size": 14,
                "sensitivity": 2.5
            },
            "metrics": ["impressions", "clicks", "conversions", "spend", "ctr", "cvr", "cpa"]
        }
    
    # Initialize anomaly detector component
    anomaly_detector = AnomalyDetectionComponent(anomaly_config)
    
    # Detect anomalies
    anomaly_results = anomaly_detector.detect_anomalies(account_data)
    
    # Extract features from anomaly results
    features = {}
    
    # Individual metric anomaly scores
    for metric, result in anomaly_results.get("metric_anomalies", {}).items():
        features[f"{metric}_anomaly_score"] = result.get("score", 0.0)
        features[f"{metric}_anomaly_zscore"] = result.get("z_score", 0.0)
    
    # Overall anomaly metrics
    features["total_anomalies"] = len(anomaly_results.get("metric_anomalies", {}))
    features["overall_anomaly_score"] = anomaly_results.get("anomaly_score", 0.0)
    features["has_multivariate_anomaly"] = 1.0 if anomaly_results.get("multivariate_anomalies", {}) else 0.0
    
    return features
```

#### 4. Account Structure Features

These features capture the organization and configuration of the advertising account:

```python
def extract_structure_features(account_structure: Dict[str, Any]) -> Dict[str, float]:
    """Extract features related to account structure and organization."""
    features = {}
    
    # Campaign counts and diversity
    campaigns = account_structure.get("campaigns", [])
    features["campaign_count"] = len(campaigns)
    features["active_campaign_count"] = sum(1 for c in campaigns if c.get("status") == "ACTIVE")
    
    # Ad group structure
    ad_group_counts = [len(c.get("ad_groups", [])) for c in campaigns]
    features["total_ad_groups"] = sum(ad_group_counts)
    features["avg_ad_groups_per_campaign"] = safe_divide(sum(ad_group_counts), len(ad_group_counts))
    features["campaign_concentration"] = gini_coefficient(ad_group_counts) if ad_group_counts else 0.0
    
    # Ad diversity
    ad_counts = []
    for campaign in campaigns:
        for ad_group in campaign.get("ad_groups", []):
            ad_counts.append(len(ad_group.get("ads", [])))
    
    features["total_ads"] = sum(ad_counts)
    features["avg_ads_per_ad_group"] = safe_divide(sum(ad_counts), len(ad_counts))
    features["ad_group_concentration"] = gini_coefficient(ad_counts) if ad_counts else 0.0
    
    # Budget utilization
    daily_budgets = [c.get("daily_budget", 0) for c in campaigns]
    features["total_daily_budget"] = sum(daily_budgets)
    features["avg_daily_budget"] = safe_divide(sum(daily_budgets), len(daily_budgets))
    features["budget_concentration"] = gini_coefficient(daily_budgets) if daily_budgets else 0.0
    
    # Targeting breadth
    targeting_count = []
    for campaign in campaigns:
        for ad_group in campaign.get("ad_groups", []):
            targeting_count.append(len(ad_group.get("targeting", [])))
    
    features["avg_targeting_criteria"] = safe_divide(sum(targeting_count), len(targeting_count))
    
    return features

def gini_coefficient(values: List[float]) -> float:
    """Calculate Gini coefficient as a measure of concentration/inequality."""
    if not values or len(values) < 2:
        return 0.0
    
    values = sorted(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum(np.arange(1, n + 1) * values) / (n * np.sum(values))) - (n + 1) / n
```

#### 5. Competitive Comparison Features

These features compare account performance to industry benchmarks:

```python
def extract_competitive_features(metrics: Dict[str, float], benchmarks: Dict[str, float]) -> Dict[str, float]:
    """Extract features comparing performance to benchmarks."""
    features = {}
    
    # Compare core metrics to benchmarks
    for metric in ["ctr", "cvr", "cpa", "cpc", "cpm", "roas"]:
        if metric in metrics and metric in benchmarks and benchmarks[metric] > 0:
            # Calculate ratio of actual to benchmark
            ratio = metrics[metric] / benchmarks[metric]
            features[f"{metric}_vs_benchmark"] = ratio
            
            # Calculate percentile if available
            if f"{metric}_percentiles" in benchmarks and isinstance(benchmarks[f"{metric}_percentiles"], list):
                percentiles = benchmarks[f"{metric}_percentiles"]
                percentile_val = find_percentile(metrics[metric], percentiles)
                features[f"{metric}_percentile"] = percentile_val
    
    # Overall benchmark score (weighted average of ratios)
    if any(key.endswith("_vs_benchmark") for key in features):
        weights = {
            "ctr_vs_benchmark": 0.15,
            "cvr_vs_benchmark": 0.25,
            "cpa_vs_benchmark": 0.3,
            "roas_vs_benchmark": 0.3
        }
        
        score_components = []
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in features:
                # For metrics where lower is better (CPA), invert the ratio
                if metric in ["cpa_vs_benchmark"]:
                    value = 1.0 / features[metric]
                else:
                    value = features[metric]
                
                score_components.append(value * weight)
                total_weight += weight
        
        if total_weight > 0:
            features["overall_benchmark_score"] = sum(score_components) / total_weight
    
    return features

def find_percentile(value: float, percentiles: List[float]) -> float:
    """Find the percentile rank of a value within a list of percentile cutoffs."""
    for i, p in enumerate(percentiles):
        if value <= p:
            # Linear interpolation between percentile points
            if i == 0:
                return i / len(percentiles)
            else:
                lower = percentiles[i-1]
                upper = percentiles[i]
                lower_pct = (i-1) / len(percentiles)
                upper_pct = i / len(percentiles)
                
                # Linear interpolation
                if upper != lower:
                    return lower_pct + (value - lower) / (upper - lower) * (upper_pct - lower_pct)
                else:
                    return lower_pct
    
    # Value is higher than all percentiles
    return 1.0
```

### Feature Selection and Importance

Not all features contribute equally to the Account Health Predictor's performance. Feature importance analysis is used to identify the most relevant features:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance(model, feature_names: List[str], top_n: int = 20) -> None:
    """Analyze and visualize feature importance."""
    # Get feature importance from model
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "feature_importance_"):
        importance = model.feature_importance_
    else:
        print("Model does not expose feature importance.")
        return
    
    # Create DataFrame with feature names and importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Display top features
    top_features = feature_importance.head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    return feature_importance
```

Based on feature importance analysis, the top 10 features for account health prediction are:

1. **overall_anomaly_score**: Overall measure of anomalies across metrics
2. **cvr_vs_benchmark**: Conversion rate compared to industry benchmark
3. **cvr_trend_28d**: 28-day trend in conversion rate
4. **cpa_vs_benchmark**: Cost per acquisition compared to benchmark
5. **ctr_vs_benchmark**: Click-through rate compared to benchmark
6. **roas_trend_14d**: 14-day trend in return on ad spend
7. **cpa_anomaly_score**: Anomaly score for cost per acquisition
8. **ctr_trend_7d**: 7-day trend in click-through rate
9. **campaign_concentration**: Measure of spend concentration across campaigns
10. **ad_group_concentration**: Measure of ad distribution across ad groups

### Feature Engineering Pipeline

The complete feature engineering pipeline integrates these components into a unified process:

```python
class FeatureEngineeringPipeline:
    """Pipeline for generating features for account health prediction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the feature engineering pipeline."""
        self.config = config or {}
        self.required_metrics = ["impressions", "clicks", "conversions", "spend", "revenue"]
    
    def generate_features(self, 
                         account_data: pd.DataFrame, 
                         account_structure: Dict[str, Any],
                         benchmarks: Dict[str, float] = None) -> Dict[str, float]:
        """
        Generate all features for account health prediction.
        
        Args:
            account_data: DataFrame with historical account metrics
            account_structure: Dictionary with account structure information
            benchmarks: Optional dictionary with industry benchmarks
            
        Returns:
            Dictionary with all generated features
        """
        features = {}
        
        # Check data availability
        if len(account_data) < self.config.get("min_days", 14):
            raise ValueError(f"Insufficient data for feature generation. Need at least {self.config.get('min_days', 14)} days.")
        
        # Check required metrics
        missing_metrics = [m for m in self.required_metrics if m not in account_data.columns]
        if missing_metrics:
            raise ValueError(f"Missing required metrics: {', '.join(missing_metrics)}")
        
        # 1. Extract performance metrics
        performance_metrics = extract_performance_metrics(account_data)
        features.update(performance_metrics)
        
        # 2. Extract trend features
        window_sizes = self.config.get("trend_windows", [7, 14, 28])
        trend_features = extract_trend_features(account_data, window_sizes)
        features.update(trend_features)
        
        # 3. Extract anomaly features
        anomaly_config = self.config.get("anomaly_detection", None)
        anomaly_features = extract_anomaly_features(account_data, anomaly_config)
        features.update(anomaly_features)
        
        # 4. Extract structure features
        structure_features = extract_structure_features(account_structure)
        features.update(structure_features)
        
        # 5. Extract competitive features if benchmarks available
        if benchmarks:
            competitive_features = extract_competitive_features(performance_metrics, benchmarks)
            features.update(competitive_features)
        
        # 6. Apply feature scaling if configured
        if self.config.get("apply_scaling", False):
            features = self._scale_features(features)
        
        return features
    
    def _scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale features using predefined scaling parameters."""
        scaling_params = self.config.get("scaling_params", {})
        scaled_features = {}
        
        for feature, value in features.items():
            if feature in scaling_params:
                # Apply min-max scaling
                min_val = scaling_params[feature].get("min", 0)
                max_val = scaling_params[feature].get("max", 1)
                
                if max_val > min_val:
                    scaled_value = (value - min_val) / (max_val - min_val)
                    scaled_features[feature] = max(0, min(1, scaled_value))
                else:
                    scaled_features[feature] = value
            else:
                # No scaling for this feature
                scaled_features[feature] = value
        
        return scaled_features
```

The feature engineering process transforms raw account data into a rich set of features that capture performance, trends, anomalies, structure, and competitive positioning, providing a comprehensive basis for account health prediction.

## Training Process

The Account Health Predictor employs a sophisticated training process that combines multiple model components and ensures robust predictive performance across diverse advertising accounts. This section details the training methodology, data preparation, model optimization, and validation approaches.

### Data Collection and Preprocessing

Before training can begin, historical account data must be collected and prepared:

```python
def prepare_training_data(accounts_data: List[Dict], 
                         benchmarks: Dict[str, Dict] = None,
                         feature_config: Dict = None,
                         label_config: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature and label data for training the Account Health Predictor.
    
    Args:
        accounts_data: List of dictionaries with account data and structure
        benchmarks: Optional benchmarks by industry/vertical
        feature_config: Configuration for feature engineering
        label_config: Configuration for label generation
        
    Returns:
        X: Feature DataFrame for training
        y: Label DataFrame for training
    """
    feature_engineering = FeatureEngineeringPipeline(config=feature_config)
    features_list = []
    labels_list = []
    
    for account in accounts_data:
        try:
            # Extract account metadata
            account_id = account.get("account_id")
            platform = account.get("platform")
            industry = account.get("industry")
            
            # Get relevant benchmark if available
            account_benchmark = None
            if benchmarks and industry in benchmarks:
                account_benchmark = benchmarks[industry]
            
            # Generate features for this account
            metrics_data = pd.DataFrame(account.get("metrics_data", []))
            account_structure = account.get("account_structure", {})
            
            if len(metrics_data) < feature_config.get("min_days", 14):
                logging.warning(f"Skipping account {account_id} due to insufficient data")
                continue
                
            # Generate features
            features = feature_engineering.generate_features(
                metrics_data, 
                account_structure,
                account_benchmark
            )
            
            # Add metadata as features
            features["account_id"] = account_id
            features["platform"] = platform
            features["industry"] = industry
            
            # Generate labels
            labels = generate_labels(account, label_config)
            
            features_list.append(features)
            labels_list.append(labels)
            
        except Exception as e:
            logging.error(f"Error processing account {account.get('account_id')}: {e}")
            continue
    
    # Convert to DataFrames
    X = pd.DataFrame(features_list)
    y = pd.DataFrame(labels_list)
    
    return X, y

def generate_labels(account: Dict, config: Dict = None) -> Dict[str, float]:
    """
    Generate training labels from account data.
    
    Args:
        account: Dictionary with account data
        config: Configuration for label generation
        
    Returns:
        Dictionary with label values
    """
    labels = {}
    metrics_data = pd.DataFrame(account.get("metrics_data", []))
    
    # Default configurations
    if config is None:
        config = {
            "performance_window": 14,  # Days to use for performance calculation
            "forecast_window": 7,      # Days to forecast for labels
            "improvement_threshold": 0.1  # Threshold for significant improvement
        }
    
    # Calculate base labels
    
    # 1. Overall Health Score (if manually labeled)
    if "health_score" in account:
        labels["health_score"] = account["health_score"]
    
    # 2. Future Performance (forecasting targets)
    if len(metrics_data) >= config["performance_window"] + config["forecast_window"]:
        # Split data into training and "future" periods
        training_data = metrics_data.iloc[:-config["forecast_window"]]
        future_data = metrics_data.iloc[-config["forecast_window"]:]
        
        # Calculate key metrics for both periods
        training_metrics = {
            "ctr": safe_divide(training_data["clicks"].sum(), training_data["impressions"].sum()),
            "cvr": safe_divide(training_data["conversions"].sum(), training_data["clicks"].sum()),
            "cpa": safe_divide(training_data["spend"].sum(), training_data["conversions"].sum()),
            "roas": safe_divide(training_data["revenue"].sum(), training_data["spend"].sum())
        }
        
        future_metrics = {
            "ctr": safe_divide(future_data["clicks"].sum(), future_data["impressions"].sum()),
            "cvr": safe_divide(future_data["conversions"].sum(), future_data["clicks"].sum()),
            "cpa": safe_divide(future_data["spend"].sum(), future_data["conversions"].sum()),
            "roas": safe_divide(future_data["revenue"].sum(), future_data["spend"].sum())
        }
        
        # Calculate percent changes
        for metric in ["ctr", "cvr", "cpa", "roas"]:
            if training_metrics[metric] > 0:
                pct_change = (future_metrics[metric] - training_metrics[metric]) / training_metrics[metric]
                
                # For CPA, invert the change (lower is better)
                if metric == "cpa":
                    pct_change = -pct_change
                
                labels[f"future_{metric}_pct_change"] = pct_change
    
    # 3. Performance Classification Labels
    performance_issues = account.get("issues", [])
    
    # Binary classification labels for common issues
    issue_types = [
        "low_quality_score", "budget_limited", "low_conversion_rate",
        "high_cpa", "low_impression_share", "poor_targeting",
        "creative_fatigue", "landing_page_issues"
    ]
    
    for issue in issue_types:
        labels[f"has_{issue}"] = 1.0 if issue in performance_issues else 0.0
    
    # 4. Optimization Opportunities
    optimizations = account.get("optimization_history", [])
    recent_optimizations = [o for o in optimizations if o.get("days_ago", 100) <= 30]
    
    # Binary labels for optimizations that led to improvements
    if recent_optimizations:
        successful_actions = set()
        
        for opt in recent_optimizations:
            action = opt.get("action")
            improvement = opt.get("improvement", 0.0)
            
            if action and improvement > config["improvement_threshold"]:
                successful_actions.add(action)
        
        # Common optimization types
        optimization_types = [
            "budget_adjustment", "bid_adjustment", "creative_refresh",
            "targeting_refinement", "campaign_restructure", "keyword_expansion",
            "negative_keyword_addition", "landing_page_optimization"
        ]
        
        for opt_type in optimization_types:
            labels[f"effective_{opt_type}"] = 1.0 if opt_type in successful_actions else 0.0
    
    return labels
```

### Model Training Components

Each component of the Account Health Predictor requires a specialized training approach. This includes training methods for:

1. **Time Series Component**: LSTM-based models to predict future performance metrics
2. **Classification Component**: Gradient boosting models to determine account health and identify issues
3. **Recommendation Engine**: Combines rule-based and ML-based approaches for action recommendations
4. **Ensemble Integration**: Methods to integrate the individual components into a unified system

The training process includes rigorous cross-validation, hyperparameter optimization, and model evaluation to ensure the highest quality predictions and recommendations.

## Evaluation Methodology

The Account Health Predictor undergoes comprehensive evaluation to ensure its predictions are accurate, reliable, and actionable. This section details the evaluation approaches used to validate the model's performance across different dimensions.

### Evaluation Framework

The evaluation framework follows a multi-faceted approach to assess all aspects of the system:

```python
class AccountHealthEvaluator:
    """Framework for evaluating Account Health Predictor performance."""
    
    def __init__(self, predictor: AccountHealthPredictor, evaluation_config: Dict[str, Any] = None):
        """
        Initialize the evaluator.
        
        Args:
            predictor: Trained Account Health Predictor to evaluate
            evaluation_config: Configuration for evaluation parameters
        """
        self.predictor = predictor
        self.config = evaluation_config or {}
        self.results = {}
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation on test data.
        
        Args:
            test_data: Dictionary with test data components
            
        Returns:
            Dictionary with evaluation results
        """
        # Evaluate each component
        self._evaluate_health_score(test_data)
        self._evaluate_issue_detection(test_data)
        self._evaluate_time_series(test_data)
        self._evaluate_recommendations(test_data)
        self._evaluate_explainability(test_data)
        
        # Calculate overall score
        self._calculate_overall_score()
        
        return self.results
    
    def _evaluate_health_score(self, test_data: Dict[str, Any]) -> None:
        """Evaluate health score prediction accuracy."""
        features = test_data.get("features")
        actual_scores = test_data.get("health_scores")
        
        if features is None or actual_scores is None:
            self.results["health_score"] = {"error": "Missing test data"}
            return
        
        # Generate predictions
        predicted_scores = self.predictor.predict_health_scores(features)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_scores, predicted_scores))
        mae = mean_absolute_error(actual_scores, predicted_scores)
        r2 = r2_score(actual_scores, predicted_scores)
        
        # Calculate error distribution
        errors = predicted_scores - actual_scores
        error_std = np.std(errors)
        error_percentiles = np.percentile(np.abs(errors), [50, 90, 95])
        
        self.results["health_score"] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "error_std": error_std,
            "median_abs_error": error_percentiles[0],
            "90th_percentile_error": error_percentiles[1],
            "95th_percentile_error": error_percentiles[2]
        }
    
    def _evaluate_issue_detection(self, test_data: Dict[str, Any]) -> None:
        """Evaluate issue detection accuracy."""
        features = test_data.get("features")
        actual_issues = test_data.get("issues")
        
        if features is None or actual_issues is None:
            self.results["issue_detection"] = {"error": "Missing test data"}
            return
        
        issue_results = {}
        avg_metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "auc": 0
        }
        
        for issue_type, actual_labels in actual_issues.items():
            # Skip if no actual instances
            if sum(actual_labels) == 0:
                continue
                
            # Generate predictions
            predicted_probs = self.predictor.predict_issue_probability(features, issue_type)
            predicted_labels = (predicted_probs >= 0.5).astype(int)
            
            # Calculate metrics
            precision = precision_score(actual_labels, predicted_labels)
            recall = recall_score(actual_labels, predicted_labels)
            f1 = f1_score(actual_labels, predicted_labels)
            auc = roc_auc_score(actual_labels, predicted_probs)
            
            # Store results
            issue_results[issue_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "confusion_matrix": confusion_matrix(actual_labels, predicted_labels).tolist(),
                "prevalence": sum(actual_labels) / len(actual_labels)
            }
            
            # Update averages
            for metric in avg_metrics:
                avg_metrics[metric] += issue_results[issue_type][metric]
        
        # Calculate averages
        if issue_results:
            for metric in avg_metrics:
                avg_metrics[metric] /= len(issue_results)
        
        self.results["issue_detection"] = {
            "by_issue": issue_results,
            "average": avg_metrics
        }
    
    def _evaluate_time_series(self, test_data: Dict[str, Any]) -> None:
        """Evaluate time series forecasting accuracy."""
        time_series_data = test_data.get("time_series")
        
        if time_series_data is None:
            self.results["time_series"] = {"error": "Missing test data"}
            return
        
        metric_results = {}
        avg_metrics = {
            "mape": 0,
            "mae": 0,
            "rmse": 0
        }
        
        for metric, series_data in time_series_data.items():
            # Create test sequences
            test_sequences = []
            for i in range(0, len(series_data) - 21, 7):  # Non-overlapping weekly sequences
                if i + 21 <= len(series_data):  # Need 14 days for input, 7 for forecast
                    test_sequences.append((series_data[i:i+14], series_data[i+14:i+21]))
            
            if not test_sequences:
                continue
            
            # Evaluate each sequence
            maes = []
            mapes = []
            rmses = []
            
            for input_seq, actual_future in test_sequences:
                # Generate forecast
                forecast = self.predictor.forecast_metric(input_seq, metric=metric)
                
                # Calculate errors
                mae = np.mean(np.abs(forecast - actual_future))
                rmse = np.sqrt(np.mean((forecast - actual_future) ** 2))
                
                # Calculate MAPE, handling zeros in actual values
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape_values = np.abs((actual_future - forecast) / actual_future)
                    mape_values = mape_values[~np.isnan(mape_values) & ~np.isinf(mape_values)]
                    mape = np.mean(mape_values) if len(mape_values) > 0 else np.nan
                
                maes.append(mae)
                mapes.append(mape)
                rmses.append(rmse)
            
            # Store results
            metric_results[metric] = {
                "mae": np.mean(maes),
                "mape": np.nanmean(mapes),
                "rmse": np.mean(rmses),
                "num_test_sequences": len(test_sequences)
            }
            
            # Update averages
            for metric_name in avg_metrics:
                if metric_name == "mape":
                    avg_metrics[metric_name] += np.nanmean(mapes)
                else:
                    avg_metrics[metric_name] += metric_results[metric][metric_name]
        
        # Calculate averages
        if metric_results:
            for metric_name in avg_metrics:
                avg_metrics[metric_name] /= len(metric_results)
        
        self.results["time_series"] = {
            "by_metric": metric_results,
            "average": avg_metrics
        }
    
    def _evaluate_recommendations(self, test_data: Dict[str, Any]) -> None:
        """Evaluate recommendation quality."""
        features = test_data.get("features")
        actual_outcomes = test_data.get("optimization_outcomes")
        
        if features is None or actual_outcomes is None:
            self.results["recommendations"] = {"error": "Missing test data"}
            return
        
        # Generate recommendations for test accounts
        all_recommendations = []
        for i in range(len(features)):
            account_features = {k: v for k, v in zip(features.columns, features.iloc[i])}
            recommendations = self.predictor.generate_recommendations(account_features)
            all_recommendations.append(recommendations)
        
        # Evaluate recommendation quality
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in [1, 3, 5]:
            precisions = []
            recalls = []
            ndcgs = []
            
            for i, recs in enumerate(all_recommendations):
                # Get top-k recommendations
                top_k_recs = [r["action"] for r in recs[:k]]
                
                # Get actual effective actions
                actual_effective = actual_outcomes[i].get("effective_actions", [])
                
                if not actual_effective:
                    continue
                
                # Calculate precision and recall
                hits = len(set(top_k_recs) & set(actual_effective))
                precision = hits / k
                recall = hits / len(actual_effective)
                
                precisions.append(precision)
                recalls.append(recall)
                
                # Calculate NDCG
                relevance = [1 if rec in actual_effective else 0 for rec in top_k_recs]
                ideal_relevance = [1] * min(k, len(actual_effective)) + [0] * max(0, k - len(actual_effective))
                
                ndcg = self._calculate_ndcg(relevance, ideal_relevance)
                ndcgs.append(ndcg)
            
            if precisions:
                precision_at_k[k] = np.mean(precisions)
                recall_at_k[k] = np.mean(recalls)
                ndcg_at_k[k] = np.mean(ndcgs)
        
        self.results["recommendations"] = {
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "ndcg_at_k": ndcg_at_k
        }
    
    def _calculate_ndcg(self, relevance: List[int], ideal_relevance: List[int]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance)))
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _evaluate_explainability(self, test_data: Dict[str, Any]) -> None:
        """Evaluate model explainability."""
        features = test_data.get("features")
        
        if features is None or len(features) == 0:
            self.results["explainability"] = {"error": "Missing test data"}
            return
        
        # Sample accounts for SHAP analysis
        sample_size = min(100, len(features))
        sample_indices = np.random.choice(len(features), sample_size, replace=False)
        sample_features = features.iloc[sample_indices]
        
        # Get feature importance
        importance = self.predictor.get_feature_importance()
        
        # Get SHAP values if available
        shap_values = None
        try:
            shap_values = self.predictor.get_shap_values(sample_features)
        except:
            pass
        
        self.results["explainability"] = {
            "feature_importance": importance,
            "shap_available": shap_values is not None
        }
    
    def _calculate_overall_score(self) -> None:
        """Calculate overall evaluation score."""
        # Default weights for components
        weights = self.config.get("component_weights", {
            "health_score": 0.3,
            "issue_detection": 0.3,
            "time_series": 0.2,
            "recommendations": 0.2
        })
        
        overall_score = 0.0
        component_scores = {}
        
        # Calculate health score component
        if "health_score" in self.results and "error" not in self.results["health_score"]:
            # R² score (0-1 scale where higher is better)
            r2 = max(0, min(1, self.results["health_score"]["r2"]))
            
            # RMSE on a 0-1 scale (lower is better)
            rmse_max = 100  # Assuming health scores are 0-100
            rmse_score = 1 - min(1, self.results["health_score"]["rmse"] / rmse_max)
            
            component_scores["health_score"] = 0.7 * r2 + 0.3 * rmse_score
        
        # Calculate issue detection component
        if "issue_detection" in self.results and "error" not in self.results["issue_detection"]:
            avg = self.results["issue_detection"]["average"]
            # F1 and AUC on 0-1 scale (higher is better)
            component_scores["issue_detection"] = 0.5 * avg["f1"] + 0.5 * avg["auc"]
        
        # Calculate time series component
        if "time_series" in self.results and "error" not in self.results["time_series"]:
            avg = self.results["time_series"]["average"]
            # MAPE on a 0-1 scale (lower is better)
            mape_score = max(0, 1 - min(1, avg["mape"] / 2))  # Cap at 200% error
            component_scores["time_series"] = mape_score
        
        # Calculate recommendations component
        if "recommendations" in self.results and "error" not in self.results["recommendations"]:
            # Average precision@k and ndcg@k (higher is better)
            precision = np.mean(list(self.results["recommendations"]["precision_at_k"].values()))
            ndcg = np.mean(list(self.results["recommendations"]["ndcg_at_k"].values()))
            component_scores["recommendations"] = 0.5 * precision + 0.5 * ndcg
        
        # Calculate weighted overall score
        for component, score in component_scores.items():
            overall_score += score * weights.get(component, 0)
        
        # Normalize if not all components were evaluated
        total_weight = sum(weights.get(component, 0) for component in component_scores)
        if total_weight > 0:
            overall_score /= total_weight
        
        self.results["overall"] = {
            "score": overall_score,
            "component_scores": component_scores
        }
```

### Evaluation Metrics

The Account Health Predictor is evaluated using a comprehensive set of metrics across its components:

#### 1. Health Score Prediction

| Metric | Description | Target |
|--------|-------------|--------|
| RMSE | Root Mean Square Error | < 10.0 |
| MAE | Mean Absolute Error | < 7.5 |
| R² | Coefficient of determination | > 0.7 |
| Error Distribution | 90th percentile of absolute errors | < 15.0 |

#### 2. Issue Detection

| Metric | Description | Target |
|--------|-------------|--------|
| Precision | Precision of issue detection | > 0.8 |
| Recall | Recall of issue detection | > 0.7 |
| F1 Score | Harmonic mean of precision and recall | > 0.75 |
| AUC | Area under ROC curve | > 0.85 |

#### 3. Time Series Forecasting

| Metric | Description | Target |
|--------|-------------|--------|
| MAPE | Mean Absolute Percentage Error | < 25% |
| MAE | Mean Absolute Error | Varies by metric |
| Forecast Bias | Systematic over/under-prediction | -5% to +5% |

#### 4. Recommendation Quality

| Metric | Description | Target |
|--------|-------------|--------|
| Precision@k | Precision of top-k recommendations | > 0.6 |
| Recall@k | Recall of top-k recommendations | > 0.4 |
| NDCG@k | Normalized Discounted Cumulative Gain | > 0.7 |

### Evaluation Process

The evaluation process follows these steps:

1. **Data Preparation**: Test data is prepared with known ground truth labels, including:
   - Manually labeled account health scores
   - Identified issues from expert review
   - Future performance metrics for forecasting evaluation
   - Historical optimization actions and their outcomes

2. **Model Prediction**: The Account Health Predictor generates predictions for:
   - Overall health scores
   - Issue detection probabilities
   - Performance forecasts
   - Recommended optimization actions

3. **Metric Calculation**: Evaluation metrics are calculated by comparing predictions to ground truth.

4. **Comparison to Baselines**: Results are compared against:
   - Previous model versions
   - Simple baseline models (e.g., moving average for time series)
   - Human expert assessments

5. **Error Analysis**: Detailed error analysis to identify:
   - Systematic biases in predictions
   - Account segments with lower performance
   - Specific issues with poor detection rates

### Fairness and Robustness Evaluation

The Account Health Predictor is also evaluated for fairness and robustness:

1. **Platform Fairness**: Ensuring consistent performance across advertising platforms (Google, Facebook, TikTok, etc.)

2. **Industry Fairness**: Evaluating performance across different industry verticals and account sizes

3. **Robustness Testing**:
   - Performance with missing data
   - Resistance to data outliers and anomalies
   - Stability of predictions over time

### Explainability Evaluation

The explainability of predictions is evaluated through:

1. **Feature Importance Analysis**: Ensuring predictions rely on relevant features

2. **SHAP Value Analysis**: Assessing individual prediction explanations

3. **Recommendation Justification**: Evaluating the quality of justifications for recommendations

### Ongoing Evaluation

The Account Health Predictor is subject to continuous evaluation:

1. **A/B Testing**: Comparing model-driven vs. expert-driven recommendations

2. **Feedback Loop**: Incorporating user feedback on prediction accuracy

3. **Monitoring Drift**: Tracking performance over time to detect model degradation

4. **Performance Dashboards**: Real-time tracking of key evaluation metrics

The comprehensive evaluation methodology ensures the Account Health Predictor delivers reliable, actionable insights that drive measurable improvements in advertising account performance.

## Inference Pipeline

The inference pipeline transforms raw account data into actionable health assessments and recommendations. This section details the end-to-end process of generating account health predictions in production.

### Overview

The Account Health Predictor's inference pipeline follows these key steps:

1. Data Ingestion and Preprocessing
2. Feature Extraction
3. Model Prediction
4. Post-processing and Contextual Enrichment
5. Results Storage and Delivery
6. Monitoring and Feedback Collection

This pipeline is designed for both batch processing of accounts and real-time assessment of individual accounts, with appropriate optimizations for each use case.

### Inference Flow

```python
def predict_account_health(
    account_data: Dict[str, Any],
    predictor: AccountHealthPredictor,
    configuration: Dict[str, Any] = None,
    context_data: Dict[str, Any] = None,
    monitoring_service: Optional[ProductionMonitoringService] = None
) -> Dict[str, Any]:
    """
    Generate health predictions for an advertising account.
    
    Args:
        account_data: Raw account data dictionary containing metrics, structure, and history
        predictor: Trained AccountHealthPredictor instance
        configuration: Optional configuration parameters for prediction
        context_data: Optional contextual data (industry benchmarks, seasonality factors)
        monitoring_service: Optional monitoring service for tracking prediction performance
        
    Returns:
        Dictionary containing health assessment, issue predictions, and recommendations
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # 1. Validate input data
        if not _validate_account_data(account_data):
            return {
                "status": "error",
                "message": "Invalid account data format or missing required fields",
                "request_id": request_id
            }
        
        # 2. Preprocess account data
        processed_data = _preprocess_account_data(account_data, configuration)
        
        # 3. Generate features
        features = predictor.feature_pipeline.transform(processed_data)
        
        # 4. Get predictions from model
        predictions = predictor.predict(features)
        
        # 5. Enrich predictions with contextual information
        if context_data:
            predictions = _enrich_predictions(predictions, context_data)
        
        # 6. Generate recommendations
        recommendations = predictor.generate_recommendations(
            account_data=processed_data,
            predictions=predictions,
            configuration=configuration
        )
        
        # 7. Assemble final response
        response = {
            "status": "success",
            "request_id": request_id,
            "predictions": predictions,
            "recommendations": recommendations,
            "metadata": {
                "model_version": predictor.version,
                "timestamp": datetime.now().isoformat(),
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        }
        
        # 8. Track prediction in monitoring service
        if monitoring_service:
            monitoring_service.record_prediction(
                model_id="account_health_predictor",
                prediction=predictions,
                features=features,
                metadata={
                    "account_id": account_data.get("account_id"),
                    "request_id": request_id
                },
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        return response
    
    except Exception as e:
        error_response = {
            "status": "error",
            "message": f"Prediction error: {str(e)}",
            "request_id": request_id
        }
        
        # Log error in monitoring service
        if monitoring_service:
            monitoring_service.send_alert(
                severity=AlertLevel.ERROR,
                message=f"Account health prediction failed: {str(e)}",
                data={"account_id": account_data.get("account_id"), "request_id": request_id}
            )
        
        return error_response
```

### Data Preprocessing

The preprocessing step ensures that input data is properly formatted and contains all required fields for prediction:

```python
def _preprocess_account_data(account_data: Dict[str, Any], 
                           configuration: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Preprocess account data for prediction.
    
    Args:
        account_data: Raw account data
        configuration: Configuration parameters
        
    Returns:
        Preprocessed account data
    """
    processed_data = copy.deepcopy(account_data)
    
    # 1. Normalize metric names
    processed_data = _normalize_metric_names(processed_data)
    
    # 2. Fill missing values using appropriate strategies
    processed_data = _fill_missing_values(processed_data, configuration)
    
    # 3. Calculate derived metrics
    processed_data = _calculate_derived_metrics(processed_data)
    
    # 4. Apply time window filtering
    time_window_days = configuration.get("time_window_days", 90) if configuration else 90
    processed_data = _filter_time_window(processed_data, time_window_days)
    
    # 5. Apply normalization/scaling if needed
    if configuration and configuration.get("normalize_metrics", True):
        processed_data = _normalize_metrics(processed_data)
    
    return processed_data
```

### Batch Inference

For processing multiple accounts efficiently, the pipeline includes a batch processing mode:

```python
def batch_predict_account_health(
    accounts_data: List[Dict[str, Any]],
    predictor: AccountHealthPredictor,
    configuration: Dict[str, Any] = None,
    context_data: Dict[str, Any] = None,
    monitoring_service: Optional[ProductionMonitoringService] = None,
    batch_size: int = 50,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Generate health predictions for multiple advertising accounts.
    
    Args:
        accounts_data: List of account data dictionaries
        predictor: Trained AccountHealthPredictor instance
        configuration: Optional configuration parameters 
        context_data: Optional contextual data
        monitoring_service: Optional monitoring service
        batch_size: Number of accounts to process in each batch
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with prediction results for all accounts
    """
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    total_accounts = len(accounts_data)
    results = {}
    
    # Process in batches to optimize memory usage
    for i in range(0, total_accounts, batch_size):
        batch = accounts_data[i:i+batch_size]
        batch_results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create future-to-account mapping
            future_to_account = {
                executor.submit(
                    predict_account_health,
                    account_data,
                    predictor,
                    configuration,
                    context_data,
                    monitoring_service
                ): account_data.get("account_id", f"unknown_{j}")
                for j, account_data in enumerate(batch)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_account):
                account_id = future_to_account[future]
                try:
                    batch_results[account_id] = future.result()
                except Exception as e:
                    batch_results[account_id] = {
                        "status": "error",
                        "message": f"Error processing account: {str(e)}",
                        "account_id": account_id
                    }
        
        # Merge batch results into overall results
        results.update(batch_results)
    
    # Compile batch summary
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    error_count = total_accounts - success_count
    
    # Track batch processing in monitoring service
    if monitoring_service:
        monitoring_service.update_model_metric(
            model_id="account_health_predictor",
            metric="batch_processing_time_ms",
            value=int((time.time() - start_time) * 1000),
            metadata={
                "batch_id": batch_id,
                "total_accounts": total_accounts,
                "success_count": success_count,
                "error_count": error_count
            }
        )
    
    return {
        "status": "complete",
        "batch_id": batch_id,
        "total_accounts": total_accounts,
        "success_count": success_count,
        "error_count": error_count,
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "results": results
    }
```

### Monitoring Integration

The predictor is tightly integrated with production monitoring to ensure prediction quality and performance over time:

```python
def setup_health_predictor_monitoring(model_id: str = "account_health_predictor") -> ProductionMonitoringService:
    """
    Set up monitoring for the Account Health Predictor.
    
    Args:
        model_id: Identifier for the model
        
    Returns:
        Configured monitoring service instance
    """
    # Create monitoring configuration
    config = ModelMonitoringConfig(
        model_id=model_id,
        performance_metrics=[
            "accuracy", 
            "latency_ms", 
            "error_rate",
            "recommendation_acceptance_rate",
            "drift_score"
        ],
        drift_detection_interval=60,  # minutes
        performance_threshold={
            "accuracy": 0.85,
            "latency_ms": 500,
            "error_rate": 0.05,
            "recommendation_acceptance_rate": 0.30,
            "drift_score": 0.15
        },
        alert_channels=["log", "email", "slack"],
        log_predictions=True,
        retention_days=90,
        sampling_rate=0.5  # Sample 50% of predictions for monitoring
    )
    
    # Create monitoring service
    monitoring_service = ProductionMonitoringService(
        storage_path="/var/log/within/model_monitoring"
    )
    
    # Register model with monitoring service
    monitoring_service.register_model(config)
    
    # Set up custom alert handlers if needed
    monitoring_service.register_alert_handler("slack", send_slack_alert)
    monitoring_service.register_alert_handler("email", send_email_alert)
    
    return monitoring_service
```

### Feedback Integration

The inference pipeline includes mechanisms to collect feedback on predictions and recommendations, which are used to continuously improve the model:

```python
def record_account_health_feedback(
    prediction_id: str,
    feedback: Dict[str, Any],
    monitoring_service: ProductionMonitoringService
) -> Dict[str, Any]:
    """
    Record feedback on account health predictions.
    
    Args:
        prediction_id: ID of the prediction
        feedback: Feedback data including corrected values, user actions, etc.
        monitoring_service: Monitoring service instance
        
    Returns:
        Status of the feedback recording
    """
    # Validate feedback
    if not _validate_feedback(feedback):
        return {
            "status": "error",
            "message": "Invalid feedback format"
        }
    
    # Record feedback in monitoring
    is_correct = feedback.get("is_correct", False)
    action_taken = feedback.get("action_taken", False)
    
    # Track accuracy metric
    monitoring_service.update_model_metric(
        model_id="account_health_predictor",
        metric="accuracy",
        value=1.0 if is_correct else 0.0,
        metadata={
            "prediction_id": prediction_id,
            "feedback_timestamp": datetime.now().isoformat(),
            "feedback_source": feedback.get("source", "unknown")
        }
    )
    
    # Track recommendation acceptance if an action was taken
    if "recommendation_id" in feedback:
        monitoring_service.update_model_metric(
            model_id="account_health_predictor",
            metric="recommendation_acceptance_rate",
            value=1.0 if action_taken else 0.0,
            metadata={
                "prediction_id": prediction_id,
                "recommendation_id": feedback.get("recommendation_id"),
                "feedback_timestamp": datetime.now().isoformat()
            }
        )
    
    # Store detailed feedback for future model improvement
    feedback_id = str(uuid.uuid4())
    feedback_record = {
        "feedback_id": feedback_id,
        "prediction_id": prediction_id,
        "timestamp": datetime.now().isoformat(),
        "is_correct": is_correct,
        "action_taken": action_taken,
        "details": feedback.get("details", {})
    }
    
    # Store feedback for later analysis and model improvement
    store_feedback(feedback_record)
    
    return {
        "status": "success",
        "message": "Feedback recorded successfully",
        "feedback_id": feedback_id
    }
```

### Deployment Considerations

The Account Health Predictor's inference pipeline is designed for deployment in various environments:

1. **Standalone API Service**: Deployed as a dedicated API service with RESTful endpoints for prediction
2. **Batch Processing Job**: Scheduled batch processing for all accounts in the system
3. **Embedded Mode**: Integrated directly into the advertising platform backend

For production deployment, the following considerations are important:

- **Scaling**: The predictor service is designed to scale horizontally, with stateless prediction instances sharing a common model store
- **Performance**: Optimized for low-latency prediction (<300ms per account) and efficient batch processing
- **Resource Requirements**: 
  - CPU: 4+ cores recommended for optimal batch processing
  - Memory: 4GB minimum, 8GB recommended for larger account datasets
  - Storage: 500MB for model artifacts and 1GB for prediction caching
- **Caching**: Predictions are cached with a configurable TTL to reduce computational load
- **Monitoring**: Built-in integration with the Production Monitoring Service for real-time performance tracking
- **Fallbacks**: Graceful degradation mechanisms in case of component failures or resource constraints

These deployment considerations ensure that the Account Health Predictor can reliably serve predictions in a production environment with appropriate performance, monitoring, and resilience.

## Integration Points

The Account Health Predictor is designed to be integrated with various components of the WITHIN platform, including:

1. **Advertising Platform**: Used to collect and process account data
2. **Monitoring Service**: Used to track prediction performance and collect feedback
3. **Recommendation Engine**: Used to generate and implement recommendations
4. **Account Health Dashboard**: Used to visualize account health and performance metrics
5. **Optimization Pipeline**: Used to implement recommended optimizations

The integration points ensure that the Account Health Predictor is seamlessly integrated into the existing advertising ecosystem, providing value to advertisers and maintaining optimal account performance.

### API Integration

The Account Health Predictor exposes a RESTful API for seamless integration with other system components:

```python
@router.post("/api/v1/account-health/predict", response_model=AccountHealthPredictionResponse)
async def predict_account_health_endpoint(
    request: AccountHealthPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    predictor: AccountHealthPredictor = Depends(get_account_health_predictor)
) -> AccountHealthPredictionResponse:
    """
    Generate account health prediction for the specified account.
    
    This endpoint requires authentication and permissions to access the account data.
    """
    # Validate authorization
    if not has_account_access(current_user, request.account_id):
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this account"
        )
    
    # Get account data
    account_data = await get_account_data(request.account_id)
    
    # Get context data if requested
    context_data = None
    if request.include_context:
        context_data = await get_context_data(
            account_id=request.account_id,
            industry=account_data.get("industry"),
            platform=account_data.get("platform")
        )
    
    # Get monitoring service
    monitoring_service = get_monitoring_service()
    
    # Generate prediction
    prediction_result = predict_account_health(
        account_data=account_data,
        predictor=predictor,
        configuration=request.configuration,
        context_data=context_data,
        monitoring_service=monitoring_service
    )
    
    # Store prediction for later reference
    background_tasks.add_task(
        store_prediction,
        account_id=request.account_id,
        prediction=prediction_result,
        user_id=current_user.id
    )
    
    # Track API usage
    background_tasks.add_task(
        track_api_usage,
        endpoint="/api/v1/account-health/predict",
        user_id=current_user.id,
        account_id=request.account_id
    )
    
    return prediction_result
```

### Data Collection Integration

The predictor integrates with data collection systems to gather the required inputs:

```python
async def get_account_data(account_id: str) -> Dict[str, Any]:
    """
    Fetch account data for health prediction.
    
    This function collects all required data from various sources,
    including performance metrics, account structure, and historical data.
    """
    # Get data service client
    data_client = get_data_service_client()
    
    # Fetch account metadata
    account_metadata = await data_client.get_account_metadata(account_id)
    
    # Fetch performance metrics
    metrics = await data_client.get_account_metrics(
        account_id=account_id,
        days=90,  # Get 90 days of historical data
        metrics=[
            "impressions", "clicks", "conversions", "spend", "revenue",
            "ctr", "cvr", "cpa", "roas", "average_position"
        ],
        granularity="daily"
    )
    
    # Fetch account structure
    structure = await data_client.get_account_structure(account_id)
    
    # Fetch optimization history
    optimization_history = await data_client.get_optimization_history(
        account_id=account_id,
        days=180  # Get 6 months of optimization history
    )
    
    # Assemble complete account data
    account_data = {
        "account_id": account_id,
        "platform": account_metadata.get("platform"),
        "industry": account_metadata.get("industry"),
        "vertical": account_metadata.get("vertical"),
        "creation_date": account_metadata.get("creation_date"),
        "spend_tier": account_metadata.get("spend_tier"),
        "metrics_data": metrics,
        "account_structure": structure,
        "optimization_history": optimization_history
    }
    
    return account_data
```

### Dashboard Integration

The Account Health Predictor is integrated with the WITHIN dashboard for visualization and reporting:

```python
def get_dashboard_health_data(account_id: str) -> Dict[str, Any]:
    """
    Generate dashboard data for account health visualization.
    
    This function formats health prediction data for dashboard display
    and includes visualization-ready metrics and charts.
    """
    # Get latest prediction for account
    prediction = get_latest_prediction(account_id)
    
    if not prediction:
        return {"status": "no_prediction_available"}
    
    # Format health score gauge data
    health_gauge = {
        "score": prediction["predictions"]["health_score"],
        "class": prediction["predictions"]["health_class"],
        "trend": calculate_health_score_trend(account_id),
        "thresholds": {
            "critical": 25,
            "poor": 50,
            "fair": 70,
            "good": 85
        }
    }
    
    # Format issue chart data
    issues_chart = {
        "categories": [],
        "scores": []
    }
    
    for issue, score in prediction["predictions"].get("issue_probabilities", {}).items():
        issues_chart["categories"].append(format_issue_name(issue))
        issues_chart["scores"].append(round(score * 100))
    
    # Format recommendations for display
    recommendations = []
    for rec in prediction.get("recommendations", []):
        recommendations.append({
            "id": rec["id"],
            "text": rec["text"],
            "priority": rec["priority"],
            "impact": rec["impact"],
            "status": get_recommendation_status(account_id, rec["id"]),
            "implementation_url": generate_implementation_url(account_id, rec["id"], rec.get("factors", []))
        })
    
    # Format forecast charts
    forecast_charts = {}
    for metric, forecast in prediction["predictions"].get("forecasts", {}).items():
        forecast_charts[metric] = {
            "dates": [d.strftime("%Y-%m-%d") for d in forecast["dates"]],
            "actual": forecast["actual"],
            "predicted": forecast["predicted"],
            "upper_bound": forecast.get("upper_bound"),
            "lower_bound": forecast.get("lower_bound")
        }
    
    # Assemble dashboard data
    dashboard_data = {
        "health_gauge": health_gauge,
        "issues_chart": issues_chart,
        "recommendations": recommendations,
        "forecast_charts": forecast_charts,
        "last_updated": prediction["metadata"]["timestamp"],
        "history": get_health_history(account_id),
        "comparison": get_industry_comparison(account_id, prediction["predictions"])
    }
    
    return dashboard_data
```

### Recommendation Engine Integration

The Account Health Predictor integrates with the Recommendation Engine to generate and track action items:

```python
def implement_recommendation(
    account_id: str,
    recommendation_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Implement a recommendation from the Account Health Predictor.
    
    This function applies the recommended changes to the account
    and tracks the implementation for feedback and evaluation.
    """
    # Get recommendation details
    recommendation = get_recommendation(account_id, recommendation_id)
    
    if not recommendation:
        return {
            "status": "error",
            "message": "Recommendation not found"
        }
    
    # Get recommendation implementation handler
    rec_type = recommendation.get("type")
    implementation_handler = get_implementation_handler(rec_type)
    
    if not implementation_handler:
        return {
            "status": "error",
            "message": f"No implementation handler for recommendation type: {rec_type}"
        }
    
    # Implement the recommendation
    try:
        implementation_result = implementation_handler.apply(
            account_id=account_id,
            recommendation=recommendation,
            user_id=user_id
        )
        
        # Record implementation
        record_recommendation_implementation(
            account_id=account_id,
            recommendation_id=recommendation_id,
            user_id=user_id,
            result=implementation_result
        )
        
        # Schedule follow-up evaluation
        schedule_recommendation_evaluation(
            account_id=account_id,
            recommendation_id=recommendation_id,
            days_to_evaluate=[1, 7, 14]  # Evaluate after 1, 7, and 14 days
        )
        
        return {
            "status": "success",
            "message": "Recommendation implemented successfully",
            "details": implementation_result
        }
        
    except Exception as e:
        logging.error(f"Error implementing recommendation {recommendation_id}: {str(e)}")
        
        return {
            "status": "error",
            "message": f"Implementation failed: {str(e)}"
        }
```

### Alerting Integration

The Account Health Predictor integrates with the platform's alerting system to notify users about critical issues:

```python
def generate_account_health_alerts(
    prediction: Dict[str, Any],
    account_id: str,
    threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Generate alerts based on account health prediction.
    
    This function identifies critical issues that require immediate attention
    and generates alerts for the appropriate users.
    """
    alerts = []
    
    # Check health score for critical issues
    health_score = prediction.get("predictions", {}).get("health_score", 0)
    health_class = prediction.get("predictions", {}).get("health_class")
    
    if health_score < 30 or health_class == "critical":
        # Generate critical health alert
        alerts.append({
            "type": "account_health_critical",
            "account_id": account_id,
            "severity": "high",
            "message": f"Account health is critical (score: {health_score})",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "health_score": health_score,
                "health_class": health_class,
                "prediction_id": prediction.get("request_id")
            }
        })
    
    # Check for specific high-probability issues
    issue_probabilities = prediction.get("predictions", {}).get("issue_probabilities", {})
    
    for issue, probability in issue_probabilities.items():
        if probability > threshold:
            # Generate issue-specific alert
            alerts.append({
                "type": f"account_health_issue_{issue}",
                "account_id": account_id,
                "severity": "medium" if probability < 0.9 else "high",
                "message": f"High probability of {format_issue_name(issue)} detected ({probability:.1%})",
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "issue": issue,
                    "probability": probability,
                    "prediction_id": prediction.get("request_id")
                }
            })
    
    # Check for anomalies
    anomalies = prediction.get("anomalies", {}).get("metric_anomalies", {})
    
    for metric, anomaly in anomalies.items():
        # Generate anomaly alert for significant anomalies
        if anomaly.get("score", 0) > threshold:
            alerts.append({
                "type": f"metric_anomaly_{metric}",
                "account_id": account_id,
                "severity": "medium",
                "message": f"Anomaly detected in {metric} metric",
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "metric": metric,
                    "anomaly_score": anomaly.get("score"),
                    "current_value": anomaly.get("current_value"),
                    "expected_value": anomaly.get("expected_value"),
                    "prediction_id": prediction.get("request_id")
                }
            })
    
    # Submit alerts to notification system
    for alert in alerts:
        send_notification(alert)
    
    return alerts
```

### Automation Integration

The Account Health Predictor can be integrated with automation tools for implementing optimizations without manual intervention:

```python
def setup_auto_optimization(
    account_id: str,
    auto_threshold: float = 0.8,
    max_daily_actions: int = 3,
    allowed_action_types: List[str] = None
) -> Dict[str, Any]:
    """
    Configure automatic optimization based on account health predictions.
    
    This function sets up rules for automatically implementing certain
    types of recommendations when confidence exceeds the threshold.
    """
    if allowed_action_types is None:
        allowed_action_types = [
            "negative_keyword_addition", 
            "bid_adjustment",
            "budget_adjustment",
            "targeting_expansion",
            "audience_exclusion"
        ]
    
    # Create auto-optimization configuration
    config = {
        "account_id": account_id,
        "enabled": True,
        "auto_threshold": auto_threshold,
        "max_daily_actions": max_daily_actions,
        "allowed_action_types": allowed_action_types,
        "notification_email": get_account_notification_email(account_id),
        "created_at": datetime.now().isoformat(),
        "created_by": "system"
    }
    
    # Store configuration
    save_auto_optimization_config(account_id, config)
    
    # Set up scheduled job for auto-optimization
    schedule_auto_optimization_job(account_id)
    
    return {
        "status": "success",
        "message": "Auto-optimization configured successfully",
        "config": config
    }
```

These integration points allow the Account Health Predictor to function as a core component of the WITHIN advertising optimization platform, providing seamless access to predictions, recommendations, and automated optimizations.

## Performance Considerations

The Account Health Predictor is designed to handle a large volume of accounts efficiently, with a focus on low-latency prediction and efficient batch processing. The system is optimized for:

1. **Scalability**: The predictor service can scale horizontally, with stateless prediction instances sharing a common model store
2. **Performance**: Optimized for low-latency prediction (<300ms per account) and efficient batch processing
3. **Resource Requirements**: 
  - CPU: 4+ cores recommended for optimal batch processing
  - Memory: 4GB minimum, 8GB recommended for larger account datasets
  - Storage: 500MB for model artifacts and 1GB for prediction caching
- **Caching**: Predictions are cached with a configurable TTL to reduce computational load
- **Monitoring**: Built-in integration with the Production Monitoring Service for real-time performance tracking
- **Fallbacks**: Graceful degradation mechanisms in case of component failures or resource constraints

These performance considerations ensure that the Account Health Predictor can reliably serve predictions in a production environment with appropriate performance, monitoring, and resilience.

## Monitoring and Maintenance

The Account Health Predictor requires ongoing monitoring and maintenance to ensure its continued effectiveness in a changing advertising landscape. This section outlines the monitoring strategies, maintenance processes, and continuous improvement mechanisms employed.

### Monitoring Strategy

The comprehensive monitoring system for the Account Health Predictor tracks several key areas to ensure the model performs consistently and reliably in production:

#### 1. Model Performance Monitoring

```python
def setup_performance_monitoring() -> None:
    """Configure monitoring dashboards for model performance."""
    
    # Create model accuracy dashboard
    create_monitoring_dashboard(
        title="Account Health Predictor - Accuracy",
        metrics=[
            "health_score_rmse",
            "health_score_mae",
            "health_classification_accuracy",
            "issue_detection_f1",
            "issue_detection_precision",
            "issue_detection_recall",
            "recommendation_acceptance_rate"
        ],
        time_range="30d",
        refresh_interval=3600,  # 1 hour
        alert_thresholds={
            "health_score_rmse": {"warning": 12.0, "critical": 15.0},
            "health_classification_accuracy": {"warning": 0.75, "critical": 0.7},
            "issue_detection_f1": {"warning": 0.7, "critical": 0.65}
        }
    )
    
    # Create prediction performance dashboard
    create_monitoring_dashboard(
        title="Account Health Predictor - Prediction Performance",
        metrics=[
            "prediction_latency_p50",
            "prediction_latency_p95",
            "prediction_error_rate",
            "prediction_volume",
            "batch_processing_time_ms"
        ],
        time_range="7d",
        refresh_interval=300,  # 5 minutes
        alert_thresholds={
            "prediction_latency_p95": {"warning": 400, "critical": 500},
            "prediction_error_rate": {"warning": 0.05, "critical": 0.1}
        }
    )
```

The model performance monitoring system tracks various metrics:

1. **Accuracy Metrics**: Measures how well the model's predictions match actual outcomes
   - RMSE (Root Mean Square Error) for health score predictions
   - Classification accuracy for health status categories
   - F1 score, precision, and recall for issue detection

2. **Operational Metrics**: Tracks the technical performance of the prediction service
   - Response times (median and 95th percentile)
   - Error rates and failure modes
   - Resource utilization during prediction

3. **Business Impact Metrics**: Evaluates how model predictions influence business outcomes
   - Recommendation acceptance rate
   - Performance improvement after implementing recommendations
   - User engagement with health insights

These metrics are displayed on dedicated dashboards and monitored continuously, with automated alerts triggered when values exceed defined thresholds.

#### 2. Data Drift Monitoring

```python
def setup_drift_detection() -> None:
    """Configure data drift detection for input features."""
    
    # Get feature engineering pipeline
    feature_pipeline = get_feature_pipeline()
    
    # Configure drift detector
    drift_detector = DataDriftDetector(
        model_id="account_health_predictor",
        reference_dataset=get_reference_dataset(),
        features=feature_pipeline.get_feature_names(),
        categorical_features=feature_pipeline.get_categorical_features(),
        numerical_features=feature_pipeline.get_numerical_features(),
        p_threshold=0.05,  # Statistical significance threshold
        distance_threshold=0.1,  # Distribution distance threshold
        monitoring_service=get_monitoring_service()
    )
    
    # Set up scheduled drift detection job
    schedule_job(
        job_id="account_health_drift_detection",
        function=drift_detector.detect_drift,
        trigger="cron",
        hour="*/6",  # Run every 6 hours
        max_instances=1
    )
    
    # Set up alert actions
    drift_detector.on_drift_detected(notify_ml_team)
    drift_detector.on_drift_detected(log_drift_event)
    drift_detector.on_severe_drift_detected(initiate_retraining)
```

Data drift monitoring is critical for ensuring the Account Health Predictor maintains its accuracy over time. The system monitors for three types of drift:

1. **Feature Drift**: Tracks changes in the statistical properties of input features
   - Distribution shifts in numerical features (using KS-test and earth mover's distance)
   - Changes in category distributions for categorical features (using chi-square tests)
   - Appearance of new categories or values outside expected ranges

2. **Concept Drift**: Detects changes in the relationship between features and target variables
   - Model error rates increasing over specific segments
   - Changes in feature importance rankings
   - Unexpected patterns in error residuals

3. **Prediction Drift**: Monitors changes in the distribution of model predictions
   - Shifts in the overall health score distribution
   - Changes in the distribution of predicted issues
   - Variations in recommendation patterns

When drift is detected, the system can trigger various responses, from alerting the ML team to automatically initiating model retraining if the drift exceeds severe thresholds.

#### 3. Feedback Loop Monitoring

```python
def setup_feedback_monitoring() -> None:
    """Configure monitoring for prediction feedback."""
    
    # Set up feedback collection endpoint
    register_feedback_handler(
        model_id="account_health_predictor",
        handler=record_account_health_feedback
    )
    
    # Configure feedback dashboard
    create_monitoring_dashboard(
        title="Account Health Predictor - Feedback Analysis",
        metrics=[
            "health_score_accuracy",
            "issue_detection_accuracy",
            "recommendation_acceptance_rate",
            "recommendation_effectiveness",
            "user_satisfaction_score"
        ],
        dimensions=[
            "platform",
            "industry",
            "account_size"
        ],
        time_range="90d",
        refresh_interval=86400  # Daily
    )
    
    # Set up automated feedback analysis
    schedule_job(
        job_id="account_health_feedback_analysis",
        function=analyze_feedback_patterns,
        trigger="cron",
        day_of_week="mon",  # Run weekly on Mondays
        hour=2  # At 2 AM
    )
```

The feedback loop monitoring system collects, analyzes, and incorporates user feedback to continuously improve the Account Health Predictor:

1. **Feedback Collection**: Multiple channels to gather user insights
   - Explicit feedback via thumbs up/down buttons on predictions
   - Surveys on recommendation quality and usefulness
   - Implementation tracking (whether users followed recommendations)
   - Account performance changes after implementing recommendations

2. **Feedback Analysis**: Regular evaluation of collected feedback
   - Pattern recognition in rejected recommendations
   - Identification of common false positives in issue detection
   - Correlation analysis between feedback and account characteristics
   - Time-based trends in model accuracy perception

3. **Continuous Improvement**: Mechanisms to incorporate feedback
   - Feedback-driven feature engineering adjustments
   - Recommendation recalibration based on acceptance patterns
   - Retraining triggers based on feedback metrics
   - UX improvements to increase feedback quality

This feedback loop creates a virtuous cycle where the model continuously learns from user interactions, enhancing its accuracy and usefulness over time.

#### 4. System Health Monitoring

```python
def setup_system_monitoring() -> None:
    """Configure system-level monitoring for the predictor service."""
    
    # Configure resource monitoring
    register_resource_metrics(
        service_name="account-health-predictor",
        metrics=[
            "cpu_utilization",
            "memory_utilization",
            "disk_utilization",
            "prediction_queue_size",
            "active_connections"
        ]
    )
    
    # Set up service health checks
    register_health_check(
        service_name="account-health-predictor",
        endpoint="/health",
        interval=60,  # Check every 60 seconds
        timeout=5,    # 5-second timeout
        unhealthy_threshold=3,  # Mark unhealthy after 3 failed checks
        healthy_threshold=2     # Mark healthy after 2 successful checks
    )
    
    # Configure alerting
    register_alerts(
        service_name="account-health-predictor",
        alerts=[
            {
                "name": "high_cpu_utilization",
                "metric": "cpu_utilization",
                "condition": "> 80",
                "duration": "5m",
                "severity": "warning"
            },
            {
                "name": "service_unhealthy",
                "metric": "health_check_status",
                "condition": "== 0",
                "duration": "2m",
                "severity": "critical"
            },
            {
                "name": "high_error_rate",
                "metric": "error_rate",
                "condition": "> 0.05",
                "duration": "5m",
                "severity": "warning"
            }
        ]
    )
```

System health monitoring ensures the Account Health Predictor infrastructure operates reliably:

1. **Resource Utilization**: Tracking of system resources to prevent degradation
   - CPU, memory, and disk utilization monitoring
   - Network bandwidth and throughput metrics
   - Database connection usage and query performance
   - Prediction service request queue lengths

2. **Service Health**: Proactive monitoring of service availability
   - Regular health check pings to validation endpoints
   - Dependency checks to ensure all components are operational
   - Response time monitoring for critical API paths
   - Status checks for batch processing jobs

3. **Error Tracking**: Comprehensive error monitoring and alerting
   - Categorized error rate tracking by error type
   - Exception logging with contextual information
   - Error clustering to identify patterns
   - Automated alerts for error spikes

4. **Capacity Planning**: Metrics to support infrastructure scaling decisions
   - Usage patterns and seasonal trend analysis
   - Growth forecasting based on historical data
   - Identification of performance bottlenecks
   - Recommendations for proactive scaling

The system health monitoring integrates with the organization's overall observability platform, providing a holistic view of the Account Health Predictor's operational status and enabling rapid response to any infrastructure issues.

### Maintenance Processes

The Account Health Predictor requires regular maintenance to ensure optimal performance:

#### 1. Scheduled Retraining

```python
def schedule_retraining() -> None:
    """Set up scheduled retraining jobs for the Account Health Predictor."""
    
    # Define retraining job
    retraining_job = TrainingJob(
        model_id="account_health_predictor",
        training_script="app/models/ml/account_health/train.py",
        data_preparation_script="app/models/ml/account_health/data_prep.py",
        config_path="configs/account_health_training.json",
        output_path="models/account_health/",
        notification_targets=["ml-team@within.co"]
    )
    
    # Schedule regular full retraining
    schedule_job(
        job_id="account_health_full_retraining",
        function=retraining_job.run_full_training,
        trigger="cron",
        day="1",     # First day of month
        hour="2",    # At 2 AM
        max_instances=1
    )
    
    # Schedule incremental retraining
    schedule_job(
        job_id="account_health_incremental_retraining",
        function=retraining_job.run_incremental_training,
        trigger="cron",
        day_of_week="mon",  # Weekly on Monday
        hour="3",           # At 3 AM
        max_instances=1
    )
```

The Account Health Predictor follows a multi-tiered retraining approach:

1. **Incremental Retraining (Weekly)**
   - Updates the model with new data while preserving core structure
   - Fine-tunes model parameters without drastic changes
   - Incorporates recent feedback to adjust issue detection sensitivity
   - Takes approximately 2-3 hours to complete
   - Changes are validated against a holdout set before deployment

2. **Full Retraining (Monthly)**
   - Complete retraining from scratch with all available data
   - Comprehensive feature selection and parameter optimization
   - In-depth evaluation on multiple test datasets
   - Takes approximately 8-10 hours to complete
   - Undergoes rigorous A/B testing before full deployment

3. **Emergency Retraining (As Needed)**
   - Triggered by significant drift detection or performance degradation
   - Focused on addressing specific issues identified in monitoring
   - Prioritizes stability and recovery over optimization
   - Subject to expedited but thorough validation
   - May be deployed initially to a subset of accounts

4. **Scheduled Feature Updates (Quarterly)**
   - Systematic evaluation of feature engineering pipeline
   - Introduction of new features based on research findings
   - Deprecation of low-importance or redundant features
   - Comprehensive documentation of feature changes
   - Extended testing period before production deployment

All retraining processes are fully automated but include manual review checkpoints for model evaluation and deployment approval.

#### 2. Model Versioning and Deployment

```python
def deploy_new_model_version(
    model_path: str,
    evaluation_results: Dict[str, Any],
    approval_required: bool = True
) -> Dict[str, Any]:
    """
    Deploy a new version of the Account Health Predictor.
    
    Args:
        model_path: Path to the model artifacts
        evaluation_results: Results of model evaluation
        approval_required: Whether manual approval is required
        
    Returns:
        Deployment status information
    """
    # Load and validate the new model
    new_model = load_account_health_predictor(model_path)
    
    # Verify model meets quality thresholds
    quality_check = verify_model_quality(new_model, evaluation_results)
    
    if not quality_check["passed"]:
        return {
            "status": "rejected",
            "reason": "Failed quality checks",
            "details": quality_check["details"]
        }
    
    # Generate model version
    version = f"{datetime.now().strftime('%Y%m%d')}-{shortuuid.uuid()[:8]}"
    
    # Create deployment package
    deployment_package = create_deployment_package(
        model=new_model,
        version=version,
        evaluation_results=evaluation_results
    )
    
    # If approval required, create approval request
    if approval_required:
        approval_request = create_approval_request(
            deployment_package=deployment_package,
            approvers=["model-approval-team"],
            deadline=datetime.now() + timedelta(days=2),
            deployment_script="scripts/deploy_account_health.py",
            rollback_script="scripts/rollback_account_health.py"
        )
        
        return {
            "status": "pending_approval",
            "version": version,
            "approval_request_id": approval_request["id"],
            "details": {
                "quality_check": quality_check,
                "deployment_package": deployment_package["id"]
            }
        }
    
    # Without approval, deploy immediately
    deployment_result = deploy_model(
        deployment_package=deployment_package,
        environment="production",
        deployment_strategy="blue_green", # Use blue-green deployment for zero downtime
        canary_percentage=10,  # Start with 10% traffic to new model
        completion_criteria={
            "error_rate_threshold": 0.01,
            "latency_threshold_p95": 500,
            "evaluation_duration_minutes": 30
        }
    )
    
    return {
        "status": "deployed" if deployment_result["success"] else "deployment_failed",
        "version": version,
        "details": deployment_result
    }
```

The Account Health Predictor employs a robust model versioning and deployment system:

1. **Versioning Strategy**
   - Semantic versioning (Major.Minor.Patch) for model releases
   - Major version changes: significant architecture or feature set changes
   - Minor version changes: performance improvements without structural changes
   - Patch version changes: bug fixes or minor updates
   - Each version is tagged with unique identifier and timestamp

2. **Deployment Pipeline**
   - Automated deployment workflow triggered by approved model updates
   - Multi-environment deployment path: Development → Staging → Production
   - Comprehensive pre-deployment checks for model quality and integrity
   - Blue-green deployment strategy for zero-downtime updates
   - Canary testing with gradual traffic shifting (10% → 25% → 50% → 100%)

3. **Quality Gates**
   - Automated quality checks against performance thresholds
   - Required manual approvals for major and minor version changes
   - A/B test results comparison before full deployment
   - Performance monitoring during canary phase with auto-rollback capability
   - Compliance and security validation for all model artifacts

4. **Rollback Mechanisms**
   - Automated rollback for quality gate failures
   - One-click emergency rollback option for production issues
   - Version history maintenance for any-point-in-time recovery
   - Comprehensive logging of deployment events for audit trails
   - Automatic data continuity during rollbacks

This versioning and deployment system ensures reliable, auditable updates to the Account Health Predictor with minimal disruption to users.

#### 3. Feature Engineering Updates

```python
def update_feature_engineering_pipeline(
    feature_config_path: str,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Update the feature engineering pipeline configuration.
    
    Args:
        feature_config_path: Path to new feature configuration
        validate: Whether to validate the changes
        
    Returns:
        Update status information
    """
    # Load new feature configuration
    with open(feature_config_path, 'r') as f:
        new_config = json.load(f)
    
    # Get current configuration
    current_config = get_current_feature_config()
    
    # Compare configurations to find changes
    config_diff = compare_feature_configs(current_config, new_config)
    
    # If validation requested, validate the changes
    if validate:
        validation_result = validate_feature_config(new_config)
        
        if not validation_result["valid"]:
            return {
                "status": "rejected",
                "reason": "Validation failed",
                "details": validation_result["errors"]
            }
    
    # Apply the new configuration
    update_result = apply_feature_config(new_config)
    
    # Record the update
    record_feature_config_update(
        new_config=new_config,
        config_diff=config_diff,
        timestamp=datetime.now().isoformat(),
        applied_by="system"
    )
    
    return {
        "status": "updated" if update_result["success"] else "update_failed",
        "details": {
            "changes": config_diff,
            "update_result": update_result
        }
    }
```

Maintaining and evolving the feature engineering pipeline is crucial for the Account Health Predictor's continued effectiveness:

1. **Feature Evolution Process**
   - Quarterly feature engineering reviews
   - Data-driven feature importance analysis
   - Experimental feature testing framework
   - User feedback incorporation into feature development
   - Incremental feature releases to minimize disruption

2. **Feature Documentation**
   - Comprehensive feature dictionary maintained with metadata
   - Automatic feature lineage tracking
   - Impact assessment for feature changes
   - Feature retirement and deprecation plans
   - Documentation of feature interactions and dependencies

3. **Feature Validation**
   - Automated validation of new features against quality metrics
   - Statistical tests for feature significance
   - Correlation analysis to prevent redundancy
   - Performance impact measurement for each feature
   - Cost-benefit analysis for computationally expensive features

4. **Feature Store Integration**
   - Central feature registry with versioning
   - Feature reuse across multiple models
   - Caching of expensive feature calculations
   - Feature serving with low-latency access
   - Feature monitoring for data quality issues

The feature engineering update process balances stability with innovation, ensuring that the Account Health Predictor continuously improves while maintaining reliability.

### Continuous Improvement

The Account Health Predictor employs several mechanisms for continuous improvement:

#### 1. Performance Analysis

```python
def analyze_prediction_performance(
    time_period: str = "30d",
    segmentation: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze the performance of the Account Health Predictor.
    
    Args:
        time_period: Time period for analysis
        segmentation: Optional dimensions for segmentation
        
    Returns:
        Analysis results
    """
    if segmentation is None:
        segmentation = ["platform", "industry", "account_size"]
    
    # Collect prediction data
    prediction_data = get_prediction_data(time_period)
    
    # Collect feedback data
    feedback_data = get_feedback_data(time_period)
    
    # Merge prediction and feedback data
    analysis_data = merge_prediction_feedback_data(
        prediction_data, 
        feedback_data
    )
    
    # Calculate overall metrics
    overall_metrics = calculate_performance_metrics(analysis_data)
    
    # Calculate segmented metrics
    segmented_metrics = {}
    for dimension in segmentation:
        segmented_metrics[dimension] = calculate_segmented_metrics(
            analysis_data, 
            dimension
        )
    
    # Identify performance issues
    performance_issues = identify_performance_issues(
        overall_metrics,
        segmented_metrics
    )
    
    # Generate optimization recommendations
    optimization_recommendations = generate_optimization_recommendations(
        performance_issues
    )
    
    return {
        "overall_metrics": overall_metrics,
        "segmented_metrics": segmented_metrics,
        "performance_issues": performance_issues,
        "optimization_recommendations": optimization_recommendations
    }
```

Systematic performance analysis is essential for identifying improvement opportunities:

1. **Holistic Performance Assessment**
   - Regular automated performance reports (daily, weekly, monthly)
   - Comprehensive metric tracking across prediction components
   - Multi-dimensional analysis by platform, industry, account size, etc.
   - Comparative analysis against baseline and previous versions
   - Statistical significance testing for performance differences

2. **Segmentation Analysis**
   - Identification of performance variations across segments
   - Detection of underperforming segments requiring specialized attention
   - Analysis of segment-specific feature importance
   - Custom threshold calibration for different segments
   - Segment-specific recommendation effectiveness analysis

3. **Root Cause Investigation**
   - Deep-dive analysis of prediction errors
   - Feature contribution analysis for misclassifications
   - Identification of systematic error patterns
   - Data quality impact assessment
   - Error clustering and categorization

4. **Optimization Roadmap**
   - Data-driven prioritization of improvement opportunities
   - ROI estimation for potential enhancements
   - Development of targeted optimization plans
   - Effort vs. impact assessment
   - Timeline recommendations for implementation

Performance analysis results are reviewed monthly by the ML team and incorporated into the development roadmap for continuous enhancement.

#### 2. Experiment Framework

```python
def setup_model_experiment(
    experiment_name: str,
    hypothesis: str,
    experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Set up an experiment for testing model improvements.
    
    Args:
        experiment_name: Name of the experiment
        hypothesis: Hypothesis being tested
        experiment_config: Configuration for the experiment
        
    Returns:
        Experiment setup information
    """
    # Define experiment variants
    variants = experiment_config.get("variants", [])
    
    if not variants:
        return {
            "status": "error",
            "message": "No variants defined for experiment"
        }
    
    # Create experiment
    experiment = ModelExperiment(
        name=experiment_name,
        hypothesis=hypothesis,
        owner=experiment_config.get("owner"),
        start_date=experiment_config.get("start_date", datetime.now()),
        end_date=experiment_config.get("end_date"),
        metrics=experiment_config.get("metrics", ["health_score_rmse", "recommendation_acceptance_rate"]),
        success_criteria=experiment_config.get("success_criteria", {})
    )
    
    # Add control variant
    experiment.add_variant(
        name="control",
        description="Current production model",
        model_id="account_health_predictor",
        model_version=get_current_model_version(),
        traffic_percentage=experiment_config.get("control_traffic_percentage", 50)
    )
    
    # Add test variants
    for variant in variants:
        experiment.add_variant(
            name=variant["name"],
            description=variant["description"],
            model_id="account_health_predictor",
            model_version=variant["model_version"],
            traffic_percentage=variant["traffic_percentage"]
        )
    
    # Register experiment
    register_result = register_model_experiment(experiment)
    
    if register_result["success"]:
        # Start experiment if configured to start immediately
        if experiment_config.get("start_immediately", False):
            start_result = start_model_experiment(experiment.id)
            return {
                "status": "started" if start_result["success"] else "registered",
                "experiment_id": experiment.id,
                "details": {
                    "experiment": experiment.to_dict(),
                    "start_result": start_result if experiment_config.get("start_immediately", False) else None
                }
            }
        else:
            return {
                "status": "registered",
                "experiment_id": experiment.id,
                "details": {
                    "experiment": experiment.to_dict()
                }
            }
    else:
        return {
            "status": "error",
            "message": "Failed to register experiment",
            "details": register_result
        }
```

The experiment framework enables systematic testing of model improvements:

1. **Experiment Design**
   - Hypothesis-driven experimentation
   - Clearly defined success metrics and criteria
   - Statistical power analysis to determine sample size
   - Randomized assignment of accounts to variants
   - Appropriate control groups for comparison

2. **Experiment Types**
   - A/B tests for model version comparisons
   - Multivariate tests for feature evaluation
   - Bandit algorithms for recommendation optimization
   - Phased rollouts for gradual implementation
   - Shadow testing for risk-free evaluation

3. **Experiment Monitoring**
   - Real-time performance tracking during experiments
   - Statistical significance monitoring
   - Early stopping rules for clear winners/losers
   - Guardrail metrics to prevent negative impacts
   - Automatic alerting for unexpected behavior

4. **Results Analysis**
   - Comprehensive statistical analysis of results
   - Segmentation analysis to identify differential effects
   - Long-term impact assessment
   - Implementation recommendations based on results
   - Documentation of learnings for future experiments

The experimentation framework provides a structured approach to testing and validating improvements, ensuring that changes to the Account Health Predictor are data-driven and demonstrably beneficial.

#### 3. Feedback Collection and Analysis

```python
def analyze_feedback_patterns(
    time_period: str = "90d",
    min_samples: int = 100
) -> Dict[str, Any]:
    """
    Analyze patterns in user feedback to identify improvement opportunities.
    
    Args:
        time_period: Time period for analysis
        min_samples: Minimum number of samples required for analysis
        
    Returns:
        Analysis results
    """
    # Collect feedback data
    feedback_data = get_feedback_data(time_period)
    
    if len(feedback_data) < min_samples:
        return {
            "status": "insufficient_data",
            "message": f"Insufficient feedback data ({len(feedback_data)} < {min_samples})"
        }
    
    # Analyze accuracy patterns
    accuracy_patterns = analyze_accuracy_patterns(feedback_data)
    
    # Analyze recommendation effectiveness
    recommendation_patterns = analyze_recommendation_patterns(feedback_data)
    
    # Analyze user comments
    comment_analysis = analyze_user_comments(feedback_data)
    
    # Identify improvement opportunities
    improvement_opportunities = identify_improvement_opportunities(
        accuracy_patterns,
        recommendation_patterns,
        comment_analysis
    )
    
    return {
        "status": "completed",
        "analysis": {
            "accuracy_patterns": accuracy_patterns,
            "recommendation_patterns": recommendation_patterns,
            "comment_analysis": comment_analysis
        },
        "improvement_opportunities": improvement_opportunities
    }
```

User feedback is a critical input for continuous improvement of the Account Health Predictor:

1. **Structured Feedback Collection**
   - In-product feedback mechanisms on predictions and recommendations
   - Periodic user surveys with targeted questions
   - Session recording and usage pattern analysis
   - Customer success team feedback channels

2. **Feedback Processing Pipeline**
   - Automated processing of structured feedback
   - NLP analysis of free-text comments
   - Sentiment analysis of user responses
   - Topic clustering to identify common themes

3. **Insight Generation**
   - Pattern identification across feedback sources
   - Correlation analysis with account characteristics
   - Trend analysis over time
   - Comparison with performance metrics

The feedback analysis process runs monthly, with critical issues escalated for immediate attention. Results are incorporated into the product roadmap and used to guide model improvements.

#### 4. Model Card Maintenance

```python
def update_model_card(model_version: str = None) -> Dict[str, Any]:
    """
    Update the model card for the Account Health Predictor.
    
    Args:
        model_version: Optional specific model version to update
        
    Returns:
        Update status information
    """
    # Get current model version if not specified
    if model_version is None:
        model_version = get_current_model_version()
    
    # Get model metadata
    model_metadata = get_model_metadata(model_version)
    
    # Get evaluation results
    evaluation_results = get_model_evaluation_results(model_version)
    
    # Get feature details
    feature_details = get_feature_details(model_version)
    
    # Get fairness assessment
    fairness_assessment = get_fairness_assessment(model_version)
    
    # Generate model card content
    model_card = generate_model_card(
        model_name="Account Health Predictor",
        version=model_version,
        description=model_metadata.get("description"),
        date=model_metadata.get("creation_date"),
        model_type=model_metadata.get("model_type"),
        primary_metrics=evaluation_results.get("primary_metrics"),
        performance_metrics=evaluation_results.get("performance_metrics"),
        features=feature_details.get("features"),
        feature_importance=feature_details.get("feature_importance"),
        training_data_summary=model_metadata.get("training_data_summary"),
        limitations=model_metadata.get("limitations"),
        fairness_assessment=fairness_assessment,
        version_history=get_model_version_history()
    )
    
    # Save model card
    save_result = save_model_card(model_card, model_version)
    
    # Update documentation
    update_result = update_model_documentation(model_card, model_version)
    
    return {
        "status": "updated" if save_result["success"] and update_result["success"] else "update_failed",
        "version": model_version
    }
```

Comprehensive model documentation is maintained through model cards:

1. **Model Card Components**
   - Model overview and purpose
   - Architecture details and component interactions
   - Training data characteristics and preprocessing steps
   - Performance metrics and evaluation results
   - Feature importance and interpretation guidance
   - Intended use cases and limitations
   - Fairness assessments and version history

2. **Automated Updates**
   - Automatic model card generation with each new version
   - Integration with CI/CD pipeline for documentation updates
   - Performance metric charts and visualizations
   - Feature importance visualization

3. **Documentation Standards**
   - Standardized format following ML model card best practices
   - Technical and non-technical sections for different audiences
   - Clear explanation of metrics and their interpretation
   - Usage guidelines and integration documentation

Model cards serve as the authoritative reference for the Account Health Predictor, ensuring that all stakeholders have access to accurate, up-to-date information about the model's capabilities, limitations, and usage guidelines.

### Maintenance Schedule

The Account Health Predictor follows a regular maintenance schedule:

| Activity | Frequency | Description |
|----------|-----------|-------------|
| Performance Monitoring | Continuous | Real-time monitoring of prediction quality and system performance |
| Data Drift Detection | Every 6 hours | Scheduled checks for drift in input features |
| Incremental Retraining | Weekly | Update model with new data while preserving core structure |
| Full Retraining | Monthly | Complete retraining from scratch with all available data |
| Feedback Analysis | Monthly | Detailed analysis of user feedback and improvement opportunities |
| Feature Engineering Review | Quarterly | Review and update of feature engineering pipeline |
| A/B Testing | As needed | Experimental validation of proposed improvements |
| Model Card Updates | With each new version | Documentation updates reflecting model changes |

This comprehensive monitoring and maintenance strategy ensures that the Account Health Predictor remains accurate, reliable, and relevant, even as advertising platforms evolve and user needs change.

---

*This document was completed on March 20, 2025. It complies with WITHIN ML documentation standards.* 