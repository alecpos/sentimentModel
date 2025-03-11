# Model Retraining Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


This guide provides detailed instructions on when, why, and how to retrain ML models in the WITHIN system. Proper retraining procedures are essential for maintaining model performance, adapting to changing data patterns, and ensuring prediction accuracy over time.

## Table of Contents

- [Overview](#overview)
- [Retraining Triggers](#retraining-triggers)
- [Retraining Workflow](#retraining-workflow)
- [Model-Specific Guidelines](#model-specific-guidelines)
- [Training Infrastructure](#training-infrastructure)
- [Validation Process](#validation-process)
- [Deployment Process](#deployment-process)
- [Rollback Procedures](#rollback-procedures)
- [Monitoring Post-Deployment](#monitoring-post-deployment)
- [Retraining Strategies](#retraining-strategies)

## Overview

Machine learning models in production require systematic retraining to maintain their performance. The WITHIN system follows a structured approach to model retraining that balances:

- **Performance optimization**: Ensuring models maintain or improve accuracy
- **Freshness**: Incorporating new data and patterns
- **Stability**: Avoiding frequent changes that may disrupt user experience
- **Resource efficiency**: Optimizing computational resources for training

The retraining lifecycle includes monitoring, trigger evaluation, training preparation, model training, validation, deployment, and post-deployment monitoring.

## Retraining Triggers

Models should be retrained when one or more of the following conditions are met:

### Data-Driven Triggers

| Trigger | Description | Threshold | Detection Method |
|---------|-------------|-----------|------------------|
| **Data Drift** | Statistical changes in input feature distribution | JS Divergence > 0.2 | Daily drift analysis |
| **Concept Drift** | Changes in the relationship between features and target | Accuracy drop > 5% | Performance monitoring |
| **Performance Degradation** | Decline in model accuracy, precision, or recall | Varies by model | A/B testing, backtesting |
| **Data Volume** | Significant increase in available training data | 20% more data | Data pipeline monitoring |
| **Feature Correlation Changes** | Changes in the correlation between features | Correlation shift > 0.3 | Quarterly feature analysis |

### Time-Based Triggers

| Model | Regular Retraining Schedule | Rationale |
|-------|----------------------------|-----------|
| Ad Score Predictor | Quarterly | Gradual changes in advertising effectiveness patterns |
| Account Health Predictor | Monthly | Frequent changes in platform metrics and benchmarks |
| Ad Sentiment Analyzer | Bi-annually | Slower evolution of language patterns in advertising |

### External Triggers

- Major platform changes (e.g., Facebook algorithm updates)
- New advertising formats or channels
- Seasonal pattern shifts
- Regulatory changes affecting advertising
- Competitive landscape changes

## Retraining Workflow

![Retraining Workflow](/docs/images/retraining_workflow.png)

### 1. Trigger Evaluation

When a retraining trigger is detected, evaluate the need for retraining:

```python
from within.ml.retraining import RetrainingEvaluator

evaluator = RetrainingEvaluator(model_name="ad_score_predictor")

# Analyze model performance
performance_analysis = evaluator.analyze_performance(
    current_version="2.1.0",
    evaluation_period="last_30_days"
)

# Check for data drift
drift_analysis = evaluator.analyze_drift(
    reference_period="2023-06-01_to_2023-06-30",
    current_period="last_30_days"
)

# Determine if retraining is needed
retraining_decision = evaluator.evaluate_retraining_need(
    performance_analysis=performance_analysis,
    drift_analysis=drift_analysis
)

if retraining_decision.should_retrain:
    print(f"Retraining recommended: {retraining_decision.reason}")
    print(f"Expected improvement: {retraining_decision.expected_improvement}")
else:
    print(f"Retraining not needed: {retraining_decision.reason}")
```

### 2. Data Preparation

Prepare the dataset for retraining:

```python
from within.data import DataPreparer

# Initialize data preparer
data_preparer = DataPreparer(model_name="ad_score_predictor")

# Define training period
training_data = data_preparer.prepare_training_data(
    start_date="2023-01-01",
    end_date="2023-09-30",
    include_features=[
        "ad_content.*",
        "historical_metrics.*",
        "platform",
        "target_audience"
    ],
    exclude_features=[
        "internal_id",
        "timestamp"
    ]
)

# Split data
train_data, validation_data, test_data = data_preparer.train_validation_test_split(
    data=training_data,
    validation_size=0.15,
    test_size=0.15,
    stratify_by="performance_category"
)

# Generate data report
data_report = data_preparer.generate_data_report(
    train_data=train_data,
    validation_data=validation_data,
    test_data=test_data
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(validation_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Class distribution: {data_report.class_distribution}")
```

### 3. Model Training

Train the model using the prepared data:

```python
from within.ml.training import ModelTrainer
from within.ml.models import AdScorePredictor

# Initialize model and trainer
model = AdScorePredictor(config={
    "architecture": "gradient_boosting",
    "max_depth": 7,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "random_state": 42
})

trainer = ModelTrainer(
    model=model,
    experiment_tracking=True,
    track_parameters=True,
    track_metrics=True
)

# Train the model
training_result = trainer.train(
    train_data=train_data,
    validation_data=validation_data,
    features=data_preparer.get_feature_columns(),
    target="effectiveness_score",
    early_stopping_rounds=50,
    verbose=100
)

# Get training metrics
print(f"Training metrics: {training_result.train_metrics}")
print(f"Validation metrics: {training_result.validation_metrics}")
print(f"Training time: {training_result.training_time} seconds")
```

### 4. Model Evaluation

Evaluate the new model against the test dataset and current production model:

```python
from within.ml.evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate on test data
test_eval = evaluator.evaluate(
    model=training_result.model,
    data=test_data,
    features=data_preparer.get_feature_columns(),
    target="effectiveness_score",
    metrics=["rmse", "mae", "r2", "accuracy_at_threshold"]
)

# Compare with current production model
comparison = evaluator.compare_models(
    new_model=training_result.model,
    production_model_version="2.1.0",
    test_data=test_data,
    features=data_preparer.get_feature_columns(),
    target="effectiveness_score"
)

print(f"Test evaluation: {test_eval}")
print(f"Model comparison: {comparison}")
print(f"Improvement: {comparison.improvement_percentage:.2f}%")
```

### 5. Model Registration

Register the new model in the model registry:

```python
from within.ml.registry import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Register the new model
model_info = registry.register_model(
    model=training_result.model,
    name="ad_score_predictor",
    version="2.2.0",
    metrics=test_eval,
    training_data_info={
        "start_date": "2023-01-01",
        "end_date": "2023-09-30",
        "samples": len(train_data)
    },
    parameters=model.get_parameters(),
    labels={
        "environment": "staging",
        "owner": "ml-team",
        "trigger": "performance_degradation"
    }
)

print(f"Registered model: {model_info.name} v{model_info.version}")
print(f"Model ID: {model_info.id}")
```

### 6. Deployment

Deploy the new model to the appropriate environment:

```python
from within.ml.deployment import ModelDeployer

# Initialize deployer
deployer = ModelDeployer()

# Deploy to staging
staging_deployment = deployer.deploy(
    model_id=model_info.id,
    environment="staging",
    deployment_strategy="blue_green",
    resource_requirements={
        "cpu": "2",
        "memory": "4Gi"
    }
)

print(f"Staging deployment: {staging_deployment.url}")
print(f"Deployment status: {staging_deployment.status}")
```

## Model-Specific Guidelines

### Ad Score Predictor

| Aspect | Guideline |
|--------|-----------|
| Training Data | Minimum 50,000 samples with balanced performance categories |
| Features | Include all standard features plus latest engagement metrics |
| Architecture | Gradient boosting (default) or deep neural network (for complex patterns) |
| Hyperparameters | Optimize via Bayesian optimization with 5-fold cross-validation |
| Validation | Must achieve at least 5% improvement in RMSE over current model |
| Fairness | Evaluate performance across platforms, industries, and ad types |

### Account Health Predictor

| Aspect | Guideline |
|--------|-----------|
| Training Data | Minimum 10,000 accounts with at least 90 days of history each |
| Time Window | Include at least 12 months of historical data to capture seasonality |
| Architecture | Ensemble of time series models and gradient boosting classifiers |
| Evaluation | Focus on early detection of account health issues (precision/recall balance) |
| Platform-specific | Train separate model components for each advertising platform |
| Anomaly Detection | Update anomaly detection thresholds based on recent distributions |

### Ad Sentiment Analyzer

| Aspect | Guideline |
|--------|-----------|
| Training Data | Minimum 100,000 labeled ad texts across industries |
| Architecture | Fine-tuned transformer model (RoBERTa base) |
| Training Approach | Mixed precision training with gradient accumulation |
| Augmentation | Apply text augmentation techniques for under-represented categories |
| Evaluation | Must maintain at least 85% accuracy on out-of-domain test sets |
| Fairness | Evaluate for bias across industries, cultures, and language patterns |

## Training Infrastructure

### Resource Requirements

| Model | CPU | RAM | GPU | Storage | Training Time |
|-------|-----|-----|-----|---------|--------------|
| Ad Score Predictor | 16 cores | 64GB | Optional | 20GB | 2-4 hours |
| Account Health Predictor | 32 cores | 128GB | Optional | 50GB | 6-12 hours |
| Ad Sentiment Analyzer | 16 cores | 64GB | 1-2 GPUs | 100GB | 12-24 hours |

### Training Environments

- **Development**: For experimentation and initial model development
  - Accessible to data scientists for interactive development
  - Limited to smaller datasets and training jobs
  
- **Training**: Dedicated environment for production model training
  - Optimized for performance and resource management
  - Supports distributed training for large models
  - Automatic logging and experiment tracking
  
- **CI/CD**: Automated training as part of continuous integration pipeline
  - Triggered by code changes or scheduled retraining
  - Validates model performance automatically

### Distributed Training

For large models like the Ad Sentiment Analyzer, use distributed training:

```python
from within.ml.distributed import DistributedTrainer

# Configure distributed training
distributed_trainer = DistributedTrainer(
    num_workers=4,
    strategy="mirrored",
    mixed_precision=True
)

# Launch distributed training job
training_job = distributed_trainer.train(
    model_class="AdSentimentAnalyzer",
    train_data_path="s3://within-training-data/sentiment/2023-09/train",
    val_data_path="s3://within-training-data/sentiment/2023-09/val",
    config_path="configs/sentiment_analyzer_v3.yml"
)

# Monitor training progress
distributed_trainer.monitor(training_job.id)
```

## Validation Process

Before a retrained model can be promoted to production, it must pass a comprehensive validation process:

### Automated Validation

```python
from within.ml.validation import ModelValidator

# Initialize validator
validator = ModelValidator(model_id=model_info.id)

# Run full validation suite
validation_result = validator.validate(
    validation_suites=[
        "performance",
        "robustness",
        "fairness",
        "compliance",
        "integration"
    ],
    required_suites=["performance", "fairness"]
)

if validation_result.passed:
    print("Model passed all required validation suites")
else:
    print(f"Validation failed: {validation_result.failures}")
```

### Validation Requirements

| Validation Type | Description | Requirements |
|-----------------|-------------|-------------|
| Performance | Accuracy metrics | Must meet minimum performance thresholds |
| A/B Testing | Live traffic comparison | Must demonstrate statistically significant improvement |
| Robustness | Stability under noise | Performance degradation < 10% with noisy inputs |
| Fairness | Bias analysis | Similar performance across segments (max 10% variation) |
| Compliance | Policy checking | Must comply with company policies and regulations |
| Integration | System compatibility | Must integrate correctly with all depending systems |

### Manual Review

In addition to automated validation, human review is required for:

1. Feature importance analysis
2. Segment-level performance evaluation
3. Edge case handling assessment
4. Model card updates
5. Final approval for production deployment

## Deployment Process

The deployment process follows a progressive rollout strategy:

### 1. Staging Deployment

Deploy to staging environment and run integration tests:

```python
# Run integration tests against staging deployment
test_results = deployer.run_integration_tests(
    deployment_id=staging_deployment.id,
    test_suite="ad_score_integration"
)

if test_results.status == "passed":
    print("Integration tests passed")
else:
    print(f"Integration test failures: {test_results.failures}")
```

### 2. Shadow Mode

Deploy to production in shadow mode (predictions are logged but not used):

```python
# Deploy in shadow mode
shadow_deployment = deployer.deploy(
    model_id=model_info.id,
    environment="production",
    deployment_strategy="shadow",
    shadow_percentage=100,  # Log 100% of traffic
    duration="7d"  # Run in shadow mode for 7 days
)

print(f"Shadow mode deployment: {shadow_deployment.id}")
```

### 3. A/B Testing

Run an A/B test to compare the new model with the current production model:

```python
from within.ml.experimentation import ABTest

# Configure A/B test
ab_test = ABTest(
    name="ad_score_v2_2_0_vs_v2_1_0",
    variants=[
        {"name": "control", "model_version": "2.1.0", "traffic_percentage": 50},
        {"name": "treatment", "model_version": "2.2.0", "traffic_percentage": 50}
    ],
    metrics=[
        {"name": "accuracy", "priority": "primary"},
        {"name": "latency_p95", "priority": "secondary"},
        {"name": "user_feedback", "priority": "secondary"}
    ],
    duration="14d"  # Run test for 14 days
)

# Start the A/B test
ab_test.start()

# Check results (after test completion)
test_results = ab_test.get_results()
print(f"A/B test results: {test_results}")

if test_results.winner == "treatment":
    print("New model outperformed current production model")
else:
    print("Current model performed better or test was inconclusive")
```

### 4. Progressive Rollout

Gradually increase traffic to the new model:

```python
# Progressive rollout
rollout = deployer.progressive_rollout(
    model_id=model_info.id,
    initial_percentage=10,
    increment=15,
    interval="1d",  # Increase by 15% every day
    max_percentage=100,
    rollback_triggers={
        "error_rate": {"threshold": 0.01, "window": "10m"},
        "latency_p95": {"threshold": 300, "window": "10m"}
    }
)

print(f"Progressive rollout initiated: {rollout.id}")
```

### 5. Full Deployment

Complete the deployment when rollout reaches 100%:

```python
# Complete deployment
deployment = deployer.finalize_deployment(
    rollout_id=rollout.id,
    make_default=True  # Make this the default model for the service
)

print(f"Deployment completed: {deployment.id}")
print(f"Model {model_info.name} v{model_info.version} is now in production")
```

## Rollback Procedures

In case issues are detected after deployment, follow these rollback procedures:

### Automated Rollbacks

The system can automatically roll back based on configured triggers:

```python
# Configure automated rollback
deployer.configure_rollback_triggers(
    deployment_id=deployment.id,
    triggers=[
        {"metric": "error_rate", "threshold": 0.02, "window": "5m"},
        {"metric": "latency_p99", "threshold": 500, "window": "5m"},
        {"metric": "accuracy", "threshold": 0.80, "window": "30m", "direction": "below"}
    ]
)
```

### Manual Rollback

For immediate rollback:

```python
# Perform manual rollback
rollback = deployer.rollback(
    deployment_id=deployment.id,
    reason="Performance degradation observed",
    target_version="2.1.0"  # Previous stable version
)

print(f"Rollback initiated: {rollback.id}")
print(f"Rolling back to version: {rollback.target_version}")
```

## Monitoring Post-Deployment

After a successful deployment, monitor the model closely:

```python
from within.monitoring import PostDeploymentMonitor

# Initialize post-deployment monitor
monitor = PostDeploymentMonitor(
    model_name="ad_score_predictor",
    version="2.2.0",
    baseline_period="7d"  # Use first 7 days as baseline
)

# Configure intensive monitoring period
monitor.configure_intensive_monitoring(
    duration="14d",  # Intensive monitoring for 14 days
    metrics=[
        "accuracy",
        "latency_p95",
        "error_rate",
        "prediction_drift",
        "feature_importance_stability"
    ],
    alert_channels=["slack-ml-team", "email-ml-engineers"]
)

# Start monitoring
monitor.start()
```

## Retraining Strategies

### Continuous Training

For models that benefit from frequent updates, implement continuous training:

```python
from within.ml.continuous import ContinuousTrainer

# Configure continuous training
continuous_trainer = ContinuousTrainer(
    model_name="ad_score_predictor",
    training_frequency="weekly",
    data_window="rolling_90d",  # Use 90-day rolling window
    evaluation_criteria={
        "minimum_improvement": 0.02,  # 2% improvement required
        "required_metrics": ["rmse", "mae"],
    },
    auto_deployment=True,  # Automatically deploy if criteria met
    max_auto_deployments_per_month=2  # Limit automatic deployments
)

# Enable continuous training
continuous_trainer.enable()
```

### Transfer Learning

For complex models like the sentiment analyzer, use transfer learning to reduce training time:

```python
from within.ml.transfer import TransferLearningTrainer

# Initialize transfer learning trainer
transfer_trainer = TransferLearningTrainer(
    base_model_name="ad_sentiment_analyzer",
    base_model_version="3.1.0",
    frozen_layers=["embeddings", "encoder.0", "encoder.1"]
)

# Train with transfer learning
transfer_result = transfer_trainer.train(
    train_data=new_sentiment_data,
    epochs=5,
    learning_rate=3e-5,
    batch_size=32
)

print(f"Transfer learning complete: {transfer_result.metrics}")
```

### Hyperparameter Optimization

Optimize hyperparameters before full retraining:

```python
from within.ml.optimization import HyperparameterOptimizer
from sklearn.metrics import mean_squared_error

# Define parameter space
param_space = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500],
    "subsample": [0.8, 0.9, 1.0]
}

# Define evaluation function
def evaluate_model(params, data):
    model = AdScorePredictor(config=params)
    model.fit(data["train_x"], data["train_y"])
    predictions = model.predict(data["val_x"])
    return mean_squared_error(data["val_y"], predictions)

# Initialize optimizer
optimizer = HyperparameterOptimizer(
    parameter_space=param_space,
    evaluation_function=evaluate_model,
    optimization_metric="minimize_mse",
    max_evaluations=50,
    strategy="bayesian"
)

# Run optimization
optimization_result = optimizer.optimize(
    data={
        "train_x": train_data[features],
        "train_y": train_data["effectiveness_score"],
        "val_x": validation_data[features],
        "val_y": validation_data["effectiveness_score"]
    }
)

print(f"Best parameters: {optimization_result.best_params}")
print(f"Best score: {optimization_result.best_score}")
```

## Additional Resources

- [Monitoring Guide](./monitoring_guide.md): Explains how to monitor model performance
- [Experiment Tracking Guide](../development/experiment_tracking.md): Details on tracking model experiments
- [Data Pipeline Guide](../implementation/data_pipeline.md): Information on data preparation
- [Model Architecture Documentation](../implementation/ml/model_architectures.md): Details on model architectures
- [Training Infrastructure Setup](../development/training_infrastructure.md): How to set up training resources 