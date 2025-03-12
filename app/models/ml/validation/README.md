# ML Model Validation

DOCUMENTATION STATUS: COMPLETE

This directory contains production validation tools for machine learning models in the WITHIN ML Prediction System.

## Purpose

The validation module provides capabilities for:
- Shadow deployment for risk-free testing of new models
- A/B testing framework for rigorous model comparison
- Canary deployment for staged, monitored rollouts
- Performance comparison with statistical significance testing
- Automated test management and reporting

## Directory Structure

- **__init__.py**: Module initialization with component exports
- **shadow_deployment.py**: Implementation of shadow, A/B, and canary deployment strategies
- **ab_test_manager.py**: Tools for managing and analyzing A/B test results
- **golden_set_validator.py**: Utilities for regression testing with golden datasets

## Key Components

### ShadowDeployment

`ShadowDeployment` is responsible for enabling risk-free testing of new models in production by running them in parallel with existing models without affecting user experience.

**Key Features:**
- Runs new models alongside production models without affecting outputs
- Collects performance metrics and logs for comparison
- Supports gradual traffic increases for testing
- Provides comprehensive performance comparison reports
- Integrates with monitoring systems for alerting

**Parameters:**
- `production_model` (BaseMLModel): The current production model
- `shadow_model` (BaseMLModel): The new model to test in shadow mode
- `metrics` (List[str]): Performance metrics to track
- `sampling_rate` (float, optional): Percentage of traffic to sample. Defaults to 1.0
- `logger` (Logger, optional): Custom logger for deployment events

**Methods:**
- `route_request(request_data)`: Routes request to both models but only returns production result
- `get_metrics()`: Returns performance comparison metrics between the models
- `generate_report()`: Generates a detailed comparison report
- `should_promote()`: Returns recommendation on whether shadow model should be promoted
- `log_prediction(request_id, pred_production, pred_shadow, actual)`: Logs predictions from both models

### ABTestDeployment

`ABTestDeployment` is responsible for systematically comparing two models by splitting traffic between them and measuring performance differences.

**Key Features:**
- Splits traffic between control and treatment models
- Ensures consistent user assignment to test groups
- Calculates statistical significance of differences
- Supports multiple evaluation metrics
- Provides guardrail metrics to detect issues

**Parameters:**
- `model_a` (BaseMLModel): Control model (typically the current production model)
- `model_b` (BaseMLModel): Treatment model (typically the new model being tested)
- `traffic_split` (float): Percentage of traffic to route to model B (0.0-1.0)
- `metrics` (Dict[str, Callable]): Metrics to evaluate with their calculation functions
- `guardrail_metrics` (Dict[str, Dict], optional): Metrics with acceptable thresholds
- `user_id_attribute` (str, optional): Attribute to use for consistent assignment

**Methods:**
- `route_request(request_data)`: Routes request to either model A or B based on assignment
- `evaluate()`: Evaluates performance metrics for both models
- `get_significance()`: Calculates statistical significance of differences
- `stop_test()`: Terminates the test and returns final results
- `get_winner()`: Returns the winning model based on primary metrics

### CanaryDeployment

`CanaryDeployment` is responsible for gradually rolling out a new model to production, increasing traffic allocation based on performance metrics.

**Key Features:**
- Gradually increases traffic to new model
- Monitors performance metrics at each stage
- Automatically rolls back if metrics deteriorate
- Supports custom evaluation functions
- Provides detailed deployment logs

**Parameters:**
- `current_model` (BaseMLModel): The currently deployed model
- `canary_model` (BaseMLModel): The new model being rolled out
- `initial_traffic_percentage` (float): Starting traffic percentage for canary model
- `target_traffic_percentage` (float): Final target traffic percentage
- `increment` (float): Percentage to increase at each step
- `evaluation_period` (int): Time (in seconds) between traffic increases
- `success_metrics` (Dict[str, Callable]): Metrics to evaluate for success

**Methods:**
- `start_deployment()`: Initiates the canary deployment process
- `evaluate_and_increase()`: Evaluates performance and increases traffic if appropriate
- `rollback()`: Reverts traffic to the current model
- `get_current_allocation()`: Returns current traffic allocation
- `get_deployment_status()`: Returns detailed status of the deployment

### ABTestManager

`ABTestManager` is responsible for managing multiple A/B tests, tracking their progress, and analyzing results.

**Key Features:**
- Manages the lifecycle of multiple A/B tests
- Schedules test start and end dates
- Calculates required sample sizes for significance
- Generates comprehensive reports with visualizations
- Integrates with experiment tracking systems

**Parameters:**
- `tests` (Dict[str, ABTestDeployment], optional): Dictionary of active tests
- `min_sample_size` (int, optional): Minimum sample size for tests
- `significance_level` (float, optional): Statistical significance level required
- `storage_backend` (str, optional): Backend for storing test results

**Methods:**
- `create_test(test_id, model_a, model_b, config)`: Creates a new A/B test
- `start_test(test_id)`: Starts an A/B test
- `stop_test(test_id)`: Stops an A/B test
- `get_test_results(test_id)`: Gets results for a specific test
- `calculate_sample_size(effect_size, power)`: Calculates required sample size
- `generate_report(test_id)`: Generates comprehensive test report

## Usage Examples

### ShadowDeployment Usage

```python
from app.models.ml.validation import ShadowDeployment
from app.models.ml.prediction import AdScorePredictor

# Initialize models
production_model = AdScorePredictor(version="v1.0")
new_model = AdScorePredictor(version="v2.0")

# Setup shadow deployment
shadow = ShadowDeployment(
    production_model=production_model,
    shadow_model=new_model,
    metrics=["rmse", "mae", "r2"],
    sampling_rate=0.5
)

# In production service
for request in requests:
    # Only production_model result is returned to user
    result = shadow.route_request(request)
    
    # Later, when actual outcomes are known
    shadow.log_prediction(
        request_id=request.id,
        pred_production=result,
        pred_shadow=None,  # Logged internally
        actual=get_actual_value(request.id)
    )

# After sufficient data collection
if shadow.should_promote():
    print("Shadow model outperforms production model!")
    report = shadow.generate_report()
    # Proceed with model promotion
```

### ABTestDeployment Usage

```python
from app.models.ml.validation import ABTestDeployment
from app.models.ml.prediction import AdScorePredictor
import numpy as np

# Define metrics
def rmse(predictions, actuals):
    return np.sqrt(np.mean((predictions - actuals) ** 2))

# Initialize models
model_a = AdScorePredictor(version="v1.0")
model_b = AdScorePredictor(version="v2.0")

# Setup A/B test with 20% traffic to model B
ab_test = ABTestDeployment(
    model_a=model_a,
    model_b=model_b,
    traffic_split=0.2,
    metrics={"rmse": rmse, "conversion_rate": lambda p, a: np.mean(a)},
    user_id_attribute="account_id"
)

# In production service
for request in requests:
    # Will route to either model A or B based on traffic split
    # and consistent user assignment
    result = ab_test.route_request(request)

# After test period
test_results = ab_test.evaluate()
significance = ab_test.get_significance()
winner = ab_test.get_winner()

print(f"Test results: {test_results}")
print(f"Statistical significance: {significance}")
print(f"Winner: {winner}")
```

## Integration Points

- **Model Registry**: Retrieves models for testing and validation
- **Feature Store**: Provides consistent features for model comparisons
- **Monitoring System**: Logs validation metrics for alerting and dashboards
- **Experiment Tracking**: Records A/B test results for historical comparison
- **Deployment Pipeline**: Integrates with CI/CD for automated testing
- **Business Metrics Dashboard**: Feeds business impact metrics from tests

## Dependencies

- **scikit-learn**: Statistical evaluation of model performance
- **statsmodels**: Statistical significance testing
- **pandas**: Data manipulation for validation datasets
- **PyTorch/TensorFlow**: Backend for model inference
- **MLflow**: Tracking experiments and model versions
- **prometheus-client**: Exporting metrics to monitoring systems
