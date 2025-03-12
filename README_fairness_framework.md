# WITHIN Fairness Framework

A comprehensive framework for evaluating, mitigating, and monitoring fairness in machine learning models with a focus on regulatory compliance.

## Overview

The WITHIN Fairness Framework is designed to address fairness concerns in machine learning models, with a particular focus on ad scoring and prediction. It provides tools for:

- **Fairness Evaluation**: Measure fairness metrics across protected groups
- **Fairness Mitigation**: Apply techniques to reduce bias in models
- **Fairness Monitoring**: Track fairness metrics over time and detect drift
- **Fairness Explainability**: Understand why models make certain predictions
- **Regulatory Compliance**: Generate model cards and documentation for regulatory frameworks

This framework follows industry best practices for responsible AI and integrates with WITHIN's existing ML model architecture.

## Architecture

The framework consists of several key components:

1. **FairnessEvaluator**: Calculates fairness metrics across protected attributes
2. **Mitigation Techniques**: Implements pre-processing, in-processing, and post-processing fairness mitigation strategies
3. **FairnessMonitor**: Tracks fairness metrics over time and detects drift
4. **FairnessExplainer**: Provides explanations for model decisions through a fairness lens
5. **ModelCardGenerator**: Creates comprehensive model cards for documentation and regulatory compliance

![Architecture Diagram](docs/images/fairness_framework_architecture.png)

## Regulatory Compliance

The framework is designed to support compliance with several regulatory frameworks:

- **EU AI Act**: Comprehensive documentation of high-risk AI systems
- **NYC Local Law 144**: Automated employment decision tool requirements
- **NIST AI Risk Management Framework**: Risk assessment and documentation
- **EEOC Guidelines**: Equal employment opportunity requirements

The `ModelCardGenerator` component produces documentation that meets these regulatory requirements, including:

- Model information and intended use
- Performance metrics with confidence intervals
- Fairness evaluation results across protected attributes
- Mitigation techniques applied and their effectiveness
- Limitations and ethical considerations
- Regulatory compliance statements

## Installation

```bash
pip install -r requirements.txt
```

## Components

### Fairness Evaluator

The `FairnessEvaluator` measures fairness across protected attributes:

```python
from fairness_evaluator import FairnessEvaluator

# Initialize with protected attributes
evaluator = FairnessEvaluator(
    protected_attributes=['gender', 'age_group', 'location'],
    fairness_threshold=0.05,
    output_dir='fairness_results'
)

# Evaluate fairness
fairness_results = evaluator.evaluate(
    X_test, y_test, y_pred, y_prob, 
    calculate_intersectional=True
)

# Generate visualizations
evaluator.plot_fairness_metrics(fairness_results)
evaluator.plot_group_metrics(fairness_results)
evaluator.plot_intersectional_fairness(fairness_results)
```

### Fairness Mitigation

Multiple mitigation techniques are available:

```python
from ad_score_predictor import AdScorePredictor

# Create a model with fairness constraints
model = AdScorePredictor(
    fairness_constraints={
        'method': 'reweighing',
        'protected_attribute': 'gender'
    }
)

# Train with fairness constraints
model.fit(X_train, y_train)
```

Available mitigation methods:
- `reweighing`: Pre-processing technique to reweight training examples
- `fairness_constraints`: In-processing technique that adds regularization terms
- `adversarial_debiasing`: In-processing technique using adversarial learning
- `calibrated_equalized_odds`: Post-processing technique (coming soon)

### Fairness Monitoring

Monitor fairness metrics over time and detect drift:

```python
from fairness_monitoring import FairnessMonitor

# Initialize monitor
monitor = FairnessMonitor(
    protected_attributes=['gender', 'age_group'],
    fairness_metrics=['demographic_parity', 'equal_opportunity'],
    alert_threshold=0.1,
    monitoring_dir='fairness_monitoring'
)

# Set baseline metrics
monitor.set_baseline_metrics(fairness_results)

# Update with new predictions from production data
monitor.update(X_new, y_pred_new, y_true_new)

# Check for alerts
alerts = monitor.check_alerts()
if alerts:
    print(f"Fairness alerts detected: {alerts}")

# Analyze trends
trend_analysis = monitor.get_trend_analysis('gender_demographic_parity')
```

### Fairness Explainability

Understand model decisions through a fairness lens:

```python
from fairness_explainability import FairnessExplainer

# Initialize explainer
explainer = FairnessExplainer(
    protected_attributes=['gender', 'age_group'],
    shap_explainer_type='tree',
    output_dir='fairness_explanations'
)

# Generate fairness explanations
explanations = explainer.explain_model(
    model, X_test, feature_names=X_test.columns
)
```

### Model Card Generator

Create comprehensive model cards for documentation and regulatory compliance:

```python
from model_card_generator import ModelCardGenerator, generate_model_card_for_ad_score_predictor

# Generate a model card from evaluation results
generator = ModelCardGenerator(
    output_dir='model_cards',
    regulatory_frameworks=[
        'EU AI Act', 
        'NYC Local Law 144',
        'NIST AI Risk Management Framework'
    ]
)

# Generate from fairness evaluation results
model_card_path = generator.generate_from_evaluation_results(
    model_info={
        'name': 'Ad Score Predictor',
        'version': '1.0.0',
        'type': 'Classification',
        'description': 'Predicts ad performance scores.'
    },
    evaluation_results=fairness_results,
    mitigation_info=mitigation_info,
    export_formats=['md', 'html', 'pdf']
)

# Or use the helper function for AdScorePredictor models
model_card_path = generate_model_card_for_ad_score_predictor(
    model=ad_score_model,
    evaluation_results=performance_metrics,
    fairness_evaluation_results=fairness_results,
    mitigation_info=mitigation_info
)
```

## Examples

The framework includes several example scripts:

- `run_fairness_framework.py`: Demonstrates the full fairness framework
- `test_fairness_mitigation.py`: Tests different fairness mitigation techniques
- `demo_model_card_generation.py`: Shows how to generate model cards
- `fairness_monitoring_example.py`: Demonstrates fairness monitoring

## Best Practices

1. **Define Protected Attributes Early**: Identify protected attributes during the data preparation phase.
2. **Evaluate Before Mitigating**: Always evaluate fairness before applying mitigation techniques.
3. **Consider Multiple Fairness Metrics**: Different fairness metrics may be appropriate for different contexts.
4. **Perform Intersectional Analysis**: Important for understanding fairness at the intersection of multiple protected attributes.
5. **Document Mitigation Decisions**: Keep records of why specific mitigation techniques were chosen.
6. **Monitor Continuously**: Fairness can degrade over time due to data drift.
7. **Generate Model Cards**: Create comprehensive documentation for regulatory compliance.

## Regulatory Compliance Checklist

- [ ] Identify protected attributes for your application domain
- [ ] Evaluate fairness across all protected attributes
- [ ] Perform intersectional fairness analysis
- [ ] Apply appropriate mitigation techniques for significant disparities
- [ ] Document mitigation decisions and effectiveness
- [ ] Implement continuous fairness monitoring
- [ ] Generate comprehensive model cards for documentation
- [ ] Review regulatory requirements specific to your use case

## Contributing

Contributions to improve the framework are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Add your feature or fix with comprehensive tests
4. Submit a pull request with a detailed description

## License

This framework is provided under the [LICENSE] license. See the LICENSE file for details.

## Contact

For questions or support, please contact the WITHIN Data Science team. 