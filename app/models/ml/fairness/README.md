# Fairness

DOCUMENTATION STATUS: COMPLETE

This directory contains the fairness module for machine learning models in the WITHIN ML Prediction System.

## Purpose

The fairness module provides capabilities for:
- Fairness evaluation metrics and assessment tools
- Bias detection and quantification
- Mitigation strategies for biased models
- Counterfactual generation and analysis
- Model auditing for fairness compliance

## Directory Structure

- **__init__.py**: Module initialization with component exports
- **evaluator.py**: Implementation of fairness evaluation metrics and tools
- **model_auditor.py**: Tools for auditing models for bias and fairness compliance
- **mitigation.py**: Strategies for mitigating bias in machine learning models
- **counterfactual.py**: Counterfactual generation and analysis for fairness evaluation

## Key Components

### FairnessEvaluator

`FairnessEvaluator` is responsible for calculating and assessing fairness metrics for machine learning models.

**Key Features:**
- Computes demographic parity, equal opportunity, and other fairness metrics
- Evaluates metrics across protected attribute groups
- Provides statistical significance tests for fairness assessments
- Generates comprehensive fairness reports

**Parameters:**
- `protected_attributes` (List[str]): List of feature names that are protected attributes
- `privileged_groups` (Dict): Dictionary mapping protected attributes to privileged values
- `unprivileged_groups` (Dict): Dictionary mapping protected attributes to unprivileged values
- `metrics` (List[str], optional): List of fairness metrics to compute

**Methods:**
- `evaluate(model, X, y)`: Evaluates fairness metrics for the provided model and data
- `compare_models(models, X, y)`: Compares fairness metrics across multiple models
- `generate_report()`: Generates a detailed fairness assessment report

### EvaluatorCFE

`EvaluatorCFE` is responsible for evaluating fairness using counterfactual examples.

**Key Features:**
- Assesses model behavior changes across counterfactual examples
- Identifies disparate treatment in model predictions
- Measures consistency in predictions for similar examples
- Supports custom counterfactual generation strategies

**Parameters:**
- `protected_attributes` (List[str]): Protected attribute names
- `counterfactual_strategy` (str): Strategy for generating counterfactuals
- `similarity_threshold` (float): Threshold for determining similar examples

**Methods:**
- `evaluate(model, X)`: Evaluates the model using counterfactual examples
- `generate_counterfactuals(X)`: Generates counterfactual examples for evaluation
- `compute_consistency_score(predictions)`: Computes prediction consistency across counterfactuals

### FairnessAuditor

`FairnessAuditor` is responsible for comprehensive auditing of ML models for fairness issues.

**Key Features:**
- Performs thorough fairness audits across multiple metrics
- Identifies intersectional fairness issues
- Provides remediation recommendations
- Generates compliance documentation

**Parameters:**
- `protected_attributes` (List[str]): Protected attribute names
- `fairness_metrics` (List[str]): Fairness metrics to evaluate
- `threshold` (float): Threshold for flagging fairness violations
- `intersectional` (bool): Whether to perform intersectional analysis

**Methods:**
- `audit(model, data)`: Performs a comprehensive fairness audit of the model
- `generate_report()`: Generates a detailed audit report
- `suggest_mitigations()`: Suggests mitigation strategies for identified issues
- `export_documentation()`: Exports compliance documentation

### BiasDetector

`BiasDetector` is responsible for detecting bias in data, features, and models.

**Key Features:**
- Detects sampling bias in training data
- Identifies biased feature representations
- Assesses label bias and disparate outcomes
- Generates detailed bias reports

**Parameters:**
- `protected_attributes` (List[str]): Protected attribute names
- `reference_data` (DataFrame, optional): Reference data for bias comparison
- `bias_threshold` (float): Threshold for flagging bias issues

**Methods:**
- `detect_sampling_bias(data)`: Detects bias in data sampling
- `detect_feature_bias(data)`: Identifies bias in feature representations
- `detect_label_bias(data, labels)`: Detects bias in outcome labels
- `generate_bias_report()`: Generates a comprehensive bias assessment report

### AdversarialDebiasing

`AdversarialDebiasing` is responsible for implementing adversarial debiasing for fairness in ML models.

**Key Features:**
- Uses adversarial networks to remove protected attribute information
- Maintains model performance while reducing bias
- Supports custom adversarial architectures
- Provides visualization of debiasing process

**Parameters:**
- `protected_attribute` (str): The protected attribute to debias
- `adversary_loss_weight` (float): Weight for the adversary's loss function
- `debias_strength` (float): Strength of the debiasing effect

**Methods:**
- `fit(X, y)`: Trains the debiasing model on the input data
- `transform(X)`: Transforms the input data to remove bias
- `fit_transform(X, y)`: Fits and transforms the input data
- `evaluate_bias_reduction(X_original, X_debiased)`: Evaluates the reduction in bias

### FairnessConstraint

`FairnessConstraint` is responsible for enforcing fairness constraints during model training.

**Key Features:**
- Integrates fairness constraints into the optimization objective
- Supports various constraint types (demographic parity, equal opportunity)
- Allows for custom constraint definitions
- Provides constraint satisfaction metrics

**Parameters:**
- `constraint_type` (str): Type of fairness constraint to enforce
- `protected_attribute` (str): Protected attribute for the constraint
- `constraint_weight` (float): Weight of the constraint in the objective
- `tolerance` (float): Tolerance for constraint violation

**Methods:**
- `compute_constraint(y_true, y_pred, sensitive_features)`: Computes the constraint value
- `get_constraint_loss(y_true, y_pred, sensitive_features)`: Gets the constraint loss
- `apply_constraint(model)`: Applies the constraint to a model
- `evaluate_constraint_satisfaction(y_true, y_pred, sensitive_features)`: Evaluates constraint satisfaction

### ReweighingMitigation

`ReweighingMitigation` is responsible for mitigating bias through instance reweighting.

**Key Features:**
- Computes instance weights to balance outcomes across groups
- Maintains overall class distributions
- Supports various reweighting strategies
- Provides visualization of weight distributions

**Parameters:**
- `protected_attribute` (str): Protected attribute for reweighting
- `favorable_label` (Any): Value representing the favorable outcome
- `unfavorable_label` (Any): Value representing the unfavorable outcome
- `reweighing_method` (str): Method for computing weights

**Methods:**
- `fit(X, y)`: Computes instance weights based on the input data
- `transform(X)`: Returns a weighted version of the input data
- `fit_transform(X, y)`: Fits and transforms the input data
- `get_weights()`: Returns the computed instance weights

### CounterfactualFairnessEvaluator

`CounterfactualFairnessEvaluator` is responsible for evaluating ML models for counterfactual fairness.

**Key Features:**
- Measures prediction consistency across counterfactual worlds
- Evaluates causal fairness metrics
- Identifies fairness violations from a causal perspective
- Provides comparative counterfactual fairness analyses

**Parameters:**
- `protected_attributes` (List[str]): Protected attribute names
- `num_counterfactuals` (int): Number of counterfactuals to generate
- `tolerance` (float): Tolerance for fairness violations

**Methods:**
- `evaluate(model, data)`: Evaluates counterfactual fairness of the model
- `compute_counterfactual_metrics(model, original, counterfactuals)`: Computes fairness metrics across counterfactuals
- `generate_report()`: Generates a detailed counterfactual fairness report

### CounterfactualGenerator

`CounterfactualGenerator` is responsible for generating counterfactual examples for fairness evaluation.

**Key Features:**
- Creates realistic counterfactual examples by changing protected attributes
- Maintains data plausibility and instance similarity
- Supports various generation methods (optimization, GAN, etc.)
- Provides quality metrics for generated counterfactuals

**Parameters:**
- `protected_attributes` (List[str]): Protected attribute names
- `num_counterfactuals` (int): Number of counterfactuals to generate per instance
- `method` (str): Method for generating counterfactuals

**Methods:**
- `generate(X)`: Generates counterfactuals for the input instances
- `generate_causal_counterfactuals(X)`: Generates causally valid counterfactuals
- `evaluate_counterfactual_quality(original, counterfactuals)`: Evaluates the quality of generated counterfactuals
- `save_counterfactuals(path)`: Saves generated counterfactuals to disk

### CounterfactualAuditor

`CounterfactualAuditor` is responsible for auditing ML models for counterfactual fairness across different metrics.

**Key Features:**
- Performs comprehensive counterfactual fairness audits
- Measures causal effects of protected attributes on predictions
- Identifies violations of counterfactual fairness principles
- Generates detailed audit reports with recommendations

**Parameters:**
- `protected_attributes` (List[str]): Protected attribute names
- `metrics` (List[str]): Counterfactual fairness metrics to compute
- `tolerance` (float): Tolerance for fairness violations

**Methods:**
- `audit(model, data)`: Performs a counterfactual fairness audit
- `measure_causal_effects(original, counterfactuals, predictions)`: Measures causal effects
- `identify_violations(causal_effects)`: Identifies counterfactual fairness violations
- `generate_report()`: Generates a detailed audit report

## Usage Examples

### FairnessEvaluator Usage

```python
from app.models.ml.fairness import FairnessEvaluator

# Initialization
evaluator = FairnessEvaluator(
    protected_attributes=["gender", "race"],
    privileged_groups={"gender": "male", "race": "white"},
    unprivileged_groups={"gender": "female", "race": "black"},
    metrics=["demographic_parity", "equal_opportunity"]
)

# Using key methods
fairness_metrics = evaluator.evaluate(model, X_test, y_test)
report = evaluator.generate_report()
```

### BiasDetector Usage

```python
from app.models.ml.fairness import BiasDetector

# Initialization
detector = BiasDetector(
    protected_attributes=["gender", "race"],
    bias_threshold=0.05
)

# Using key methods
sampling_bias = detector.detect_sampling_bias(training_data)
feature_bias = detector.detect_feature_bias(training_data)
bias_report = detector.generate_bias_report()
```

## Integration Points

- **ML Pipeline**: Integrates with the ML training pipeline to enforce fairness constraints
- **Model Registry**: Provides fairness metrics for models stored in the registry
- **Reporting System**: Feeds fairness reports into the system's reporting infrastructure
- **Monitoring Service**: Supplies fairness metrics for ongoing model monitoring
- **Compliance Documentation**: Generates documentation for regulatory compliance

## Dependencies

- **scikit-learn**: Base for ML models and metrics calculations
- **AIF360**: IBM's AI Fairness 360 toolkit for fairness metrics and mitigations
- **PyTorch**: Used for adversarial debiasing and some fairness constraints
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations for fairness metrics
- **matplotlib/plotly**: Visualization of fairness metrics and reports
