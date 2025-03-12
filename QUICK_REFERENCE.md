# Enhanced Fairness Features: Quick Reference Guide

This document provides a quick reference for using the enhanced fairness features implemented in the sentiment analysis system.

## Command-Line Arguments

### Post-Processing Fairness

```bash
# Apply threshold optimization with equalized odds constraint
python run_enhanced_training.py --fairness_postprocessing threshold_optimization --fairness_metric equalized_odds

# Apply threshold optimization with demographic parity constraint
python run_enhanced_training.py --fairness_postprocessing threshold_optimization --fairness_metric demographic_parity

# Apply rejection option classification
python run_enhanced_training.py --fairness_postprocessing rejection_option
```

### Explainability Features

```bash
# Generate all types of explanations for 5 random test examples
python run_enhanced_training.py --generate_explanations --explanation_samples 5 --explanation_types all

# Generate only feature importance explanations
python run_enhanced_training.py --generate_explanations --explanation_types feature

# Generate only counterfactual explanations
python run_enhanced_training.py --generate_explanations --explanation_types counterfactual
```

### Complete Example

```bash
# Full example with all fairness features
python run_enhanced_training.py \
  --model_type logistic \
  --train_mode traditional \
  --max_samples 50000 \
  --fairness_evaluation \
  --bias_mitigation \
  --fairness_postprocessing threshold_optimization \
  --fairness_metric equalized_odds \
  --generate_explanations \
  --explanation_samples 10
```

## API Usage

### Using Post-Processing in Code

```python
from fairness_postprocessing import ThresholdOptimizer, RejectionOptionClassifier

# Create a threshold optimizer
optimizer = ThresholdOptimizer(fairness_metric="equalized_odds")

# Fit the optimizer on validation data
optimizer.fit(y_true, y_pred_proba, protected_attributes)

# Apply optimized thresholds to new predictions
adjusted_predictions = optimizer.adjust(new_y_pred_proba, new_protected_attributes)
```

### Using Explainability Features in Code

```python
from fairness_explainer import FairnessAwareExplainer, CounterfactualExplainer

# Create counterfactual explainer
cf_explainer = CounterfactualExplainer(model=sentiment_analyzer)

# Generate gender counterfactual
explanation = cf_explainer.generate_counterfactuals(
    text="He was very happy with the service.",
    demographic_type="gender"
)

# Visualize the counterfactual
cf_explainer.plot_counterfactual_comparison(
    explanation, 
    save_path="counterfactual_example.png"
)
```

## Output Files

### Fairness Evaluation Files

- `enhanced_sentiment_results/enhanced/fairness/fairness_metrics.json`: Detailed fairness metrics
- `enhanced_sentiment_results/enhanced/fairness/fairness_report.md`: Human-readable fairness report
- `enhanced_sentiment_results/enhanced/fairness/plots/`: Directory containing fairness visualizations

### Explanation Files

- `enhanced_sentiment_results/enhanced/explanations/example_N.png`: Visualization of explanation for example N
- `enhanced_sentiment_results/enhanced/explanations/example_N_metadata.json`: Metadata for explanation N
- `enhanced_sentiment_results/enhanced/explanations/explanations_report.md`: Summary report of all explanations

## Interpreting Results

### Fairness Metrics

- **Disparate Impact Ratio**: Values between 0.8-1.2 are generally considered acceptable
- **Equalized Odds Difference**: Lower values indicate better fairness (< 0.1 is good)
- **Demographic Parity Difference**: Lower values indicate better fairness (< 0.1 is good)

### Explanation Warnings

- **High influence of demographic terms**: Indicates that demographic terms have a significant impact on the prediction
- **Large prediction difference in counterfactuals**: Indicates that changing demographic terms significantly alters the prediction

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Post-processing reduces accuracy | Try different fairness metrics or adjust constraints |
| Missing demographic information | Use synthetic demographic data generation |
| No demographic terms found in text | Check demographic term dictionary and add domain-specific terms |
| High rejection rate | Adjust uncertainty threshold in RejectionOptionClassifier |

## Further Resources

- [fairlearn](https://fairlearn.org/): Python package for fairness assessment and mitigation
- [LIME](https://github.com/marcotcr/lime): Local Interpretable Model-agnostic Explanations
- [SHAP](https://github.com/slundberg/shap): SHapley Additive exPlanations
- [AI Fairness 360](https://aif360.mybluemix.net/): Comprehensive toolkit for fairness 