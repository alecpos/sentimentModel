# Fairness Assessment Guidelines

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document outlines the guidelines and methodologies for assessing and ensuring fairness in machine learning models within the WITHIN platform. These standards should be applied across all ML models to minimize bias and ensure equitable performance across different user segments.

## Table of Contents

1. [Introduction](#introduction)
2. [Fairness Definitions](#fairness-definitions)
3. [Fairness Metrics](#fairness-metrics)
4. [Assessment Methodology](#assessment-methodology)
5. [Threshold Selection](#threshold-selection)
6. [Bias Mitigation Techniques](#bias-mitigation-techniques)
7. [Documentation Requirements](#documentation-requirements)
8. [Ongoing Monitoring](#ongoing-monitoring)
9. [Special Considerations](#special-considerations)

## Introduction

Fairness in machine learning is the principle that an ML system should not discriminate against certain groups or individuals based on sensitive attributes such as platform, account size, industry vertical, or geography. This document establishes the standards, metrics, and processes for evaluating and ensuring fairness in all production ML models at WITHIN.

### Objectives

The primary objectives of these fairness guidelines are to:

1. Provide clear definitions and metrics for assessing fairness
2. Establish standardized testing methodologies
3. Define minimum fairness requirements for production models
4. Guide the selection and implementation of bias mitigation techniques
5. Ensure compliance with ethical AI principles and best practices

### Scope

These guidelines apply to all machine learning models deployed in the WITHIN platform, with particular emphasis on:

- Account Health Prediction models
- Ad Score Prediction models
- Recommendation systems
- Optimization algorithms
- Classification models used for decision support

## Fairness Definitions

WITHIN adopts multiple definitions of fairness to address different aspects of equity in ML systems:

### Group Fairness

Group fairness ensures that the model treats different segments fairly at an aggregate level.

#### Demographic Parity

A model satisfies demographic parity if the probability of a certain prediction is the same across all groups:

P(Ŷ = 1 | G = g) = P(Ŷ = 1) for all groups g

#### Equal Opportunity

A model satisfies equal opportunity if it has equal true positive rates across groups:

P(Ŷ = 1 | Y = 1, G = g) = P(Ŷ = 1 | Y = 1) for all groups g

#### Equalized Odds

A model satisfies equalized odds if it has equal true positive rates and false positive rates across groups:

P(Ŷ = 1 | Y = y, G = g) = P(Ŷ = 1 | Y = y) for all y ∈ {0,1} and all groups g

### Individual Fairness

Individual fairness ensures that similar individuals receive similar predictions, regardless of group membership.

#### Consistency

A prediction is consistent if similar individuals receive similar predictions, as measured by:

1/n ∑ₙ |f(xₙ) - 1/k ∑ₖ f(xₙₖ)| < ε

where xₙₖ are the k nearest neighbors of xₙ.

## Fairness Metrics

WITHIN employs the following metrics to quantify fairness across various dimensions:

### Disparate Impact

Disparate impact measures the ratio of positive prediction rates between different groups:

DI = min(P(Ŷ = 1 | G = g₁) / P(Ŷ = 1 | G = g₂))

**Target**: DI ≥ 0.80 for all group pairs

### Statistical Parity Difference

The difference in positive prediction rates between groups:

SPD = |P(Ŷ = 1 | G = g₁) - P(Ŷ = 1 | G = g₂)|

**Target**: SPD ≤ 0.10 for all group pairs

### Equal Opportunity Difference

The difference in true positive rates between groups:

EOD = |P(Ŷ = 1 | Y = 1, G = g₁) - P(Ŷ = 1 | Y = 1, G = g₂)|

**Target**: EOD ≤ 0.10 for all group pairs

### Average Absolute Odds Difference

The average of differences in false positive rates and true positive rates between groups:

AAOD = (|FPR_diff| + |TPR_diff|) / 2

**Target**: AAOD ≤ 0.10 for all group pairs

### Theil Index

A measure of inequality in model benefit distribution:

T = (1/n) ∑ᵢ (b_i/μ) ln(b_i/μ)

where b_i is the benefit received by individual i and μ is the mean benefit.

**Target**: T ≤ 0.20

## Assessment Methodology

### Required Segmentation Analysis

All models must be evaluated for fairness across these dimensions:

1. **Platform Segments**
   - Facebook, Google, Amazon, TikTok, Snapchat, Pinterest, LinkedIn, etc.

2. **Account Size Segments**
   - Small (<$1K/mo)
   - Medium ($1K-10K/mo)
   - Large ($10K-100K/mo)
   - Enterprise (>$100K/mo)

3. **Industry Vertical Segments**
   - E-commerce
   - Retail
   - B2B Services
   - Finance
   - Travel & Hospitality
   - Entertainment & Media
   - Healthcare
   - Education
   - Other significant verticals

4. **Geographic Segments**
   - North America
   - Europe
   - Asia-Pacific
   - Latin America
   - Other significant regions

### Testing Procedure

1. **Data Preparation**
   - Identify sensitive attributes for segmentation
   - Ensure sufficient representation of all groups
   - Create balanced test datasets for fairness evaluation

2. **Model Evaluation**
   - Evaluate model performance metrics for each segment
   - Calculate fairness metrics for all segment pairs
   - Identify any fairness violations according to thresholds

3. **Statistical Significance**
   - Apply statistical tests to determine if differences are significant
   - Consider sample size effects in fairness assessments
   - Use confidence intervals to account for uncertainty

4. **Documentation**
   - Document fairness metrics across all segments
   - Note any observed disparities and potential causes
   - Recommend mitigation strategies for identified issues

### Fairness Evaluation Code Example

```python
from within.fairness import FairnessEvaluator
from within.models import MLModel
from within.data import TestDataset

# Initialize fairness evaluator with desired metrics
evaluator = FairnessEvaluator(
    metrics=["disparate_impact", "statistical_parity_difference", 
             "equal_opportunity_difference", "average_odds_difference"],
    significance_level=0.05
)

# Load model and test data
model = MLModel.load("path/to/model")
test_data = TestDataset.load("path/to/test_data")

# Define sensitive attributes for evaluation
sensitive_attributes = {
    "platform": ["facebook", "google", "amazon", "tiktok", "snapchat", "pinterest", "linkedin"],
    "account_size": ["small", "medium", "large", "enterprise"],
    "industry": ["ecommerce", "retail", "b2b", "finance", "travel", "media", "healthcare", "education"],
    "region": ["north_america", "europe", "apac", "latam"]
}

# Evaluate fairness
results = evaluator.evaluate(
    model=model,
    data=test_data,
    sensitive_attributes=sensitive_attributes,
    prediction_task="classification",
    target_variable="conversion"
)

# Generate fairness report
report = evaluator.generate_report(
    results=results,
    thresholds={
        "disparate_impact": 0.80,
        "statistical_parity_difference": 0.10,
        "equal_opportunity_difference": 0.10,
        "average_odds_difference": 0.10
    },
    output_format="html"
)

# Save results
evaluator.save_results("fairness_evaluation_results.json")
```

## Threshold Selection

Proper threshold selection is critical for ensuring fairness in classification models:

### Group-Specific Thresholds

When appropriate, use group-specific classification thresholds to equalize error rates:

1. Determine the threshold that maximizes the model's performance on the entire dataset
2. Evaluate fairness metrics using this threshold
3. If fairness violations are detected, consider group-specific thresholds:
   - Adjust thresholds to equalize false positive rates across groups
   - Adjust thresholds to equalize false negative rates across groups
   - Evaluate the impact of adjusted thresholds on overall performance

### ROC Curve Analysis

Use ROC curves to visualize the fairness-accuracy tradeoff:

1. Generate ROC curves for each group
2. Identify threshold values that minimize disparities
3. Select thresholds that balance fairness and accuracy objectives

## Bias Mitigation Techniques

When fairness issues are identified, apply these mitigation techniques:

### Pre-processing Techniques

Apply these techniques before model training:

1. **Resampling**
   - Oversample underrepresented groups
   - Create balanced training sets across sensitive attributes

2. **Feature Transformation**
   - Remove or transform biased features
   - Apply dimensionality reduction techniques

3. **Data Augmentation**
   - Generate synthetic samples for minority groups
   - Use techniques like SMOTE for balanced representation

### In-processing Techniques

Apply these techniques during model training:

1. **Adversarial Debiasing**
   - Train the model to maximize performance while minimizing ability to predict sensitive attributes

2. **Constraint Optimization**
   - Add fairness constraints to the optimization objective
   - Penalize unfair solutions during training

3. **Fair Representation Learning**
   - Learn representations that are invariant to sensitive attributes
   - Apply variational fair autoencoders

### Post-processing Techniques

Apply these techniques after model training:

1. **Calibration**
   - Apply group-specific calibration to equalize error rates
   - Use Platt scaling or isotonic regression

2. **Reject Option Classification**
   - Identify and handle cases near the decision boundary
   - Apply more careful processing for borderline cases

3. **Weighted Predictions**
   - Apply different weights to predictions based on group membership
   - Adjust for known biases in historical data

## Documentation Requirements

All models must include comprehensive fairness documentation:

### Model Cards

Each model card must include a fairness section with:

1. Fairness definitions considered
2. Fairness metrics and results
3. Identified disparities and limitations
4. Mitigation techniques applied
5. Recommendations for fair use

### Fairness Statement Template

```markdown
## Fairness Assessment

### Fairness Metrics
- Disparate Impact: [Value] (Target: ≥ 0.80)
- Statistical Parity Difference: [Value] (Target: ≤ 0.10)
- Equal Opportunity Difference: [Value] (Target: ≤ 0.10)
- Average Absolute Odds Difference: [Value] (Target: ≤ 0.10)

### Segment Analysis
- Platform Fairness: [Pass/Fail]
- Account Size Fairness: [Pass/Fail]
- Industry Vertical Fairness: [Pass/Fail]
- Geographic Fairness: [Pass/Fail]

### Known Limitations
- [Description of identified fairness limitations]
- [Groups with potential suboptimal performance]

### Mitigation Techniques Applied
- [List of techniques applied to address fairness issues]

### Recommendations for Fair Use
- [Guidelines for users to ensure fair application]
```

## Ongoing Monitoring

Fairness assessment is not a one-time activity but requires ongoing monitoring:

### Monitoring Requirements

1. **Periodic Re-evaluation**
   - Evaluate fairness metrics at least quarterly
   - Re-evaluate after significant model updates

2. **Drift Detection**
   - Monitor for fairness drift over time
   - Alert when fairness metrics fall below thresholds

3. **Feedback Loops**
   - Collect user feedback related to fairness
   - Create channels for reporting perceived unfairness

### Implementation Example

```python
from within.monitoring import FairnessMonitor

# Initialize fairness monitor
monitor = FairnessMonitor(
    model_id="account_health_predictor_v1",
    metrics=["disparate_impact", "equal_opportunity_difference"],
    segments=["platform", "account_size", "industry", "region"],
    thresholds={
        "disparate_impact": 0.80,
        "equal_opportunity_difference": 0.10
    },
    evaluation_frequency="weekly"
)

# Start monitoring
monitor.start()

# Register alert handler
def fairness_alert_handler(alert):
    """Handle fairness alerts"""
    if alert["severity"] == "critical":
        notify_team(alert)
        trigger_mitigation_workflow(alert["model_id"])
    
    log_fairness_incident(alert)

monitor.register_alert_handler(fairness_alert_handler)
```

## Special Considerations

### Small Sample Sizes

When evaluating fairness across segments with small sample sizes:

1. Use appropriate statistical techniques accounting for sample size
2. Calculate confidence intervals for fairness metrics
3. Employ regularization in fairness calculations
4. Consider pooling similar groups with small samples

### Multi-class Classification

For multi-class classification models:

1. Evaluate fairness for each class separately
2. Use macro-averaging of fairness metrics across classes
3. Consider fairness in the confusion matrix across all class pairs

### Regression Models

For regression models:

1. Use equality of mean squared error across groups
2. Evaluate residual distributions for different segments
3. Apply quantile-based fairness metrics

## References

1. Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning. fairmlbook.org
2. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. NIPS 2016.
3. Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. ITCS 2012.
4. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys.
5. Mitchell, S., Potash, E., Barocas, S., D'Amour, A., & Lum, K. (2021). Algorithmic fairness: Choices, assumptions, and definitions. Annual Review of Statistics and Its Application.

---

*Last updated: March 15, 2025* 