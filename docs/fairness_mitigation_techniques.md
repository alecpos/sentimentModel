# Fairness Mitigation Techniques

This document provides comprehensive documentation for all fairness mitigation techniques implemented in the WITHIN fairness framework. It explains how each technique works, when to use it, and its strengths and limitations.

## Table of Contents

1. [Introduction](#introduction)
2. [Pre-processing Techniques](#pre-processing-techniques)
   - [Reweighing](#reweighing)
   - [Disparate Impact Remover](#disparate-impact-remover)
3. [In-processing Techniques](#in-processing-techniques)
   - [Fairness Constraints](#fairness-constraints)
   - [Adversarial Debiasing](#adversarial-debiasing)
4. [Post-processing Techniques](#post-processing-techniques)
   - [Equalized Odds Post-processing](#equalized-odds-post-processing)
   - [Calibrated Equalized Odds](#calibrated-equalized-odds)
5. [Comparing Mitigation Approaches](#comparing-mitigation-approaches)
6. [Implementation Details](#implementation-details)
7. [Best Practices](#best-practices)
8. [References](#references)

## Introduction

Machine learning models can inadvertently learn and amplify biases present in training data. Fairness mitigation techniques aim to reduce these biases and ensure equitable outcomes across different demographic groups.

Our fairness framework implements three categories of bias mitigation techniques:

1. **Pre-processing techniques**: Modify the training data to reduce bias before model training
2. **In-processing techniques**: Incorporate fairness constraints during model training
3. **Post-processing techniques**: Adjust model outputs after training to ensure fairness

Each technique makes different trade-offs between fairness and model performance. This document helps you select the appropriate technique for your specific use case.

## Pre-processing Techniques

### Reweighing

**Implementation**: `ReweighingMitigation` in `app/models/ml/fairness/mitigation.py`

#### How It Works

Reweighing assigns different weights to training examples to ensure fairness across protected groups. The technique calculates weights for each combination of (protected attribute value, label) to balance their representation in the training data.

For each instance with protected attribute value `s` and label `y`, the weight is calculated as:

```
w(s,y) = (n_s/n) / (n_{s,y}/n_y)
```

Where:
- `n_s` is the number of instances with protected attribute value `s`
- `n` is the total number of instances
- `n_{s,y}` is the number of instances with protected attribute value `s` and label `y`
- `n_y` is the number of instances with label `y`

#### When to Use

Reweighing is most effective when:
- The bias in the dataset is due to imbalanced representation across groups
- You want to preserve the original features without transformation
- The underlying model supports instance weights during training

#### Strengths
- Simple to implement and interpret
- Does not modify features or labels
- Can be applied to any model that supports instance weights
- Preserves overall data distribution

#### Limitations
- Only addresses representation bias, not more complex forms of bias
- May not be sufficient if bias is embedded in feature values
- Can increase variance in underrepresented groups
- Not applicable to models that don't support instance weights

#### Implementation Example

```python
from app.models.ml.fairness.mitigation import ReweighingMitigation

# Initialize reweighing for gender
reweigher = ReweighingMitigation(protected_attribute='gender')

# Calculate instance weights
reweigher.fit(X_train, y_train, protected_attributes={'gender': gender_train})

# Get the reweighted data
X_train_reweighted, sample_weights = reweigher.transform(X_train)

# Train a model with sample weights
model.fit(X_train_reweighted, y_train, sample_weight=sample_weights)
```

### Disparate Impact Remover

**Implementation**: Not currently implemented, planned for future release

#### How It Works

Disparate Impact Remover is a pre-processing technique that transforms feature values to remove their correlation with protected attributes while preserving rank-ordering within groups. It ensures that the transformed features have a similar distribution across protected groups.

#### When to Use

This technique will be implemented in a future release of the fairness framework.

## In-processing Techniques

### Fairness Constraints

**Implementation**: `FairnessConstraint` in `app/models/ml/fairness/mitigation.py`

#### How It Works

The Fairness Constraint technique adds regularization terms to the model's objective function during training. These constraints penalize violations of fairness metrics, guiding the optimization process toward both accurate and fair models.

We implement fairness constraints for different fairness definitions:
- **Demographic Parity**: Equalizes positive prediction rates across groups
- **Equal Opportunity**: Equalizes true positive rates across groups
- **Equalized Odds**: Equalizes both true positive and false positive rates

#### When to Use

Fairness constraints are most effective when:
- You need tight control over specific fairness metrics
- You're training a new model from scratch
- You have sufficient data to optimize for both accuracy and fairness

#### Strengths
- Directly optimizes for the fairness metric of interest
- Provides control over the accuracy-fairness trade-off via constraint strength
- Can be applied to various model types
- Often achieves better accuracy-fairness trade-offs than pre/post-processing

#### Limitations
- Requires modifying the training algorithm
- Can be computationally expensive
- May not converge if constraints are too strict
- Typically requires more data than unconstrained training

#### Implementation Example

```python
from app.models.ml.fairness.mitigation import FairnessConstraint
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor

# Initialize a model with fairness constraints
model = AdScorePredictor(
    model_config={
        'learning_rate': 0.01,
        'hidden_dim': 64,
        'dropout': 0.2
    },
    fairness_constraints=[
        FairnessConstraint(
            constraint_type='demographic_parity',
            protected_attribute='gender',
            epsilon=0.05  # Maximum allowed disparity
        )
    ]
)

# Train with fairness constraints
model.fit(X_train, y_train, protected_attributes={
    'gender': gender_train
})
```

### Adversarial Debiasing

**Implementation**: `AdversarialDebiasing` in `app/models/ml/fairness/mitigation.py`

#### How It Works

Adversarial Debiasing employs an adversarial learning approach where two networks compete:
1. **Predictor Network**: Tries to predict the target label
2. **Adversary Network**: Tries to predict the protected attribute from the predictor's representations

During training, the predictor attempts to maximize prediction accuracy while minimizing the adversary's ability to determine the protected attribute from its internal representations. This forces the model to learn representations that are informative for the prediction task but not for identifying the protected attribute.

#### When to Use

Adversarial Debiasing is most effective when:
- The bias might be complex and embedded in non-linear interactions between features
- You're working with high-dimensional data like images or text
- You want strong guarantees that protected information is not being used
- You have sufficient data for adversarial training

#### Strengths
- Can handle complex, non-linear relationships between features and protected attributes
- Removes information about protected attributes from learned representations
- Often achieves good fairness-accuracy trade-offs
- Works well with deep learning architectures

#### Limitations
- Complex to implement and tune
- Computationally expensive
- Training can be unstable
- Requires careful balancing of predictor and adversary losses
- May not work well with small datasets

#### Implementation Example

```python
from app.models.ml.fairness.mitigation import AdversarialDebiasing
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor

# Initialize a model with adversarial debiasing
model = AdScorePredictor(
    model_config={
        'learning_rate': 0.01,
        'hidden_dim': 64,
        'dropout': 0.2
    },
    fairness_method='adversarial',
    adversarial_config={
        'protected_attribute': 'gender',
        'adversary_loss_weight': 0.1
    }
)

# Train with adversarial debiasing
model.fit(X_train, y_train, protected_attributes={
    'gender': gender_train
})
```

## Post-processing Techniques

### Equalized Odds Post-processing

**Implementation**: Planned for future release

#### How It Works

Equalized Odds Post-processing adjusts the decision threshold differently for each protected group after the model has been trained. It determines the optimal thresholds that satisfy equalized odds constraints while minimizing overall error.

#### When to Use

This technique will be implemented in a future release of the fairness framework.

### Calibrated Equalized Odds

**Implementation**: Planned for future release

#### How It Works

Calibrated Equalized Odds is a post-processing technique that finds a derived predictor that minimizes error subject to equalized odds and calibration constraints. It works by mixing the original classifier's predictions with a group-dependent random prediction.

#### When to Use

This technique will be implemented in a future release of the fairness framework.

## Comparing Mitigation Approaches

| Technique | Stage | Works with any model | Preserves data | Computational cost | Fairness-performance trade-off | Implementation status |
|-----------|-------|----------------------|----------------|-------------------|--------------------------------|----------------------|
| Reweighing | Pre-processing | ✓ (if weights supported) | ✓ | Low | Moderate | Implemented |
| Disparate Impact Remover | Pre-processing | ✓ | ✗ | Moderate | Moderate | Planned |
| Fairness Constraints | In-processing | ✗ | ✓ | High | Good | Implemented |
| Adversarial Debiasing | In-processing | ✗ | ✓ | Very High | Very Good | Implemented |
| Equalized Odds Post-processing | Post-processing | ✓ | ✓ | Low | Moderate | Planned |
| Calibrated Equalized Odds | Post-processing | ✓ | ✓ | Low | Moderate-Good | Planned |

## Implementation Details

### Integration with AdScorePredictor

Our fairness mitigation techniques are deeply integrated with the `AdScorePredictor` class. When initializing an `AdScorePredictor`, you can specify:

1. A fairness method:
```python
model = AdScorePredictor(
    fairness_method='reweighing',  # or 'adversarial', 'constraints'
    protected_attribute='gender'
)
```

2. Specific fairness constraints:
```python
from app.models.ml.fairness.mitigation import FairnessConstraint

model = AdScorePredictor(
    fairness_constraints=[
        FairnessConstraint(
            constraint_type='demographic_parity',
            protected_attribute='gender',
            epsilon=0.05
        )
    ]
)
```

3. Multiple protected attributes:
```python
model = AdScorePredictor(
    fairness_method='reweighing',
    protected_attributes=['gender', 'location']
)
```

### Logging and Reporting

All fairness mitigation techniques in our framework provide detailed logging of their operations and impact on model behavior. They also integrate with our reporting framework to document:

- Before and after fairness metrics
- Mitigation technique configuration
- Impact on model performance

## Best Practices

### Selecting a Technique

1. **Start with evaluation**: Before applying mitigation, evaluate your model to understand the nature and extent of bias.

2. **Consider your constraints**:
   - If you can't modify the training process, use pre or post-processing techniques
   - If you need the strongest fairness guarantees, use in-processing techniques
   - If computational resources are limited, prefer simpler methods like reweighing

3. **Consider multiple techniques**: Different techniques may address different aspects of bias. Sometimes a combination works best.

4. **Be aware of trade-offs**: All mitigation techniques involve some trade-off between fairness and overall model performance.

### Setting Parameters

1. **Start conservative**: Begin with mild constraints and gradually increase them if needed.

2. **For Reweighing**: No parameters required, as weights are determined by data statistics.

3. **For Fairness Constraints**:
   - `epsilon`: Controls the maximum allowed disparity. Start with 0.1 and adjust as needed.
   - `constraint_type`: Select based on the fairness definition most relevant to your application.

4. **For Adversarial Debiasing**:
   - `adversary_loss_weight`: Controls the strength of the adversarial component. Start with 0.1.
   - If training is unstable, reduce this weight.
   - If fairness isn't improving, increase this weight.

### Monitoring and Validation

1. **Always validate on a separate test set**: Mitigation techniques may overfit to the training data.

2. **Monitor multiple fairness metrics**: Improving one fairness metric might worsen others.

3. **Check impact on subgroups**: Overall fairness might improve while certain subgroups are still disadvantaged.

4. **Implement continuous monitoring**: Fairness can degrade over time due to data or concept drift.

## References

1. Kamiran, F., & Calders, T. (2012). "Data preprocessing techniques for classification without discrimination." Knowledge and Information Systems, 33(1), 1-33.

2. Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). "A reductions approach to fair classification." In International Conference on Machine Learning (pp. 60-69).

3. Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). "Mitigating unwanted biases with adversarial learning." In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (pp. 335-340).

4. Hardt, M., Price, E., & Srebro, N. (2016). "Equality of opportunity in supervised learning." In Advances in Neural Information Processing Systems (pp. 3315-3323).

5. Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). "On fairness and calibration." In Advances in Neural Information Processing Systems (pp. 5680-5689). 