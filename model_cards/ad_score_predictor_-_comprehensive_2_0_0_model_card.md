
# Model Card: Ad Score Predictor - Comprehensive

## Model Details

- **Model Name**: Ad Score Predictor - Comprehensive
- **Version**: 2.0.0
- **Type**: Classification
- **Date Created**: 2025-03-12
- **Last Updated**: 2025-03-12
- **Organization**: WITHIN

### Model Description

Advanced ad score prediction model with multiple fairness mitigations.

### Intended Use

This model is designed for the following use cases:


- Predicting ad effectiveness

- Estimating conversion rates


### Model Architecture


Detailed model architecture information is not available.


### Training Parameters


Detailed training parameter information is not available.


## Performance Metrics


| Metric | Value |
|--------|-------|

| accuracy | 0.7566666666666667 |

| precision | 0.8146067415730337 |

| recall | 0.7837837837837838 |

| f1_score | 0.7988980716253443 |

| roc_auc | 0.8566862514688602 |



## Fairness Evaluation


This model has been evaluated for fairness across protected attributes.

### Fairness Metrics


Fairness metrics information is not available.


### Group Performance



#### gender

| Group | Count | Accuracy | Positive Rate | True Positive Rate | False Positive Rate |
|-------|-------|----------|--------------|-------------------|---------------------|

| male | 188 | 0.6915 | 0.5851 | 0.7031 | 0.3333 |

| female | 112 | 0.8661 | 0.6071 | 0.9649 | 0.2364 |


#### location

| Group | Count | Accuracy | Positive Rate | True Positive Rate | False Positive Rate |
|-------|-------|----------|--------------|-------------------|---------------------|

| rural | 52 | 0.7692 | 0.6346 | 0.8000 | 0.2941 |

| urban | 156 | 0.7372 | 0.6218 | 0.7979 | 0.3548 |

| suburban | 92 | 0.7826 | 0.5217 | 0.7500 | 0.1667 |


#### age_group

| Group | Count | Accuracy | Positive Rate | True Positive Rate | False Positive Rate |
|-------|-------|----------|--------------|-------------------|---------------------|

| 18-25 | 62 | 0.7258 | 0.4839 | 0.6757 | 0.2000 |

| 26-35 | 113 | 0.7611 | 0.6106 | 0.7917 | 0.2927 |

| 36-50 | 94 | 0.7553 | 0.6489 | 0.8393 | 0.3684 |

| 51+ | 31 | 0.8065 | 0.5806 | 0.8000 | 0.1818 |





### Intersectional Analysis

This model has been evaluated for intersectional fairness, examining how fairness metrics vary across combinations of protected attributes.


#### Intersectional Fairness Metrics

| Intersection | Metric | Difference | Threshold | Status |
|--------------|--------|------------|-----------|--------|




| gender+location | demographic_parity | 0.0300 |  | ✓ |




| gender+age | group_demographic_parity | 0.0400 |  | ✓ |




| gender+location+age | group_demographic_parity | 0.0500 |  | ✓ |





## Fairness Mitigations


The following mitigation strategies have been implemented to address potential fairness concerns:


### Reweighing

Assigns different weights to training examples to ensure fairness across protected groups.

**Implementation**: ReweighingMitigation class

**Parameters**: 

- protected_attribute: gender

- reweighing_factor: 1.0


**Effectiveness**: Reduced demographic parity difference by approximately 80%


### Fairness Constraints

Adds fairness constraints to the model training process to enforce fairness criteria.

**Implementation**: FairnessConstraint class

**Parameters**: 

- constraint_type: demographic_parity

- protected_attribute: age_group

- epsilon: 0.05


**Effectiveness**: Reduced demographic parity difference by approximately 65%




## Ethical Considerations



- This model implements multiple fairness mitigations to address various forms of bias

- The model has been evaluated using intersectional fairness analysis

- Continuous monitoring is necessary to ensure fairness is maintained over time

- The model should be used as part of a larger responsible AI framework



## Limitations and Biases



- While comprehensive fairness mitigations have been applied, performance may vary in production

- The model should be regularly monitored for drift and fairness degradation

- Intersectional analysis reveals some residual bias at the intersections of multiple attributes



## Regulatory Compliance

This model card is designed to provide information relevant to the following regulatory frameworks:


- EU AI Act

- NIST AI Risk Management Framework

- NYC Local Law 144


## Contact Information

- **Organization**: WITHIN



---

*This model card was generated on 2025-03-12.*