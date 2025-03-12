# Fairness Evaluation Report

## Summary
- Total samples: 20000
- Average positive prediction rate: 0.5067
- Problematic groups identified: 0
- Fairness concern level: Low

- Overall accuracy: 0.5067
- Maximum accuracy disparity: 0.013603122745819651
- Accuracy disparity level: Low

## Intersectional Analysis

Analysis of prediction patterns across intersections of demographic variables:

- **Intersection 1: Unknown-Unknown**
  - Disparate impact: 0.95 (lower rate by 4.9%)
  - Sample size: 1657

- **Intersection 2: Unknown-Unknown**
  - Disparate impact: 1.04 (higher rate by 4.4%)
  - Sample size: 1595

- **Intersection 3: Unknown-Unknown**
  - Disparate impact: 0.96 (lower rate by 4.3%)
  - Sample size: 1653

- **Intersection 4: 1-urban**
  - Disparate impact: 0.96 (lower rate by 3.7%)
  - Sample size: 3261

- **Intersection 5: 0-Unknown**
  - Disparate impact: 1.03 (higher rate by 3.4%)
  - Sample size: 2518

## Recommendations

The model shows generally balanced predictions across demographic groups. Continue to monitor for fairness as the model is deployed and updated.