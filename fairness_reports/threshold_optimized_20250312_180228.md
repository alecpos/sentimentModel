# Fairness Evaluation Report

Generated on: 2025-03-12 18:02:28

## Summary

- Average Demographic Parity Disparity: 0.0082
- Maximum Demographic Parity Disparity: 0.0082
- Average Equalized Odds Disparity: 0.0555
- Maximum Equalized Odds Disparity: 0.0555

## Protected Attribute: gender

- Demographic Parity Disparity: 0.0082
- Equalized Odds Disparity: 0.0555

### Group-level Metrics

| Group | Sample Size | Accuracy | Pos. Rate | TPR | FPR | Acc. Disparity |
|-------|-------------|----------|-----------|-----|-----|---------------|
| 0 | 517 | 0.7872 | 0.5280 | 0.8030 | 0.2298 | 0.0132 |
| 1 | 483 | 0.7598 | 0.5362 | 0.7849 | 0.2672 | -0.0142 |

### Recommendations

- **Low Demographic Parity Disparity**: No immediate action needed
- **Low Equalized Odds Disparity**: No immediate action needed

## Overall Recommendations

- **Fairness Concern Level**: Low
- **No Immediate Action Required**: Continue monitoring fairness metrics
