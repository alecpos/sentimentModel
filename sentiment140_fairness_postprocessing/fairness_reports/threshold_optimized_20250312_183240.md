# Fairness Evaluation Report

Generated on: 2025-03-12 18:32:40

## Summary

- Average Demographic Parity Disparity: 0.0124
- Maximum Demographic Parity Disparity: 0.0314
- Average Equalized Odds Disparity: 0.0532
- Maximum Equalized Odds Disparity: 0.0711

## Protected Attribute: gender

- Demographic Parity Disparity: 0.0045
- Equalized Odds Disparity: 0.0626

### Group-level Metrics

| Group | Sample Size | Accuracy | Pos. Rate | TPR | FPR | Acc. Disparity |
|-------|-------------|----------|-----------|-----|-----|---------------|
| Female | 139045 | 0.7661 | 0.5849 | 0.8124 | 0.2932 | -0.0109 |
| Male | 100955 | 0.7919 | 0.5805 | 0.8673 | 0.2856 | 0.0149 |

### Recommendations

- **Low Demographic Parity Disparity**: No immediate action needed
- **Low Equalized Odds Disparity**: No immediate action needed

## Protected Attribute: age_group

- Demographic Parity Disparity: 0.0314
- Equalized Odds Disparity: 0.0711

### Group-level Metrics

| Group | Sample Size | Accuracy | Pos. Rate | TPR | FPR | Acc. Disparity |
|-------|-------------|----------|-----------|-----|-----|---------------|
| 51+ | 35007 | 0.7458 | 0.5657 | 0.8050 | 0.3159 | -0.0312 |
| 26-35 | 109134 | 0.7846 | 0.5831 | 0.8429 | 0.2828 | 0.0076 |
| 36-50 | 37463 | 0.7680 | 0.5770 | 0.8213 | 0.2938 | -0.0090 |
| 18-25 | 58396 | 0.7872 | 0.5972 | 0.8423 | 0.2833 | 0.0102 |

### Recommendations

- **Low Demographic Parity Disparity**: No immediate action needed
- **Low Equalized Odds Disparity**: No immediate action needed

## Protected Attribute: location

- Demographic Parity Disparity: 0.0014
- Equalized Odds Disparity: 0.0261

### Group-level Metrics

| Group | Sample Size | Accuracy | Pos. Rate | TPR | FPR | Acc. Disparity |
|-------|-------------|----------|-----------|-----|-----|---------------|
| Urban | 114253 | 0.7721 | 0.5824 | 0.8252 | 0.2916 | -0.0049 |
| Rural | 49152 | 0.7840 | 0.5833 | 0.8455 | 0.2858 | 0.0070 |
| Suburban | 76595 | 0.7798 | 0.5838 | 0.8404 | 0.2896 | 0.0028 |

### Recommendations

- **Low Demographic Parity Disparity**: No immediate action needed
- **Low Equalized Odds Disparity**: No immediate action needed

## Overall Recommendations

- **Fairness Concern Level**: Low
- **No Immediate Action Required**: Continue monitoring fairness metrics
