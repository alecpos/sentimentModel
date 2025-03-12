# Fairness Evaluation Report

Generated on: 2025-03-12 18:32:40

## Summary

- Average Demographic Parity Disparity: 0.0138
- Maximum Demographic Parity Disparity: 0.0360
- Average Equalized Odds Disparity: 0.0578
- Maximum Equalized Odds Disparity: 0.0819

## Protected Attribute: gender

- Demographic Parity Disparity: 0.0036
- Equalized Odds Disparity: 0.0660

### Group-level Metrics

| Group | Sample Size | Accuracy | Pos. Rate | TPR | FPR | Acc. Disparity |
|-------|-------------|----------|-----------|-----|-----|---------------|
| Female | 139045 | 0.7689 | 0.5275 | 0.7638 | 0.2245 | -0.0143 |
| Male | 100955 | 0.8029 | 0.5239 | 0.8224 | 0.2171 | 0.0197 |

### Recommendations

- **Low Demographic Parity Disparity**: No immediate action needed
- **Low Equalized Odds Disparity**: No immediate action needed

## Protected Attribute: age_group

- Demographic Parity Disparity: 0.0360
- Equalized Odds Disparity: 0.0819

### Group-level Metrics

| Group | Sample Size | Accuracy | Pos. Rate | TPR | FPR | Acc. Disparity |
|-------|-------------|----------|-----------|-----|-----|---------------|
| 51+ | 35007 | 0.7512 | 0.5058 | 0.7516 | 0.2491 | -0.0320 |
| 26-35 | 109134 | 0.7918 | 0.5258 | 0.7961 | 0.2132 | 0.0086 |
| 36-50 | 37463 | 0.7740 | 0.5209 | 0.7747 | 0.2268 | -0.0092 |
| 18-25 | 58396 | 0.7922 | 0.5418 | 0.7975 | 0.2145 | 0.0090 |

### Recommendations

- **Low Demographic Parity Disparity**: No immediate action needed
- **Low Equalized Odds Disparity**: No immediate action needed

## Protected Attribute: location

- Demographic Parity Disparity: 0.0020
- Equalized Odds Disparity: 0.0255

### Group-level Metrics

| Group | Sample Size | Accuracy | Pos. Rate | TPR | FPR | Acc. Disparity |
|-------|-------------|----------|-----------|-----|-----|---------------|
| Urban | 114253 | 0.7776 | 0.5253 | 0.7779 | 0.2228 | -0.0056 |
| Rural | 49152 | 0.7909 | 0.5272 | 0.7992 | 0.2186 | 0.0077 |
| Suburban | 76595 | 0.7867 | 0.5263 | 0.7930 | 0.2206 | 0.0035 |

### Recommendations

- **Low Demographic Parity Disparity**: No immediate action needed
- **Low Equalized Odds Disparity**: No immediate action needed

## Overall Recommendations

- **Fairness Concern Level**: Low
- **No Immediate Action Required**: Continue monitoring fairness metrics
