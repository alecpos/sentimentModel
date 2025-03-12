# Fairness Improvement Comparison

Generated on: 2025-03-12 18:02:28

## Summary of Improvements

| Protected Attribute | Metric | Before | After | Change | % Improvement |
|---------------------|--------|--------|-------|--------|---------------|
| gender | Demographic Parity | 0.0270 | 0.0082 | 0.0188 | 69.7% |
| gender | Equalized Odds | 0.0517 | 0.0555 | -0.0038 | -7.4% |

## Detailed Group-level Changes

### gender

| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| 0 | Accuracy | 0.7872 | 0.7872 | 0.0000 |
| 0 | Positive Rate | 0.5280 | 0.5280 | 0.0000 |
| 0 | TPR | 0.8030 | 0.8030 | 0.0000 |
| 0 | FPR | 0.2298 | 0.2298 | 0.0000 |
| 1 | Accuracy | 0.7660 | 0.7598 | -0.0062 |
| 1 | Positive Rate | 0.5010 | 0.5362 | 0.0352 |
| 1 | TPR | 0.7570 | 0.7849 | 0.0279 |
| 1 | FPR | 0.2241 | 0.2672 | 0.0431 |

## Conclusion

- Demographic Parity disparity improved by an average of 69.7%
- Equalized Odds disparity improved by an average of -7.4%

The fairness adjustments have made **significant improvements** to the model's fairness metrics.
