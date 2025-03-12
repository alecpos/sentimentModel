# Fairness Improvement Comparison

Generated on: 2025-03-12 18:02:28

## Summary of Improvements

| Protected Attribute | Metric | Before | After | Change | % Improvement |
|---------------------|--------|--------|-------|--------|---------------|
| gender | Demographic Parity | 0.0270 | 0.0183 | 0.0087 | 32.4% |
| gender | Equalized Odds | 0.0517 | 0.0605 | -0.0088 | -17.0% |

## Detailed Group-level Changes

### gender

| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| 0 | Accuracy | 0.7872 | 0.7872 | 0.0000 |
| 0 | Positive Rate | 0.5280 | 0.5242 | -0.0039 |
| 0 | TPR | 0.8030 | 0.7993 | -0.0037 |
| 0 | FPR | 0.2298 | 0.2258 | -0.0040 |
| 1 | Accuracy | 0.7660 | 0.7578 | -0.0083 |
| 1 | Positive Rate | 0.5010 | 0.5424 | 0.0414 |
| 1 | TPR | 0.7570 | 0.7888 | 0.0319 |
| 1 | FPR | 0.2241 | 0.2759 | 0.0517 |

## Conclusion

- Demographic Parity disparity improved by an average of 32.4%
- Equalized Odds disparity improved by an average of -17.0%

The fairness adjustments have made **significant improvements** to the model's fairness metrics.
