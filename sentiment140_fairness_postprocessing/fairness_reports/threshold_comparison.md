# Fairness Improvement Comparison

Generated on: 2025-03-12 18:32:40

## Summary of Improvements

| Protected Attribute | Metric | Before | After | Change | % Improvement |
|---------------------|--------|--------|-------|--------|---------------|
| gender | Demographic Parity | 0.0026 | 0.0045 | -0.0019 | -72.6% |
| gender | Equalized Odds | 0.0666 | 0.0626 | 0.0040 | 6.1% |
| age_group | Demographic Parity | 0.0450 | 0.0314 | 0.0136 | 30.1% |
| age_group | Equalized Odds | 0.0852 | 0.0711 | 0.0141 | 16.5% |
| location | Demographic Parity | 0.0047 | 0.0014 | 0.0033 | 70.9% |
| location | Equalized Odds | 0.0271 | 0.0261 | 0.0010 | 3.8% |

## Detailed Group-level Changes

### gender

| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| Male | Accuracy | 0.8033 | 0.7919 | -0.0114 |
| Male | Positive Rate | 0.5230 | 0.5805 | 0.0575 |
| Male | TPR | 0.8219 | 0.8673 | 0.0454 |
| Male | FPR | 0.2157 | 0.2856 | 0.0698 |
| Female | Accuracy | 0.7688 | 0.7661 | -0.0027 |
| Female | Positive Rate | 0.5256 | 0.5849 | 0.0593 |
| Female | TPR | 0.7620 | 0.8124 | 0.0504 |
| Female | FPR | 0.2225 | 0.2932 | 0.0707 |

### age_group

| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| 36-50 | Accuracy | 0.7741 | 0.7680 | -0.0061 |
| 36-50 | Positive Rate | 0.5158 | 0.5770 | 0.0611 |
| 36-50 | TPR | 0.7701 | 0.8213 | 0.0512 |
| 36-50 | FPR | 0.2212 | 0.2938 | 0.0726 |
| 18-25 | Accuracy | 0.7926 | 0.7872 | -0.0054 |
| 18-25 | Positive Rate | 0.5443 | 0.5972 | 0.0529 |
| 18-25 | TPR | 0.8000 | 0.8423 | 0.0423 |
| 18-25 | FPR | 0.2169 | 0.2833 | 0.0664 |
| 26-35 | Accuracy | 0.7918 | 0.7846 | -0.0072 |
| 26-35 | Positive Rate | 0.5250 | 0.5831 | 0.0582 |
| 26-35 | TPR | 0.7954 | 0.8429 | 0.0475 |
| 26-35 | FPR | 0.2123 | 0.2828 | 0.0705 |
| 51+ | Accuracy | 0.7512 | 0.7458 | -0.0053 |
| 51+ | Positive Rate | 0.4993 | 0.5657 | 0.0664 |
| 51+ | TPR | 0.7452 | 0.8050 | 0.0598 |
| 51+ | FPR | 0.2426 | 0.3159 | 0.0733 |

### location

| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| Rural | Accuracy | 0.7914 | 0.7840 | -0.0074 |
| Rural | Positive Rate | 0.5269 | 0.5833 | 0.0564 |
| Rural | TPR | 0.7994 | 0.8455 | 0.0461 |
| Rural | FPR | 0.2177 | 0.2858 | 0.0681 |
| Suburban | Accuracy | 0.7869 | 0.7798 | -0.0071 |
| Suburban | Positive Rate | 0.5263 | 0.5838 | 0.0575 |
| Suburban | TPR | 0.7932 | 0.8404 | 0.0473 |
| Suburban | FPR | 0.2203 | 0.2896 | 0.0693 |
| Urban | Accuracy | 0.7775 | 0.7721 | -0.0054 |
| Urban | Positive Rate | 0.5223 | 0.5824 | 0.0602 |
| Urban | TPR | 0.7750 | 0.8252 | 0.0503 |
| Urban | FPR | 0.2196 | 0.2916 | 0.0720 |

## Conclusion

- Demographic Parity disparity improved by an average of 9.5%
- Equalized Odds disparity improved by an average of 8.8%

The fairness adjustments have made **minor improvements** to the model's fairness metrics. Consider trying different fairness constraints or adjusting hyperparameters.
