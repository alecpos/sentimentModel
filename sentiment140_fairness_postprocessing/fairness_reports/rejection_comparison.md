# Fairness Improvement Comparison

Generated on: 2025-03-12 18:32:40

## Summary of Improvements

| Protected Attribute | Metric | Before | After | Change | % Improvement |
|---------------------|--------|--------|-------|--------|---------------|
| gender | Demographic Parity | 0.0026 | 0.0036 | -0.0010 | -38.3% |
| gender | Equalized Odds | 0.0666 | 0.0660 | 0.0006 | 0.8% |
| age_group | Demographic Parity | 0.0450 | 0.0360 | 0.0090 | 19.9% |
| age_group | Equalized Odds | 0.0852 | 0.0819 | 0.0033 | 3.9% |
| location | Demographic Parity | 0.0047 | 0.0020 | 0.0027 | 58.2% |
| location | Equalized Odds | 0.0271 | 0.0255 | 0.0015 | 5.7% |

## Detailed Group-level Changes

### gender

| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| Male | Accuracy | 0.8033 | 0.8029 | -0.0004 |
| Male | Positive Rate | 0.5230 | 0.5239 | 0.0009 |
| Male | TPR | 0.8219 | 0.8224 | 0.0005 |
| Male | FPR | 0.2157 | 0.2171 | 0.0013 |
| Female | Accuracy | 0.7688 | 0.7689 | 0.0001 |
| Female | Positive Rate | 0.5256 | 0.5275 | 0.0019 |
| Female | TPR | 0.7620 | 0.7638 | 0.0018 |
| Female | FPR | 0.2225 | 0.2245 | 0.0021 |

### age_group

| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| 36-50 | Accuracy | 0.7741 | 0.7740 | -0.0001 |
| 36-50 | Positive Rate | 0.5158 | 0.5209 | 0.0051 |
| 36-50 | TPR | 0.7701 | 0.7747 | 0.0046 |
| 36-50 | FPR | 0.2212 | 0.2268 | 0.0056 |
| 18-25 | Accuracy | 0.7926 | 0.7922 | -0.0004 |
| 18-25 | Positive Rate | 0.5443 | 0.5418 | -0.0025 |
| 18-25 | TPR | 0.8000 | 0.7975 | -0.0026 |
| 18-25 | FPR | 0.2169 | 0.2145 | -0.0024 |
| 26-35 | Accuracy | 0.7918 | 0.7918 | -0.0000 |
| 26-35 | Positive Rate | 0.5250 | 0.5258 | 0.0008 |
| 26-35 | TPR | 0.7954 | 0.7961 | 0.0008 |
| 26-35 | FPR | 0.2123 | 0.2132 | 0.0009 |
| 51+ | Accuracy | 0.7512 | 0.7512 | 0.0001 |
| 51+ | Positive Rate | 0.4993 | 0.5058 | 0.0065 |
| 51+ | TPR | 0.7452 | 0.7516 | 0.0064 |
| 51+ | FPR | 0.2426 | 0.2491 | 0.0065 |

### location

| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| Rural | Accuracy | 0.7914 | 0.7909 | -0.0005 |
| Rural | Positive Rate | 0.5269 | 0.5272 | 0.0003 |
| Rural | TPR | 0.7994 | 0.7992 | -0.0002 |
| Rural | FPR | 0.2177 | 0.2186 | 0.0009 |
| Suburban | Accuracy | 0.7869 | 0.7867 | -0.0002 |
| Suburban | Positive Rate | 0.5263 | 0.5263 | 0.0000 |
| Suburban | TPR | 0.7932 | 0.7930 | -0.0002 |
| Suburban | FPR | 0.2203 | 0.2206 | 0.0002 |
| Urban | Accuracy | 0.7775 | 0.7776 | 0.0001 |
| Urban | Positive Rate | 0.5223 | 0.5253 | 0.0030 |
| Urban | TPR | 0.7750 | 0.7779 | 0.0029 |
| Urban | FPR | 0.2196 | 0.2228 | 0.0032 |

## Conclusion

- Demographic Parity disparity improved by an average of 13.2%
- Equalized Odds disparity improved by an average of 3.5%

The fairness adjustments have made **moderate improvements** to the model's fairness metrics.
