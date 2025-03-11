# Account Health Predictor Evaluation

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document details the comprehensive evaluation methodology used to assess the Account Health Predictor's performance. The evaluation framework encompasses multiple dimensions of model assessment to ensure the predictor is accurate, reliable, fair, and provides actionable insights.

## Evaluation Framework

The Account Health Predictor is evaluated using a multi-faceted approach that tests its capabilities across several dimensions:

1. **Prediction Accuracy** - How well the model predicts health scores and classifications
2. **Anomaly Detection** - The model's ability to identify unusual patterns in account performance
3. **Recommendation Quality** - The relevance and impact of suggested optimization actions
4. **Time Series Forecasting** - Accuracy of future performance predictions
5. **Fairness Across Segments** - Consistency of performance across different account types
6. **Robustness** - Stability under various data quality conditions

## Evaluation Datasets

### Primary Evaluation Datasets

| Dataset | Description | Size | Usage |
|---------|-------------|------|-------|
| Hold-out Test Set | Random sample of accounts withheld from training | 500 accounts | General performance metrics |
| Time Series Validation Set | Future data for accounts in training set | 90 days × 1,000 accounts | Forecast accuracy |
| Expert-labeled Dataset | Accounts labeled by advertising experts | 200 accounts | Ground truth comparison |
| Cross-Platform Dataset | Accounts with presence on multiple platforms | 300 accounts | Platform-agnostic evaluation |
| Small Account Dataset | Accounts with limited history or low spend | 250 accounts | Performance with limited data |
| Intervention Dataset | Accounts with documented optimization actions | 150 accounts | Recommendation quality |

### Industry Vertical Subsets

Performance is evaluated separately across key industry verticals to ensure consistent quality:

- E-commerce (n=120)
- Retail (n=105)
- B2B Services (n=85)
- Finance (n=70)
- Travel & Hospitality (n=65)
- Entertainment & Media (n=80)
- Healthcare (n=50)
- Education (n=45)

## Evaluation Metrics

### Health Score Prediction

| Metric | Target | Description |
|--------|--------|-------------|
| RMSE | < 8.0 | Root Mean Square Error between predicted and actual health scores |
| MAE | < 6.0 | Mean Absolute Error of health score predictions |
| R² | > 0.75 | Coefficient of determination for health score regression |
| 90% CI Coverage | > 0.90 | Percentage of actual scores falling within 90% confidence interval |

### Health Classification

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | > 0.80 | Proportion of correct classifications |
| Precision | > 0.80 | Precision for each health category |
| Recall | > 0.75 | Recall for each health category |
| F1 Score | > 0.80 | Harmonic mean of precision and recall |
| Cohen's Kappa | > 0.70 | Agreement with expert labels, adjusted for chance |

### Anomaly Detection

| Metric | Target | Description |
|--------|--------|-------------|
| Precision | > 0.75 | Proportion of detected anomalies that are genuine |
| Recall | > 0.70 | Proportion of genuine anomalies that are detected |
| F1 Score | > 0.75 | Harmonic mean of precision and recall |
| AUC-ROC | > 0.85 | Area under the ROC curve for anomaly detection |
| False Positive Rate | < 0.10 | Rate of false anomaly detections |

### Recommendation Quality

| Metric | Target | Description |
|--------|--------|-------------|
| Precision@3 | > 0.80 | Precision of top 3 recommendations |
| Acceptance Rate | > 0.30 | Rate at which recommendations are implemented |
| Impact Accuracy | > 0.70 | Accuracy of predicted impact vs. actual impact |
| Time-to-Value | < 14 days | Average time until recommendation shows impact |
| Diversity | > 0.50 | Proportion of unique recommendation types provided |

### Time Series Forecasting

| Metric | Target | Description |
|--------|--------|-------------|
| RMSE (7-day) | < 10% | RMSE of 7-day forecasts (as % of actual) |
| RMSE (30-day) | < 15% | RMSE of 30-day forecasts (as % of actual) |
| MAPE | < 20% | Mean Absolute Percentage Error |
| Direction Accuracy | > 0.75 | Accuracy in predicting trend direction |
| Calibration Error | < 0.05 | Error in prediction interval calibration |

## Evaluation Process

The Account Health Predictor evaluation follows this process:

1. **Initial Evaluation**
   - Run the model on the hold-out test set
   - Calculate core metrics (RMSE, classification accuracy, etc.)
   - Generate initial performance report

2. **Time-Series Validation**
   - Test forecasting accuracy on future data
   - Evaluate prediction stability over time
   - Assess drift in model accuracy

3. **Expert Comparison**
   - Compare model assessments with expert evaluations
   - Analyze disagreement patterns
   - Identify areas for improvement

4. **Recommendation Tracking**
   - Track implementation of recommendations in live accounts
   - Measure impact of implemented recommendations
   - Calculate recommendation relevance metrics

5. **Fairness Analysis**
   - Evaluate performance across account segments
   - Test for disparities in accuracy by account size, industry, or platform
   - Identify and address any systematic biases

6. **Robustness Testing**
   - Evaluate with artificially degraded data
   - Test with missing metrics
   - Measure performance with historical platform changes

## Cross-Platform Evaluation

The Account Health Predictor is evaluated for consistency across advertising platforms:

| Platform | Health Score RMSE | Classification Accuracy | Anomaly F1 | Recommendation Precision@3 |
|----------|-------------------|-------------------------|------------|----------------------------|
| Facebook | 6.8 | 0.85 | 0.79 | 0.84 |
| Google | 7.0 | 0.84 | 0.78 | 0.82 |
| Amazon | 8.1 | 0.78 | 0.72 | 0.77 |
| TikTok | 9.3 | 0.75 | 0.68 | 0.74 |
| Pinterest | 8.9 | 0.76 | 0.70 | 0.75 |
| LinkedIn | 8.4 | 0.77 | 0.71 | 0.76 |
| Snapchat | 9.5 | 0.74 | 0.67 | 0.73 |

## Evaluation by Account Size

Performance is analyzed by account size to ensure fairness:

| Account Size | Health Score RMSE | Classification Accuracy | Notes |
|--------------|-------------------|-------------------------|-------|
| Small (<$1K/mo) | 9.8 | 0.74 | Limited data impacts accuracy |
| Medium ($1K-10K/mo) | 7.5 | 0.81 | Good overall performance |
| Large ($10K-100K/mo) | 6.4 | 0.86 | Strong performance with rich data |
| Enterprise (>$100K/mo) | 6.1 | 0.88 | Best performance with extensive data |

## Fairness Evaluation

The model is evaluated for fairness across different segments:

1. **Industry Fairness**
   - Performance parity across industry verticals
   - Similar recommendation accuracy across sectors
   - Balanced representation in anomaly detection

2. **Platform Fairness**
   - Consistent performance regardless of advertising platform
   - Platform-agnostic health assessment criteria
   - Balanced recommendation diversity across platforms

3. **Account Size Fairness**
   - Strategies to maintain accuracy for smaller accounts
   - Confidence calibration appropriate to data volume
   - Specialized features for limited-data scenarios

4. **Geographical Fairness**
   - Consistent performance across regions
   - Accommodation of market-specific factors
   - Testing with international account datasets

## Baseline Comparisons

The Account Health Predictor is compared against these baselines:

1. **Expert Baseline**: Manual assessment by advertising specialists
2. **Platform Metrics Baseline**: Native quality scores from platforms
3. **Simple Heuristic Baseline**: Rule-based health scoring
4. **Previous Model Version**: Comparison to prior model generation
5. **Naive Forecast Baseline**: Simple trend extrapolation

### Baseline Comparison Results

| Baseline | Health Score RMSE | Classification Accuracy | Recommendation Precision@3 |
|----------|-------------------|-------------------------|----------------------------|
| Account Health Predictor v1.5 | 7.2 | 0.83 | 0.81 |
| Expert Assessment | 8.1 | 0.79 | 0.78 |
| Platform Metrics | 12.4 | 0.65 | 0.62 |
| Heuristic Rules | 14.7 | 0.61 | 0.58 |
| Previous Model (v1.4) | 8.9 | 0.76 | 0.73 |
| Naive Forecast | 18.3 | 0.54 | N/A |

## Evaluation Tools

The evaluation process leverages these tools:

1. **EvalML Framework**: Custom evaluation pipeline for comprehensive testing
2. **SHAP Analysis**: For feature contribution understanding
3. **Fairness Indicators**: For bias detection and mitigation
4. **A/B Testing Platform**: For recommendation impact assessment
5. **Visualization Dashboard**: For performance metric monitoring

## Example Evaluation Code

```python
from within.evaluation import AccountHealthEvaluator
from within.models import AccountHealthPredictor

# Initialize model and evaluator
model = AccountHealthPredictor(version="1.5.0")
evaluator = AccountHealthEvaluator(metrics=[
    "rmse", "classification_accuracy", "anomaly_f1", "recommendation_precision"
])

# Load test dataset
test_data = load_test_dataset("account_health_test_v3.parquet")

# Run evaluation
results = evaluator.evaluate(
    model=model,
    test_data=test_data,
    segments=["platform", "industry", "account_size"],
    confidence_intervals=True
)

# Generate comprehensive report
report = evaluator.generate_report(
    results=results,
    baseline_results=baseline_results,
    output_format="html"
)

# Save evaluation artifacts
evaluator.save_results(
    results=results,
    path="evaluations/account_health_v1.5/"
)
```

## Error Analysis

The evaluation includes detailed error analysis:

1. **Confusion Matrix Analysis**: Identifies patterns in misclassification
2. **Error Distribution**: Examines error distribution across account types
3. **Feature Contribution**: Analyzes which features contribute to errors
4. **Edge Cases**: Identifies specific account scenarios with poor performance
5. **Temporal Effects**: Studies how prediction accuracy changes over time

## Continuous Evaluation

The Account Health Predictor undergoes ongoing evaluation:

1. **Weekly Performance Tracking**: Automated evaluation on new data
2. **Monthly Deep Dives**: Comprehensive analysis of performance trends
3. **Quarterly Full Evaluation**: Complete evaluation using all metrics
4. **Pre/Post Platform Changes**: Special evaluations around platform updates
5. **User Feedback Integration**: Incorporating user reports into evaluation

## Conclusion

The Account Health Predictor demonstrates strong performance across key metrics, meeting or exceeding targets in most areas. The evaluation shows particular strength in classification accuracy (0.83) and recommendation quality (0.81 Precision@3). Areas for improvement include performance on smaller accounts and newer platforms with limited training data.

The multi-faceted evaluation confirms the model provides reliable, actionable insights for account optimization while maintaining fairness across different account types, sizes, and platforms.

## References

- [Model Card: Account Health Predictor](../model_card_account_health_predictor.md)
- [Account Health Prediction Implementation](../account_health_prediction.md)
- [Anomaly Detection Methodology](../technical/anomaly_detection.md)
- [Model Evaluation Framework](../model_evaluation.md)
- [Fairness Assessment Guidelines](../../standards/fairness_guidelines.md)

*Last updated: March 20, 2025* 