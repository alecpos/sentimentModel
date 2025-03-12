# Model Explanation Report

## Model Information

- Model Type: Random Forest Regressor
- Number of Features: 14

## Performance Metrics

- Mean Squared Error: 0.0003
- Mean Absolute Error: 0.0134
- R² Score: 0.9409

## Feature Importance

The following features have the most impact on model predictions:

| Feature | Importance |
|---------|------------|
| ad_age_days | 0.4085 |
| ctr | 0.2075 |
| conversion_rate | 0.2034 |
| device_mobile | 0.0462 |
| daily_budget | 0.0460 |
| cpc | 0.0223 |
| device_desktop | 0.0139 |
| image_count | 0.0134 |
| description_length | 0.0119 |
| video_length | 0.0108 |


![Feature Importance](plots/feature_importance.png)

## SHAP Summary Plot

This plot shows how each feature affects model predictions.

![SHAP Summary Dot Plot](plots/summary_plot_dot.png)

## Feature Dependence Plots

These plots show how the impact of a feature varies with its value.

### ad_age_days

![ad_age_days Dependence Plot](plots/dependence_plot_ad_age_days.png)

### ctr

![ctr Dependence Plot](plots/dependence_plot_ctr.png)

### conversion_rate

![conversion_rate Dependence Plot](plots/dependence_plot_conversion_rate.png)

## Interpretation Guidelines

- **SHAP Values**: SHAP (SHapley Additive exPlanations) values represent the impact of each feature on the model's prediction.
- **Feature Importance**: Higher values indicate features with a larger overall impact on the model's predictions.
- **Dependence Plots**: Show how the effect of a feature varies with its value, and how it interacts with other features.
- **Color Coding**: In the dot plot, red indicates higher feature values, blue indicates lower values.
