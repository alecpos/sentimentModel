#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP Model Explainability Demonstration

This script demonstrates the use of SHAP for explaining model predictions
with a simple ad score prediction model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap  # Import SHAP library

# Import our SimpleAdScorePredictor
from app.models.ml.prediction.simple_ad_score_predictor import SimpleAdScorePredictor

# Create output directory if it doesn't exist
os.makedirs('model_explanations', exist_ok=True)
os.makedirs('model_explanations/plots', exist_ok=True)

print("Loading and preparing data...")
# Load the data
data = pd.read_csv('data/sample_ad_data.csv')

# Separate features and target
X = data.drop(['ad_score', 'campaign_id', 'ad_text'], axis=1)
y = data['ad_score']

# Convert target_gender to numeric (one-hot encode)
X = pd.get_dummies(X, columns=['target_gender'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training the model...")
# Create and train the model
model = SimpleAdScorePredictor(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
results = model.predict(X_test)
predictions = results['score']

# Calculate performance
mse = np.mean((predictions - y_test) ** 2)
mae = np.mean(np.abs(predictions - y_test))
r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

print(f"Model performance:")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Mean Absolute Error: {mae:.4f}")
print(f"  R² Score: {r2:.4f}")

# Print feature importance
importance_df = model.feature_importance()
print("\nTop 5 feature importances:")
print(importance_df.head(5))

print("\nGenerating SHAP explanations...")

# Get numeric features from the test data
X_test_numeric = X_test[model.numeric_cols]

# Standardize the data as the model expects
X_test_scaled = model.scaler.transform(X_test_numeric)

# Create a SHAP TreeExplainer for our random forest model
explainer = shap.TreeExplainer(model.model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_scaled)

# Create feature importance plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_numeric, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('model_explanations/plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("- Created feature importance plot")

# Create SHAP summary dot plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_numeric, show=False)
plt.tight_layout()
plt.savefig('model_explanations/plots/summary_plot_dot.png', dpi=300, bbox_inches='tight')
plt.close()
print("- Created SHAP summary dot plot")

# Create dependence plots for top features
top_features = importance_df.head(3)['feature'].values
for feature in top_features:
    plt.figure(figsize=(10, 6))
    feature_idx = list(X_test_numeric.columns).index(feature)
    shap.dependence_plot(feature_idx, shap_values, X_test_numeric, show=False)
    plt.tight_layout()
    plt.savefig(f'model_explanations/plots/dependence_plot_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"- Created dependence plot for {feature}")

# Create a simple explanation report in Markdown
report_path = 'model_explanations/model_explanation_report.md'
with open(report_path, 'w') as f:
    f.write("# Model Explanation Report\n\n")
    
    f.write("## Model Information\n\n")
    f.write("- Model Type: Random Forest Regressor\n")
    f.write(f"- Number of Features: {len(model.numeric_cols)}\n\n")
    
    f.write("## Performance Metrics\n\n")
    f.write(f"- Mean Squared Error: {mse:.4f}\n")
    f.write(f"- Mean Absolute Error: {mae:.4f}\n")
    f.write(f"- R² Score: {r2:.4f}\n\n")
    
    f.write("## Feature Importance\n\n")
    f.write("The following features have the most impact on model predictions:\n\n")
    f.write("| Feature | Importance |\n")
    f.write("|---------|------------|\n")
    
    # Add top features
    for _, row in importance_df.head(10).iterrows():
        f.write(f"| {row['feature']} | {row['importance']:.4f} |\n")
    
    f.write("\n\n")
    f.write("![Feature Importance](plots/feature_importance.png)\n\n")
    
    f.write("## SHAP Summary Plot\n\n")
    f.write("This plot shows how each feature affects model predictions.\n\n")
    f.write("![SHAP Summary Dot Plot](plots/summary_plot_dot.png)\n\n")
    
    f.write("## Feature Dependence Plots\n\n")
    f.write("These plots show how the impact of a feature varies with its value.\n\n")
    
    for feature in top_features:
        f.write(f"### {feature}\n\n")
        f.write(f"![{feature} Dependence Plot](plots/dependence_plot_{feature}.png)\n\n")
    
    f.write("## Interpretation Guidelines\n\n")
    f.write("- **SHAP Values**: SHAP (SHapley Additive exPlanations) values represent the impact of each feature on the model's prediction.\n")
    f.write("- **Feature Importance**: Higher values indicate features with a larger overall impact on the model's predictions.\n")
    f.write("- **Dependence Plots**: Show how the effect of a feature varies with its value, and how it interacts with other features.\n")
    f.write("- **Color Coding**: In the dot plot, red indicates higher feature values, blue indicates lower values.\n")

print(f"\nExplanation report generated: {report_path}")
print("\nSHAP explanations complete!")
print("Check the 'model_explanations' directory for results.")
print("Key files to examine:")
print("  - model_explanation_report.md: Comprehensive explanation report")
print("  - plots/feature_importance.png: Visual representation of feature importance")
print("  - plots/summary_plot_dot.png: SHAP summary dot plot")
print("  - plots/dependence_plot_*.png: Feature dependence plots") 