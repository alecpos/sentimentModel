#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Fairness Evaluation Demonstration

This script demonstrates the evaluation of model fairness across different demographic groups.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import our SimpleAdScorePredictor
from app.models.ml.prediction.simple_ad_score_predictor import SimpleAdScorePredictor

# Create output directory if it doesn't exist
os.makedirs('fairness_evaluation', exist_ok=True)
os.makedirs('fairness_evaluation/plots', exist_ok=True)

print("Loading and preparing data...")
# Load the data
data = pd.read_csv('data/sample_ad_data.csv')

# For demonstration, let's create a binary target by thresholding the ad_score
# Scores >= 0.75 are considered "high performing ads"
threshold = 0.75
data['high_performing'] = (data['ad_score'] >= threshold).astype(int)

# Let's use target_gender as our protected attribute
# Let's ensure we have a more diverse dataset for demonstration
data['target_gender'].replace('all', np.random.choice(['male', 'female', 'all'], p=[0.3, 0.3, 0.4]), inplace=True)

# Separate features and target
X = data.drop(['ad_score', 'high_performing', 'campaign_id', 'ad_text'], axis=1)
y = data['high_performing']

# Keep protected attribute in X for fairness evaluation
protected_attr = 'target_gender'

# Convert target_gender to one-hot encoded columns
X = pd.get_dummies(X, columns=[protected_attr], drop_first=False)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training the model...")
# Create and train the model
model = SimpleAdScorePredictor(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
results = model.predict(X_test)
predictions = results['score']
y_pred_binary = (predictions >= 0.5).astype(int)

# Calculate performance
accuracy = np.mean(y_pred_binary == y_test)
cm = confusion_matrix(y_test, y_pred_binary)
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Model performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1_score:.4f}")

print("\nEvaluating fairness metrics...")

# Reconstruct the protected attributes from one-hot encoding
# Look for the columns that start with "target_gender_"
protected_columns = [col for col in X_test.columns if col.startswith("target_gender_")]
protected_values = [col.replace("target_gender_", "") for col in protected_columns]

# Create a function to identify the demographic group
def get_demographic(row):
    for col, val in zip(protected_columns, protected_values):
        if row[col] == 1:
            return val
    return "unknown"

# Add demographic group back to the test data
X_test_with_demographics = X_test.copy()
X_test_with_demographics['demographic'] = X_test.apply(get_demographic, axis=1)

# Calculate demographic parity (equal positive prediction rates)
demo_metrics = {}
overall_positive_rate = y_pred_binary.mean()

for demo in protected_values:
    # Get predictions for this demographic group
    demo_mask = X_test_with_demographics['demographic'] == demo
    demo_pred = y_pred_binary[demo_mask]
    demo_true = y_test[demo_mask]
    
    # Skip if no samples in this group
    if len(demo_pred) == 0:
        continue
    
    # Calculate positive prediction rate
    positive_rate = demo_pred.mean()
    
    # Calculate true positive rate (equal opportunity)
    if sum(demo_true) > 0:  # Only if there are positive examples
        tp = sum((demo_pred == 1) & (demo_true == 1))
        tpr = tp / sum(demo_true)
    else:
        tpr = float('nan')
    
    demo_metrics[demo] = {
        'count': sum(demo_mask),
        'positive_rate': positive_rate,
        'true_positive_rate': tpr
    }

print("\nDemographic parity analysis:")
print(f"Overall positive prediction rate: {overall_positive_rate:.4f}")
for demo, metrics in demo_metrics.items():
    print(f"  {demo}: {metrics['positive_rate']:.4f} ({metrics['count']} samples)")

# Find min and max positive rates
positive_rates = [m['positive_rate'] for m in demo_metrics.values()]
if len(positive_rates) >= 2:
    min_rate = min(positive_rates)
    max_rate = max(positive_rates)
    disparity_ratio = min_rate / max_rate if max_rate > 0 else 1.0
    
    # Check if the model passes the 80% rule
    passes_80pct_rule = disparity_ratio >= 0.8
    
    print(f"\nDisparity ratio (min/max): {disparity_ratio:.4f}")
    print(f"Passes 80% rule: {passes_80pct_rule}")
    
    # Create bar chart of positive prediction rates
    plt.figure(figsize=(10, 6))
    demos = list(demo_metrics.keys())
    rates = [demo_metrics[d]['positive_rate'] for d in demos]
    
    # Sort by rate
    sorted_idx = np.argsort(rates)
    demos = [demos[i] for i in sorted_idx]
    rates = [rates[i] for i in sorted_idx]
    
    plt.bar(demos, rates, color='skyblue')
    plt.axhline(y=overall_positive_rate, color='r', linestyle='-', label='Overall Rate')
    plt.axhline(y=max_rate * 0.8, color='orange', linestyle='--', label='80% of Max Rate')
    
    plt.xlabel('Demographic Group')
    plt.ylabel('Positive Prediction Rate')
    plt.title('Demographic Parity Comparison')
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(rates):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('fairness_evaluation/plots/demographic_parity.png', dpi=300)
    plt.close()

# Evaluate equal opportunity (equal true positive rates)
print("\nEqual opportunity analysis:")
tpr_values = [m['true_positive_rate'] for m in demo_metrics.values() 
             if not np.isnan(m['true_positive_rate'])]

if len(tpr_values) >= 2:
    min_tpr = min(tpr_values)
    max_tpr = max(tpr_values)
    tpr_disparity_ratio = min_tpr / max_tpr if max_tpr > 0 else 1.0
    
    # Check if the model passes the 80% rule for TPR
    passes_tpr_80pct_rule = tpr_disparity_ratio >= 0.8
    
    for demo, metrics in demo_metrics.items():
        if not np.isnan(metrics['true_positive_rate']):
            print(f"  {demo}: {metrics['true_positive_rate']:.4f}")
        else:
            print(f"  {demo}: No positive examples")
    
    print(f"\nTPR disparity ratio (min/max): {tpr_disparity_ratio:.4f}")
    print(f"Passes 80% rule for equal opportunity: {passes_tpr_80pct_rule}")
    
    # Create equal opportunity plot for groups with TPR values
    plt.figure(figsize=(10, 6))
    demos_with_tpr = [d for d in demos if not np.isnan(demo_metrics[d]['true_positive_rate'])]
    tpr_values = [demo_metrics[d]['true_positive_rate'] for d in demos_with_tpr]
    
    # Sort by TPR
    sorted_idx = np.argsort(tpr_values)
    demos_with_tpr = [demos_with_tpr[i] for i in sorted_idx]
    tpr_values = [tpr_values[i] for i in sorted_idx]
    
    plt.bar(demos_with_tpr, tpr_values, color='lightgreen')
    plt.axhline(y=np.mean(tpr_values), color='r', linestyle='-', label='Average TPR')
    plt.axhline(y=max_tpr * 0.8, color='orange', linestyle='--', label='80% of Max TPR')
    
    plt.xlabel('Demographic Group')
    plt.ylabel('True Positive Rate')
    plt.title('Equal Opportunity Comparison')
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(tpr_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('fairness_evaluation/plots/equal_opportunity.png', dpi=300)
    plt.close()

# Create a fairness report
report_path = 'fairness_evaluation/fairness_report.md'
with open(report_path, 'w') as f:
    f.write("# Model Fairness Evaluation Report\n\n")
    
    f.write("## Summary\n\n")
    f.write(f"**Fairness Threshold:** 0.80 (80% rule)\n\n")
    
    if 'disparity_ratio' in locals():
        if passes_80pct_rule and (not 'passes_tpr_80pct_rule' or passes_tpr_80pct_rule):
            f.write("**Overall Verdict:** FAIR\n\n")
        else:
            f.write("**Overall Verdict:** UNFAIR\n\n")
    
    # Add summary table
    f.write("| Metric | Worst Case Ratio | Threshold | Status |\n")
    f.write("|--------|-----------------|-----------|--------|\n")
    
    if 'disparity_ratio' in locals():
        status = "✅ PASS" if passes_80pct_rule else "❌ FAIL"
        f.write(f"| Demographic Parity | {disparity_ratio:.2f} | 0.80 | {status} |\n")
    
    if 'tpr_disparity_ratio' in locals():
        status = "✅ PASS" if passes_tpr_80pct_rule else "❌ FAIL"
        f.write(f"| Equal Opportunity | {tpr_disparity_ratio:.2f} | 0.80 | {status} |\n")
    
    f.write("\n")
    
    # Protected attributes evaluated
    f.write("### Protected Attributes Evaluated\n\n")
    f.write(f"- target_gender\n\n")
    
    # Include plots if available
    if os.path.exists('fairness_evaluation/plots/demographic_parity.png'):
        f.write("![Demographic Parity](plots/demographic_parity.png)\n\n")
    
    if os.path.exists('fairness_evaluation/plots/equal_opportunity.png'):
        f.write("![Equal Opportunity](plots/equal_opportunity.png)\n\n")
    
    # Detailed results
    f.write("## Detailed Results\n\n")
    f.write("### Target Gender\n\n")
    
    # Demographic parity details
    f.write("#### Demographic Parity\n\n")
    f.write("Demographic parity measures whether the model predicts positive outcomes at the same rate across different demographic groups.\n\n")
    f.write(f"- Overall positive prediction rate: {overall_positive_rate:.4f}\n")
    f.write("- Group positive prediction rates:\n")
    
    for demo, metrics in demo_metrics.items():
        f.write(f"  - {demo}: {metrics['positive_rate']:.4f}\n")
    
    if 'disparity_ratio' in locals():
        f.write(f"- Disparity ratio: {disparity_ratio:.4f}\n")
        f.write(f"- Maximum disparity: {max_rate - min_rate:.4f}\n\n")
    
    # Equal opportunity details
    f.write("#### Equal Opportunity\n\n")
    f.write("Equal opportunity measures whether the model has the same true positive rate (recall) across different demographic groups.\n\n")
    if 'tpr_values' in locals() and len(tpr_values) > 0:
        f.write(f"- Average true positive rate: {np.mean(tpr_values):.4f}\n")
        f.write("- Group true positive rates:\n")
        
        for demo, metrics in demo_metrics.items():
            if not np.isnan(metrics['true_positive_rate']):
                f.write(f"  - {demo}: {metrics['true_positive_rate']:.4f}\n")
            else:
                f.write(f"  - {demo}: No positive examples\n")
        
        if 'tpr_disparity_ratio' in locals():
            f.write(f"- TPR disparity ratio: {tpr_disparity_ratio:.4f}\n")
            f.write(f"- Maximum TPR disparity: {max_tpr - min_tpr:.4f}\n\n")
    
    # Model performance
    f.write("## Model Performance\n\n")
    f.write(f"- Accuracy: {accuracy:.4f}\n")
    f.write(f"- Precision: {precision:.4f}\n")
    f.write(f"- Recall: {recall:.4f}\n")
    f.write(f"- F1 Score: {f1_score:.4f}\n\n")
    
    # Recommendations
    f.write("## Recommendations\n\n")
    
    if 'passes_80pct_rule' in locals() and passes_80pct_rule and (not 'passes_tpr_80pct_rule' or passes_tpr_80pct_rule):
        f.write("The model meets the minimum fairness criteria (80% rule) for all protected attributes. However, consider the following recommendations to further improve fairness:\n\n")
    else:
        f.write("The model does not meet the minimum fairness criteria for all protected attributes. Consider the following recommendations to address fairness issues:\n\n")
    
    f.write("1. **Bias Mitigation Techniques**: Consider implementing pre-processing, in-processing, or post-processing bias mitigation techniques.\n")
    f.write("2. **Data Collection**: Review data collection procedures to ensure representative sampling across all demographic groups.\n")
    f.write("3. **Feature Engineering**: Review features that may introduce or amplify biases in the model.\n")
    f.write("4. **Model Selection**: Consider using different model architectures that may be less prone to learning biased patterns.\n")
    f.write("5. **Threshold Adjustment**: Consider group-specific thresholds to balance error rates across groups.\n")

print(f"\nFairness evaluation complete!")
print(f"Fairness report generated: {report_path}")
print("Check the 'fairness_evaluation' directory for results.")
print("Key files to examine:")
print("  - fairness_report.md: Comprehensive fairness evaluation report")
print("  - plots/demographic_parity.png: Visual comparison of positive prediction rates")
print("  - plots/equal_opportunity.png: Visual comparison of true positive rates") 