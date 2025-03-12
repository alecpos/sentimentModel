#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Fairness Bias Mitigation Demonstration

This script demonstrates the application of bias mitigation techniques
to improve fairness across different demographic groups.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Import our fairness components
from app.models.ml.fairness.mitigation import ReweighingMitigation
from app.models.ml.fairness.evaluator import FairnessEvaluator

# Create output directories if they don't exist
os.makedirs('fairness_evaluation/mitigation', exist_ok=True)
os.makedirs('fairness_evaluation/mitigation/plots', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def evaluate_model(model, X_test, y_test, protected_attributes, threshold=0.5):
    """
    Evaluate model performance and fairness metrics.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test labels
        protected_attributes: Dictionary of protected attribute values
        threshold: Decision threshold for binary classification
        
    Returns:
        Dictionary with performance and fairness metrics
    """
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
        
    y_pred = (y_prob >= threshold).astype(int)
    
    # Performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate fairness metrics manually since we're using a stub implementation
    # Demographic parity
    demo_metrics = {}
    overall_positive_rate = y_pred.mean()
    
    # Group metrics by protected attribute
    group_metrics = {}
    for attr_name, attr_values in protected_attributes.items():
        unique_values = np.unique(attr_values)
        for value in unique_values:
            mask = (attr_values == value)
            group_preds = y_pred[mask]
            group_labels = y_test.iloc[mask] if hasattr(y_test, 'iloc') else y_test[mask]
            
            # Skip if no samples in this group
            if len(group_preds) == 0:
                continue
            
            # Demographic parity - positive prediction rate
            positive_rate = group_preds.mean()
            
            # Equal opportunity - true positive rate
            positives = (group_labels == 1)
            if sum(positives) > 0:
                true_positives = group_preds[positives]
                true_positive_rate = true_positives.mean()
            else:
                true_positive_rate = float('nan')
            
            group_name = f"{attr_name}_{value}"
            group_metrics[group_name] = {
                'positive_rate': positive_rate,
                'true_positive_rate': true_positive_rate,
                'count': len(group_preds)
            }
    
    # Calculate disparity ratios
    positive_rates = [m['positive_rate'] for m in group_metrics.values()]
    min_rate = min(positive_rates) if positive_rates else 0
    max_rate = max(positive_rates) if positive_rates else 1
    demographic_parity_ratio = min_rate / max_rate if max_rate > 0 else 1.0
    
    # Calculate equal opportunity disparity ratio
    tpr_values = [m['true_positive_rate'] for m in group_metrics.values() 
                 if not np.isnan(m['true_positive_rate'])]
    min_tpr = min(tpr_values) if tpr_values else 0
    max_tpr = max(tpr_values) if tpr_values else 1
    equal_opportunity_ratio = min_tpr / max_tpr if max_tpr > 0 else 1.0
    
    return {
        'performance': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        },
        'fairness': {
            'demographic_parity': {
                'overall_rate': overall_positive_rate,
                'group_metrics': group_metrics,
                'disparity_ratio': demographic_parity_ratio,
                'min_rate': min_rate,
                'max_rate': max_rate
            },
            'equal_opportunity': {
                'group_metrics': group_metrics,
                'disparity_ratio': equal_opportunity_ratio,
                'min_rate': min_tpr,
                'max_rate': max_tpr
            }
        }
    }

def plot_fairness_comparison(original_results, mitigated_results, metric_name, output_path):
    """
    Create comparison plots for fairness metrics before and after mitigation.
    
    Args:
        original_results: Dictionary with original model results
        mitigated_results: Dictionary with mitigated model results
        metric_name: Name of the fairness metric to plot
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Extract relevant data
    original_metric = original_results['fairness'][metric_name]
    mitigated_metric = mitigated_results['fairness'][metric_name]
    
    groups = list(original_metric['group_metrics'].keys())
    
    # Get rates for each group
    original_rates = [original_metric['group_metrics'][group]['rate'] for group in groups]
    mitigated_rates = [mitigated_metric['group_metrics'][group]['rate'] for group in groups]
    
    # Set up the bar positions
    bar_width = 0.35
    r1 = np.arange(len(groups))
    r2 = [x + bar_width for x in r1]
    
    # Create the bars
    plt.bar(r1, original_rates, width=bar_width, label='Original Model', color='indianred', alpha=0.7)
    plt.bar(r2, mitigated_rates, width=bar_width, label='Mitigated Model', color='seagreen', alpha=0.7)
    
    # Add horizontal lines for thresholds
    plt.axhline(y=original_metric['overall_rate'], color='red', linestyle='--', 
                label='Original Overall Rate')
    plt.axhline(y=mitigated_metric['overall_rate'], color='green', linestyle='--',
                label='Mitigated Overall Rate')
    
    # Calculate improvement in disparity ratio
    original_disparity = original_metric['disparity_ratio']
    mitigated_disparity = mitigated_metric['disparity_ratio']
    improvement = ((mitigated_disparity - original_disparity) / original_disparity) * 100
    
    # Add annotations
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison\n'
              f'Disparity Ratio: {original_disparity:.2f} → {mitigated_disparity:.2f} '
              f'({improvement:.1f}% improvement)', fontsize=14)
    
    plt.xlabel('Demographic Groups', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.xticks([r + bar_width/2 for r in range(len(groups))], groups)
    plt.ylim(0, 1.0)
    
    # Add values on bars
    for i, (orig, mitig) in enumerate(zip(original_rates, mitigated_rates)):
        plt.text(i, orig + 0.02, f'{orig:.2f}', ha='center', va='bottom', color='black')
        plt.text(i + bar_width, mitig + 0.02, f'{mitig:.2f}', ha='center', va='bottom', color='black')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison(original_results, mitigated_results, output_path):
    """
    Create comparison plot for performance metrics before and after mitigation.
    
    Args:
        original_results: Dictionary with original model results
        mitigated_results: Dictionary with mitigated model results
        output_path: Path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    original_scores = [original_results['performance'][m] for m in metrics]
    mitigated_scores = [mitigated_results['performance'][m] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    r1 = np.arange(len(metrics))
    r2 = [x + bar_width for x in r1]
    
    ax.bar(r1, original_scores, width=bar_width, label='Original Model', color='indianred', alpha=0.7)
    ax.bar(r2, mitigated_scores, width=bar_width, label='Mitigated Model', color='seagreen', alpha=0.7)
    
    # Calculate average performance change
    perf_changes = [(mitigated_scores[i] - original_scores[i]) / original_scores[i] * 100 
                   for i in range(len(metrics))]
    avg_change = sum(perf_changes) / len(perf_changes)
    
    # Add annotations
    ax.set_title(f'Performance Metrics Comparison\nAverage change: {avg_change:.1f}%', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks([r + bar_width/2 for r in range(len(metrics))])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1.0)
    
    # Add values on bars
    for i, (orig, mitig) in enumerate(zip(original_scores, mitigated_scores)):
        change = ((mitig - orig) / orig) * 100
        color = 'green' if change >= 0 else 'red'
        ax.text(i, orig + 0.02, f'{orig:.3f}', ha='center', va='bottom', color='black')
        ax.text(i + bar_width, mitig + 0.02, 
                f'{mitig:.3f}\n({change:+.1f}%)', ha='center', va='bottom', color=color)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_bias_mitigation_report(original_results, mitigated_results, mitigation_technique, output_path):
    """
    Create a markdown report comparing original and mitigated model fairness.
    
    Args:
        original_results: Dictionary with original model results
        mitigated_results: Dictionary with mitigated model results
        mitigation_technique: Name of the mitigation technique used
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write(f"# Bias Mitigation Report - {mitigation_technique}\n\n")
        
        # Summary section
        f.write("## Summary\n\n")
        
        fairness_threshold = 0.8  # 80% rule
        
        # Compute fairness status for original and mitigated models
        metrics = ['demographic_parity', 'equal_opportunity']
        original_status = all(original_results['fairness'][m]['disparity_ratio'] >= fairness_threshold 
                              for m in metrics)
        mitigated_status = all(mitigated_results['fairness'][m]['disparity_ratio'] >= fairness_threshold 
                               for m in metrics)
        
        f.write(f"**Fairness Threshold:** {fairness_threshold:.2f} (80% rule)\n\n")
        f.write(f"**Original Model Verdict:** {'FAIR' if original_status else 'UNFAIR'}\n\n")
        f.write(f"**Mitigated Model Verdict:** {'FAIR' if mitigated_status else 'UNFAIR'}\n\n")
        
        # Add summary table for fairness metrics
        f.write("### Fairness Metrics\n\n")
        f.write("| Metric | Original Ratio | Mitigated Ratio | Improvement | Status |\n")
        f.write("|--------|---------------|----------------|------------|--------|\n")
        
        for metric in metrics:
            original_ratio = original_results['fairness'][metric]['disparity_ratio']
            mitigated_ratio = mitigated_results['fairness'][metric]['disparity_ratio']
            improvement = ((mitigated_ratio - original_ratio) / original_ratio) * 100
            
            original_passes = original_ratio >= fairness_threshold
            mitigated_passes = mitigated_ratio >= fairness_threshold
            
            status = ""
            if not original_passes and mitigated_passes:
                status = "✅ FIXED"
            elif not original_passes and not mitigated_passes:
                status = "⚠️ IMPROVED" if improvement > 0 else "❌ NO IMPROVEMENT"
            elif original_passes and mitigated_passes:
                status = "✅ MAINTAINED"
            
            f.write(f"| {metric.replace('_', ' ').title()} | {original_ratio:.2f} | {mitigated_ratio:.2f} | {improvement:+.1f}% | {status} |\n")
        
        f.write("\n")
        
        # Add summary table for performance metrics
        f.write("### Performance Metrics\n\n")
        f.write("| Metric | Original Score | Mitigated Score | Change |\n")
        f.write("|--------|---------------|----------------|--------|\n")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            original_score = original_results['performance'][metric]
            mitigated_score = mitigated_results['performance'][metric]
            change = ((mitigated_score - original_score) / original_score) * 100
            
            f.write(f"| {metric.replace('_', ' ').title()} | {original_score:.4f} | {mitigated_score:.4f} | {change:+.1f}% |\n")
        
        f.write("\n")
        
        # Include visualization references
        f.write("## Visualizations\n\n")
        f.write("### Demographic Parity Comparison\n\n")
        f.write("![Demographic Parity Comparison](plots/demographic_parity_comparison.png)\n\n")
        
        f.write("### Equal Opportunity Comparison\n\n")
        f.write("![Equal Opportunity Comparison](plots/equal_opportunity_comparison.png)\n\n")
        
        f.write("### Performance Metrics Comparison\n\n")
        f.write("![Performance Metrics Comparison](plots/performance_comparison.png)\n\n")
        
        # Implementation details
        f.write("## Implementation Details\n\n")
        f.write(f"### Mitigation Technique: {mitigation_technique}\n\n")
        
        if mitigation_technique == "Reweighing":
            f.write("Reweighing is a preprocessing technique that assigns weights to training examples to ensure fairness. ")
            f.write("The method calculates weights for each instance based on its protected attribute value and label, ")
            f.write("effectively balancing the representation of protected groups in the training data.\n\n")
            f.write("**Implementation:** `app.models.ml.fairness.mitigation.ReweighingMitigation`\n\n")
        elif mitigation_technique == "Adversarial Debiasing":
            f.write("Adversarial Debiasing is an in-processing technique that uses adversarial training to remove information ")
            f.write("about protected attributes from the model's predictions. The method employs an adversary model that attempts ")
            f.write("to predict the protected attribute from the main model's predictions, while the main model is trained to ")
            f.write("minimize its error while maximizing the adversary's error.\n\n")
            f.write("**Implementation:** `app.models.ml.fairness.mitigation.AdversarialDebiasing`\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if mitigated_status:
            f.write("The mitigation technique successfully addressed the fairness issues. Consider the following next steps:\n\n")
            f.write("1. **Monitor Performance in Production:** Continue monitoring fairness metrics when the model is deployed\n")
            f.write("2. **Test on Additional Data:** Validate the mitigation results on different test sets\n")
            f.write("3. **Consider Intersectional Fairness:** Evaluate and address intersectional fairness concerns\n")
        else:
            f.write("The mitigation technique improved fairness but didn't fully resolve all issues. Consider the following next steps:\n\n")
            f.write("1. **Try Combined Techniques:** Apply multiple mitigation techniques in combination\n")
            f.write("2. **Adjust Hyperparameters:** Fine-tune the mitigation method parameters\n")
            f.write("3. **Data Collection:** Collect more balanced training data\n")
            f.write("4. **Feature Engineering:** Develop less biased features\n")

def main():
    print("Loading and preparing data...")
    # Load the data
    data = pd.read_csv('data/sample_ad_data.csv')
    
    # Create binary target by thresholding the ad_score
    threshold = 0.75
    data['high_performing'] = (data['ad_score'] >= threshold).astype(int)
    
    # Make gender distribution more imbalanced for demo purposes
    data['target_gender'] = np.random.choice(['male', 'female', 'all'], size=len(data), p=[0.3, 0.3, 0.4])
    
    # Separate features and target
    X = data.drop(['ad_score', 'high_performing', 'campaign_id', 'ad_text'], axis=1)
    y = data['high_performing']
    
    # Store the protected attribute separately
    protected_attr = 'target_gender'
    protected_attr_values = X[protected_attr].copy()
    
    # Create one-hot encoded features for the model
    X_encoded = pd.get_dummies(X, drop_first=False)
    
    # Create a dictionary mapping each sample to its protected attribute value
    protected_attributes = {
        protected_attr: protected_attr_values.reset_index(drop=True)
    }
    
    # Split the data
    X_train, X_test, y_train, y_test, train_protected, test_protected = train_test_split(
        X_encoded, y, protected_attr_values, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create test protected attributes dictionary
    test_protected_dict = {
        protected_attr: test_protected.reset_index(drop=True)
    }
    
    print("Training the original model...")
    # Create and train the original model (using RandomForestClassifier directly)
    original_model = RandomForestClassifier(n_estimators=100, random_state=42)
    original_model.fit(X_train, y_train)
    
    # Evaluate the original model
    print("Evaluating the original model...")
    original_results = evaluate_model(
        original_model, X_test, y_test, test_protected_dict
    )
    
    # Print original results
    print(f"\nOriginal Model Performance:")
    for metric, value in original_results['performance'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nOriginal Model Fairness:")
    for metric, results in original_results['fairness'].items():
        print(f"  {metric} disparity ratio: {results['disparity_ratio']:.4f}")
    
    # Apply mitigation technique: Reweighing
    print("\nApplying Reweighing mitigation technique...")
    reweighing = ReweighingMitigation(protected_attribute=protected_attr)
    reweighing.fit(X_train, y_train, {protected_attr: train_protected.reset_index(drop=True)})
    X_train_reweighed, sample_weights = reweighing.transform(X_train)
    
    # Train a mitigated model with the reweighted data
    mitigated_model = RandomForestClassifier(n_estimators=100, random_state=42)
    mitigated_model.fit(X_train_reweighed, y_train, sample_weight=sample_weights)
    
    # Evaluate the mitigated model
    print("Evaluating the mitigated model...")
    mitigated_results = evaluate_model(
        mitigated_model, X_test, y_test, test_protected_dict
    )
    
    # Print mitigated results
    print(f"\nMitigated Model Performance:")
    for metric, value in mitigated_results['performance'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nMitigated Model Fairness:")
    for metric, results in mitigated_results['fairness'].items():
        print(f"  {metric} disparity ratio: {results['disparity_ratio']:.4f}")
    
    # Create comparison visualizations
    print("\nGenerating visualizations...")
    plot_fairness_comparison(
        original_results, 
        mitigated_results, 
        'demographic_parity',
        'fairness_evaluation/mitigation/plots/demographic_parity_comparison.png'
    )
    
    plot_fairness_comparison(
        original_results, 
        mitigated_results, 
        'equal_opportunity',
        'fairness_evaluation/mitigation/plots/equal_opportunity_comparison.png'
    )
    
    plot_performance_comparison(
        original_results,
        mitigated_results,
        'fairness_evaluation/mitigation/plots/performance_comparison.png'
    )
    
    # Create comprehensive report
    print("Generating bias mitigation report...")
    create_bias_mitigation_report(
        original_results,
        mitigated_results,
        "Reweighing",
        'fairness_evaluation/mitigation/bias_mitigation_report.md'
    )
    
    print("\nBias mitigation demo completed successfully!")
    print("Results and visualizations available in the 'fairness_evaluation/mitigation/' directory")

if __name__ == "__main__":
    main() 