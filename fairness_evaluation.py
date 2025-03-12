#!/usr/bin/env python
"""
Fairness Evaluation Module for Sentiment Analysis

This module provides functions for evaluating fairness and bias in sentiment analysis models:
1. Calculating fairness metrics across demographic groups and intersections
2. Visualizing bias patterns using heatmaps and other plots
3. Identifying problematic demographic intersections
4. Generating fairness reports with recommendations

Usage:
    from fairness_evaluation import evaluate_fairness, plot_fairness_metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from sklearn.metrics import confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_positive_rates(data, outcome_col, dem1_col, dem2_col):
    """
    Calculate positive rates across intersectional demographic groups.
    
    Parameters:
    - data: pandas DataFrame containing the data
    - outcome_col: column name for the outcome variable (1 = positive, 0 = negative)
    - dem1_col: first demographic variable column name
    - dem2_col: second demographic variable column name
    
    Returns:
    - pivot table with positive rates for each intersection
    """
    # Group by demographic intersections and calculate positive rate
    positive_rates = data.groupby([dem1_col, dem2_col])[outcome_col].mean().reset_index()
    
    # Pivot the data for heatmap visualization
    pivot_data = positive_rates.pivot(index=dem1_col, columns=dem2_col, values=outcome_col)
    
    return pivot_data

def plot_intersectional_heatmap(pivot_data, title, save_path=None, cmap='Blues', annotate=True):
    """
    Plot a heatmap of positive rates across intersectional groups.
    
    Parameters:
    - pivot_data: pivot table with rates for each intersection
    - title: title for the heatmap
    - save_path: path to save the figure (if None, just displays it)
    - cmap: colormap for the heatmap
    - annotate: whether to include annotations on the heatmap
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        pivot_data, 
        annot=annotate, 
        fmt='.3f', 
        cmap=cmap,
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Positive Rate'}
    )
    
    plt.title(f"{title}\nIntersectional Analysis of {pivot_data.index.name} and {pivot_data.columns.name}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to {save_path}")
    
    plt.close()
    
    return ax

def calculate_group_metrics(data, outcome_col, label_col, group_col):
    """
    Calculate classification metrics for different demographic groups.
    
    Parameters:
    - data: pandas DataFrame containing the data
    - outcome_col: column name for the predicted outcome
    - label_col: column name for the true label
    - group_col: column name for the demographic group
    
    Returns:
    - DataFrame with metrics for each group
    """
    groups = data[group_col].unique()
    results = []
    
    for group in groups:
        group_data = data[data[group_col] == group]
        
        # Skip groups with too few samples
        if len(group_data) < 10:
            continue
        
        # Calculate metrics
        true_labels = group_data[label_col]
        predictions = group_data[outcome_col]
        
        # True positive rate (sensitivity/recall for positive class)
        tp = ((predictions == 1) & (true_labels == 1)).sum()
        actual_positives = (true_labels == 1).sum()
        tpr = tp / actual_positives if actual_positives > 0 else 0
        
        # True negative rate (specificity)
        tn = ((predictions == 0) & (true_labels == 0)).sum()
        actual_negatives = (true_labels == 0).sum()
        tnr = tn / actual_negatives if actual_negatives > 0 else 0
        
        # False positive rate
        fp = ((predictions == 1) & (true_labels == 0)).sum()
        fpr = fp / actual_negatives if actual_negatives > 0 else 0
        
        # False negative rate
        fn = ((predictions == 0) & (true_labels == 1)).sum()
        fnr = fn / actual_positives if actual_positives > 0 else 0
        
        # Positive predictive value (precision)
        predicted_positives = (predictions == 1).sum()
        ppv = tp / predicted_positives if predicted_positives > 0 else 0
        
        # Accuracy
        accuracy = (tp + tn) / len(group_data)
        
        # Predicted positive rate
        ppr = predicted_positives / len(group_data)
        
        # Actual positive rate
        apr = actual_positives / len(group_data)
        
        # Store results
        results.append({
            'group': group,
            'count': len(group_data),
            'accuracy': accuracy,
            'true_positive_rate': tpr,
            'true_negative_rate': tnr,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'precision': ppv,
            'predicted_positive_rate': ppr,
            'actual_positive_rate': apr
        })
    
    return pd.DataFrame(results)

def calculate_disparate_impact(data, outcome_col, protected_col, privileged_value):
    """
    Calculate disparate impact ratio between privileged and unprivileged groups.
    Values below 0.8 or above 1.25 typically indicate concerning bias.
    
    Parameters:
    - data: pandas DataFrame
    - outcome_col: binary outcome column (1 = positive)
    - protected_col: protected attribute column
    - privileged_value: value representing the privileged group
    
    Returns:
    - disparate impact ratio
    """
    privileged = data[data[protected_col] == privileged_value]
    unprivileged = data[data[protected_col] != privileged_value]
    
    priv_rate = privileged[outcome_col].mean()
    unpriv_rate = unprivileged[outcome_col].mean()
    
    # Avoid division by zero
    if priv_rate == 0:
        return float('inf')
    
    return unpriv_rate / priv_rate

def calculate_equalized_odds_difference(data, outcome_col, protected_col, privileged_value, actual_col):
    """
    Calculate the maximum difference in true positive and false positive rates.
    Values closer to 0 indicate better fairness.
    
    Parameters:
    - data: pandas DataFrame
    - outcome_col: predicted outcome column (1 = positive)
    - protected_col: protected attribute column
    - privileged_value: value representing the privileged group
    - actual_col: column with actual outcomes
    
    Returns:
    - maximum difference in error rates
    """
    privileged = data[data[protected_col] == privileged_value]
    unprivileged = data[data[protected_col] != privileged_value]
    
    # True positive rates
    priv_positives = privileged[privileged[actual_col] == 1]
    unpriv_positives = unprivileged[unprivileged[actual_col] == 1]
    
    # False positive rates
    priv_negatives = privileged[privileged[actual_col] == 0]
    unpriv_negatives = unprivileged[unprivileged[actual_col] == 0]
    
    # Handle empty groups
    if len(priv_positives) == 0 or len(unpriv_positives) == 0:
        tpr_diff = 1.0  # Maximum difference
    else:
        priv_tpr = priv_positives[outcome_col].mean()
        unpriv_tpr = unpriv_positives[outcome_col].mean()
        tpr_diff = abs(priv_tpr - unpriv_tpr)
    
    if len(priv_negatives) == 0 or len(unpriv_negatives) == 0:
        fpr_diff = 1.0  # Maximum difference
    else:
        priv_fpr = priv_negatives[outcome_col].mean()
        unpriv_fpr = unpriv_negatives[outcome_col].mean()
        fpr_diff = abs(priv_fpr - unpriv_fpr)
    
    return max(tpr_diff, fpr_diff)

def calculate_intersectional_metrics(data, outcome_col, dem1_col, dem2_col, actual_col=None):
    """
    Calculate fairness metrics for all intersectional groups.
    
    Parameters:
    - data: pandas DataFrame
    - outcome_col: predicted outcome column
    - dem1_col: first demographic column
    - dem2_col: second demographic column
    - actual_col: column with actual outcomes (optional)
    
    Returns:
    - DataFrame with fairness metrics for each intersection
    """
    results = []
    
    # Get all unique combinations of demographic variables
    intersections = data.groupby([dem1_col, dem2_col]).size().reset_index()
    
    # Calculate metrics for each intersection vs. rest
    for _, row in intersections.iterrows():
        dem1_val = row[dem1_col]
        dem2_val = row[dem2_col]
        
        # Create binary column for this intersection
        intersection_mask = ((data[dem1_col] == dem1_val) & (data[dem2_col] == dem2_val))
        data_group = data[intersection_mask]
        data_rest = data[~intersection_mask]
        
        # Skip groups with too few samples
        if len(data_group) < 10:
            continue
        
        # Calculate predicted positive rate for this intersection and the rest
        group_ppr = data_group[outcome_col].mean()
        rest_ppr = data_rest[outcome_col].mean()
        
        # Calculate disparate impact
        di = group_ppr / rest_ppr if rest_ppr > 0 else float('inf')
        
        # Default values
        group_tpr = None
        group_fpr = None
        rest_tpr = None
        rest_fpr = None
        tpr_diff = None
        fpr_diff = None
        eod = None
        
        # Calculate equalized odds difference if actual outcomes are available
        if actual_col is not None:
            # Group TPR and FPR
            group_positives = data_group[data_group[actual_col] == 1]
            group_negatives = data_group[data_group[actual_col] == 0]
            
            if len(group_positives) > 0:
                group_tpr = group_positives[outcome_col].mean()
            
            if len(group_negatives) > 0:
                group_fpr = group_negatives[outcome_col].mean()
            
            # Rest TPR and FPR
            rest_positives = data_rest[data_rest[actual_col] == 1]
            rest_negatives = data_rest[data_rest[actual_col] == 0]
            
            if len(rest_positives) > 0:
                rest_tpr = rest_positives[outcome_col].mean()
            
            if len(rest_negatives) > 0:
                rest_fpr = rest_negatives[outcome_col].mean()
            
            # Calculate differences
            if group_tpr is not None and rest_tpr is not None:
                tpr_diff = abs(group_tpr - rest_tpr)
            
            if group_fpr is not None and rest_fpr is not None:
                fpr_diff = abs(group_fpr - rest_fpr)
            
            # Equalized odds difference is the maximum of TPR and FPR differences
            if tpr_diff is not None and fpr_diff is not None:
                eod = max(tpr_diff, fpr_diff)
        
        # Calculate accuracy if actual outcomes are available
        accuracy = None
        if actual_col is not None:
            correct = (data_group[outcome_col] == data_group[actual_col]).sum()
            accuracy = correct / len(data_group)
        
        results.append({
            dem1_col: dem1_val,
            dem2_col: dem2_val,
            'group_size': len(data_group),
            'group_positive_rate': group_ppr,
            'rest_positive_rate': rest_ppr,
            'disparate_impact': di,
            'group_tpr': group_tpr,
            'group_fpr': group_fpr,
            'rest_tpr': rest_tpr,
            'rest_fpr': rest_fpr,
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'equalized_odds_diff': eod,
            'accuracy': accuracy
        })
    
    return pd.DataFrame(results)

def evaluate_fairness(data, predictions_col, demographics_cols, label_col=None, output_dir=None):
    """
    Comprehensive fairness evaluation of model predictions across demographic groups.
    
    Parameters:
    - data: pandas DataFrame with predictions and demographic information
    - predictions_col: column name for model predictions
    - demographics_cols: list of column names for demographic variables
    - label_col: column name for true labels (optional)
    - output_dir: directory to save visualizations and reports
    
    Returns:
    - Dictionary with fairness metrics and detection of problematic groups
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
    
    fairness_results = {
        "univariate_metrics": {},
        "intersectional_metrics": {},
        "problematic_groups": [],
        "summary": {}
    }
    
    # Evaluate univariate demographic variables
    for col in demographics_cols:
        logger.info(f"Evaluating fairness for demographic variable: {col}")
        
        # Calculate metrics for each group
        if label_col:
            group_metrics = calculate_group_metrics(data, predictions_col, label_col, col)
        else:
            # If no ground truth, just compute the predicted positive rate
            group_metrics = data.groupby(col)[predictions_col].mean().reset_index()
            group_metrics.columns = [col, 'predicted_positive_rate']
            group_metrics['count'] = data.groupby(col).size().values
        
        fairness_results["univariate_metrics"][col] = group_metrics.to_dict(orient='records')
        
        # Identify problematic groups (if they exist)
        if 'predicted_positive_rate' in group_metrics.columns:
            mean_ppr = data[predictions_col].mean()
            for _, row in group_metrics.iterrows():
                ppr = row.get('predicted_positive_rate')
                if ppr is not None:
                    # Check if positive rate deviates significantly from average
                    if ppr < mean_ppr * 0.8 or ppr > mean_ppr * 1.25:
                        fairness_results["problematic_groups"].append({
                            "demographic": col,
                            "group": row[col],
                            "positive_rate": ppr,
                            "average_positive_rate": mean_ppr,
                            "ratio": ppr / mean_ppr,
                            "count": row['count']
                        })
        
        # Plot positive rates by group
        if output_dir:
            plt.figure(figsize=(10, 6))
            if 'predicted_positive_rate' in group_metrics.columns and col in group_metrics.columns:
                ax = sns.barplot(x=col, y='predicted_positive_rate', data=group_metrics)
                plt.title(f"Positive Prediction Rate by {col}")
                plt.ylabel("Positive Rate")
                plt.xticks(rotation=45)
                plt.axhline(y=data[predictions_col].mean(), color='r', linestyle='--', label="Overall Mean")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{col}_positive_rates.png"), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                logger.warning(f"Could not create barplot for {col}. Column not found in metrics data.")
    
    # Evaluate pairwise intersections
    for i, col1 in enumerate(demographics_cols):
        for col2 in demographics_cols[i+1:]:
            logger.info(f"Evaluating intersectional fairness for: {col1} × {col2}")
            
            # Calculate positive rates by intersection
            positive_rates = calculate_positive_rates(data, predictions_col, col1, col2)
            
            # Calculate fairness metrics if ground truth is available
            if label_col:
                metrics_df = calculate_intersectional_metrics(
                    data, predictions_col, col1, col2, label_col
                )
                
                # Store intersectional metrics
                fairness_results["intersectional_metrics"][f"{col1}_{col2}"] = metrics_df.to_dict(orient='records')
            
            # Plot heatmap
            if output_dir:
                plot_intersectional_heatmap(
                    positive_rates,
                    f"Positive Prediction Rate by {col1} and {col2}",
                    save_path=os.path.join(plots_dir, f"{col1}_{col2}_heatmap.png")
                )
    
    # Calculate overall fairness summary
    problematic_count = len(fairness_results["problematic_groups"])
    fairness_results["summary"] = {
        "total_samples": len(data),
        "average_positive_rate": float(data[predictions_col].mean()),
        "problematic_groups_count": problematic_count
    }
    
    if problematic_count > 0:
        fairness_results["summary"]["fairness_concern_level"] = "High" if problematic_count > 5 else "Medium"
    else:
        fairness_results["summary"]["fairness_concern_level"] = "Low"
    
    # Add evaluation based on label if available
    if label_col:
        fairness_results["summary"]["accuracy"] = float((data[predictions_col] == data[label_col]).mean())
        
        # Check if there are significant accuracy disparities
        if any(col in fairness_results["univariate_metrics"] for col in demographics_cols):
            accuracy_disparities = []
            
            for col in demographics_cols:
                if col in fairness_results["univariate_metrics"]:
                    group_metrics = pd.DataFrame(fairness_results["univariate_metrics"][col])
                    if 'accuracy' in group_metrics.columns:
                        acc_values = group_metrics['accuracy'].values
                        if len(acc_values) > 1:
                            acc_disparity = max(acc_values) - min(acc_values)
                            accuracy_disparities.append(acc_disparity)
            
            if accuracy_disparities:
                max_acc_disparity = max(accuracy_disparities)
                fairness_results["summary"]["max_accuracy_disparity"] = float(max_acc_disparity)
                
                if max_acc_disparity > 0.1:
                    fairness_results["summary"]["accuracy_disparity_level"] = "High"
                elif max_acc_disparity > 0.05:
                    fairness_results["summary"]["accuracy_disparity_level"] = "Medium"
                else:
                    fairness_results["summary"]["accuracy_disparity_level"] = "Low"
    
    # Save fairness results if output_dir is provided
    if output_dir:
        with open(os.path.join(output_dir, "fairness_metrics.json"), 'w') as f:
            import json
            # Convert numpy types to Python native types for JSON serialization
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            
            json.dump(fairness_results, f, indent=2, cls=NpEncoder)
    
    return fairness_results

def plot_fairness_summary(fairness_results, save_path=None):
    """
    Create a summary visualization of fairness metrics.
    
    Parameters:
    - fairness_results: Dictionary with fairness metrics (output of evaluate_fairness)
    - save_path: Path to save the visualization
    
    Returns:
    - Figure object
    """
    # Extract problematic groups
    problematic_groups = pd.DataFrame(fairness_results["problematic_groups"])
    
    if len(problematic_groups) == 0:
        logger.info("No problematic groups found for visualization")
        return None
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Plot 1: Problematic groups by ratio
    plt.subplot(2, 1, 1)
    problematic_groups['group_label'] = (
        problematic_groups['demographic'] + ': ' + problematic_groups['group'].astype(str)
    )
    problematic_groups = problematic_groups.sort_values('ratio')
    
    colors = ['red' if r < 0.8 else 'orange' if r < 0.9 else 'green' if r > 1.1 else 'blue' 
             for r in problematic_groups['ratio']]
    
    bars = plt.barh(
        problematic_groups['group_label'],
        problematic_groups['ratio'],
        color=colors
    )
    
    # Add a vertical line at ratio=1
    plt.axvline(x=1, color='gray', linestyle='--')
    
    # Add region indicators
    plt.axvline(x=0.8, color='red', linestyle=':', alpha=0.5)
    plt.axvline(x=1.2, color='red', linestyle=':', alpha=0.5)
    
    plt.xlabel('Ratio to Average Positive Rate')
    plt.title('Fairness Concerns by Demographic Group')
    
    # Add counts as annotations
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height()/2,
            f"n={problematic_groups.iloc[i]['count']}",
            va='center'
        )
    
    # Plot 2: Summary of fairness metrics
    if any('univariate_metrics' in fairness_results and col in fairness_results['univariate_metrics'] 
          for col in ['gender', 'age_group', 'location']):
        plt.subplot(2, 1, 2)
        
        # Prepare data for demographic variables
        demo_data = []
        for col in ['gender', 'age_group', 'location']:
            if col in fairness_results.get('univariate_metrics', {}):
                for item in fairness_results['univariate_metrics'][col]:
                    if 'predicted_positive_rate' in item:
                        demo_data.append({
                            'demographic': col,
                            'group': item[col],
                            'positive_rate': item['predicted_positive_rate'],
                            'count': item.get('count', 0)
                        })
        
        if demo_data:
            demo_df = pd.DataFrame(demo_data)
            
            # Create a grouped bar chart
            pivot_df = demo_df.pivot(index='group', columns='demographic', values='positive_rate')
            pivot_df.plot(kind='bar', ax=plt.gca())
            
            plt.title('Positive Prediction Rates Across Demographic Groups')
            plt.xlabel('Group')
            plt.ylabel('Positive Rate')
            plt.legend(title='Demographic')
            plt.axhline(y=fairness_results['summary']['average_positive_rate'], 
                       color='r', linestyle='--', label="Overall Mean")
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Fairness summary plot saved to {save_path}")
    
    return fig

def generate_fairness_report(fairness_results, output_path=None):
    """
    Generate a human-readable fairness report with recommendations.
    
    Parameters:
    - fairness_results: Dictionary with fairness metrics (output of evaluate_fairness)
    - output_path: Path to save the report
    
    Returns:
    - Report text
    """
    report_lines = [
        "# Fairness Evaluation Report",
        "",
        "## Summary",
        f"- Total samples: {fairness_results['summary']['total_samples']}",
        f"- Average positive prediction rate: {fairness_results['summary']['average_positive_rate']:.4f}",
        f"- Problematic groups identified: {fairness_results['summary']['problematic_groups_count']}",
        f"- Fairness concern level: {fairness_results['summary']['fairness_concern_level']}",
        ""
    ]
    
    if 'accuracy' in fairness_results['summary']:
        report_lines.extend([
            f"- Overall accuracy: {fairness_results['summary']['accuracy']:.4f}",
            f"- Maximum accuracy disparity: {fairness_results['summary'].get('max_accuracy_disparity', 'N/A')}"
        ])
        
        if 'accuracy_disparity_level' in fairness_results['summary']:
            report_lines.append(f"- Accuracy disparity level: {fairness_results['summary']['accuracy_disparity_level']}")
        
        report_lines.append("")
    
    # Add problematic groups section if any exist
    if fairness_results['problematic_groups']:
        report_lines.extend([
            "## Problematic Groups",
            "",
            "The following demographic groups show significant disparities in positive prediction rates:",
            ""
        ])
        
        for group in fairness_results['problematic_groups']:
            ratio = group['ratio']
            direction = "lower" if ratio < 1 else "higher"
            magnitude = abs(1 - ratio) * 100
            
            report_lines.append(f"- **{group['demographic']}: {group['group']}**")
            report_lines.append(f"  - Positive rate: {group['positive_rate']:.4f} ({direction} than average by {magnitude:.1f}%)")
            report_lines.append(f"  - Sample size: {group['count']}")
            report_lines.append("")
    
    # Add intersectional analysis results if available
    if fairness_results['intersectional_metrics']:
        report_lines.extend([
            "## Intersectional Analysis",
            "",
            "Analysis of prediction patterns across intersections of demographic variables:",
            ""
        ])
        
        # Identify most imbalanced intersections
        all_intersections = []
        for intersection_key, metrics in fairness_results['intersectional_metrics'].items():
            for item in metrics:
                if 'disparate_impact' in item and not pd.isna(item['disparate_impact']):
                    all_intersections.append({
                        'intersection': f"{item.get(intersection_key.split('_')[0], 'Unknown')}-{item.get(intersection_key.split('_')[1], 'Unknown')}",
                        'disparate_impact': item['disparate_impact'],
                        'group_size': item.get('group_size', 0),
                        'details': item
                    })
        
        # Sort by deviation from 1.0
        all_intersections.sort(key=lambda x: abs(1 - x['disparate_impact']), reverse=True)
        
        # Report on top 5 most imbalanced intersections
        top_intersections = all_intersections[:5]
        for i, intersection in enumerate(top_intersections):
            di = intersection['disparate_impact']
            direction = "lower" if di < 1 else "higher"
            magnitude = abs(1 - di) * 100
            
            report_lines.append(f"- **Intersection {i+1}: {intersection['intersection']}**")
            report_lines.append(f"  - Disparate impact: {di:.2f} ({direction} rate by {magnitude:.1f}%)")
            report_lines.append(f"  - Sample size: {intersection['group_size']}")
            report_lines.append("")
    
    # Add recommendations based on fairness concerns
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    if fairness_results['summary']['fairness_concern_level'] == "Low":
        report_lines.append("The model shows generally balanced predictions across demographic groups. Continue to monitor for fairness as the model is deployed and updated.")
    else:
        # Add specific recommendations based on the identified issues
        if fairness_results['summary']['fairness_concern_level'] == "High":
            report_lines.append("**High fairness concerns detected.** Consider implementing the following mitigation strategies:")
        else:
            report_lines.append("**Medium fairness concerns detected.** Consider the following improvements:")
        
        report_lines.extend([
            "",
            "1. **Data Augmentation:** Increase representation of underrepresented groups in the training data.",
            "2. **Bias Mitigation Techniques:** Implement pre-processing, in-processing, or post-processing bias mitigation techniques:",
            "   - Pre-processing: Reweighing, disparate impact remover",
            "   - In-processing: Adversarial debiasing, prejudice remover",
            "   - Post-processing: Calibrated equal odds, reject option classification",
            "3. **Fairness Constraints:** Incorporate fairness constraints during model training to enforce balanced predictions.",
            "4. **Model Architecture:** Experiment with different model architectures that may be less prone to learning biased patterns.",
            "5. **Feature Selection:** Review features to identify and remove those that may be proxies for protected attributes.",
            "",
            "Implementing a combination of these strategies may be necessary to address the identified disparities effectively."
        ])
    
    # Write report to file if output_path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"Fairness report saved to {output_path}")
    
    return '\n'.join(report_lines)

if __name__ == "__main__":
    # Example usage
    print("This module provides fairness evaluation functions for sentiment analysis models.")
    print("Import this module in your main script to use its functionality.") 