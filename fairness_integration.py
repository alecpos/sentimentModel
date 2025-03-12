#!/usr/bin/env python
"""
Fairness Integration Module

This module connects the enhanced fairness metrics and visualizations 
with the existing sentiment analysis pipeline.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_enhanced_fairness_evaluation(test_df: pd.DataFrame, 
                                   predictions_column: str,
                                   label_column: str,
                                   demographic_columns: List[str],
                                   output_dir: str,
                                   model_name: str = "sentiment_model") -> Dict[str, Any]:
    """
    Perform enhanced fairness evaluation using advanced metrics and visualizations.
    
    Args:
        test_df: DataFrame containing test data
        predictions_column: Name of column containing model predictions
        label_column: Name of column containing true labels
        demographic_columns: List of demographic attribute columns
        output_dir: Directory to save outputs
        model_name: Name of the model being evaluated
    
    Returns:
        Dictionary with paths to generated artifacts
    """
    try:
        from enhanced_fairness_metrics import EnhancedFairnessMetrics
        from fairness_visualizations import FairnessVisualizer
    except ImportError:
        logger.error("Enhanced fairness modules not found. Make sure they are in the Python path.")
        return {"status": "error", "message": "Required modules not found"}
    
    logger.info("Starting enhanced fairness evaluation")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Extract data for fairness evaluation
    y_true = test_df[label_column].values
    y_pred = test_df[predictions_column].values
    
    # Check that demographic columns exist
    missing_cols = [col for col in demographic_columns if col not in test_df.columns]
    if missing_cols:
        logger.warning(f"Missing demographic columns: {missing_cols}")
        logger.warning("Adding synthetic demographic data for missing columns")
        test_df = add_synthetic_demographics(test_df, missing_cols)
    
    # Extract protected attributes
    protected_attributes = test_df[demographic_columns].copy()
    
    # Compute enhanced fairness metrics
    metrics_calculator = EnhancedFairnessMetrics()
    fairness_metrics = metrics_calculator.compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        protected_attributes=protected_attributes
    )
    
    # Save metrics to file
    metrics_file = output_path / f"{model_name}_fairness_metrics.json"
    metrics_calculator.save_metrics(fairness_metrics, metrics_file)
    
    # Create fairness visualizations
    visualizations_dir = output_path / "plots"
    visualizations_dir.mkdir(exist_ok=True)
    
    visualizer = FairnessVisualizer(output_dir=str(visualizations_dir))
    visualization_paths = visualizer.visualize_all_metrics(fairness_metrics, prefix=model_name)
    
    # Generate a human-readable fairness report
    report_path = output_path / f"{model_name}_fairness_report.md"
    generate_fairness_report(fairness_metrics, report_path, model_name)
    
    # Return paths to all generated artifacts
    artifacts = {
        "status": "success",
        "metrics_file": str(metrics_file),
        "report_file": str(report_path),
        "visualizations": visualization_paths,
        "summary": fairness_metrics["summary"]
    }
    
    logger.info(f"Enhanced fairness evaluation completed. Results saved to {output_path}")
    return artifacts

def add_synthetic_demographics(df: pd.DataFrame, 
                             missing_columns: List[str]) -> pd.DataFrame:
    """
    Add synthetic demographic data for missing columns.
    
    Args:
        df: DataFrame to add columns to
        missing_columns: List of missing demographic columns
    
    Returns:
        DataFrame with added synthetic demographic data
    """
    np.random.seed(42)  # For reproducibility
    result_df = df.copy()
    n_samples = len(df)
    
    for col in missing_columns:
        if col == 'gender':
            result_df[col] = np.random.randint(0, 2, size=n_samples)
        elif col == 'age_group':
            result_df[col] = np.random.choice(
                ['18-25', '26-35', '36-50', '51+'], size=n_samples
            )
        elif col == 'location':
            result_df[col] = np.random.choice(
                ['urban', 'suburban', 'rural'], size=n_samples
            )
        elif col == 'ethnicity':
            result_df[col] = np.random.choice(
                ['white', 'black', 'hispanic', 'asian', 'other'], size=n_samples
            )
        else:
            # Generic fallback for other demographic columns
            result_df[col] = np.random.choice(
                [f'group_{i}' for i in range(1, 5)], size=n_samples
            )
    
    logger.info(f"Added synthetic demographic data for columns: {missing_columns}")
    return result_df

def generate_fairness_report(fairness_metrics: Dict[str, Any], 
                           output_file: str,
                           model_name: str = "sentiment_model") -> None:
    """
    Generate a human-readable fairness report in Markdown format.
    
    Args:
        fairness_metrics: Dictionary of fairness metrics
        output_file: Path to save the report
        model_name: Name of the model being evaluated
    """
    # Extract summary metrics
    summary = fairness_metrics["summary"]
    univariate_metrics = fairness_metrics["univariate_metrics"]
    intersectional_metrics = fairness_metrics["intersectional_metrics"]
    
    # Start building the report
    report_lines = [
        f"# Fairness Evaluation Report: {model_name}",
        "",
        "## Summary",
        "",
        f"- Total samples: {summary['total_samples']}",
        f"- Average positive prediction rate: {summary['average_positive_rate']:.4f}",
    ]
    
    # Check for problematic groups
    problematic_groups = fairness_metrics.get("problematic_groups", [])
    if problematic_groups:
        report_lines.append(f"- Problematic groups identified: {len(problematic_groups)}")
        fairness_concern_level = "High" if len(problematic_groups) > 2 else "Medium"
    else:
        report_lines.append("- Problematic groups identified: 0")
        fairness_concern_level = "Low"
    
    # Add fairness concern level
    report_lines.append(f"- Fairness concern level: {fairness_concern_level}")
    
    # Add accuracy metrics
    report_lines.extend([
        f"- Overall accuracy: {summary['accuracy']:.4f}"
    ])
    
    # Add section for univariate analysis
    report_lines.extend([
        "",
        "## Univariate Analysis",
        ""
    ])
    
    for attr, metrics in univariate_metrics.items():
        report_lines.extend([
            f"### {attr.title()}",
            "",
            "| Group | Count | Positive Rate | Accuracy |",
            "|-------|-------|--------------|----------|"
        ])
        
        for group_metrics in metrics:
            group = group_metrics["group"]
            count = group_metrics["count"]
            positive_rate = group_metrics["predicted_positive_rate"]
            accuracy = group_metrics["accuracy"]
            
            report_lines.append(
                f"| {group} | {count} | {positive_rate:.4f} | {accuracy:.4f} |"
            )
        
        report_lines.append("")
    
    # Add section for intersectional analysis
    report_lines.extend([
        "## Intersectional Analysis",
        ""
    ])
    
    for intersection_key, metrics in intersectional_metrics.items():
        attr1, attr2 = intersection_key.split('_')
        report_lines.extend([
            f"### {attr1.title()} × {attr2.title()}",
            "",
            f"This analysis examines how the intersection of {attr1} and {attr2} affects prediction outcomes.",
            "",
            "| Group | Size | Positive Rate | Disparate Impact | Accuracy |",
            "|-------|------|---------------|------------------|----------|"
        ])
        
        for group_metrics in metrics:
            group1 = group_metrics[attr1]
            group2 = group_metrics[attr2]
            group_size = group_metrics["group_size"]
            positive_rate = group_metrics["group_positive_rate"]
            disparate_impact = group_metrics.get("disparate_impact", "N/A")
            accuracy = group_metrics["accuracy"]
            
            disparate_impact_str = f"{disparate_impact:.4f}" if isinstance(disparate_impact, (int, float)) else "N/A"
            
            report_lines.append(
                f"| {group1} × {group2} | {group_size} | {positive_rate:.4f} | {disparate_impact_str} | {accuracy:.4f} |"
            )
        
        report_lines.append("")
    
    # Add recommendations section
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    # Check for disparate impact issues
    disparate_impact_issues = []
    for intersection_key, metrics in intersectional_metrics.items():
        attr1, attr2 = intersection_key.split('_')
        for group_metrics in metrics:
            disparate_impact = group_metrics.get("disparate_impact")
            if disparate_impact is not None:
                # Disparate impact below 0.8 or above 1.25 indicates potential issues
                if disparate_impact < 0.8 or disparate_impact > 1.25:
                    group1 = group_metrics[attr1]
                    group2 = group_metrics[attr2]
                    disparate_impact_issues.append({
                        "intersection": f"{attr1}={group1}, {attr2}={group2}",
                        "disparate_impact": disparate_impact
                    })
    
    if disparate_impact_issues:
        report_lines.append("### Disparate Impact Concerns")
        report_lines.append("")
        report_lines.append("The following demographic intersections show concerning disparate impact:")
        report_lines.append("")
        
        for issue in disparate_impact_issues:
            direction = "higher" if issue["disparate_impact"] > 1 else "lower"
            report_lines.append(
                f"- {issue['intersection']}: {direction} positive prediction rate " +
                f"(disparate impact = {issue['disparate_impact']:.2f})"
            )
        
        report_lines.append("")
    
    # General recommendations based on findings
    report_lines.append("### General Recommendations")
    report_lines.append("")
    
    if fairness_concern_level == "Low":
        report_lines.append(
            "The model shows generally balanced predictions across demographic groups. " +
            "Continue monitoring for fairness as the model is deployed and updated."
        )
    elif fairness_concern_level == "Medium":
        report_lines.append(
            "Some demographic groups show concerning prediction patterns. " +
            "Consider implementing fairness mitigation techniques such as:"
        )
        report_lines.append("")
        report_lines.append("1. Reweighting training examples to address representation issues")
        report_lines.append("2. Applying post-processing fairness adjustments with equalized odds constraints")
        report_lines.append("3. Conducting a feature importance analysis to identify potentially problematic features")
    else:  # High
        report_lines.append(
            "Significant fairness concerns have been identified. " +
            "Immediate intervention is recommended:"
        )
        report_lines.append("")
        report_lines.append("1. Investigate and address training data bias")
        report_lines.append("2. Implement robust fairness constraints during model training")
        report_lines.append("3. Consider model redesign with fairness as an explicit objective")
        report_lines.append("4. Apply post-processing fairness adjustments")
    
    # Write the report
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Fairness report generated and saved to {output_file}")


# Example standalone usage
if __name__ == "__main__":
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic test DataFrame
    test_df = pd.DataFrame({
        "text": [f"Sample text {i}" for i in range(n_samples)],
        "target": np.random.randint(0, 2, size=n_samples),
        "prediction": np.random.randint(0, 2, size=n_samples),
        "gender": np.random.randint(0, 2, size=n_samples),
        "age_group": np.random.choice(['18-25', '26-35', '36-50', '51+'], size=n_samples),
        "location": np.random.choice(['urban', 'suburban', 'rural'], size=n_samples)
    })
    
    # Run enhanced fairness evaluation
    results = run_enhanced_fairness_evaluation(
        test_df=test_df,
        predictions_column="prediction",
        label_column="target",
        demographic_columns=["gender", "age_group", "location"],
        output_dir="demo_fairness_results",
        model_name="synthetic_demo"
    )
    
    print(f"Enhanced fairness evaluation completed. Results saved to demo_fairness_results")
    print(f"Fairness metrics file: {results['metrics_file']}")
    print(f"Fairness report file: {results['report_file']}")
    print(f"Number of visualizations created: {len(results['visualizations'])}") 