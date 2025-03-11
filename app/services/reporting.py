"""
Reporting services for ML monitoring.

This module provides tools for generating reports on drift detection,
model performance, and other monitoring metrics.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_drift_report(
    model_id: str,
    drift_results: Dict[str, Any],
    include_features: bool = True,
    include_metrics: bool = True,
    report_format: str = 'json'
) -> Dict[str, Any]:
    """
    Generate a report for detected drift.
    
    Args:
        model_id: ID of the model
        drift_results: Results from drift detection
        include_features: Whether to include feature-level details
        include_metrics: Whether to include drift metrics
        report_format: Format of the report ('json', 'html', 'markdown')
        
    Returns:
        Dictionary containing the generated report
    """
    report = {
        'model_id': model_id,
        'generated_at': datetime.now().isoformat(),
        'drift_detected': drift_results.get('drift_detected', False),
        'drift_types': drift_results.get('drift_types', []),
        'summary': _generate_drift_summary(drift_results)
    }
    
    # Add feature details if requested
    if include_features:
        report['feature_details'] = _extract_feature_details(drift_results)
        
    # Add metrics if requested
    if include_metrics:
        report['metrics'] = _extract_drift_metrics(drift_results)
        
    # Add recommendations
    report['recommendations'] = _generate_recommendations(drift_results)
    
    # Format report if needed
    if report_format == 'html':
        report['html'] = _convert_to_html(report)
    elif report_format == 'markdown':
        report['markdown'] = _convert_to_markdown(report)
        
    logger.info(f"Generated drift report for model {model_id}")
    return report

def _generate_drift_summary(drift_results: Dict[str, Any]) -> str:
    """Generate a summary of drift detection results."""
    if not drift_results.get('drift_detected', False):
        return "No significant drift detected."
        
    drift_types = drift_results.get('drift_types', [])
    if 'data_drift' in drift_types:
        return "Data drift detected. Distribution of input features has changed significantly."
    elif 'concept_drift' in drift_types:
        return "Concept drift detected. Relationship between features and target has changed."
    elif 'prediction_drift' in drift_types:
        return "Prediction drift detected. Distribution of model outputs has changed."
    else:
        return "Drift detected in model behavior."

def _extract_feature_details(drift_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract feature-level drift details."""
    details = {}
    
    # Extract drifted features
    if 'drifted_features' in drift_results:
        details['drifted_features'] = drift_results['drifted_features']
        
    # Extract drift scores by feature
    if 'drift_scores' in drift_results:
        details['drift_scores'] = drift_results['drift_scores']
        
    # Extract feature importance in drift
    if 'feature_importance' in drift_results:
        details['feature_importance'] = drift_results['feature_importance']
        
    return details

def _extract_drift_metrics(drift_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract drift metrics."""
    metrics = {}
    
    # Extract overall drift score
    if 'drift_score' in drift_results:
        metrics['overall_drift_score'] = drift_results['drift_score']
        
    # Extract multivariate drift score
    if 'multivariate_drift_score' in drift_results:
        metrics['multivariate_drift_score'] = drift_results['multivariate_drift_score']
        
    # Extract threshold
    if 'drift_threshold' in drift_results:
        metrics['drift_threshold'] = drift_results['drift_threshold']
        
    return metrics

def _generate_recommendations(drift_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on drift results."""
    recommendations = []
    
    if not drift_results.get('drift_detected', False):
        recommendations.append("Continue monitoring model performance.")
        return recommendations
        
    drift_types = drift_results.get('drift_types', [])
    
    if 'data_drift' in drift_types:
        recommendations.append("Investigate source of data drift and consider retraining model with recent data.")
        
    if 'concept_drift' in drift_types:
        recommendations.append("Model retraining recommended to address changes in the relationship between features and target.")
        
    if 'prediction_drift' in drift_types:
        recommendations.append("Validate model outputs against ground truth to assess impact on business metrics.")
        
    # Add feature-specific recommendations
    if 'drifted_features' in drift_results and drift_results['drifted_features']:
        recommendations.append(f"Focus analysis on drifted features: {', '.join(drift_results['drifted_features'][:5])}.")
        
    return recommendations

def _convert_to_html(report: Dict[str, Any]) -> str:
    """Convert report to HTML format."""
    # Simple HTML conversion for testing purposes
    html = "<html><body>"
    html += f"<h1>Drift Report for Model {report['model_id']}</h1>"
    html += f"<p><strong>Generated:</strong> {report['generated_at']}</p>"
    html += f"<p><strong>Drift Detected:</strong> {report['drift_detected']}</p>"
    html += f"<p><strong>Summary:</strong> {report['summary']}</p>"
    html += "</body></html>"
    return html

def _convert_to_markdown(report: Dict[str, Any]) -> str:
    """Convert report to Markdown format."""
    # Simple Markdown conversion for testing purposes
    markdown = f"# Drift Report for Model {report['model_id']}\n\n"
    markdown += f"**Generated:** {report['generated_at']}\n\n"
    markdown += f"**Drift Detected:** {report['drift_detected']}\n\n"
    markdown += f"**Summary:** {report['summary']}\n\n"
    return markdown 