#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Card Generator

This module generates comprehensive model cards with fairness documentation
for regulatory compliance, following best practices from Google, Microsoft,
and other industry leaders.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import jinja2
import markdown
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelCardGenerator:
    """
    Generate comprehensive model cards with fairness documentation.
    
    Model cards provide detailed information about machine learning models, 
    including their intended use, limitations, performance characteristics,
    fairness considerations, and ethical considerations.
    """
    
    def __init__(
        self,
        output_dir: str = 'model_cards',
        template_dir: str = None,
        organization: str = 'WITHIN',
        regulatory_frameworks: List[str] = None
    ):
        """
        Initialize the ModelCardGenerator.
        
        Args:
            output_dir: Directory to save model cards
            template_dir: Directory containing custom templates
            organization: Organization name
            regulatory_frameworks: List of regulatory frameworks to comply with
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        self.organization = organization
        self.regulatory_frameworks = regulatory_frameworks or [
            'EU AI Act', 
            'NIST AI Risk Management Framework',
            'NYC Local Law 144'
        ]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'assets'), exist_ok=True)
        
        # Set up Jinja2 environment
        if template_dir and os.path.exists(template_dir):
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir)
            )
        else:
            # Use default template
            self.jinja_env = jinja2.Environment(
                loader=jinja2.DictLoader({'model_card.md': DEFAULT_TEMPLATE})
            )
    
    def generate_model_card(
        self,
        model_info: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        fairness_metrics: Dict[str, Any] = None,
        evaluation_datasets: Dict[str, Any] = None,
        limitations: List[str] = None,
        ethical_considerations: List[str] = None,
        mitigation_strategies: Dict[str, Any] = None,
        visualizations: Dict[str, str] = None,
        export_formats: List[str] = None
    ) -> str:
        """
        Generate a comprehensive model card.
        
        Args:
            model_info: Information about the model
            performance_metrics: Model performance metrics
            fairness_metrics: Fairness evaluation metrics
            evaluation_datasets: Information about evaluation datasets
            limitations: Model limitations
            ethical_considerations: Ethical considerations
            mitigation_strategies: Fairness mitigation strategies
            visualizations: Paths to visualization assets
            export_formats: Export formats (md, html, pdf)
            
        Returns:
            Path to the generated model card
        """
        # Default export formats
        export_formats = export_formats or ['md', 'html']
        
        # Ensure required fields are present
        required_model_fields = [
            'name', 'version', 'type', 'description', 'use_cases', 
            'developers', 'date_created'
        ]
        
        for field in required_model_fields:
            if field not in model_info:
                logger.warning(f"Required field '{field}' missing from model_info")
                model_info[field] = "Not specified"
        
        # Add timestamp if not present
        if 'date_created' not in model_info or not model_info['date_created']:
            model_info['date_created'] = datetime.now().strftime('%Y-%m-%d')
        
        if 'last_updated' not in model_info or not model_info['last_updated']:
            model_info['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        # Prepare template data
        template_data = {
            'model_info': model_info,
            'performance_metrics': performance_metrics,
            'fairness_metrics': fairness_metrics or {},
            'evaluation_datasets': evaluation_datasets or {},
            'limitations': limitations or [],
            'ethical_considerations': ethical_considerations or [],
            'mitigation_strategies': mitigation_strategies or {},
            'visualizations': visualizations or {},
            'regulatory_frameworks': self.regulatory_frameworks,
            'organization': self.organization,
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'has_fairness_data': bool(fairness_metrics)
        }
        
        # Copy visualizations to assets directory
        if visualizations:
            self._copy_visualizations(visualizations)
        
        # Generate model card using template
        try:
            template = self.jinja_env.get_template('model_card.md')
            model_card_md = template.render(**template_data)
        except jinja2.exceptions.TemplateNotFound:
            logger.warning("Template 'model_card.md' not found, using default template")
            template = jinja2.Template(DEFAULT_TEMPLATE)
            model_card_md = template.render(**template_data)
        
        # Save model card
        model_name = model_info.get('name', 'model').replace(' ', '_').lower()
        model_version = model_info.get('version', 'v1').replace('.', '_')
        
        base_filename = f"{model_name}_{model_version}_model_card"
        md_path = os.path.join(self.output_dir, f"{base_filename}.md")
        
        with open(md_path, 'w') as f:
            f.write(model_card_md)
        
        logger.info(f"Model card saved to {md_path}")
        
        # Generate other formats
        if 'html' in export_formats:
            html_path = os.path.join(self.output_dir, f"{base_filename}.html")
            self._export_html(model_card_md, html_path)
        
        if 'pdf' in export_formats:
            pdf_path = os.path.join(self.output_dir, f"{base_filename}.pdf")
            self._export_pdf(model_card_md, pdf_path)
        
        # Save metadata
        metadata = {
            'model_info': model_info,
            'performance_metrics': performance_metrics,
            'fairness_metrics': fairness_metrics,
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'exported_formats': export_formats
        }
        
        metadata_path = os.path.join(self.output_dir, f"{base_filename}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return md_path
    
    def generate_from_evaluation_results(
        self,
        model_info: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        model_metrics: Dict[str, Any] = None,
        mitigation_info: Dict[str, Any] = None,
        export_formats: List[str] = None
    ) -> str:
        """
        Generate a model card from evaluation results.
        
        Args:
            model_info: Information about the model
            evaluation_results: Results from fairness evaluation
            model_metrics: Additional model metrics
            mitigation_info: Information about mitigation strategies used
            export_formats: Export formats
            
        Returns:
            Path to the generated model card
        """
        # Extract performance metrics
        performance_metrics = evaluation_results.get('overall', {})
        
        # Add additional model metrics if provided
        if model_metrics:
            performance_metrics.update(model_metrics)
        
        # Extract fairness metrics
        fairness_metrics = {
            'metrics': evaluation_results.get('fairness_metrics', {}),
            'group_metrics': evaluation_results.get('group_metrics', {}),
            'threshold': evaluation_results.get('fairness_threshold', 0.2)
        }
        
        # Add intersectional metrics if available
        if 'intersectional' in evaluation_results:
            fairness_metrics['intersectional'] = evaluation_results['intersectional']
        
        # Extract visualizations if available
        visualizations = {}
        
        # Generate model card
        return self.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            fairness_metrics=fairness_metrics,
            mitigation_strategies=mitigation_info,
            export_formats=export_formats
        )
    
    def _copy_visualizations(self, visualizations):
        """
        Copy visualization files to assets directory.
        
        Args:
            visualizations: Dictionary mapping visualization names to file paths
        """
        import shutil
        assets_dir = os.path.join(self.output_dir, 'assets')
        
        for name, path in visualizations.items():
            if os.path.exists(path):
                # Get filename
                filename = os.path.basename(path)
                dest_path = os.path.join(assets_dir, filename)
                
                # Only copy if source and destination are different
                if os.path.abspath(path) != os.path.abspath(dest_path):
                    # Copy file
                    shutil.copy2(path, dest_path)
                
                # Update path in visualizations dictionary
                visualizations[name] = os.path.join('assets', filename)
            else:
                logger.warning(f"Visualization file {path} not found")
    
    def _export_html(self, model_card_md, output_path):
        """
        Export model card to HTML format.
        
        Args:
            model_card_md: Model card in Markdown format
            output_path: Path to save HTML file
        """
        try:
            # Convert markdown to HTML
            html_content = markdown.markdown(
                model_card_md,
                extensions=['tables', 'fenced_code', 'codehilite']
            )
            
            # Add basic styles
            styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Card</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
        }}
        h1, h2, h3, h4 {{
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #2c3e50;
        }}
        h1 {{ font-size: 2.2rem; border-bottom: 2px solid #eaecef; padding-bottom: 0.3rem; }}
        h2 {{ font-size: 1.8rem; border-bottom: 1px solid #eaecef; padding-bottom: 0.3rem; }}
        h3 {{ font-size: 1.5rem; }}
        h4 {{ font-size: 1.3rem; }}
        img {{ max-width: 100%; height: auto; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.5rem;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
            background-color: #f5f5f5;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
        }}
        pre code {{
            display: block;
            padding: 1rem;
            overflow-x: auto;
            line-height: 1.45;
        }}
        blockquote {{
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            color: #6a737d;
            border-left: 0.25rem solid #dfe2e5;
        }}
        .warning {{
            background-color: #fffbea;
            border-left: 0.25rem solid #f0ad4e;
            padding: 1rem;
            margin: 1rem 0;
        }}
        .info {{
            background-color: #e7f5ff;
            border-left: 0.25rem solid #1c7ed6;
            padding: 1rem;
            margin: 1rem 0;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
            
            # Save HTML
            with open(output_path, 'w') as f:
                f.write(styled_html)
            
            logger.info(f"HTML model card saved to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to HTML: {e}")
    
    def _export_pdf(self, model_card_md, output_path):
        """
        Export model card to PDF format.
        
        Args:
            model_card_md: Model card in Markdown format
            output_path: Path to save PDF file
        """
        try:
            import weasyprint
            
            # First convert to HTML
            html_path = output_path.replace('.pdf', '_temp.html')
            self._export_html(model_card_md, html_path)
            
            # Convert HTML to PDF
            weasyprint.HTML(filename=html_path).write_pdf(output_path)
            
            # Remove temporary HTML file
            os.remove(html_path)
            
            logger.info(f"PDF model card saved to {output_path}")
        except ImportError:
            logger.error("Error exporting to PDF: weasyprint not installed. Install with 'pip install weasyprint'")
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")

def generate_model_card_for_ad_score_predictor(
    model,
    evaluation_results,
    model_name="AdScorePredictor",
    model_version="1.0.0",
    output_dir="model_cards",
    fairness_evaluation_results=None,
    mitigation_info=None
):
    """
    Generate a model card for an AdScorePredictor model.
    
    Args:
        model: Trained AdScorePredictor model
        evaluation_results: Evaluation results
        model_name: Name of the model
        model_version: Version of the model
        output_dir: Directory to save model card
        fairness_evaluation_results: Fairness evaluation results
        mitigation_info: Information about mitigation strategies used
        
    Returns:
        Path to the generated model card
    """
    # Create model info
    model_info = {
        'name': model_name,
        'version': model_version,
        'type': 'Regression/Classification',
        'description': 'Predicts ad performance scores for digital advertising campaigns.',
        'use_cases': [
            'Predicting ad effectiveness before campaign launch',
            'Estimating click-through and conversion rates',
            'Optimizing ad content for target audiences'
        ],
        'developers': [
            {'name': 'WITHIN Data Science Team', 'role': 'Model Development'}
        ],
        'date_created': datetime.now().strftime('%Y-%m-%d'),
        'last_updated': datetime.now().strftime('%Y-%m-%d'),
        'license': 'Proprietary',
        'contact': 'data-science@within.co',
        'model_architecture': {
            'framework': 'PyTorch',
            'type': 'Deep Neural Network + Tree Ensemble',
            'hidden_layers': getattr(model, 'hidden_dim', 64),
            'activation': 'ReLU',
            'dropout': getattr(model, 'dropout', 0.2)
        },
        'training_parameters': {
            'learning_rate': getattr(model, 'learning_rate', 0.01),
            'batch_size': getattr(model, 'batch_size', 32),
            'epochs': getattr(model, 'epochs', 100),
            'early_stopping': True
        },
        'fairness_considerations': bool(fairness_evaluation_results or mitigation_info)
    }
    
    # Create additional performance metrics
    model_metrics = {
        'training_time': getattr(model, 'training_time', 'Not recorded'),
        'model_size_mb': 'Not recorded',
        'inference_time_ms': 'Not recorded',
    }
    
    # Create model card generator
    generator = ModelCardGenerator(output_dir=output_dir)
    
    # Generate model card
    if fairness_evaluation_results:
        model_card_path = generator.generate_from_evaluation_results(
            model_info=model_info,
            evaluation_results=fairness_evaluation_results,
            model_metrics=model_metrics,
            mitigation_info=mitigation_info,
            export_formats=['md', 'html']
        )
    else:
        # Use general evaluation results
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=evaluation_results,
            mitigation_strategies=mitigation_info,
            export_formats=['md', 'html']
        )
    
    return model_card_path

# Default template for model cards
DEFAULT_TEMPLATE = """
# Model Card: {{ model_info.name }}

## Model Details

- **Model Name**: {{ model_info.name }}
- **Version**: {{ model_info.version }}
- **Type**: {{ model_info.type }}
- **Date Created**: {{ model_info.date_created }}
- **Last Updated**: {{ model_info.last_updated }}
- **Organization**: {{ organization }}

### Model Description

{{ model_info.description }}

### Intended Use

This model is designed for the following use cases:

{% for use_case in model_info.use_cases %}
- {{ use_case }}
{% endfor %}

### Model Architecture

{% if model_info.model_architecture %}
- **Framework**: {{ model_info.model_architecture.framework }}
- **Type**: {{ model_info.model_architecture.type }}
- **Hidden Layers**: {{ model_info.model_architecture.hidden_layers }}
- **Activation**: {{ model_info.model_architecture.activation }}
- **Dropout**: {{ model_info.model_architecture.dropout }}
{% else %}
Detailed model architecture information is not available.
{% endif %}

### Training Parameters

{% if model_info.training_parameters %}
- **Learning Rate**: {{ model_info.training_parameters.learning_rate }}
- **Batch Size**: {{ model_info.training_parameters.batch_size }}
- **Epochs**: {{ model_info.training_parameters.epochs }}
- **Early Stopping**: {{ "Yes" if model_info.training_parameters.early_stopping else "No" }}
{% else %}
Detailed training parameter information is not available.
{% endif %}

## Performance Metrics

{% if performance_metrics %}
| Metric | Value |
|--------|-------|
{% for metric, value in performance_metrics.items() %}
| {{ metric }} | {{ value }} |
{% endfor %}
{% else %}
Performance metrics information is not available.
{% endif %}

## Fairness Evaluation

{% if has_fairness_data %}
This model has been evaluated for fairness across protected attributes.

### Fairness Metrics

{% if fairness_metrics.metrics %}
| Attribute | Metric | Difference | Threshold | Status |
|-----------|--------|------------|-----------|--------|
{% for metric_key, metric_value in fairness_metrics.metrics.items() %}
{% set parts = metric_key.split('_') %}
{% set attribute = parts[0] %}
{% set metric_name = '_'.join(parts[1:]) %}
| {{ attribute }} | {{ metric_name }} | {{ "%.4f"|format(metric_value.difference) }} | {{ fairness_metrics.threshold }} | {{ "✓" if metric_value.passes_threshold else "✗" }} |
{% endfor %}
{% else %}
Fairness metrics information is not available.
{% endif %}

### Group Performance

{% if fairness_metrics.group_metrics %}
{% for attr_name, groups in fairness_metrics.group_metrics.items() %}
#### {{ attr_name }}

| Group | Count | Accuracy | Positive Rate | True Positive Rate | False Positive Rate |
|-------|-------|----------|--------------|-------------------|---------------------|
{% for group_name, group_metrics in groups.items() %}
| {{ group_name }} | {{ group_metrics.count }} | {{ "%.4f"|format(group_metrics.accuracy) if group_metrics.accuracy is defined else "N/A" }} | {{ "%.4f"|format(group_metrics.positive_rate) if group_metrics.positive_rate is defined else "N/A" }} | {{ "%.4f"|format(group_metrics.true_positive_rate) if group_metrics.true_positive_rate is defined else "N/A" }} | {{ "%.4f"|format(group_metrics.false_positive_rate) if group_metrics.false_positive_rate is defined else "N/A" }} |
{% endfor %}
{% endfor %}
{% else %}
Group performance information is not available.
{% endif %}

{% if fairness_metrics.intersectional %}
### Intersectional Analysis

This model has been evaluated for intersectional fairness, examining how fairness metrics vary across combinations of protected attributes.

{% if fairness_metrics.intersectional.fairness_metrics %}
#### Intersectional Fairness Metrics

| Intersection | Metric | Difference | Threshold | Status |
|--------------|--------|------------|-----------|--------|
{% for metric_key, metric_value in fairness_metrics.intersectional.fairness_metrics.items() %}
{% set parts = metric_key.split('_') %}
{% set intersection = parts[0] %}
{% set metric_name = '_'.join(parts[1:]) %}
| {{ intersection }} | {{ metric_name }} | {{ "%.4f"|format(metric_value.difference) }} | {{ fairness_metrics.threshold }} | {{ "✓" if metric_value.passes_threshold else "✗" }} |
{% endfor %}
{% endif %}
{% endif %}
{% else %}
This model has not been evaluated for fairness considerations.
{% endif %}

## Fairness Mitigations

{% if mitigation_strategies %}
The following mitigation strategies have been implemented to address potential fairness concerns:

{% for name, strategy in mitigation_strategies.items() %}
### {{ name }}

{{ strategy.description }}

**Implementation**: {{ strategy.implementation }}

**Parameters**: 
{% for param_name, param_value in strategy.parameters.items() %}
- {{ param_name }}: {{ param_value }}
{% endfor %}

**Effectiveness**: {{ strategy.effectiveness }}

{% endfor %}
{% else %}
{% if has_fairness_data %}
No specific fairness mitigation strategies have been implemented for this model.
{% else %}
Fairness mitigation information is not available for this model.
{% endif %}
{% endif %}

## Ethical Considerations

{% if ethical_considerations %}
{% for consideration in ethical_considerations %}
- {{ consideration }}
{% endfor %}
{% else %}
No specific ethical considerations have been documented for this model.
{% endif %}

## Limitations and Biases

{% if limitations %}
{% for limitation in limitations %}
- {{ limitation }}
{% endfor %}
{% else %}
No specific limitations or biases have been documented for this model.
{% endif %}

## Regulatory Compliance

This model card is designed to provide information relevant to the following regulatory frameworks:

{% for framework in regulatory_frameworks %}
- {{ framework }}
{% endfor %}

## Contact Information

- **Organization**: {{ organization }}
{% if model_info.contact %}
- **Contact**: {{ model_info.contact }}
{% endif %}
{% if model_info.license %}
- **License**: {{ model_info.license }}
{% endif %}

---

*This model card was generated on {{ generation_date }}.*
"""

def main():
    """
    Run a demonstration of the model card generator.
    """
    # Create sample data
    model_info = {
        'name': 'Ad Score Predictor',
        'version': '1.0.0',
        'type': 'Regression/Classification',
        'description': 'Predicts ad performance scores for digital advertising campaigns.',
        'use_cases': [
            'Predicting ad effectiveness before campaign launch',
            'Estimating click-through and conversion rates',
            'Optimizing ad content for target audiences'
        ],
        'developers': [
            {'name': 'WITHIN Data Science Team', 'role': 'Model Development'}
        ],
        'date_created': '2023-10-15',
        'last_updated': '2023-11-01',
        'license': 'Proprietary',
        'contact': 'data-science@within.co',
        'model_architecture': {
            'framework': 'PyTorch',
            'type': 'Deep Neural Network + Tree Ensemble',
            'hidden_layers': 64,
            'activation': 'ReLU',
            'dropout': 0.2
        },
        'training_parameters': {
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping': True
        }
    }
    
    performance_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85,
        'mse': 0.042,
        'rmse': 0.205,
        'mae': 0.156,
        'r2': 0.76
    }
    
    fairness_metrics = {
        'metrics': {
            'gender_demographic_parity': {
                'difference': 0.01,
                'passes_threshold': True
            },
            'gender_equal_opportunity': {
                'difference': 0.02,
                'passes_threshold': True
            },
            'location_demographic_parity': {
                'difference': 0.04,
                'passes_threshold': True
            },
            'age_group_demographic_parity': {
                'difference': 0.08,
                'passes_threshold': False
            }
        },
        'group_metrics': {
            'gender': {
                'male': {
                    'count': 600,
                    'accuracy': 0.86,
                    'positive_rate': 0.75,
                    'true_positive_rate': 0.88,
                    'false_positive_rate': 0.16
                },
                'female': {
                    'count': 400,
                    'accuracy': 0.83,
                    'positive_rate': 0.74,
                    'true_positive_rate': 0.86,
                    'false_positive_rate': 0.15
                }
            },
            'location': {
                'urban': {
                    'count': 500,
                    'accuracy': 0.85,
                    'positive_rate': 0.76,
                    'true_positive_rate': 0.89,
                    'false_positive_rate': 0.16
                },
                'suburban': {
                    'count': 300,
                    'accuracy': 0.84,
                    'positive_rate': 0.75,
                    'true_positive_rate': 0.87,
                    'false_positive_rate': 0.17
                },
                'rural': {
                    'count': 200,
                    'accuracy': 0.83,
                    'positive_rate': 0.72,
                    'true_positive_rate': 0.85,
                    'false_positive_rate': 0.15
                }
            }
        },
        'threshold': 0.05
    }
    
    mitigation_strategies = {
        'Reweighing': {
            'description': 'Assigns different weights to training examples to ensure fairness across protected groups.',
            'implementation': 'ReweighingMitigation in app/models/ml/fairness/mitigation.py',
            'parameters': {
                'protected_attribute': 'gender',
                'reweighing_factor': 1.0
            },
            'effectiveness': 'Reduced demographic parity difference from 0.08 to 0.01'
        },
        'Fairness Constraints': {
            'description': 'Adds fairness constraints to the model training process to enforce fairness criteria.',
            'implementation': 'FairnessConstraint in app/models/ml/fairness/mitigation.py',
            'parameters': {
                'constraint_type': 'demographic_parity',
                'protected_attribute': 'age_group',
                'epsilon': 0.05
            },
            'effectiveness': 'Reduced demographic parity difference from 0.15 to 0.08'
        }
    }
    
    limitations = [
        'The model was trained primarily on data from North American advertising campaigns and may not generalize well to other regions.',
        'Performance may vary across different demographic groups, with lower accuracy for underrepresented groups.',
        'The model does not account for changes in consumer behavior over time and may require regular retraining.',
        'Limited evaluation on intersectional fairness (combinations of protected attributes).'
    ]
    
    ethical_considerations = [
        'This model may influence advertising strategy decisions that impact users from various demographic groups.',
        'Care should be taken to regularly audit for unintended biases in predictions across different groups.',
        'The model should not be used as the sole decision-maker for high-stakes advertising budget allocations without human oversight.',
        'Users of this model should be aware of and mitigate potential disparate impact across demographic groups.'
    ]
    
    # Create model card generator
    generator = ModelCardGenerator()
    
    # Generate model card
    model_card_path = generator.generate_model_card(
        model_info=model_info,
        performance_metrics=performance_metrics,
        fairness_metrics=fairness_metrics,
        limitations=limitations,
        ethical_considerations=ethical_considerations,
        mitigation_strategies=mitigation_strategies,
        export_formats=['md', 'html']
    )
    
    print(f"Model card generated at {model_card_path}")
    print("Key sections in the model card:")
    print("- Model Details")
    print("- Performance Metrics")
    print("- Fairness Evaluation")
    print("- Fairness Mitigations")
    print("- Ethical Considerations")
    print("- Limitations and Biases")
    print("- Regulatory Compliance")

if __name__ == "__main__":
    main() 