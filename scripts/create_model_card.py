#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML Model Card Generator (Simple File Version)

This script generates a model card markdown file by directly reading 
the class definition from a file rather than importing it.

Usage:
    python create_model_card.py --file path/to/ad_score_predictor.py --class AdScorePredictor --output docs/models/ad_score_predictor.md
"""

import os
import sys
import re
import ast
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

def get_class_docstring(file_path: str, class_name: str) -> Optional[str]:
    """Extract docstring from a class in a Python file.
    
    Args:
        file_path: Path to the Python file
        class_name: Name of the class
        
    Returns:
        The class docstring if found, None otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        module = ast.parse(source)
        
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                docstring = ast.get_docstring(node)
                return docstring
        
        return None
    
    except Exception as e:
        print(f"Error extracting class docstring: {e}")
        return None

def get_method_docstring(file_path: str, class_name: str, method_name: str) -> Optional[str]:
    """Extract docstring from a method in a class.
    
    Args:
        file_path: Path to the Python file
        class_name: Name of the class
        method_name: Name of the method
        
    Returns:
        The method docstring if found, None otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        module = ast.parse(source)
        
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        docstring = ast.get_docstring(item)
                        return docstring
        
        return None
    
    except Exception as e:
        print(f"Error extracting method docstring: {e}")
        return None

def extract_docstring_sections(docstring: str) -> Dict[str, Any]:
    """Extract different sections from a Google-style docstring.
    
    Args:
        docstring: The docstring to parse
        
    Returns:
        Dictionary with extracted sections
    """
    if not docstring:
        return {}
    
    sections = {
        'description': '',
        'args': {},
        'returns': None,
        'raises': [],
        'examples': []
    }
    
    # Extract description (content before any section)
    lines = docstring.split('\n')
    description = []
    current_section = 'description'
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines at the beginning
        if not line and not description and current_section == 'description':
            continue
        
        # Detect section headers
        if line.lower().startswith('args:') or line.lower().startswith('arguments:'):
            current_section = 'args'
            continue
        elif line.lower().startswith('returns:'):
            current_section = 'returns'
            continue
        elif line.lower().startswith('raises:'):
            current_section = 'raises'
            continue
        elif line.lower().startswith('examples:'):
            current_section = 'examples'
            continue
        elif line.lower().startswith('attributes:'):
            current_section = 'attributes'
            continue
        
        # Process line based on current section
        if current_section == 'description':
            description.append(line)
        elif current_section == 'args' or current_section == 'attributes':
            # Parse parameter
            match = re.match(r'\s*(\w+)(?:\s*\(([^)]+)\))?\s*:\s*(.*)', line)
            if match:
                param_name, param_type, param_desc = match.groups()
                sections['args'][param_name] = {
                    'type': param_type,
                    'description': param_desc
                }
        elif current_section == 'returns':
            if sections['returns'] is None:
                sections['returns'] = line
            else:
                sections['returns'] += ' ' + line
        elif current_section == 'raises':
            sections['raises'].append(line)
        elif current_section == 'examples':
            sections['examples'].append(line)
    
    # Join description lines
    sections['description'] = '\n'.join(description).strip()
    
    return sections

def generate_model_card(file_path: str, class_name: str, output_path: str) -> None:
    """Generate a model card markdown file.
    
    Args:
        file_path: Path to the Python file
        class_name: Name of the class
        output_path: Path to save the model card
    """
    # Extract docstrings
    class_docstring = get_class_docstring(file_path, class_name)
    init_docstring = get_method_docstring(file_path, class_name, '__init__')
    predict_docstring = get_method_docstring(file_path, class_name, 'predict')
    fit_docstring = get_method_docstring(file_path, class_name, 'fit')
    
    # Parse docstrings
    class_sections = extract_docstring_sections(class_docstring)
    init_sections = extract_docstring_sections(init_docstring)
    predict_sections = extract_docstring_sections(predict_docstring)
    fit_sections = extract_docstring_sections(fit_docstring)
    
    # Generate model card content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# Model Card: {class_name}

## Model Overview

- **Model Name**: {class_name}
- **Version**: 1.0.0
- **Type**: Hybrid ML model for ad score prediction
- **Created**: {timestamp}
- **Last Updated**: {timestamp}

## Description

{class_sections.get('description', 'No description available')}

## Intended Use

This model is designed for predicting advertisement performance scores to help optimize ad campaigns
and improve targeting efficiency.

## Model Architecture

The {class_name} implements a hybrid machine learning approach, combining:

- Tree-based models for handling categorical features and capturing non-linear relationships
- Neural network components for complex feature interactions and representation learning

## Model Parameters

### Initialization Parameters

"""
    
    # Add initialization parameters
    if init_sections.get('args'):
        for param_name, param_info in init_sections['args'].items():
            param_type = param_info.get('type', 'Not specified')
            param_desc = param_info.get('description', 'No description')
            content += f"- **{param_name}** ({param_type}): {param_desc}\n"
    else:
        content += "No initialization parameters documented.\n"
    
    # Add prediction information
    content += "\n## Prediction Interface\n\n"
    
    if predict_sections.get('description'):
        content += f"{predict_sections['description']}\n\n"
    
    if predict_sections.get('args'):
        content += "### Input Parameters\n\n"
        for param_name, param_info in predict_sections['args'].items():
            param_type = param_info.get('type', 'Not specified')
            param_desc = param_info.get('description', 'No description')
            content += f"- **{param_name}** ({param_type}): {param_desc}\n"
    
    if predict_sections.get('returns'):
        content += f"\n### Output\n\n{predict_sections['returns']}\n"
    
    # Add training information
    if fit_sections:
        content += "\n## Training Interface\n\n"
        
        if fit_sections.get('description'):
            content += f"{fit_sections['description']}\n\n"
        
        if fit_sections.get('args'):
            content += "### Training Parameters\n\n"
            for param_name, param_info in fit_sections['args'].items():
                param_type = param_info.get('type', 'Not specified')
                param_desc = param_info.get('description', 'No description')
                content += f"- **{param_name}** ({param_type}): {param_desc}\n"
    
    # Add examples if available
    if class_sections.get('examples') or predict_sections.get('examples'):
        content += "\n## Usage Examples\n\n"
        
        examples = class_sections.get('examples', []) or predict_sections.get('examples', [])
        if examples:
            content += "```python\n"
            content += "\n".join(examples)
            content += "\n```\n"
    
    # Add limitations and ethical considerations
    content += """
## Limitations and Ethical Considerations

### Limitations
- The model assumes input data follows the same distribution as the training data
- Performance may degrade with significant data drift
- Not designed to handle real-time streaming data without proper optimization

### Ethical Considerations
- The model should be regularly monitored for bias in predictions
- Does not collect or store personally identifiable information
- Intended for business metrics optimization, not for decisions about individuals
"""

    # Add maintenance information
    content += """
## Maintenance and Governance

- **Owner**: ML Team
- **Review Schedule**: Quarterly
- **Retraining Schedule**: Monthly or upon significant data drift detection
- **Feedback Mechanism**: File issues in the project repository or contact ml-team@example.com
"""

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model card
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Model card successfully saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate a model card markdown file")
    parser.add_argument('--file', type=str, required=True, help="Path to the Python file")
    parser.add_argument('--class', dest='class_name', type=str, required=True, help="Name of the class")
    parser.add_argument('--output', type=str, required=True, help="Path to save the model card")
    
    args = parser.parse_args()
    
    try:
        generate_model_card(args.file, args.class_name, args.output)
    except Exception as e:
        print(f"Error generating model card: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 