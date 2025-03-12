#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML Model Documentation Generator

This script generates comprehensive documentation for ML models following
best practices in model documentation, transparency, and explainability.

Usage:
    python document_ml_model.py --model_path path/to/model.pkl --output path/to/output.md
    python document_ml_model.py --module app.models.ml.prediction.ad_score_predictor --class AdScorePredictor --output docs/models/ad_score_predictor.md
"""

import os
import sys
import argparse
import importlib
import inspect
import json
import logging
import numpy as np
import pandas as pd
import re
import textwrap
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_model_documenter')

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available. Some model loading features may be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch model documentation features will be disabled.")

try:
    import sklearn
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. scikit-learn model documentation features will be disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not available. SHAP explainability features will be disabled.")


class MLModelDocumenter:
    """Generates comprehensive documentation for ML models."""
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 model_object: Optional[Any] = None,
                 model_metadata: Optional[Dict[str, Any]] = None):
        """Initialize the ML model documenter.
        
        Args:
            model_path: Path to the model file
            model_object: Pre-loaded model object
            model_metadata: Additional metadata about the model
        """
        self.model_path = model_path
        self.model_object = model_object
        self.model_metadata = model_metadata or {}
        self.model = None
        self._feature_importances = None
        self._model_type = None
        self._model_parameters = None
        
        if model_path and not model_object:
            self._load_model()
        elif model_object:
            self.model = model_object
            self._analyze_model()
        
    def _load_model(self) -> None:
        """Load the model from the specified path."""
        if not self.model_path:
            raise ValueError("Model path not specified")
        
        try:
            # Try to determine the model format from file extension
            path = Path(self.model_path)
            extension = path.suffix.lower()
            
            if extension == '.pkl' or extension == '.pickle':
                self._load_pickle_model()
            elif extension == '.joblib':
                self._load_joblib_model()
            elif extension == '.pt' or extension == '.pth':
                self._load_torch_model()
            elif extension == '.json':
                self._load_json_model()
            elif extension == '.yaml' or extension == '.yml':
                self._load_yaml_model()
            else:
                logger.warning(f"Unknown model format: {extension}. Trying pickle as default.")
                self._load_pickle_model()
                
            logger.info(f"Successfully loaded model from {self.model_path}")
            self._analyze_model()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_pickle_model(self) -> None:
        """Load a model from a pickle file."""
        import pickle
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def _load_joblib_model(self) -> None:
        """Load a model from a joblib file."""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required to load .joblib files")
        
        self.model = joblib.load(self.model_path)
    
    def _load_torch_model(self) -> None:
        """Load a PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load .pt/.pth files")
        
        self.model = torch.load(self.model_path)
    
    def _load_json_model(self) -> None:
        """Load a model from a JSON file."""
        with open(self.model_path, 'r') as f:
            self.model = json.load(f)
    
    def _load_yaml_model(self) -> None:
        """Load a model from a YAML file."""
        with open(self.model_path, 'r') as f:
            self.model = yaml.safe_load(f)
    
    def _analyze_model(self) -> None:
        """Analyze the model to extract relevant information."""
        self._determine_model_type()
        self._extract_model_parameters()
        self._extract_feature_importances()
    
    def _determine_model_type(self) -> None:
        """Determine the type and framework of the model."""
        if self.model is None:
            self._model_type = "Unknown"
            return
        
        model_class = self.model.__class__.__name__
        module_name = self.model.__class__.__module__
        
        if SKLEARN_AVAILABLE and isinstance(self.model, BaseEstimator):
            if 'ensemble' in module_name:
                self._model_type = f"scikit-learn Ensemble ({model_class})"
            elif 'linear_model' in module_name:
                self._model_type = f"scikit-learn Linear Model ({model_class})"
            elif 'tree' in module_name:
                self._model_type = f"scikit-learn Tree-based Model ({model_class})"
            elif 'svm' in module_name:
                self._model_type = f"scikit-learn SVM ({model_class})"
            elif 'neural_network' in module_name:
                self._model_type = f"scikit-learn Neural Network ({model_class})"
            else:
                self._model_type = f"scikit-learn Model ({model_class})"
        elif TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
            self._model_type = f"PyTorch Neural Network ({model_class})"
        elif hasattr(self.model, 'keras_model'):
            self._model_type = f"Keras/TensorFlow Model ({model_class})"
        elif hasattr(self.model, 'predict') and callable(self.model.predict):
            self._model_type = f"Custom ML Model with predict() ({model_class})"
        elif hasattr(self.model, 'forward') and callable(self.model.forward):
            self._model_type = f"Custom Neural Network with forward() ({model_class})"
        else:
            self._model_type = f"Unknown Model Type ({model_class})"
    
    def _extract_model_parameters(self) -> None:
        """Extract parameters and hyperparameters from the model."""
        if self.model is None:
            self._model_parameters = {}
            return
        
        params = {}
        
        # For scikit-learn models
        if SKLEARN_AVAILABLE and isinstance(self.model, BaseEstimator):
            # Get hyperparameters
            params['hyperparameters'] = self.model.get_params()
            
            # Get fitted parameters if available
            fitted_attrs = {}
            for attr in dir(self.model):
                if attr.endswith('_') and not attr.startswith('__'):
                    try:
                        value = getattr(self.model, attr)
                        if isinstance(value, (int, float, str, bool, list, dict, tuple)):
                            fitted_attrs[attr] = value
                        elif isinstance(value, np.ndarray):
                            if value.size < 1000:  # Don't include large arrays
                                fitted_attrs[attr] = value.shape
                    except:
                        pass
            
            params['fitted_attributes'] = fitted_attrs
            
        # For PyTorch models
        elif TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            params['total_parameters'] = total_params
            params['trainable_parameters'] = trainable_params
            
            # Get model architecture
            architecture = []
            for name, module in self.model.named_children():
                architecture.append(f"{name}: {module.__class__.__name__}")
            
            params['architecture'] = architecture
        
        # For any model with custom attributes
        custom_attrs = {}
        for attr in dir(self.model):
            if not attr.startswith('_') and not callable(getattr(self.model, attr)):
                try:
                    value = getattr(self.model, attr)
                    if isinstance(value, (int, float, str, bool)):
                        custom_attrs[attr] = value
                except:
                    pass
        
        if custom_attrs:
            params['attributes'] = custom_attrs
            
        self._model_parameters = params
    
    def _extract_feature_importances(self) -> None:
        """Extract feature importances if available in the model."""
        if self.model is None:
            return
        
        importances = {}
        
        # scikit-learn feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances['feature_importances'] = self.model.feature_importances_
            
            # If feature names are stored in the model
            if hasattr(self.model, 'feature_names_in_'):
                importances['feature_names'] = self.model.feature_names_in_
        
        # Linear models coefficients
        if hasattr(self.model, 'coef_'):
            importances['coefficients'] = self.model.coef_
            
            # If intercept is available
            if hasattr(self.model, 'intercept_'):
                importances['intercept'] = self.model.intercept_
        
        # Tree models feature importance
        if hasattr(self.model, 'tree_') and hasattr(self.model.tree_, 'feature_importance'):
            importances['tree_feature_importance'] = self.model.tree_.feature_importance
        
        self._feature_importances = importances
    
    def generate_model_card(self, 
                            output_path: str, 
                            include_code_examples: bool = True,
                            include_performance_metrics: bool = True,
                            include_limitations: bool = True) -> None:
        """Generate a comprehensive model card in Markdown format.
        
        Args:
            output_path: Path to save the model card
            include_code_examples: Whether to include code examples
            include_performance_metrics: Whether to include performance metrics
            include_limitations: Whether to include model limitations
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start building the markdown content
        content = f"""# Model Card: {self.model_metadata.get('name', 'Unnamed Model')}

## Model Overview

- **Model Name**: {self.model_metadata.get('name', 'Not specified')}
- **Version**: {self.model_metadata.get('version', '1.0.0')}
- **Type**: {self._model_type}
- **Created**: {self.model_metadata.get('created_at', 'Not specified')}
- **Last Updated**: {timestamp}
- **Authors**: {self.model_metadata.get('authors', 'Not specified')}
- **License**: {self.model_metadata.get('license', 'Not specified')}
- **Repository**: {self.model_metadata.get('repository', 'Not specified')}

## Intended Use

{self.model_metadata.get('intended_use', 'This model is designed for general-purpose use. Specific usage guidelines should be documented here.')}

## Model Architecture

"""
        
        # Model architecture details
        if 'architecture' in self._model_parameters:
            content += "```\n"
            for layer in self._model_parameters['architecture']:
                content += f"{layer}\n"
            content += "```\n\n"
        else:
            content += f"{self.model_metadata.get('architecture_description', 'Not available')}\n\n"
        
        # Model parameters
        content += "## Model Parameters\n\n"
        
        if 'total_parameters' in self._model_parameters:
            content += f"- **Total Parameters**: {self._model_parameters['total_parameters']:,}\n"
            content += f"- **Trainable Parameters**: {self._model_parameters['trainable_parameters']:,}\n\n"
        
        if 'hyperparameters' in self._model_parameters:
            content += "### Hyperparameters\n\n"
            content += "```\n"
            for param, value in self._model_parameters['hyperparameters'].items():
                content += f"{param}: {value}\n"
            content += "```\n\n"
        
        # Feature importance
        if self._feature_importances:
            content += "## Feature Importance\n\n"
            
            if 'feature_importances' in self._feature_importances and 'feature_names' in self._feature_importances:
                importances = self._feature_importances['feature_importances']
                names = self._feature_importances['feature_names']
                
                content += "| Feature | Importance |\n"
                content += "|---------|------------|\n"
                
                # Sort by importance
                sorted_idx = np.argsort(importances)[::-1]
                
                for idx in sorted_idx[:20]:  # Show top 20 features
                    content += f"| {names[idx]} | {importances[idx]:.4f} |\n"
                
                if len(names) > 20:
                    content += f"| ... | ... |\n"
                
                content += "\n"
            elif 'feature_importances' in self._feature_importances:
                importances = self._feature_importances['feature_importances']
                
                content += "| Feature Index | Importance |\n"
                content += "|--------------|------------|\n"
                
                # Sort by importance
                sorted_idx = np.argsort(importances)[::-1]
                
                for i, idx in enumerate(sorted_idx[:20]):  # Show top 20 features
                    content += f"| Feature_{idx} | {importances[idx]:.4f} |\n"
                
                if len(importances) > 20:
                    content += f"| ... | ... |\n"
                
                content += "\n"
            elif 'coefficients' in self._feature_importances:
                coef = self._feature_importances['coefficients']
                
                if len(coef.shape) == 1:  # Binary classification or regression
                    content += "| Feature Index | Coefficient |\n"
                    content += "|--------------|-------------|\n"
                    
                    # Sort by absolute value
                    sorted_idx = np.argsort(np.abs(coef))[::-1]
                    
                    for i, idx in enumerate(sorted_idx[:20]):  # Show top 20 features
                        content += f"| Feature_{idx} | {coef[idx]:.4f} |\n"
                    
                    if len(coef) > 20:
                        content += f"| ... | ... |\n"
                else:  # Multiclass
                    content += "Coefficient matrix shape: {}\n\n".format(coef.shape)
                
                if 'intercept' in self._feature_importances:
                    content += f"\nIntercept: {self._feature_importances['intercept']}\n\n"
        
        # Training data
        content += "## Training Data\n\n"
        if 'training_data' in self.model_metadata:
            content += self.model_metadata['training_data']
        else:
            content += """
The model was trained on [DATASET NAME]. Details include:
- **Size**: [SIZE]
- **Features**: [FEATURE COUNT]
- **Time Period**: [TIME PERIOD]
- **Data Source**: [SOURCE]
- **Data Preparation**: [DATA PREPARATION STEPS]
"""
        
        # Performance metrics
        if include_performance_metrics:
            content += "\n## Performance Metrics\n\n"
            
            if 'performance_metrics' in self.model_metadata:
                metrics = self.model_metadata['performance_metrics']
                
                if isinstance(metrics, dict):
                    content += "| Metric | Value |\n"
                    content += "|--------|-------|\n"
                    
                    for metric, value in metrics.items():
                        content += f"| {metric} | {value} |\n"
                else:
                    content += metrics
            else:
                content += """
Performance metrics on test data:
- **Accuracy**: [ACCURACY]
- **Precision**: [PRECISION]
- **Recall**: [RECALL]
- **F1 Score**: [F1]
- **AUC-ROC**: [AUC]

Cross-validation results:
- **CV Score (mean)**: [CV_MEAN]
- **CV Score (std)**: [CV_STD]
"""
        
        # Limitations and ethical considerations
        if include_limitations:
            content += "\n## Limitations and Ethical Considerations\n\n"
            
            if 'limitations' in self.model_metadata:
                content += self.model_metadata['limitations']
            else:
                content += """
### Limitations
- [KNOWN LIMITATIONS]
- [EDGE CASES]
- [PERFORMANCE VARIATIONS]

### Ethical Considerations
- [POTENTIAL BIASES]
- [FAIRNESS CONSIDERATIONS]
- [PRIVACY IMPLICATIONS]
"""
        
        # Code examples
        if include_code_examples:
            content += "\n## Usage Examples\n\n"
            
            if 'code_examples' in self.model_metadata:
                content += self.model_metadata['code_examples']
            else:
                model_class = self.model.__class__.__name__
                content += f"""
### Loading the Model

```python
import joblib

# Load the {model_class} model
model = joblib.load('path/to/model.pkl')
```

### Making Predictions

```python
# Prepare your input features
import pandas as pd
data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # Add other features
})

# Make a prediction
prediction = model.predict(data)
print(f"Prediction: {prediction}")

# For probability estimates (if applicable)
try:
    proba = model.predict_proba(data)
    print(f"Probability: {proba}")
except:
    print("Probability estimation not available for this model")
```
"""
        
        # Maintenance and governance
        content += "\n## Maintenance and Governance\n\n"
        
        if 'maintenance' in self.model_metadata:
            content += self.model_metadata['maintenance']
        else:
            content += """
- **Owner**: [TEAM/INDIVIDUAL RESPONSIBLE]
- **Review Schedule**: [REVIEW FREQUENCY]
- **Retraining Schedule**: [RETRAINING FREQUENCY]
- **Feedback Mechanism**: [HOW TO PROVIDE FEEDBACK]
"""
        
        # Citation
        content += "\n## Citation\n\n"
        
        if 'citation' in self.model_metadata:
            content += self.model_metadata['citation']
        else:
            authors = self.model_metadata.get('authors', 'Author')
            name = self.model_metadata.get('name', 'Model')
            year = datetime.now().year
            
            content += f"""
If you use this model in your work, please cite:

```
{authors} ({year}). {name}. [Organization/Repository URL]
```
"""
        
        # Save the model card
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Model card successfully saved to {output_path}")
        
    @classmethod
    def from_class(cls, 
                   module_path: str, 
                   class_name: str, 
                   init_params: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> 'MLModelDocumenter':
        """Create a documenter from a Python class.
        
        Args:
            module_path: Import path to the module (e.g., 'app.models.ml.prediction.ad_score_predictor')
            class_name: Name of the class to document
            init_params: Parameters to pass to the class constructor
            metadata: Additional metadata about the model
            
        Returns:
            An instance of MLModelDocumenter
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the class
            model_class = getattr(module, class_name)
            
            # Create an instance of the class
            init_params = init_params or {}
            model_instance = model_class(**init_params)
            
            # Create the documenter
            return cls(model_object=model_instance, model_metadata=metadata)
            
        except ImportError:
            logger.error(f"Failed to import module: {module_path}")
            raise
        except AttributeError:
            logger.error(f"Class {class_name} not found in module {module_path}")
            raise
        except Exception as e:
            logger.error(f"Error creating model instance: {e}")
            raise


def extract_docstring_metadata(module_path: str, class_name: str) -> Dict[str, Any]:
    """Extract metadata from class and method docstrings.
    
    Args:
        module_path: Import path to the module
        class_name: Name of the class
        
    Returns:
        Dictionary of metadata extracted from docstrings
    """
    metadata = {}
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class
        model_class = getattr(module, class_name)
        
        # Extract class docstring
        class_doc = inspect.getdoc(model_class)
        if class_doc:
            # First line as name
            lines = class_doc.split('\n')
            if lines:
                metadata['name'] = lines[0].strip()
            
            # Extract intended use
            intended_use = []
            capture = False
            for line in lines:
                if line.strip().lower() in ('intended use:', 'purpose:'):
                    capture = True
                    continue
                
                if capture and line.strip() and not any(line.strip().lower().startswith(s) for s in ('attributes:', 'parameters:', 'returns:', 'raises:')):
                    intended_use.append(line.strip())
                elif capture and any(line.strip().lower().startswith(s) for s in ('attributes:', 'parameters:', 'returns:', 'raises:')):
                    capture = False
            
            if intended_use:
                metadata['intended_use'] = '\n'.join(intended_use)
        
        # Extract metadata from __init__ docstring
        init_method = getattr(model_class, '__init__', None)
        if init_method:
            init_doc = inspect.getdoc(init_method)
            if init_doc:
                # Extract parameters from docstring
                params_dict = {}
                
                # Simple regex to extract parameter descriptions
                param_pattern = r'(\w+):\s*(.*?)(?=\n\s*\w+:|$)'
                params_section = re.search(r'Args:(.*?)(?:Returns:|Raises:|$)', init_doc, re.DOTALL)
                
                if params_section:
                    params_text = params_section.group(1)
                    params = re.findall(param_pattern, params_text, re.DOTALL)
                    
                    for param_name, param_desc in params:
                        params_dict[param_name.strip()] = param_desc.strip()
                
                if params_dict:
                    metadata['parameters'] = params_dict
        
        # Extract metadata from predict method
        predict_method = getattr(model_class, 'predict', None)
        if predict_method:
            predict_doc = inspect.getdoc(predict_method)
            if predict_doc:
                # Extract return value description
                returns_section = re.search(r'Returns:(.*?)(?:Raises:|$)', predict_doc, re.DOTALL)
                if returns_section:
                    returns_text = returns_section.group(1).strip()
                    metadata['prediction_output'] = returns_text
        
        return metadata
    
    except Exception as e:
        logger.warning(f"Error extracting docstring metadata: {e}")
        return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate comprehensive ML model documentation")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--model_path', type=str, help="Path to the model file")
    input_group.add_argument('--module', type=str, help="Python module containing the model class")
    
    # Module-specific options
    parser.add_argument('--class', dest='class_name', type=str, help="Class name within the module")
    
    # Output options
    parser.add_argument('--output', type=str, required=True, help="Output file path for the model card")
    parser.add_argument('--format', choices=['markdown', 'md', 'json'], default='markdown', 
                        help="Output format (default: markdown)")
    
    # Content options
    parser.add_argument('--no-code-examples', action='store_false', dest='include_code',
                        help="Don't include code examples in the model card")
    parser.add_argument('--no-performance', action='store_false', dest='include_performance',
                        help="Don't include performance metrics in the model card")
    parser.add_argument('--no-limitations', action='store_false', dest='include_limitations',
                        help="Don't include limitations section in the model card")
    
    args = parser.parse_args()
    
    try:
        # Validate arguments
        if args.module and not args.class_name:
            parser.error("--class is required when using --module")
        
        if args.model_path:
            # Create documenter from model file
            documenter = MLModelDocumenter(model_path=args.model_path)
            
        elif args.module and args.class_name:
            # Extract metadata from docstrings
            metadata = extract_docstring_metadata(args.module, args.class_name)
            
            # Create documenter from class
            documenter = MLModelDocumenter.from_class(
                module_path=args.module,
                class_name=args.class_name,
                metadata=metadata
            )
        
        # Generate the model card
        documenter.generate_model_card(
            output_path=args.output,
            include_code_examples=args.include_code,
            include_performance_metrics=args.include_performance,
            include_limitations=args.include_limitations
        )
        
        logger.info(f"Model documentation successfully generated at {args.output}")
        
    except Exception as e:
        logger.error(f"Error generating model documentation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 