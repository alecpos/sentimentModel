#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML Model Explainability Generator

This script generates model explanations using SHAP values to understand feature importance
and model predictions. It creates visualizations and explanations that can be included in
model documentation.

Usage:
    python generate_model_explanations.py --model=path/to/model.pkl --data=path/to/data.csv --output=path/to/output_dir
    python generate_model_explanations.py --module=app.models.ml.prediction.ad_score_predictor --class=AdScorePredictor --data=path/to/data.csv --output=path/to/output_dir
"""

import os
import sys
import json
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_explainer')

# Try to import optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with 'pip install shap' for model explanations.")

try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some functionality may be limited.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available. Install with 'pip install joblib' for model loading.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some functionality for deep learning models may be limited.")


class ModelExplainer:
    """Generate explanations for ML models using SHAP."""
    
    def __init__(self, model_path: Optional[str] = None, model_object: Optional[Any] = None):
        """Initialize the model explainer.
        
        Args:
            model_path: Path to the serialized model file
            model_object: Pre-loaded model object
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for model explanations. Install with 'pip install shap'.")
        
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.num_features = None
        self.model_type = None
        self.explanation_summary = {}
        
        if model_path:
            self._load_model(model_path)
        elif model_object:
            self.model = model_object
            self._identify_model_type()
    
    def _load_model(self, model_path: str) -> None:
        """Load a model from a file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            # Determine file type by extension
            path = Path(model_path)
            extension = path.suffix.lower()
            
            if extension in ('.pkl', '.pickle'):
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif extension == '.joblib':
                if not JOBLIB_AVAILABLE:
                    raise ImportError("joblib is required to load .joblib files.")
                self.model = joblib.load(model_path)
            elif extension in ('.pt', '.pth'):
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch is required to load .pt/.pth files.")
                self.model = torch.load(model_path)
            else:
                logger.warning(f"Unknown model format: {extension}. Attempting to load with pickle.")
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
            logger.info(f"Successfully loaded model from {model_path}")
            self._identify_model_type()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _identify_model_type(self) -> None:
        """Identify the type of model for appropriate explainer selection."""
        if not self.model:
            raise ValueError("No model loaded")
            
        model_class = self.model.__class__.__name__
        module_name = self.model.__class__.__module__
        
        if SKLEARN_AVAILABLE and isinstance(self.model, BaseEstimator):
            if 'ensemble' in module_name:
                self.model_type = "tree_ensemble"
            elif 'linear_model' in module_name:
                self.model_type = "linear"
            elif 'tree' in module_name:
                self.model_type = "tree"
            else:
                self.model_type = "sklearn"
        elif hasattr(self.model, 'predict') and callable(self.model.predict):
            # Custom model with predict method
            self.model_type = "custom"
        elif TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
            self.model_type = "deep_learning"
        else:
            logger.warning(f"Model type {model_class} not recognized for explainability.")
            self.model_type = "unknown"

    def create_explainer(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> None:
        """Create a SHAP explainer based on the model type.
        
        Args:
            X: Training data to create the explainer
            **kwargs: Additional parameters for the explainer
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for model explanations.")
            
        if X is None:
            raise ValueError("Data required to create explainer")
            
        # Extract feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_values = X.values
        else:
            X_values = X
            self.feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        self.num_features = X_values.shape[1]
        
        # Create appropriate explainer based on model type
        if self.model_type in ("tree", "tree_ensemble"):
            self.explainer = shap.TreeExplainer(self.model, **kwargs)
        elif self.model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, X_values, **kwargs)
        elif self.model_type == "deep_learning":
            # Deep learning models require a specialized approach
            background = shap.sample(X_values, 100)  # Sample data for background
            self.explainer = shap.DeepExplainer(self.model, background, **kwargs)
        else:
            # Kernel explainer as fallback for any model
            self.explainer = shap.KernelExplainer(
                model=self.model.predict if hasattr(self.model, 'predict') else self.model,
                data=shap.sample(X_values, 100),
                **kwargs
            )
        
        logger.info(f"Created {type(self.explainer).__name__} for model explanation")
    
    def generate_explanations(self, X: Union[pd.DataFrame, np.ndarray], sample_size: Optional[int] = None) -> None:
        """Generate SHAP explanations for a dataset.
        
        Args:
            X: Data to explain predictions for
            sample_size: Number of samples to use (for large datasets)
        """
        if not self.explainer:
            raise ValueError("Explainer not created. Call create_explainer first.")
            
        # Extract data values
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # Sample data if specified
        if sample_size and sample_size < X_values.shape[0]:
            indices = np.random.choice(X_values.shape[0], sample_size, replace=False)
            X_sampled = X_values[indices]
            logger.info(f"Using {sample_size} samples for explanation")
        else:
            X_sampled = X_values
        
        # Generate SHAP values
        try:
            if self.model_type == "deep_learning":
                # Convert to tensor if needed for deep learning models
                if TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
                    if not isinstance(X_sampled, torch.Tensor):
                        X_tensor = torch.tensor(X_sampled, dtype=torch.float32)
                    else:
                        X_tensor = X_sampled
                    self.shap_values = self.explainer.shap_values(X_tensor)
                else:
                    self.shap_values = self.explainer.shap_values(X_sampled)
            else:
                self.shap_values = self.explainer.shap_values(X_sampled)
            
            # Extract explanation summary
            self._extract_explanation_summary()
            
            logger.info("Successfully generated SHAP explanations")
        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            raise
    
    def _extract_explanation_summary(self) -> None:
        """Extract a summary of the explanations."""
        if self.shap_values is None:
            raise ValueError("No SHAP values generated")
            
        # Handle different formats of SHAP values
        if isinstance(self.shap_values, list):
            # Multi-class case
            feature_importances = np.abs(np.array(self.shap_values)).mean(axis=1).mean(axis=0)
        else:
            # Binary classification or regression
            feature_importances = np.abs(self.shap_values).mean(axis=0)
        
        # Create a mapping of feature importances
        feature_importance_dict = {}
        for i, importance in enumerate(feature_importances):
            if i < len(self.feature_names):
                feature_importance_dict[self.feature_names[i]] = float(importance)
            else:
                feature_importance_dict[f"Feature_{i}"] = float(importance)
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Store in explanation summary
        self.explanation_summary = {
            "model_type": self.model_type,
            "num_features": self.num_features,
            "feature_importance": dict(sorted_features),
            "top_features": dict(sorted_features[:10]),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_feature_importance_plot(self, output_path: str, max_features: int = 20) -> str:
        """Save a feature importance plot.
        
        Args:
            output_path: Directory to save the plot
            max_features: Maximum number of features to include
            
        Returns:
            Path to the saved plot
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values generated")
            
        os.makedirs(output_path, exist_ok=True)
        plot_path = os.path.join(output_path, "feature_importance.png")
        
        plt.figure(figsize=(10, 8))
        
        # Get feature importances in the right format
        if isinstance(self.shap_values, list):
            # For multi-class models, use the mean absolute SHAP value
            # across all classes and samples
            shap_values_summary = np.abs(np.array(self.shap_values)).mean(axis=1).mean(axis=0)
        else:
            # For binary classification or regression
            shap_values_summary = np.abs(self.shap_values).mean(axis=0)
        
        # Create a DataFrame for plotting
        if len(self.feature_names) == len(shap_values_summary):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': shap_values_summary
            })
        else:
            # Handle mismatch (should not happen in normal cases)
            logger.warning("Feature name length mismatch. Using generic feature names.")
            importance_df = pd.DataFrame({
                'Feature': [f"Feature_{i}" for i in range(len(shap_values_summary))],
                'Importance': shap_values_summary
            })
        
        # Sort and limit to max_features
        importance_df = importance_df.sort_values('Importance', ascending=False).head(max_features)
        
        # Create the plot
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Feature')
        plt.title('Feature Importance (based on SHAP values)')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
        return plot_path
    
    def save_summary_plot(self, output_path: str, plot_type: str = 'bar') -> str:
        """Save a SHAP summary plot.
        
        Args:
            output_path: Directory to save the plot
            plot_type: Type of summary plot ('bar', 'dot', or 'violin')
            
        Returns:
            Path to the saved plot
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values generated")
            
        os.makedirs(output_path, exist_ok=True)
        plot_path = os.path.join(output_path, f"summary_plot_{plot_type}.png")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Get the correct feature names
        feature_names = self.feature_names if len(self.feature_names) == self.num_features else None
        
        # Create the SHAP summary plot
        if plot_type == 'bar':
            shap.summary_plot(
                self.shap_values,
                plot_type='bar',
                feature_names=feature_names,
                show=False
            )
        else:
            # Dot plot (default) or violin
            shap.summary_plot(
                self.shap_values,
                feature_names=feature_names,
                show=False
            )
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {plot_path}")
        return plot_path
    
    def save_dependence_plots(self, output_path: str, X: Union[pd.DataFrame, np.ndarray], top_n: int = 5) -> List[str]:
        """Save dependence plots for top features.
        
        Args:
            output_path: Directory to save the plots
            X: Data used for plotting
            top_n: Number of top features to create plots for
            
        Returns:
            List of paths to the saved plots
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values generated")
            
        os.makedirs(output_path, exist_ok=True)
        plot_paths = []
        
        # Get the top features
        if isinstance(self.shap_values, list):
            # Multi-class case
            feature_importances = np.abs(np.array(self.shap_values)).mean(axis=1).mean(axis=0)
        else:
            # Binary classification or regression
            feature_importances = np.abs(self.shap_values).mean(axis=0)
        
        # Get indices of top features
        top_indices = np.argsort(-feature_importances)[:top_n]
        
        # Convert X to a DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            if len(self.feature_names) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X
        
        # Create dependence plots for each top feature
        for idx in top_indices:
            if idx >= len(self.feature_names):
                continue
                
            feature_name = self.feature_names[idx]
            plot_path = os.path.join(output_path, f"dependence_plot_{feature_name.replace(' ', '_')}.png")
            
            plt.figure(figsize=(10, 8))
            
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                # Use the first class for multi-class models
                shap.dependence_plot(
                    idx, 
                    self.shap_values[0], 
                    X_df,
                    show=False
                )
            else:
                shap.dependence_plot(
                    idx, 
                    self.shap_values, 
                    X_df,
                    show=False
                )
            
            plt.title(f"SHAP Dependence Plot for {feature_name}")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(plot_path)
            
        logger.info(f"Generated {len(plot_paths)} dependence plots")
        return plot_paths
    
    def generate_explanation_report(self, output_path: str, X: Union[pd.DataFrame, np.ndarray]) -> str:
        """Generate a comprehensive explanation report.
        
        Args:
            output_path: Directory to save the report and plots
            X: Data used for plotting
            
        Returns:
            Path to the explanation report
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values generated")
            
        os.makedirs(output_path, exist_ok=True)
        plots_dir = os.path.join(output_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate all plots
        feature_importance_plot = self.save_feature_importance_plot(plots_dir)
        summary_bar_plot = self.save_summary_plot(plots_dir, 'bar')
        summary_dot_plot = self.save_summary_plot(plots_dir, 'dot')
        dependence_plots = self.save_dependence_plots(plots_dir, X, top_n=5)
        
        # Create a report in Markdown format
        report_path = os.path.join(output_path, "model_explanation_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Model Explanation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Model Information\n\n")
            f.write(f"- Model Type: {self.model_type}\n")
            f.write(f"- Number of Features: {self.num_features}\n\n")
            
            f.write("## Feature Importance\n\n")
            f.write("The following features have the most impact on model predictions:\n\n")
            f.write("| Feature | Importance |\n")
            f.write("|---------|------------|\n")
            
            # Add top 10 features
            top_features = list(self.explanation_summary["top_features"].items())
            for feature, importance in top_features:
                f.write(f"| {feature} | {importance:.4f} |\n")
            
            f.write("\n\n")
            f.write(f"![Feature Importance](plots/{os.path.basename(feature_importance_plot)})\n\n")
            
            f.write("## SHAP Summary Plots\n\n")
            f.write("These plots show how each feature affects model predictions.\n\n")
            f.write(f"![SHAP Summary Bar Plot](plots/{os.path.basename(summary_bar_plot)})\n\n")
            f.write(f"![SHAP Summary Dot Plot](plots/{os.path.basename(summary_dot_plot)})\n\n")
            
            f.write("## Feature Dependence Plots\n\n")
            f.write("These plots show how the impact of a feature varies with its value.\n\n")
            
            for plot_path in dependence_plots:
                feature_name = os.path.basename(plot_path).replace("dependence_plot_", "").replace(".png", "").replace("_", " ")
                f.write(f"### {feature_name}\n\n")
                f.write(f"![{feature_name} Dependence Plot](plots/{os.path.basename(plot_path)})\n\n")
            
            f.write("## Interpretation Guidelines\n\n")
            f.write("- **SHAP Values**: SHAP (SHapley Additive exPlanations) values represent the impact of each feature on the model's prediction.\n")
            f.write("- **Feature Importance**: Higher values indicate features with a larger overall impact on the model's predictions.\n")
            f.write("- **Dependence Plots**: Show how the effect of a feature varies with its value, and how it interacts with other features.\n")
            f.write("- **Color Coding**: In the dot plot, red indicates higher feature values, blue indicates lower values.\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on this analysis, consider the following recommendations:\n\n")
            f.write("1. Focus on the top features for model refinement and feature engineering.\n")
            f.write("2. Examine unusual patterns in the dependence plots that might indicate data issues.\n")
            f.write("3. Consider removing or transforming features with low importance for model simplification.\n")
            f.write("4. Check for potential biases in how the model uses important features.\n")
        
        # Save explanation summary as JSON
        summary_path = os.path.join(output_path, "explanation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.explanation_summary, f, indent=2)
        
        logger.info(f"Explanation report saved to {report_path}")
        return report_path


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a file.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Loaded data as a DataFrame
    """
    try:
        # Infer file type from extension
        extension = Path(data_path).suffix.lower()
        
        if extension == '.csv':
            data = pd.read_csv(data_path)
        elif extension == '.parquet':
            data = pd.read_parquet(data_path)
        elif extension in ('.xls', '.xlsx'):
            data = pd.read_excel(data_path)
        elif extension == '.json':
            data = pd.read_json(data_path)
        elif extension == '.pkl':
            import pickle
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        logger.info(f"Successfully loaded data from {data_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_model_from_module(module_path: str, class_name: str, **kwargs) -> Any:
    """Load a model by importing a module and instantiating a class.
    
    Args:
        module_path: Import path for the module
        class_name: Name of the class to instantiate
        **kwargs: Arguments to pass to the class constructor
        
    Returns:
        Instantiated model object
    """
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model = model_class(**kwargs)
        
        logger.info(f"Successfully loaded {class_name} from {module_path}")
        return model
    
    except ImportError:
        logger.error(f"Could not import module: {module_path}")
        raise
    except AttributeError:
        logger.error(f"Class {class_name} not found in module {module_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from module: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate ML model explanations using SHAP")
    
    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data file for generating explanations")
    
    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                       help="Directory to save explanations and visualizations")
    
    # Model loading arguments (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str,
                          help="Path to serialized model file")
    model_group.add_argument("--module", type=str,
                          help="Python module path containing the model class")
    
    # Additional arguments
    parser.add_argument("--class", dest="class_name", type=str,
                       help="Class name to instantiate (required with --module)")
    parser.add_argument("--sample", type=int, default=None,
                       help="Number of samples to use for explanations")
    parser.add_argument("--target", type=str, default=None,
                       help="Target column name for splitting features and target")
    
    args = parser.parse_args()
    
    # Check SHAP availability
    if not SHAP_AVAILABLE:
        logger.error("SHAP is required for model explanations. Install with 'pip install shap'")
        return 1
    
    try:
        # Load data
        data = load_data(args.data)
        logger.info(f"Loaded data with shape {data.shape}")
        
        # Split features and target if specified
        if args.target and args.target in data.columns:
            X = data.drop(columns=[args.target])
            y = data[args.target]
            logger.info(f"Split data into features (shape {X.shape}) and target")
        else:
            X = data
            y = None
            logger.info("Using all columns as features")
        
        # Load model
        if args.model:
            explainer = ModelExplainer(model_path=args.model)
        elif args.module:
            if not args.class_name:
                logger.error("--class is required when using --module")
                return 1
                
            model = load_model_from_module(args.module, args.class_name)
            explainer = ModelExplainer(model_object=model)
        
        # Create explainer and generate explanations
        explainer.create_explainer(X)
        explainer.generate_explanations(X, sample_size=args.sample)
        
        # Generate report
        report_path = explainer.generate_explanation_report(args.output, X)
        
        logger.info(f"Model explanation completed successfully. Report saved to {report_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating model explanations: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 