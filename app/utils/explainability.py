"""
Model Explainability Utilities

This module provides utilities for generating explanations for ML model predictions
in the WITHIN ML Prediction System. It implements SHAP-based feature importance
and other explainability methods.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import abc
import time
import numpy as np
from dataclasses import dataclass

# Import ML exceptions
from app.utils.ml_exceptions import ExplanationError


@dataclass
class ExplanationResult:
    """Data class for storing explanation results."""
    
    method: str
    feature_importance: Dict[str, float]
    baseline_score: float
    processing_time_ms: int
    feature_interactions: Optional[Dict[str, Dict[str, float]]] = None
    partial_dependence: Optional[Dict[str, List[float]]] = None


class ModelExplainer(abc.ABC):
    """Base class for model explainers."""
    
    def __init__(self):
        """Initialize the model explainer."""
        self.logger = logging.getLogger(__name__)
    
    @property
    @abc.abstractmethod
    def method(self) -> str:
        """Get the explanation method name."""
        pass
    
    @abc.abstractmethod
    async def explain(self, model: Any, features: Dict[str, Any], prediction: Dict[str, Any]) -> ExplanationResult:
        """
        Generate explanation for a prediction.
        
        Args:
            model: The ML model
            features: The input features
            prediction: The prediction result
            
        Returns:
            Explanation result
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass


class ShapExplainer(ModelExplainer):
    """SHAP-based model explainer."""
    
    @property
    def method(self) -> str:
        """Get the explanation method name."""
        return "shap"
    
    async def explain(self, model: Any, features: Dict[str, Any], prediction: Dict[str, Any]) -> ExplanationResult:
        """
        Generate SHAP-based explanation for a prediction.
        
        Args:
            model: The ML model
            features: The input features
            prediction: The prediction result
            
        Returns:
            Explanation result with SHAP values
            
        Raises:
            ExplanationError: If SHAP explanation generation fails
        """
        try:
            # Record start time
            start_time = time.time()
            
            # In a real implementation, this would use SHAP library to generate explanations
            # For this example, we'll create a simple mock implementation
            
            # Extract feature types
            feature_importance = {}
            
            # Process text features
            if features.get("text_features"):
                for i, text in enumerate(features["text_features"]):
                    feature_importance[f"text_features[{i}]"] = 0.3 + (0.1 * i)
            
            # Process numeric features
            if features.get("numeric_features"):
                for i, value in enumerate(features["numeric_features"]):
                    feature_importance[f"numeric_features[{i}]"] = 0.2 + (value * 0.5)
            
            # Process categorical features
            if features.get("categorical_features"):
                for i, category in enumerate(features["categorical_features"]):
                    feature_importance[f"categorical_features[{i}]"] = 0.15 + (0.05 * i)
            
            # Normalize feature importance to sum to 1
            total = sum(feature_importance.values())
            feature_importance = {k: v / total for k, v in feature_importance.items()}
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create explanation result
            explanation = ExplanationResult(
                method=self.method,
                feature_importance=feature_importance,
                baseline_score=0.5,  # In a real implementation, this would be the model's base prediction
                processing_time_ms=processing_time_ms
            )
            
            return explanation
            
        except Exception as e:
            # Log error
            self.logger.exception(f"Error generating SHAP explanation: {str(e)}")
            
            # Raise explanation error
            raise ExplanationError(
                message=f"Failed to generate SHAP explanation: {str(e)}",
                details={"error_type": str(type(e).__name__)}
            )


class LimeExplainer(ModelExplainer):
    """LIME-based model explainer."""
    
    @property
    def method(self) -> str:
        """Get the explanation method name."""
        return "lime"
    
    async def explain(self, model: Any, features: Dict[str, Any], prediction: Dict[str, Any]) -> ExplanationResult:
        """
        Generate LIME-based explanation for a prediction.
        
        Args:
            model: The ML model
            features: The input features
            prediction: The prediction result
            
        Returns:
            Explanation result with LIME values
            
        Raises:
            ExplanationError: If LIME explanation generation fails
        """
        try:
            # Record start time
            start_time = time.time()
            
            # In a real implementation, this would use LIME library to generate explanations
            # For this example, we'll create a simple mock implementation
            
            # Extract feature types
            feature_importance = {}
            
            # Process text features
            if features.get("text_features"):
                for i, text in enumerate(features["text_features"]):
                    feature_importance[f"text_features[{i}]"] = 0.25 + (0.15 * i)
            
            # Process numeric features
            if features.get("numeric_features"):
                for i, value in enumerate(features["numeric_features"]):
                    feature_importance[f"numeric_features[{i}]"] = 0.15 + (value * 0.6)
            
            # Process categorical features
            if features.get("categorical_features"):
                for i, category in enumerate(features["categorical_features"]):
                    feature_importance[f"categorical_features[{i}]"] = 0.1 + (0.08 * i)
            
            # Normalize feature importance to sum to 1
            total = sum(feature_importance.values())
            feature_importance = {k: v / total for k, v in feature_importance.items()}
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create explanation result
            explanation = ExplanationResult(
                method=self.method,
                feature_importance=feature_importance,
                baseline_score=0.5,  # In a real implementation, this would be the model's base prediction
                processing_time_ms=processing_time_ms
            )
            
            return explanation
            
        except Exception as e:
            # Log error
            self.logger.exception(f"Error generating LIME explanation: {str(e)}")
            
            # Raise explanation error
            raise ExplanationError(
                message=f"Failed to generate LIME explanation: {str(e)}",
                details={"error_type": str(type(e).__name__)}
            )


class EnsembleExplainer(ModelExplainer):
    """Ensemble model explainer that combines multiple explanation methods."""
    
    def __init__(self, explainers: Optional[List[ModelExplainer]] = None):
        """
        Initialize ensemble explainer.
        
        Args:
            explainers: List of model explainers to use
        """
        super().__init__()
        self.explainers = explainers or [ShapExplainer(), LimeExplainer()]
    
    @property
    def method(self) -> str:
        """Get the explanation method name."""
        return "ensemble"
    
    async def explain(self, model: Any, features: Dict[str, Any], prediction: Dict[str, Any]) -> ExplanationResult:
        """
        Generate ensemble explanation by combining multiple methods.
        
        Args:
            model: The ML model
            features: The input features
            prediction: The prediction result
            
        Returns:
            Explanation result with combined values
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        try:
            # Record start time
            start_time = time.time()
            
            # Generate explanations from all explainers
            explanations = []
            for explainer in self.explainers:
                try:
                    explanation = await explainer.explain(model, features, prediction)
                    explanations.append(explanation)
                except Exception as e:
                    self.logger.warning(f"Error with explainer {explainer.method}: {str(e)}")
            
            # If no explanations were generated, raise error
            if not explanations:
                raise ExplanationError(
                    message="All explainers failed to generate explanations",
                    details={"explainers": [e.method for e in self.explainers]}
                )
            
            # Combine feature importance from all explanations
            combined_importance = {}
            
            # For each feature in any explanation
            all_features = set()
            for explanation in explanations:
                all_features.update(explanation.feature_importance.keys())
            
            # Average importance across all explanations that include the feature
            for feature in all_features:
                importances = []
                for explanation in explanations:
                    if feature in explanation.feature_importance:
                        importances.append(explanation.feature_importance[feature])
                
                if importances:
                    combined_importance[feature] = sum(importances) / len(importances)
            
            # Normalize feature importance to sum to 1
            total = sum(combined_importance.values())
            combined_importance = {k: v / total for k, v in combined_importance.items()}
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create explanation result
            explanation = ExplanationResult(
                method=self.method,
                feature_importance=combined_importance,
                baseline_score=sum(e.baseline_score for e in explanations) / len(explanations),
                processing_time_ms=processing_time_ms,
                feature_interactions={},  # Would combine from individual explainers in real impl
                partial_dependence={}  # Would combine from individual explainers in real impl
            )
            
            return explanation
            
        except Exception as e:
            # Log error
            self.logger.exception(f"Error generating ensemble explanation: {str(e)}")
            
            # Raise explanation error
            raise ExplanationError(
                message=f"Failed to generate ensemble explanation: {str(e)}",
                details={"error_type": str(type(e).__name__)}
            )


async def create_explanation(
    model: Any, 
    features: Dict[str, Any], 
    prediction: Dict[str, Any],
    method: str = "shap"
) -> Dict[str, Any]:
    """
    Create explanation for a prediction using the specified method.
    
    Args:
        model: The ML model
        features: The input features
        prediction: The prediction result
        method: Explanation method to use ('shap', 'lime', 'ensemble')
        
    Returns:
        Explanation dictionary
        
    Raises:
        ExplanationError: If explanation generation fails
        ValueError: If the specified method is not supported
    """
    # Get appropriate explainer for the specified method
    explainer = get_explainer(method)
    
    # Generate explanation
    explanation = await explainer.explain(model, features, prediction)
    
    # Convert to dictionary
    explanation_dict = {
        "method": explanation.method,
        "feature_importance": explanation.feature_importance,
        "baseline_score": explanation.baseline_score,
        "processing_time_ms": explanation.processing_time_ms
    }
    
    # Add additional details if available
    if explanation.feature_interactions:
        explanation_dict["feature_interactions"] = explanation.feature_interactions
        
    if explanation.partial_dependence:
        explanation_dict["partial_dependence"] = explanation.partial_dependence
    
    return explanation_dict


def get_explainer(method: str) -> ModelExplainer:
    """
    Get explainer for the specified method.
    
    Args:
        method: Explanation method ('shap', 'lime', 'ensemble')
        
    Returns:
        Model explainer
        
    Raises:
        ValueError: If the specified method is not supported
    """
    explainers = {
        "shap": ShapExplainer(),
        "lime": LimeExplainer(),
        "ensemble": EnsembleExplainer()
    }
    
    if method not in explainers:
        raise ValueError(f"Unsupported explanation method: {method}")
    
    return explainers[method]


def format_feature_importance(feature_importance: Dict[str, float], top_k: int = 10) -> Dict[str, float]:
    """
    Format feature importance for API response.
    
    Args:
        feature_importance: Feature importance dictionary
        top_k: Number of top features to include
        
    Returns:
        Formatted feature importance dictionary
    """
    # Sort features by importance (descending)
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Take top k features
    top_features = sorted_features[:top_k]
    
    # Convert to dictionary
    return dict(top_features) 