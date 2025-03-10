#!/usr/bin/env python3
"""
Model Interpretation Example

This example demonstrates how to interpret model predictions using SHAP values,
helping to understand which features contribute most to individual predictions.

Requirements:
    - within-sdk>=1.2.0
    - shap>=0.41.0
    - matplotlib>=3.5.0
    - pandas>=1.3.0
    - numpy>=1.21.0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('within-model-interpretation')

# Load environment variables
load_dotenv()

# Try to import the WITHIN SDK
try:
    from within import Client
    HAS_WITHIN_SDK = True
except ImportError:
    logger.warning("WITHIN SDK not found. Will use simulated data for demonstration.")
    HAS_WITHIN_SDK = False

def get_client() -> Any:
    """Initialize WITHIN client with API credentials."""
    if not HAS_WITHIN_SDK:
        return None
        
    access_key = os.getenv("WITHIN_ACCESS_KEY")
    secret_key = os.getenv("WITHIN_SECRET_KEY")
    
    if not access_key or not secret_key:
        raise ValueError("API credentials not found. Set WITHIN_ACCESS_KEY and WITHIN_SECRET_KEY environment variables.")
    
    return Client(access_key=access_key, secret_key=secret_key)

def get_feature_values(ad_id: str) -> Dict[str, Any]:
    """
    Get feature values for a specific ad for SHAP interpretation.
    
    Args:
        ad_id: Identifier for the ad
        
    Returns:
        Dictionary of feature values
    """
    client = get_client()
    
    if client:
        # Real implementation using SDK
        try:
            return client.get_features(ad_id=ad_id)
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            raise
    else:
        # Simulated data for demonstration
        return {
            "ad_length": 85,
            "contains_emoji": True,
            "exclamation_count": 2,
            "question_count": 0,
            "call_to_action_strength": 0.78,
            "sentiment_score": 0.65,
            "emotional_appeal": 0.58,
            "urgency_level": 0.72,
            "benefit_clarity": 0.81,
            "uses_social_proof": False,
            "uses_scarcity": True,
            "uses_price_mention": True,
            "uses_numbers": True,
            "reading_ease": 68.5,
            "target_audience_relevance": 0.76,
            "historical_ctr": 0.042,
            "image_quality_score": 0.85,
            "video_length": 0,
            "platform_suitability": 0.79,
            "audience_size": 850000
        }

def get_shap_values(ad_id: str, model_name: str = "ad_score_predictor") -> Dict[str, Any]:
    """
    Get SHAP values for model interpretation.
    
    Args:
        ad_id: Identifier for the ad
        model_name: Name of the model to interpret
        
    Returns:
        Dictionary with SHAP values and base value
    """
    client = get_client()
    
    if client:
        # Real implementation using SDK
        try:
            return client.get_prediction_explanation(
                ad_id=ad_id,
                model_name=model_name,
                explanation_type="shap"
            )
        except Exception as e:
            logger.error(f"Error getting SHAP values: {e}")
            raise
    else:
        # Simulated SHAP values for demonstration
        features = get_feature_values(ad_id)
        
        # Creating simulated SHAP values roughly proportional to feature values
        # In a real scenario, these would come from the model's explanation
        shap_values = {}
        base_value = 5.0  # Base prediction value
        
        # Assign SHAP values (simulated)
        shap_values = {
            "ad_length": -0.12,
            "contains_emoji": 0.08,
            "exclamation_count": -0.05,
            "call_to_action_strength": 0.42,
            "sentiment_score": 0.38,
            "emotional_appeal": 0.25,
            "urgency_level": 0.31,
            "benefit_clarity": 0.45,
            "uses_scarcity": 0.22,
            "uses_price_mention": 0.18,
            "historical_ctr": 0.55,
            "image_quality_score": 0.48,
            "platform_suitability": 0.35
        }
        
        return {
            "shap_values": shap_values,
            "base_value": base_value
        }

def plot_shap_values(shap_data: Dict[str, Any], title: str = "Feature Impact on Prediction") -> None:
    """
    Plot SHAP values as a waterfall chart.
    
    Args:
        shap_data: Dictionary with SHAP values and base value
        title: Title for the plot
    """
    shap_values = shap_data["shap_values"]
    base_value = shap_data["base_value"]
    
    # Sort features by absolute SHAP value
    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Prepare data for plotting
    features = [item[0].replace('_', ' ').title() for item in sorted_features]
    values = [item[1] for item in sorted_features]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot bars
    colors = ['#ff4d4d' if x < 0 else '#4daf4a' for x in values]
    y_pos = np.arange(len(features))
    
    plt.barh(y_pos, values, color=colors)
    plt.yticks(y_pos, features)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.title(title)
    
    # Add value labels to bars
    for i, v in enumerate(values):
        label = f"{v:.3f}"
        plt.text(
            v + (0.01 if v >= 0 else -0.08),
            i,
            label,
            va='center'
        )
    
    # Calculate total prediction
    total_shap = sum(values)
    prediction = base_value + total_shap
    
    # Add prediction info
    plt.figtext(
        0.5, 0.01,
        f"Base value: {base_value:.2f} + Feature contributions: {total_shap:.2f} = Final prediction: {prediction:.2f}",
        ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('shap_explanation.png')
    plt.show()

def print_shap_explanation(shap_data: Dict[str, Any]) -> None:
    """
    Print textual explanation of SHAP values.
    
    Args:
        shap_data: Dictionary with SHAP values and base value
    """
    shap_values = shap_data["shap_values"]
    base_value = shap_data["base_value"]
    
    # Sort features by absolute SHAP value
    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Calculate total prediction
    total_shap = sum(shap_values.values())
    prediction = base_value + total_shap
    
    print("\n===== Model Prediction Explanation =====")
    print(f"Base prediction value: {base_value:.2f}/10")
    print(f"Final prediction: {prediction:.2f}/10")
    print("\nTop factors contributing to this prediction:")
    
    for feature, value in sorted_features[:5]:
        impact = "increased" if value > 0 else "decreased"
        print(f"  • {feature.replace('_', ' ').title()} {impact} the prediction by {abs(value):.2f} points")
    
    print("\nDetailed feature contributions:")
    for feature, value in sorted_features:
        impact = "+" if value > 0 else ""
        print(f"  {feature.replace('_', ' ').title()}: {impact}{value:.3f}")
    
    print("=======================================")

def analyze_ad_performance(ad_id: str, model_name: str = "ad_score_predictor") -> None:
    """
    Complete analysis of ad performance with model interpretation.
    
    Args:
        ad_id: Identifier for the ad
        model_name: Name of the model to interpret
    """
    try:
        # Get feature values
        features = get_feature_values(ad_id)
        logger.info(f"Retrieved {len(features)} features for ad ID: {ad_id}")
        
        # Get SHAP values for interpretation
        shap_data = get_shap_values(ad_id, model_name)
        
        # Print textual explanation
        print_shap_explanation(shap_data)
        
        # Plot visualization
        plot_shap_values(shap_data, title=f"Feature Impact on {model_name.title().replace('_', ' ')} Prediction")
        
        # Provide recommendations based on SHAP values
        provide_recommendations(features, shap_data)
        
    except Exception as e:
        logger.error(f"Error analyzing ad performance: {e}")
        raise

def provide_recommendations(features: Dict[str, Any], shap_data: Dict[str, Any]) -> None:
    """
    Provide actionable recommendations based on SHAP values.
    
    Args:
        features: Dictionary of feature values
        shap_data: Dictionary with SHAP values and base value
    """
    shap_values = shap_data["shap_values"]
    
    # Find negative contributors that could be improved
    negative_impacts = {
        k: v for k, v in shap_values.items() 
        if v < 0 and k in features
    }
    
    # Find top positive contributors to maintain
    positive_impacts = {
        k: v for k, v in shap_values.items() 
        if v > 0 and k in features
    }
    
    print("\n===== Recommendations =====")
    
    # Recommendations for improvement
    if negative_impacts:
        print("\nAreas for improvement:")
        for feature, value in sorted(negative_impacts.items(), key=lambda x: x[1]):
            if feature == "ad_length":
                if features[feature] < 50:
                    print(f"  • Consider making your ad longer for more information")
                else:
                    print(f"  • Your ad may be too long. Consider making it more concise")
            elif feature == "exclamation_count":
                print(f"  • Reduce the number of exclamation marks for a more professional tone")
            elif feature == "sentiment_score":
                print(f"  • Adjust the sentiment of your ad - current emotional tone may not resonate with audience")
            elif feature == "benefit_clarity":
                print(f"  • Make the benefits of your offer clearer to the audience")
            elif feature == "call_to_action_strength":
                print(f"  • Strengthen your call to action to drive more engagement")
            else:
                print(f"  • Consider improving {feature.replace('_', ' ')}")
    else:
        print("\nNo significant areas for improvement identified.")
    
    # Strengths to maintain
    if positive_impacts:
        print("\nStrengths to maintain:")
        for feature, value in sorted(positive_impacts.items(), key=lambda x: -x[1])[:3]:
            print(f"  • {feature.replace('_', ' ').title()} is positively impacting performance")
    
    print("===========================")

def main():
    """Main function demonstrating model interpretation."""
    print("WITHIN Model Interpretation Example")
    
    # Example 1: Analyze an example ad
    print("\nExample 1: Analyzing ad performance")
    try:
        # In a real scenario, you would use an actual ad ID
        ad_id = "ad_12345"
        analyze_ad_performance(ad_id, "ad_score_predictor")
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
    
    # Example 2: Compare two different models on the same ad
    print("\nExample 2: Comparing model interpretations")
    try:
        ad_id = "ad_12345"
        # In a real scenario, you would use actual model names
        for model in ["ad_score_predictor", "conversion_predictor"]:
            print(f"\nAnalyzing with model: {model}")
            shap_data = get_shap_values(ad_id, model)
            print_shap_explanation(shap_data)
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")

if __name__ == "__main__":
    main() 