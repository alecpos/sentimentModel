# Fairness Implementation Guide

## Introduction

This implementation guide provides detailed technical instructions for enhancing the sentiment analysis system with advanced fairness techniques. Building on the gameplan outlined in ADVANCED_FAIRNESS_GAMEPLAN.md, this document focuses on practical implementation steps for the highest-priority enhancements.

The guide is organized by component, with each section providing:
- Detailed implementation instructions
- Code snippets and examples
- Integration requirements
- Testing procedures

This document focuses on Phase 1 priorities that can provide immediate improvements to the fairness capabilities of our system.

## 1. Post-Processing Adjustments

Post-processing techniques modify model outputs to enforce fairness constraints without changing the underlying model. These techniques are particularly valuable because they can be applied to existing trained models without retraining.

### Implementation Steps

#### 1.1 Create the `fairness_postprocessing.py` Module

```python
#!/usr/bin/env python
"""
Fairness Post-Processing Module

This module implements post-processing techniques to mitigate fairness issues
in sentiment analysis predictions after model inference.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """Optimizes classification thresholds to satisfy fairness constraints."""
    
    def __init__(self, fairness_metric: str = "equalized_odds", 
                 grid_size: int = 100):
        """
        Initialize the threshold optimizer.
        
        Args:
            fairness_metric: The fairness metric to optimize for.
                Options: "demographic_parity", "equalized_odds", "equal_opportunity"
            grid_size: Number of threshold values to try during optimization
        """
        self.fairness_metric = fairness_metric
        self.grid_size = grid_size
        self.group_thresholds = {}
        self.base_threshold = 0.5
        
    def compute_confusion_rates(self, y_true: np.ndarray, 
                               y_pred_proba: np.ndarray, 
                               threshold: float) -> Tuple[float, float]:
        """
        Compute true positive rate and false positive rate for given threshold.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Tuple of (true positive rate, false positive rate)
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Handle division by zero
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return tpr, fpr
    
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
           protected_attributes: pd.DataFrame) -> None:
        """
        Learn optimal thresholds for each demographic group.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
        """
        # Initialize base threshold using overall data
        thresholds = np.linspace(0.01, 0.99, self.grid_size)
        best_threshold = 0.5
        best_f1 = 0
        
        # Find best overall threshold with F1 optimization
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate precision and recall
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            false_positives = np.sum((y_true == 0) & (y_pred == 1))
            false_negatives = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.base_threshold = best_threshold
        logger.info(f"Base threshold set to {self.base_threshold:.3f} with F1 score {best_f1:.3f}")
        
        # Process each protected attribute column
        for col in protected_attributes.columns:
            unique_groups = protected_attributes[col].unique()
            
            # If the metric is demographic parity
            if self.fairness_metric == "demographic_parity":
                self._optimize_demographic_parity(y_true, y_pred_proba, 
                                                protected_attributes, col)
            
            # If the metric is equalized odds or equal opportunity
            elif self.fairness_metric in ["equalized_odds", "equal_opportunity"]:
                self._optimize_equalized_odds(y_true, y_pred_proba, 
                                            protected_attributes, col)
    
    def _optimize_demographic_parity(self, y_true, y_pred_proba, 
                                   protected_attributes, col):
        """Optimize thresholds for demographic parity."""
        unique_groups = protected_attributes[col].unique()
        thresholds = {}
        
        # Calculate overall positive rate with base threshold
        overall_pred = (y_pred_proba >= self.base_threshold).astype(int)
        overall_positive_rate = np.mean(overall_pred)
        
        # Find thresholds that give similar positive rates for each group
        for group in unique_groups:
            group_mask = protected_attributes[col] == group
            if sum(group_mask) < 10:  # Skip groups with too few samples
                thresholds[group] = self.base_threshold
                continue
                
            group_proba = y_pred_proba[group_mask]
            
            # Try different thresholds
            best_threshold = self.base_threshold
            min_diff = float('inf')
            
            for threshold in np.linspace(0.01, 0.99, self.grid_size):
                group_pred = (group_proba >= threshold).astype(int)
                group_positive_rate = np.mean(group_pred)
                diff = abs(group_positive_rate - overall_positive_rate)
                
                if diff < min_diff:
                    min_diff = diff
                    best_threshold = threshold
            
            thresholds[group] = best_threshold
        
        # Store thresholds for this protected attribute
        self.group_thresholds[col] = thresholds
        logger.info(f"Optimized demographic parity thresholds for {col}: {thresholds}")
    
    def _optimize_equalized_odds(self, y_true, y_pred_proba, 
                               protected_attributes, col):
        """Optimize thresholds for equalized odds or equal opportunity."""
        unique_groups = protected_attributes[col].unique()
        thresholds = {}
        
        # Calculate overall TPR and FPR with base threshold
        overall_tpr, overall_fpr = self.compute_confusion_rates(
            y_true, y_pred_proba, self.base_threshold
        )
        
        # Find thresholds that give similar TPR and FPR for each group
        for group in unique_groups:
            group_mask = protected_attributes[col] == group
            if sum(group_mask) < 10:  # Skip groups with too few samples
                thresholds[group] = self.base_threshold
                continue
                
            group_y = y_true[group_mask]
            group_proba = y_pred_proba[group_mask]
            
            # Try different thresholds
            best_threshold = self.base_threshold
            min_diff = float('inf')
            
            for threshold in np.linspace(0.01, 0.99, self.grid_size):
                group_tpr, group_fpr = self.compute_confusion_rates(
                    group_y, group_proba, threshold
                )
                
                # For equalized odds, consider both TPR and FPR
                if self.fairness_metric == "equalized_odds":
                    tpr_diff = abs(group_tpr - overall_tpr)
                    fpr_diff = abs(group_fpr - overall_fpr)
                    diff = tpr_diff + fpr_diff
                
                # For equal opportunity, consider only TPR
                else:  # equal_opportunity
                    diff = abs(group_tpr - overall_tpr)
                
                if diff < min_diff:
                    min_diff = diff
                    best_threshold = threshold
            
            thresholds[group] = best_threshold
        
        # Store thresholds for this protected attribute
        self.group_thresholds[col] = thresholds
        logger.info(f"Optimized {self.fairness_metric} thresholds for {col}: {thresholds}")
    
    def adjust(self, y_pred_proba: np.ndarray, 
              protected_attributes: pd.DataFrame) -> np.ndarray:
        """
        Apply group-specific thresholds to predictions.
        
        Args:
            y_pred_proba: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Adjusted binary predictions
        """
        # Start with base threshold predictions
        y_pred = (y_pred_proba >= self.base_threshold).astype(int)
        
        # Apply group-specific thresholds for each protected attribute
        for col, thresholds in self.group_thresholds.items():
            if col not in protected_attributes.columns:
                continue
                
            for group, threshold in thresholds.items():
                group_mask = protected_attributes[col] == group
                y_pred[group_mask] = (y_pred_proba[group_mask] >= threshold).astype(int)
        
        return y_pred
    
    def save(self, filepath: str) -> None:
        """Save the threshold configuration to a file."""
        import json
        
        config = {
            "fairness_metric": self.fairness_metric,
            "base_threshold": self.base_threshold,
            "group_thresholds": self.group_thresholds
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved threshold configuration to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load threshold configuration from a file."""
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.fairness_metric = config["fairness_metric"]
        self.base_threshold = config["base_threshold"]
        self.group_thresholds = config["group_thresholds"]
        
        logger.info(f"Loaded threshold configuration from {filepath}")


class RejectionOptionClassifier:
    """
    Implements the Rejection Option Classification post-processing technique.
    
    This technique identifies instances near the decision boundary that may
    contribute to discrimination and defers decisions on these instances.
    """
    
    def __init__(self, uncertainty_threshold: float = 0.15,
                fairness_gap_threshold: float = 0.1):
        """
        Initialize the rejection option classifier.
        
        Args:
            uncertainty_threshold: Threshold for prediction uncertainty
                (closer to 0.5 is more uncertain)
            fairness_gap_threshold: Threshold for acceptable fairness gap
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.fairness_gap_threshold = fairness_gap_threshold
        
    def adjust(self, y_pred_proba: np.ndarray, 
              protected_attributes: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rejection option classification.
        
        Args:
            y_pred_proba: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Tuple of (adjusted predictions, rejection mask)
        """
        # Calculate uncertainty (distance from 0.5)
        uncertainty = 0.5 - np.abs(y_pred_proba - 0.5)
        
        # Initialize rejection mask
        rejection_mask = uncertainty >= self.uncertainty_threshold
        
        # Default predictions using 0.5 threshold
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Analyze each protected attribute
        for col in protected_attributes.columns:
            unique_groups = protected_attributes[col].unique()
            
            # Skip if there's only one group
            if len(unique_groups) <= 1:
                continue
            
            # Calculate positive prediction rate for each group
            group_rates = {}
            for group in unique_groups:
                group_mask = protected_attributes[col] == group
                group_pred = (y_pred_proba[group_mask] >= 0.5).astype(int)
                group_rates[group] = np.mean(group_pred)
            
            # Find maximum fairness gap
            max_gap = max(group_rates.values()) - min(group_rates.values())
            
            # If gap exceeds threshold, apply rejection to uncertain predictions
            if max_gap > self.fairness_gap_threshold:
                logger.info(f"Fairness gap for {col} is {max_gap:.3f}, applying rejection option")
                
                # Find advantaged and disadvantaged groups
                advantaged = max(group_rates.items(), key=lambda x: x[1])[0]
                disadvantaged = min(group_rates.items(), key=lambda x: x[1])[0]
                
                # For advantaged group, reject uncertain positive predictions
                adv_mask = (protected_attributes[col] == advantaged) & (y_pred == 1) & (uncertainty >= self.uncertainty_threshold)
                rejection_mask = rejection_mask | adv_mask
                
                # For disadvantaged group, reject uncertain negative predictions
                disadv_mask = (protected_attributes[col] == disadvantaged) & (y_pred == 0) & (uncertainty >= self.uncertainty_threshold)
                rejection_mask = rejection_mask | disadv_mask
        
        # Set rejected predictions to -1 (or other special value)
        y_pred_adjusted = y_pred.copy()
        y_pred_adjusted[rejection_mask] = -1
        
        return y_pred_adjusted, rejection_mask


# Example usage if run as script
if __name__ == "__main__":
    print("Fairness post-processing module. Import this in your main script.")
```

#### 1.2 Add Post-Processing to Enhanced Sentiment Analysis

Modify `enhanced_sentiment_analysis.py` to include post-processing in the prediction pipeline:

```python
# Add to imports at the top of the file
from fairness_postprocessing import ThresholdOptimizer, RejectionOptionClassifier

# Add to EnhancedSentimentAnalyzer class
class EnhancedSentimentAnalyzer:
    # ... existing code ...
    
    def __init__(self, model_type='logistic', use_advanced_features=True, 
                fairness_postprocessing=None):
        # ... existing initialization ...
        self.fairness_postprocessor = None
        self.fairness_postprocessing = fairness_postprocessing
        
        # Initialize post-processor if specified
        if fairness_postprocessing:
            if fairness_postprocessing == 'threshold_optimization':
                self.fairness_postprocessor = ThresholdOptimizer()
            elif fairness_postprocessing == 'rejection_option':
                self.fairness_postprocessor = RejectionOptionClassifier()
    
    # Add this method to the class
    def fit_fairness_postprocessor(self, texts, labels, demographic_df):
        """
        Train the fairness post-processor using validation data.
        
        Args:
            texts: List of text samples
            labels: True labels
            demographic_df: DataFrame with demographic information
        """
        if not self.fairness_postprocessor:
            return
            
        # Transform texts to features
        X = self.transform(texts)
        
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Fit the post-processor
        self.fairness_postprocessor.fit(
            np.array(labels),
            y_pred_proba,
            demographic_df
        )
        
        logging.info(f"Fitted fairness post-processor: {self.fairness_postprocessing}")
    
    # Modify the predict method to include post-processing
    def predict(self, texts, demographic_df=None):
        """
        Predict sentiment with fairness post-processing if available.
        
        Args:
            texts: List of text samples
            demographic_df: Optional DataFrame with demographic information
        
        Returns:
            Array of predictions (0 for negative, 1 for positive)
        """
        # Transform texts to features
        X = self.transform(texts)
        
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # If no post-processor or no demographic info, use standard threshold
        if not self.fairness_postprocessor or demographic_df is None:
            return (y_pred_proba >= 0.5).astype(int)
        
        # Apply post-processing
        if isinstance(self.fairness_postprocessor, ThresholdOptimizer):
            return self.fairness_postprocessor.adjust(y_pred_proba, demographic_df)
        elif isinstance(self.fairness_postprocessor, RejectionOptionClassifier):
            predictions, rejection_mask = self.fairness_postprocessor.adjust(
                y_pred_proba, demographic_df
            )
            # Log rejection rate
            rejection_rate = rejection_mask.mean() * 100
            logging.info(f"Rejection rate: {rejection_rate:.2f}%")
            return predictions
```

#### 1.3 Update Command-Line Interface in run_enhanced_training.py

Add fairness post-processing options to the argument parser:

```python
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run enhanced sentiment analysis training')
    
    # ... existing arguments ...
    
    # Add fairness post-processing options
    parser.add_argument('--fairness_postprocessing', type=str, 
                      choices=['threshold_optimization', 'rejection_option', 'none'],
                      default='none',
                      help='Fairness post-processing method to apply')
    parser.add_argument('--fairness_metric', type=str,
                      choices=['demographic_parity', 'equalized_odds', 'equal_opportunity'],
                      default='equalized_odds',
                      help='Fairness metric to optimize for in post-processing')
    
    return parser.parse_args()
```

#### 1.4 Update the Training Function in run_enhanced_training.py

Modify `run_traditional_training` function to include post-processing:

```python
def run_traditional_training(df, args, output_dir):
    """Run traditional ML model training with enhancements."""
    # ... existing code ...
    
    # Initialize model with post-processing if requested
    fairness_postprocessing = None
    if args.fairness_postprocessing != 'none':
        fairness_postprocessing = args.fairness_postprocessing
    
    analyzer = EnhancedSentimentAnalyzer(
        model_type=args.model_type if args.model_type != 'logistic' else 'logistic',
        use_advanced_features=use_advanced_features,
        fairness_postprocessing=fairness_postprocessing
    )
    
    # ... existing code for bias mitigation ...
    
    # Train model
    logger.info("Training enhanced traditional model...")
    metrics, X_test, y_test, y_pred = analyzer.train(
        texts=df['text'].tolist(),
        labels=df['target'].tolist(),
        test_size=0.2,
        perform_cv=True
    )
    
    # If post-processing is enabled, fit the post-processor
    if fairness_postprocessing != 'none':
        # Get validation indices (last 10% of training data)
        val_size = int(len(df) * 0.1)
        val_indices = list(range(len(df)))[-(val_size + int(len(df) * 0.2)):-int(len(df) * 0.2)]
        val_df = df.iloc[val_indices].copy()
        
        logger.info(f"Fitting fairness post-processor with {len(val_df)} validation samples")
        analyzer.fit_fairness_postprocessor(
            texts=val_df['text'].tolist(),
            labels=val_df['target'].tolist(),
            demographic_df=val_df[['gender', 'age_group', 'location']]
        )
        
        # Get test indices
        test_indices = list(range(len(df)))[int(len(df) * 0.8):]
        test_df = df.iloc[test_indices].copy()
        
        # Re-predict with post-processing
        y_pred_post = analyzer.predict(
            texts=test_df['text'].tolist(),
            demographic_df=test_df[['gender', 'age_group', 'location']]
        )
        
        # Compare original and post-processed predictions
        original_accuracy = metrics['accuracy']
        post_accuracy = np.mean(y_pred_post[y_pred_post != -1] == test_df['target'][y_pred_post != -1])
        
        logger.info(f"Original accuracy: {original_accuracy:.4f}")
        logger.info(f"Post-processed accuracy: {post_accuracy:.4f}")
        
        # If rejection option was used, report rejection rate
        if fairness_postprocessing == 'rejection_option':
            rejection_rate = np.mean(y_pred_post == -1) * 100
            logger.info(f"Rejection rate: {rejection_rate:.2f}%")
            
            # Update predictions, excluding rejections
            valid_mask = y_pred_post != -1
            y_pred = y_pred_post[valid_mask]
            test_df = test_df[valid_mask].reset_index(drop=True)
        else:
            y_pred = y_pred_post
    
    # ... rest of the existing code ...
```

## 2. Explainability Enhancements

Explainability techniques help stakeholders understand model predictions and identify potential sources of bias. By enhancing the existing explainability capabilities, we can provide fairness-aware explanations that highlight how protected attributes might influence predictions.

### Overview of Enhancements

Building on the existing explainability logic, we will implement:

1. **Fairness-aware feature importance**: Highlighting demographic-sensitive features
2. **Counterfactual explanations**: Showing how predictions would change with demographic shifts
3. **Disparate impact visualization**: Identifying which features contribute most to fairness concerns

These enhancements will work with both LIME and SHAP explainers that may already exist in the codebase.

### Implementation Steps

#### 2.1 Create FairnessExplainer Extension Module

Create a new file `fairness_explainer.py` that extends the existing explainability framework:

```python
#!/usr/bin/env python
"""
Fairness-Aware Explainability Extension

This module extends existing explainability methods (LIME, SHAP) with
fairness-specific enhancements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FairnessAwareExplainer:
    """
    Extends explainers with fairness-aware capabilities.
    
    This class wraps existing explainers (LIME, SHAP) and adds
    fairness-specific functionality such as:
    - Highlighting demographic-sensitive features
    - Generating counterfactual explanations
    - Visualizing disparate impact of features
    """
    
    def __init__(self, base_explainer, demographic_terms=None):
        """
        Initialize the fairness-aware explainer.
        
        Args:
            base_explainer: The base explainer (LIME or SHAP)
            demographic_terms: List of demographic-sensitive terms to highlight
        """
        self.base_explainer = base_explainer
        
        # Default demographic terms if none provided
        self.demographic_terms = demographic_terms or {
            'gender': [
                'he', 'she', 'his', 'her', 'him', 'man', 'woman', 'male', 'female',
                'boy', 'girl', 'guys', 'ladies', 'gentleman', 'gentlemen', 'lady'
            ],
            'age': [
                'young', 'old', 'elderly', 'teenager', 'teenage', 'adolescent', 
                'senior', 'youth', 'adult', 'kid', 'child', 'boomer', 'millennial'
            ],
            'race/ethnicity': [
                'black', 'white', 'asian', 'hispanic', 'latino', 'latina', 'african',
                'european', 'middle eastern', 'native', 'indigenous'
            ]
        }
    
    def explain_instance(self, text, predict_fn, num_features=10):
        """
        Generate fairness-aware explanation for a text instance.
        
        Args:
            text: The text to explain
            predict_fn: Prediction function
            num_features: Number of features to include in explanation
            
        Returns:
            Dict containing the explanation with fairness enhancements
        """
        # Get the base explanation
        base_explanation = self.base_explainer.explain_instance(
            text, predict_fn, num_features=num_features
        )
        
        # Extend with fairness awareness
        fairness_explanation = self._enhance_explanation(base_explanation, text)
        
        return fairness_explanation
    
    def _enhance_explanation(self, explanation, text):
        """Add fairness-aware enhancements to the explanation."""
        # Extract feature importance from base explanation
        feature_importance = {}
        for feature, weight in explanation:
            feature_importance[feature] = weight
        
        # Identify demographic-sensitive features
        demographic_features = self._identify_demographic_features(
            list(feature_importance.keys())
        )
        
        # Create enhanced explanation
        enhanced = {
            "base_explanation": explanation,
            "demographic_features": demographic_features,
            "fairness_warnings": self._generate_fairness_warnings(
                feature_importance, demographic_features
            )
        }
        
        return enhanced
    
    def _identify_demographic_features(self, features):
        """
        Identify features that may be related to demographic attributes.
        
        Args:
            features: List of features from the explanation
            
        Returns:
            Dict mapping demographic categories to related features
        """
        demographic_features = {}
        
        # Check each demographic category
        for category, terms in self.demographic_terms.items():
            related_features = []
            
            for feature in features:
                # Check if any demographic term is part of this feature
                if any(term in feature.lower() for term in terms):
                    related_features.append(feature)
            
            if related_features:
                demographic_features[category] = related_features
        
        return demographic_features
    
    def _generate_fairness_warnings(self, feature_importance, demographic_features):
        """
        Generate warnings about potential fairness issues.
        
        Args:
            feature_importance: Dict mapping features to importance values
            demographic_features: Dict mapping demographic categories to features
            
        Returns:
            List of fairness warnings
        """
        warnings = []
        
        # Check each demographic category
        for category, features in demographic_features.items():
            # Calculate total importance of demographic features
            total_importance = sum(abs(feature_importance.get(f, 0)) for f in features)
            
            # If demographic features have high importance, generate warning
            if total_importance > 0.2:  # Threshold can be adjusted
                warnings.append({
                    "category": category,
                    "importance": total_importance,
                    "features": features,
                    "message": f"High influence of {category}-related terms detected."
                })
        
        return warnings
    
    def plot_fairness_aware_explanation(self, explanation, save_path=None):
        """
        Create fairness-aware visualization of feature importance.
        
        Args:
            explanation: Enhanced explanation from explain_instance
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        base_explanation = explanation["base_explanation"]
        demographic_features = explanation["demographic_features"]
        
        # Extract features and weights
        features = []
        weights = []
        colors = []
        
        for feature, weight in base_explanation:
            features.append(feature)
            weights.append(weight)
            
            # Check if feature is demographic-related
            is_demographic = any(
                feature in demo_features 
                for demo_features in demographic_features.values()
            )
            
            # Set color based on demographic relation
            if is_demographic:
                colors.append('red')  # Demographic features in red
            else:
                colors.append('blue')  # Other features in blue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, weights, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Display top-to-bottom
        ax.set_xlabel('Feature Importance')
        ax.set_title('Fairness-Aware Feature Importance')
        
        # Add legend
        import matplotlib.patches as mpatches
        demographic_patch = mpatches.Patch(color='red', label='Demographic-sensitive')
        other_patch = mpatches.Patch(color='blue', label='Other features')
        ax.legend(handles=[demographic_patch, other_patch])
        
        # Add fairness warnings
        if explanation["fairness_warnings"]:
            warning_text = "Fairness Warnings:\n"
            for warning in explanation["fairness_warnings"]:
                warning_text += f"- {warning['message']}\n"
            
            plt.figtext(0.5, 0.01, warning_text, ha='center', 
                      bbox={'facecolor':'lightyellow', 'alpha':0.5, 'pad':5})
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved fairness-aware explanation to {save_path}")
        
        return fig
```

#### 2.2 Implement Counterfactual Explanations

Add the following class to `fairness_explainer.py` to generate counterfactual explanations:

```python
class CounterfactualExplainer:
    """
    Generates counterfactual explanations for fairness assessment.
    
    Counterfactual explanations show how the prediction would change
    if demographic attributes or terms were modified.
    """
    
    def __init__(self, model, text_processor=None):
        """
        Initialize the counterfactual explainer.
        
        Args:
            model: The sentiment analysis model
            text_processor: Function to preprocess text before prediction
        """
        self.model = model
        self.text_processor = text_processor or (lambda x: x)
        
        # Define demographic replacements
        self.gender_replacements = {
            'he': 'she', 'she': 'he',
            'his': 'her', 'her': 'his',
            'him': 'her', 'himself': 'herself', 'herself': 'himself',
            'man': 'woman', 'woman': 'man',
            'male': 'female', 'female': 'male',
            'boy': 'girl', 'girl': 'boy',
            'Mr.': 'Ms.', 'Ms.': 'Mr.', 'Mrs.': 'Mr.',
            'guys': 'ladies', 'ladies': 'guys',
            'men': 'women', 'women': 'men',
            'gentleman': 'lady', 'lady': 'gentleman'
        }
        
        self.age_replacements = {
            'young': 'old', 'old': 'young',
            'elderly': 'young', 'teenager': 'senior', 'senior': 'teenager',
            'youth': 'senior', 'kid': 'adult', 'adult': 'kid',
            'child': 'adult', 'boomer': 'millennial', 'millennial': 'boomer'
        }
    
    def generate_counterfactuals(self, text, demographic_type='gender'):
        """
        Generate counterfactual text by modifying demographic terms.
        
        Args:
            text: Original text
            demographic_type: Type of demographic to modify ('gender' or 'age')
            
        Returns:
            Tuple of (original prediction, counterfactual text, counterfactual prediction)
        """
        # Preprocess text
        processed_text = self.text_processor(text)
        
        # Get original prediction
        original_prediction = self.predict(processed_text)
        
        # Generate counterfactual text
        if demographic_type == 'gender':
            counterfactual_text = self._swap_gender_terms(text)
        elif demographic_type == 'age':
            counterfactual_text = self._swap_age_terms(text)
        else:
            raise ValueError(f"Unsupported demographic type: {demographic_type}")
        
        # Preprocess counterfactual text
        processed_counterfactual = self.text_processor(counterfactual_text)
        
        # Get counterfactual prediction
        counterfactual_prediction = self.predict(processed_counterfactual)
        
        return {
            'original_text': text,
            'original_prediction': original_prediction,
            'counterfactual_text': counterfactual_text,
            'counterfactual_prediction': counterfactual_prediction,
            'prediction_difference': counterfactual_prediction - original_prediction,
            'demographic_type': demographic_type
        }
    
    def predict(self, text):
        """Make a prediction for a text."""
        if hasattr(self.model, 'predict_proba'):
            # If model has predict_proba, use it to get probability of positive class
            try:
                # Handle different input formats (list vs single string)
                if isinstance(text, str):
                    proba = self.model.predict_proba([text])[0][1]
                else:
                    proba = self.model.predict_proba(text)[0][1]
                return proba
            except:
                # Fallback to binary prediction
                if isinstance(text, str):
                    return self.model.predict([text])[0]
                else:
                    return self.model.predict(text)[0]
        else:
            # If model only has predict, use it
            if isinstance(text, str):
                return self.model.predict([text])[0]
            else:
                return self.model.predict(text)[0]
    
    def _swap_gender_terms(self, text):
        """Swap gender-specific terms in the text."""
        words = text.split()
        swapped_words = []
        
        for word in words:
            # Check if word needs to be lowercased for comparison
            lower_word = word.lower()
            
            # Check if we have a replacement
            if lower_word in self.gender_replacements:
                replacement = self.gender_replacements[lower_word]
                
                # Preserve capitalization
                if word.istitle():
                    replacement = replacement.title()
                elif word.isupper():
                    replacement = replacement.upper()
                
                swapped_words.append(replacement)
            else:
                swapped_words.append(word)
        
        return ' '.join(swapped_words)
    
    def _swap_age_terms(self, text):
        """Swap age-specific terms in the text."""
        words = text.split()
        swapped_words = []
        
        for word in words:
            # Check if word needs to be lowercased for comparison
            lower_word = word.lower()
            
            # Check if we have a replacement
            if lower_word in self.age_replacements:
                replacement = self.age_replacements[lower_word]
                
                # Preserve capitalization
                if word.istitle():
                    replacement = replacement.title()
                elif word.isupper():
                    replacement = replacement.upper()
                
                swapped_words.append(replacement)
            else:
                swapped_words.append(word)
        
        return ' '.join(swapped_words)
    
    def plot_counterfactual_comparison(self, counterfactual_result, save_path=None):
        """
        Create visualization comparing original and counterfactual predictions.
        
        Args:
            counterfactual_result: Result from generate_counterfactuals
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Extract data
        original_text = counterfactual_result['original_text']
        counterfactual_text = counterfactual_result['counterfactual_text']
        original_pred = counterfactual_result['original_prediction']
        counterfactual_pred = counterfactual_result['counterfactual_prediction']
        demographic_type = counterfactual_result['demographic_type']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot original prediction
        ax1.barh(['Original'], [original_pred], color='blue')
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Positive Sentiment Probability')
        ax1.set_title('Original Text')
        ax1.text(0.02, 0.5, f"Text: {original_text[:50]}...", 
               transform=ax1.transAxes, fontsize=10, verticalalignment='center')
        
        # Plot counterfactual prediction
        ax2.barh(['Counterfactual'], [counterfactual_pred], color='orange')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Positive Sentiment Probability')
        ax2.set_title(f'Counterfactual Text ({demographic_type} swap)')
        ax2.text(0.02, 0.5, f"Text: {counterfactual_text[:50]}...", 
               transform=ax2.transAxes, fontsize=10, verticalalignment='center')
        
        # Add difference annotation
        diff = counterfactual_pred - original_pred
        diff_text = f"Difference: {diff:.3f}"
        fig.suptitle(f"Counterfactual Explanation ({demographic_type})\n{diff_text}", 
                   fontsize=14)
        
        # Add fairness assessment
        if abs(diff) > 0.1:
            warning = (f"Warning: Large difference ({abs(diff):.3f}) in predictions when "
                     f"{demographic_type} terms are changed. This may indicate bias.")
            plt.figtext(0.5, 0.01, warning, ha='center', fontsize=12,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved counterfactual explanation to {save_path}")
        
        return fig
```

#### 2.3 Integrate the Explainers with the Existing System

Add the following method to the `EnhancedSentimentAnalyzer` class in `enhanced_sentiment_analysis.py`:

```python
def explain_prediction(self, text, demographic_info=None, explanation_type='all', 
                     num_features=10, save_path=None):
    """
    Generate a fairness-aware explanation for a prediction.
    
    Args:
        text: Text to explain
        demographic_info: Optional demographic information
        explanation_type: Type of explanation ('feature', 'counterfactual', or 'all')
        num_features: Number of features to include in explanation
        save_path: Optional path to save visualizations
        
    Returns:
        Dict containing explanation information
    """
    # Import required modules
    from fairness_explainer import FairnessAwareExplainer, CounterfactualExplainer
    
    # Create base explainer (use existing explainer if available)
    if hasattr(self, 'explainer'):
        base_explainer = self.explainer
    else:
        try:
            # Try to use LIME explainer
            from lime.lime_text import LimeTextExplainer
            
            # Define LIME class names
            class_names = ['Negative', 'Positive']
            
            # Create LIME explainer
            lime_explainer = LimeTextExplainer(class_names=class_names)
            base_explainer = lime_explainer
        except ImportError:
            logging.warning("LIME not available. Using basic feature importance.")
            base_explainer = None
    
    # Initialize results dictionary
    explanation = {
        'text': text,
        'prediction': None,
        'feature_importance': None,
        'counterfactual': None,
        'fairness_warnings': []
    }
    
    # Get prediction
    X = self.transform([text])
    prediction_proba = self.model.predict_proba(X)[0]
    prediction = int(prediction_proba[1] > 0.5)
    confidence = prediction_proba[prediction]
    
    explanation['prediction'] = {
        'class': prediction,
        'class_name': 'Positive' if prediction == 1 else 'Negative',
        'confidence': confidence
    }
    
    # Get feature importance explanation if requested
    if explanation_type in ['feature', 'all'] and base_explainer:
        fairness_explainer = FairnessAwareExplainer(base_explainer)
        
        # Define prediction function for explainer
        def predict_fn(texts):
            X_batch = self.transform(texts)
            return self.model.predict_proba(X_batch)
        
        # Generate fairness-aware explanation
        feature_exp = fairness_explainer.explain_instance(
            text, predict_fn, num_features=num_features
        )
        
        explanation['feature_importance'] = feature_exp
        explanation['fairness_warnings'].extend(feature_exp['fairness_warnings'])
        
        # Generate visualization if save path provided
        if save_path:
            feature_path = save_path.replace('.png', '_features.png')
            fairness_explainer.plot_fairness_aware_explanation(
                feature_exp, save_path=feature_path
            )
    
    # Get counterfactual explanation if requested
    if explanation_type in ['counterfactual', 'all']:
        counterfactual_explainer = CounterfactualExplainer(
            model=self, text_processor=self.transform
        )
        
        # Generate gender counterfactual
        gender_cf = counterfactual_explainer.generate_counterfactuals(
            text, demographic_type='gender'
        )
        
        # Generate age counterfactual
        age_cf = counterfactual_explainer.generate_counterfactuals(
            text, demographic_type='age'
        )
        
        explanation['counterfactual'] = {
            'gender': gender_cf,
            'age': age_cf
        }
        
        # Check for large differences and add warnings
        for cf_type, cf_result in explanation['counterfactual'].items():
            diff = abs(cf_result['prediction_difference'])
            if diff > 0.1:
                explanation['fairness_warnings'].append({
                    'category': cf_type,
                    'message': f"Large prediction difference ({diff:.3f}) when {cf_type} terms are changed.",
                    'severity': 'high' if diff > 0.2 else 'medium'
                })
        
        # Generate visualizations if save path provided
        if save_path:
            gender_path = save_path.replace('.png', '_gender_cf.png')
            age_path = save_path.replace('.png', '_age_cf.png')
            
            counterfactual_explainer.plot_counterfactual_comparison(
                gender_cf, save_path=gender_path
            )
            
            counterfactual_explainer.plot_counterfactual_comparison(
                age_cf, save_path=age_path
            )
    
    return explanation
```

#### 2.4 Update Command-Line Interface to Include Explainability Options

Add the following options to the `parse_args` function in `run_enhanced_training.py`:

```python
# Add explainability options
parser.add_argument('--generate_explanations', action='store_true',
                  help='Generate fairness-aware explanations for test examples')
parser.add_argument('--explanation_samples', type=int, default=5,
                  help='Number of test examples to explain')
parser.add_argument('--explanation_types', type=str,
                  choices=['feature', 'counterfactual', 'all'],
                  default='all',
                  help='Types of explanations to generate')
```

#### 2.5 Update the Training Function to Generate Explanations

Add the following code to the end of the `run_traditional_training` function in `run_enhanced_training.py`:

```python
# Generate explanations if requested
if args.generate_explanations:
    logger.info("Generating fairness-aware explanations for test examples...")
    
    # Create output directory
    explanations_dir = output_dir / "explanations"
    explanations_dir.mkdir(exist_ok=True)
    
    # Select a subset of test samples
    n_samples = min(args.explanation_samples, len(test_df))
    explanation_indices = np.random.choice(
        len(test_df), n_samples, replace=False
    )
    
    # Generate explanations
    for i, idx in enumerate(explanation_indices):
        sample = test_df.iloc[idx]
        text = sample['text']
        true_label = sample['target']
        
        logger.info(f"Generating explanation for example {i+1}/{n_samples}")
        
        # Generate explanation
        explanation = analyzer.explain_prediction(
            text=text,
            demographic_info=sample[['gender', 'age_group', 'location']],
            explanation_type=args.explanation_types,
            save_path=str(explanations_dir / f"example_{i+1}.png")
        )
        
        # Save explanation metadata
        with open(explanations_dir / f"example_{i+1}_metadata.json", 'w') as f:
            import json
            
            # Remove non-serializable components
            serializable_explanation = {
                'text': explanation['text'],
                'prediction': explanation['prediction'],
                'fairness_warnings': explanation['fairness_warnings']
            }
            
            # Include counterfactual results if available
            if explanation['counterfactual']:
                serializable_explanation['counterfactual'] = explanation['counterfactual']
            
            json.dump(serializable_explanation, f, indent=2)
    
    logger.info(f"Explanations saved to {explanations_dir}")
```

#### 2.6 Add a Function to Generate an Explanations Report

Create a new function in `run_enhanced_training.py` to generate an explanations report:

```python
def generate_explanations_report(explanations_dir):
    """
    Generate a summary report of fairness-aware explanations.
    
    Args:
        explanations_dir: Path to the directory containing explanations
    
    Returns:
        Path to the generated report
    """
    import json
    import glob
    from pathlib import Path
    
    explanations_dir = Path(explanations_dir)
    metadata_files = glob.glob(str(explanations_dir / "*_metadata.json"))
    
    if not metadata_files:
        logger.warning("No explanation metadata files found.")
        return None
    
    # Load all explanations
    explanations = []
    for file_path in metadata_files:
        with open(file_path, 'r') as f:
            explanation = json.load(f)
            explanations.append(explanation)
    
    # Generate Markdown report
    report_lines = [
        "# Fairness-Aware Explanations Report",
        "",
        f"Number of examples analyzed: {len(explanations)}",
        "",
        "## Summary of Fairness Warnings",
        ""
    ]
    
    # Aggregate warnings
    warning_counts = {}
    for explanation in explanations:
        for warning in explanation.get('fairness_warnings', []):
            category = warning.get('category', 'unknown')
            message = warning.get('message', 'Unknown warning')
            key = f"{category}: {message}"
            warning_counts[key] = warning_counts.get(key, 0) + 1
    
    # Add warnings summary
    if warning_counts:
        report_lines.append("| Warning | Count |")
        report_lines.append("| ------- | ----- |")
        for warning, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"| {warning} | {count} |")
    else:
        report_lines.append("No fairness warnings detected.")
    
    report_lines.extend([
        "",
        "## Counterfactual Analysis",
        ""
    ])
    
    # Analyze counterfactual results
    cf_diffs = {
        'gender': [],
        'age': []
    }
    
    for explanation in explanations:
        if explanation.get('counterfactual'):
            for cf_type, cf_result in explanation['counterfactual'].items():
                if cf_type in cf_diffs:
                    cf_diffs[cf_type].append(cf_result.get('prediction_difference', 0))
    
    # Add counterfactual summary
    for cf_type, diffs in cf_diffs.items():
        if diffs:
            avg_diff = sum(abs(d) for d in diffs) / len(diffs)
            max_diff = max(abs(d) for d in diffs)
            report_lines.extend([
                f"### {cf_type.title()} Counterfactuals",
                "",
                f"- Number of examples: {len(diffs)}",
                f"- Average absolute difference: {avg_diff:.4f}",
                f"- Maximum absolute difference: {max_diff:.4f}",
                ""
            ])
    
    # Write report
    report_path = explanations_dir / "explanations_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Explanations report saved to {report_path}")
    return report_path
```

Then call this function at the end of the `run_traditional_training` function:

```python
# Generate explanations report
if args.generate_explanations:
    generate_explanations_report(explanations_dir)
```

### Integration and Testing

To test the explainability implementation:

1. **Unit Testing**:
   - Test the FairnessAwareExplainer on simple examples with known demographic terms
   - Verify that counterfactual generation correctly swaps demographic terms
   - Ensure warnings are generated when demographic features have high importance

2. **Integration Testing**:
   - Run the system with explanations enabled
   - Verify that explanations highlight demographic features
   - Check that counterfactual examples show the impact of changing demographic terms

3. **Example Command**:
   ```bash
   python run_enhanced_training.py --model_type logistic --train_mode traditional \
     --fairness_evaluation --bias_mitigation --max_samples 10000 \
     --generate_explanations --explanation_samples 10
   ```

4. **Expected Results**:
   - Detailed explanations for selected test examples
   - Visualization of feature importance with demographic features highlighted
   - Counterfactual examples showing the impact of changing demographic terms
   - Summary report of fairness warnings and demographic sensitivity
```

## 3. Disparate Impact Mitigation

Disparate impact mitigation techniques aim to reduce the difference in outcomes between different demographic groups. These techniques are particularly valuable because they can be applied to existing trained models without retraining.

### Implementation Steps

#### 3.1 Create the `disparate_impact_mitigation.py` Module

```python
#!/usr/bin/env python
"""
Disparate Impact Mitigation Module

This module implements techniques to mitigate disparate impact in sentiment analysis predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DisparateImpactMitigator:
    """Implements techniques to mitigate disparate impact in sentiment analysis predictions."""
    
    def __init__(self, fairness_metric: str = "equalized_odds"):
        """
        Initialize the disparate impact mitigator.
        
        Args:
            fairness_metric: The fairness metric to optimize for.
                Options: "demographic_parity", "equalized_odds", "equal_opportunity"
        """
        self.fairness_metric = fairness_metric
    
    def adjust(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
              protected_attributes: pd.DataFrame) -> np.ndarray:
        """
        Apply disparate impact mitigation techniques to predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            protected_attributes: DataFrame with protected attribute columns
            
        Returns:
            Adjusted binary predictions
        """
        # Start with base threshold predictions
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Apply disparate impact mitigation techniques
        if self.fairness_metric == "demographic_parity":
            self._apply_demographic_parity(y_true, y_pred_proba, protected_attributes)
        elif self.fairness_metric == "equalized_odds":
            self._apply_equalized_odds(y_true, y_pred_proba, protected_attributes)
        elif self.fairness_metric == "equal_opportunity":
            self._apply_equal_opportunity(y_true, y_pred_proba, protected_attributes)
        
        return y_pred
    
    def _apply_demographic_parity(self, y_true, y_pred_proba, protected_attributes):
        """Apply demographic parity techniques."""
        # Implement demographic parity techniques
        pass
    
    def _apply_equalized_odds(self, y_true, y_pred_proba, protected_attributes):
        """Apply equalized odds techniques."""
        # Implement equalized odds techniques
        pass
    
    def _apply_equal_opportunity(self, y_true, y_pred_proba, protected_attributes):
        """Apply equal opportunity techniques."""
        # Implement equal opportunity techniques
        pass

# Example usage if run as script
if __name__ == "__main__":
    print("Disparate impact mitigation module. Import this in your main script.")
```

### Integration and Testing

To test the disparate impact mitigation implementation:

1. **Unit Testing**:
   - Test the mitigation techniques on a synthetic dataset with known bias
   - Verify that the techniques reduce the difference in outcomes between different demographic groups

2. **Integration Testing**:
   - Run the system with and without mitigation
   - Compare fairness metrics before and after mitigation
   - Verify that accuracy doesn't degrade significantly

3. **Example Command**:
   ```bash
   python run_enhanced_training.py --model_type logistic --train_mode traditional \
     --fairness_evaluation --bias_mitigation --max_samples 80000 \
     --fairness_postprocessing threshold_optimization --fairness_metric equalized_odds
   ```

4. **Expected Results**:
   - Improved fairness metrics (especially equalized odds)
   - Minimal change in overall accuracy
   - Detailed fairness evaluation report showing the effects of mitigation 
```

## 4. Conclusion and Next Steps

This implementation guide has outlined detailed steps for enhancing our sentiment analysis system with advanced fairness techniques. By implementing these enhancements, we can create a more robust and equitable system that serves all demographic groups fairly.

### Key Benefits

1. **Fairness Guarantees**: The post-processing techniques provide mathematical guarantees for fairness metrics such as equalized odds and demographic parity.

2. **Transparency**: The explainability enhancements make model decisions more transparent and help identify potential sources of bias.

3. **Continuous Improvement**: The feedback mechanisms enable ongoing refinement of the system based on real-world performance and user feedback.

### Implementation Roadmap

1. **Phase 1 (Current Focus)**:
   - Post-processing adjustments for immediate fairness improvements
   - Explainability enhancements for model transparency
   - Counterfactual testing for robustness evaluation

2. **Phase 2 (Future Work)**:
   - User feedback integration via API endpoints
   - Interactive visualization tools for exploring fairness metrics
   - Multiple model ensemble for improved performance and fairness

3. **Phase 3 (Long-term Vision)**:
   - Causal fairness assessment
   - Reinforcement learning from human feedback
   - Academic publication of results and open-source contributions

### Getting Started

To begin implementing these enhancements, follow these steps:

1. Create the required modules:
   - `fairness_postprocessing.py` for post-processing techniques
   - `fairness_explainer.py` for explainability enhancements

2. Modify the existing codebase:
   - Update `enhanced_sentiment_analysis.py` with new methods
   - Extend `run_enhanced_training.py` with new command-line options

3. Test the implementation:
   - Start with small-scale tests on synthetic data
   - Gradually scale up to the full dataset
   - Compare fairness metrics before and after enhancements

By implementing these fairness enhancements, we can create a sentiment analysis system that not only achieves high accuracy but also ensures equitable treatment across all demographic groups.