#!/usr/bin/env python
"""
Local Sentiment Fairness Analysis Example

This script demonstrates fairness evaluation on a sentiment analysis task
with a small sample dataset, without requiring Kaggle. It shows:

1. Creating a synthetic sentiment analysis dataset
2. Adding synthetic demographic attributes
3. Training a sentiment analysis model
4. Evaluating fairness metrics across demographic groups
5. Applying fairness mitigation techniques
6. Generating fairness visualizations and reports

This is an example implementation for the WITHIN Ad Score & Account Health Predictor system.
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.fairness import (
    FairnessEvaluator, FairnessResults, FairnessMetric, FairnessThreshold,
    Reweighing, FairDataTransformer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample sentiment data
SAMPLE_DATA = [
    {"text": "I love this product, it's amazing!", "target": 1},
    {"text": "This is great, would recommend to everyone", "target": 1},
    {"text": "Not bad, but could be better", "target": 0},
    {"text": "Absolutely terrible experience, avoid at all costs", "target": 0},
    {"text": "It's okay, nothing special", "target": 0},
    {"text": "Fantastic service and great value", "target": 1},
    {"text": "Disappointed with the quality", "target": 0},
    {"text": "Exceeded my expectations, very happy", "target": 1},
    {"text": "Wouldn't buy again, waste of money", "target": 0},
    {"text": "Average performance for the price", "target": 0},
    {"text": "Best purchase I've made all year!", "target": 1},
    {"text": "Complete garbage, don't waste your time", "target": 0},
    {"text": "Really pleased with this product", "target": 1},
    {"text": "Not worth the money at all", "target": 0},
    {"text": "Exactly what I needed, perfect", "target": 1},
    {"text": "Customer service was horrible", "target": 0},
    {"text": "Highly recommend to anyone", "target": 1},
    {"text": "Just okay, nothing to write home about", "target": 0},
    {"text": "Incredible quality and fast delivery", "target": 1},
    {"text": "Very dissatisfied with my purchase", "target": 0},
]

def create_synthetic_dataset(samples: int = 100) -> pd.DataFrame:
    """Create a synthetic sentiment dataset with demographic attributes.
    
    Args:
        samples: Number of samples to generate
        
    Returns:
        DataFrame with text, sentiment labels, and demographic features
    """
    logger.info(f"Creating synthetic dataset with {samples} samples...")
    
    # Create base dataset from sample data
    np.random.seed(42)
    sample_indices = np.random.randint(0, len(SAMPLE_DATA), size=samples)
    data = [SAMPLE_DATA[i] for i in sample_indices]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add synthetic demographic features
    df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
    df['age_group'] = np.random.choice(['18-25', '26-35', '36-50', '51+'], size=len(df))
    df['location'] = np.random.choice(['Urban', 'Suburban', 'Rural'], size=len(df))
    df['education'] = np.random.choice(['High School', 'College', 'Graduate'], size=len(df))
    
    # Add slight bias toward certain demographics for testing fairness metrics
    # Make females slightly more likely to get positive sentiment
    bias_indices = df[df['gender'] == 'Female'].index
    df.loc[bias_indices, 'target'] = np.where(
        np.random.random(size=len(bias_indices)) < 0.65,  # 65% chance of positive
        1, 
        df.loc[bias_indices, 'target']
    )
    
    # Make urban locations slightly less likely to get positive sentiment
    bias_indices = df[df['location'] == 'Urban'].index
    df.loc[bias_indices, 'target'] = np.where(
        np.random.random(size=len(bias_indices)) < 0.4,  # 40% chance of negative
        0, 
        df.loc[bias_indices, 'target']
    )
    
    logger.info(f"Created dataset with {len(df)} samples")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into training, validation, and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Split into train+val and test
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['target'])
    
    # Further split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=42, stratify=train_val_df['target'])
    
    logger.info(f"Split dataset - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def apply_fairness_mitigation(train_df: pd.DataFrame, protected_attr: str = 'gender', 
                             mitigation_technique: str = 'reweighing') -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply fairness mitigation to the dataset.
    
    Args:
        train_df: Training DataFrame
        protected_attr: Protected attribute to mitigate bias for
        mitigation_technique: Name of mitigation technique to apply
        
    Returns:
        Tuple of (updated_df, sample_weights)
    """
    if mitigation_technique is None:
        logger.info("No fairness mitigation applied")
        return train_df, None
    
    logger.info(f"Applying {mitigation_technique} for protected attribute: {protected_attr}")
    
    X = train_df.copy()
    y = X.pop('target')
    
    # Apply reweighing
    if mitigation_technique == 'reweighing':
        reweigher = Reweighing(protected_attribute=protected_attr)
        reweigher.fit(X, y)
        
        # Get sample weights for training
        sample_weights = reweigher.get_sample_weights(X, y)
        
        # Log weight statistics
        logger.info(f"Reweighing sample weight stats: min={sample_weights.min():.4f}, "
                   f"max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
        
        return train_df, sample_weights
        
    # Apply fair data transformation
    elif mitigation_technique == 'fair_transform':
        transformer = FairDataTransformer(
            protected_attribute=protected_attr,
            method='decorrelation'
        )
        
        # Fit and transform the data
        X_transformed = transformer.fit_transform(X, y)
        
        # Combine transformed features with original labels
        X_transformed['target'] = y.values
        
        logger.info(f"Applied fair data transformation to remove bias for {protected_attr}")
        
        return X_transformed, None
    
    return train_df, None

def train_sentiment_model(train_df: pd.DataFrame, sample_weights: Optional[np.ndarray] = None) -> Pipeline:
    """Train a sentiment analysis model.
    
    Args:
        train_df: Training DataFrame
        sample_weights: Optional sample weights for training
        
    Returns:
        Trained model pipeline
    """
    logger.info("Training sentiment analysis model...")
    
    # Create a pipeline with TF-IDF and logistic regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=200))
    ])
    
    # Extract features and target
    X = train_df['text']
    y = train_df['target']
    
    # Train the model
    pipeline.fit(X, y, **{
        'classifier__sample_weight': sample_weights
    } if sample_weights is not None else {})
    
    logger.info("Model training completed")
    
    return pipeline

def evaluate_model(model: Pipeline, test_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluate the model performance.
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        
    Returns:
        Tuple of (predictions, metrics)
    """
    logger.info("Evaluating model performance...")
    
    # Extract features and target
    X = test_df['text']
    y = test_df['target']
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    # Print classification report
    logger.info(f"Classification Report:\n{classification_report(y, y_pred)}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return y_pred, {'accuracy': accuracy}

def evaluate_fairness(test_df: pd.DataFrame, y_pred: np.ndarray, output_dir: Path) -> Dict[str, FairnessResults]:
    """Evaluate fairness metrics for the model predictions.
    
    Args:
        test_df: Test DataFrame
        y_pred: Model predictions
        output_dir: Directory to save fairness results
        
    Returns:
        Dictionary of fairness results by protected attribute
    """
    logger.info("Evaluating fairness...")
    
    # Create combined DataFrame for fairness evaluation
    eval_df = test_df.copy()
    eval_df['prediction'] = y_pred
    
    # Create output directory for fairness results
    fairness_dir = output_dir / 'fairness'
    fairness_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize fairness evaluator
    evaluator = FairnessEvaluator(
        output_dir=str(fairness_dir),
        threshold=0.1,  # Standard fairness threshold
        save_visualizations=True
    )
    
    # Protected attributes to evaluate
    protected_attributes = ['gender', 'age_group', 'location', 'education']
    
    # Evaluate fairness for each protected attribute
    fairness_results = {}
    for attr in protected_attributes:
        if attr in eval_df.columns:
            logger.info(f"Evaluating fairness for protected attribute: {attr}")
            
            # Run evaluation
            results = evaluator.evaluate(
                df=eval_df,
                protected_attribute=attr,
                target_column='target',
                prediction_column='prediction'
            )
            
            # Store results
            fairness_results[attr] = results
            
            # Log summary
            logger.info(f"\nFairness Evaluation Summary for {attr}:")
            logger.info(results.summary())
    
    # Generate intersectional fairness visualizations
    if all(attr in eval_df.columns for attr in ['gender', 'age_group']):
        logger.info("Generating intersectional fairness visualization: gender × age_group")
        evaluator._plot_intersectional_heatmap(
            df=eval_df,
            attr1='gender',
            attr2='age_group',
            target_column='prediction',
            output_dir=fairness_dir
        )
    
    if all(attr in eval_df.columns for attr in ['gender', 'location']):
        logger.info("Generating intersectional fairness visualization: gender × location")
        evaluator._plot_intersectional_heatmap(
            df=eval_df,
            attr1='gender',
            attr2='location',
            target_column='prediction',
            output_dir=fairness_dir
        )
    
    return fairness_results

def main() -> None:
    """Run the local sentiment fairness analysis example."""
    # Create output directory
    output_dir = Path('sentiment_fairness_example_local')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic dataset with demographic attributes
    df = create_synthetic_dataset(samples=500)
    
    # Split into train, validation, and test sets
    train_df, val_df, test_df = split_dataset(df)
    
    # Apply fairness mitigation (reweighing)
    train_df_mitigated, sample_weights = apply_fairness_mitigation(
        train_df=train_df,
        protected_attr='gender',
        mitigation_technique='reweighing'
    )
    
    # Train sentiment analysis model
    model = train_sentiment_model(train_df_mitigated, sample_weights)
    
    # Evaluate model on test set
    y_pred, metrics = evaluate_model(model, test_df)
    
    # Evaluate fairness
    fairness_results = evaluate_fairness(
        test_df=test_df,
        y_pred=y_pred,
        output_dir=output_dir
    )
    
    # Save fairness results as JSON
    for attr, results in fairness_results.items():
        results_path = output_dir / 'fairness' / f"{attr}_results.json"
        
        with open(results_path, "w") as f:
            import json
            json.dump(results.to_dict(), f, indent=2)
            
        logger.info(f"Saved fairness results to {results_path}")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main() 