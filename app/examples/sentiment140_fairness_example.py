#!/usr/bin/env python
"""
Sentiment140 Dataset Fairness Analysis Example

This script demonstrates how to use the Kaggle dataset integration pipeline
with the Sentiment140 dataset, adding synthetic demographic data for fairness
evaluation. It shows the complete workflow for:

1. Downloading and processing the Sentiment140 dataset from Kaggle
2. Adding synthetic demographic attributes for fairness analysis
3. Training a sentiment analysis model
4. Evaluating fairness metrics across (synthetic) demographic groups
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

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.data_integration.kaggle_pipeline import (
    KaggleDatasetPipeline, DatasetConfig, ProcessedDataset, DatasetCategory
)
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

def add_synthetic_demographics(dataset: ProcessedDataset) -> ProcessedDataset:
    """Add synthetic demographic features for fairness analysis.
    
    Args:
        dataset: Processed dataset without demographic features
        
    Returns:
        Dataset with added synthetic demographic features
    """
    logger.info("Adding synthetic demographic features...")
    
    # Make a copy of the training, validation, and test datasets
    X_train = dataset.X_train.copy()
    X_val = dataset.X_val.copy()
    X_test = dataset.X_test.copy()
    
    # Use a fixed random seed for reproducibility
    np.random.seed(42)
    
    # Add synthetic demographic features to training data
    X_train['gender'] = np.random.choice(['Male', 'Female'], size=len(X_train))
    X_train['age_group'] = np.random.choice(['18-25', '26-35', '36-50', '51+'], size=len(X_train))
    X_train['location'] = np.random.choice(['Urban', 'Suburban', 'Rural'], size=len(X_train))
    
    # Add synthetic demographic features to validation data
    X_val['gender'] = np.random.choice(['Male', 'Female'], size=len(X_val))
    X_val['age_group'] = np.random.choice(['18-25', '26-35', '36-50', '51+'], size=len(X_val))
    X_val['location'] = np.random.choice(['Urban', 'Suburban', 'Rural'], size=len(X_val))
    
    # Add synthetic demographic features to test data
    X_test['gender'] = np.random.choice(['Male', 'Female'], size=len(X_test))
    X_test['age_group'] = np.random.choice(['18-25', '26-35', '36-50', '51+'], size=len(X_test))
    X_test['location'] = np.random.choice(['Urban', 'Suburban', 'Rural'], size=len(X_test))
    
    # Update dataset with new features
    dataset.X_train = X_train
    dataset.X_val = X_val
    dataset.X_test = X_test
    
    # Update protected attributes
    dataset.protected_attributes = {
        'gender': X_test['gender'],
        'age_group': X_test['age_group'],
        'location': X_test['location']
    }
    
    logger.info("Synthetic demographic features added")
    logger.info(f"Added protected attributes: {list(dataset.protected_attributes.keys())}")
    
    return dataset

def train_sentiment_model(X_train: pd.DataFrame, y_train: pd.Series, 
                         sample_weights: Optional[np.ndarray] = None) -> Pipeline:
    """Train a sentiment analysis model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_weights: Optional sample weights for training
        
    Returns:
        Trained model pipeline
    """
    logger.info("Training sentiment analysis model...")
    
    # Create a pipeline with TF-IDF and logistic regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=200))
    ])
    
    # Train the model (only use the text column)
    pipeline.fit(X_train['text'], y_train, **{
        'classifier__sample_weight': sample_weights
    } if sample_weights is not None else {})
    
    logger.info("Model training completed")
    
    return pipeline

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluate the model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (predictions, metrics)
    """
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test['text'])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print classification report
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return y_pred, {'accuracy': accuracy}

def apply_fairness_mitigation(processed_dataset: ProcessedDataset, mitigation_technique: str = 'reweighing') -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """Apply fairness mitigation to the dataset.
    
    Args:
        processed_dataset: Processed dataset with protected attributes
        mitigation_technique: Name of the mitigation technique to apply
        
    Returns:
        Tuple of (mitigated X_train, y_train, sample_weights)
    """
    X_train = processed_dataset.X_train
    y_train = processed_dataset.y_train
    
    # No protected attributes or mitigation
    if not processed_dataset.protected_attributes or mitigation_technique is None:
        logger.info("No fairness mitigation applied")
        return X_train, y_train, None
    
    # Choose primary protected attribute for mitigation
    protected_attr = 'gender'  # Using gender as the primary attribute
    
    if protected_attr not in X_train.columns:
        logger.warning(f"Protected attribute '{protected_attr}' not found in dataset")
        return X_train, y_train, None
    
    logger.info(f"Applying {mitigation_technique} for protected attribute: {protected_attr}")
    
    # Apply reweighing
    if mitigation_technique == 'reweighing':
        reweigher = Reweighing(protected_attribute=protected_attr)
        reweigher.fit(X_train, y_train)
        
        # Get sample weights for training
        sample_weights = reweigher.get_sample_weights(X_train, y_train)
        
        # Log weight statistics
        logger.info(f"Reweighing sample weight stats: min={sample_weights.min():.4f}, "
                   f"max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
        
        return X_train, y_train, sample_weights
        
    # Apply fair data transformation
    elif mitigation_technique == 'fair_transform':
        transformer = FairDataTransformer(
            protected_attribute=protected_attr,
            method='decorrelation'
        )
        
        # Fit and transform the data
        X_train_transformed, y_train = transformer.fit_transform(X_train, y_train)
        
        logger.info(f"Applied fair data transformation to remove bias for {protected_attr}")
        
        return X_train_transformed, y_train, None
    
    return X_train, y_train, None

def evaluate_fairness(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, output_dir: Path) -> Dict[str, FairnessResults]:
    """Evaluate fairness metrics for the model predictions.
    
    Args:
        X_test: Test features with demographic attributes
        y_test: True labels
        y_pred: Model predictions
        output_dir: Directory to save fairness results
        
    Returns:
        Dictionary of fairness results by protected attribute
    """
    logger.info("Evaluating fairness...")
    
    # Create combined DataFrame for fairness evaluation
    eval_df = X_test.copy()
    eval_df['target'] = y_test
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
    protected_attributes = ['gender', 'age_group', 'location']
    
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
    
    if all(attr in eval_df.columns for attr in ['age_group', 'location']):
        logger.info("Generating intersectional fairness visualization: age_group × location")
        evaluator._plot_intersectional_heatmap(
            df=eval_df,
            attr1='age_group',
            attr2='location',
            target_column='prediction',
            output_dir=fairness_dir
        )
    
    return fairness_results

def main() -> None:
    """Run the Sentiment140 fairness analysis example."""
    # Create output directory
    output_dir = Path('sentiment140_fairness_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the Kaggle dataset pipeline
    logger.info("Initializing Kaggle dataset pipeline...")
    pipeline = KaggleDatasetPipeline(
        data_dir=str(output_dir / 'data'),
        cache_dir=str(output_dir / 'cache'),
        validate_fairness=True
    )
    
    # Get dataset configs
    configs = pipeline.get_dataset_configs()
    
    # Select the Sentiment140 dataset
    logger.info("Using Sentiment140 dataset from Kaggle...")
    dataset_config = configs['sentiment140']
    
    # Process the dataset
    logger.info(f"Processing dataset: {dataset_config.dataset_slug}")
    processed_dataset = pipeline.process_dataset(dataset_config)
    
    # Log dataset information
    logger.info(f"Dataset loaded: {processed_dataset.metadata.dataset_name}")
    logger.info(f"Training set: {len(processed_dataset.X_train)} samples")
    logger.info(f"Validation set: {len(processed_dataset.X_val)} samples")
    logger.info(f"Test set: {len(processed_dataset.X_test)} samples")
    
    # Add synthetic demographic features for fairness analysis
    processed_dataset = add_synthetic_demographics(processed_dataset)
    
    # Apply fairness mitigation (reweighing)
    X_train, y_train, sample_weights = apply_fairness_mitigation(
        processed_dataset=processed_dataset,
        mitigation_technique='reweighing'
    )
    
    # Train sentiment analysis model
    model = train_sentiment_model(X_train, y_train, sample_weights)
    
    # Evaluate model on test set
    y_pred, metrics = evaluate_model(model, processed_dataset.X_test, processed_dataset.y_test)
    
    # Evaluate fairness
    fairness_results = evaluate_fairness(
        X_test=processed_dataset.X_test,
        y_test=processed_dataset.y_test,
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