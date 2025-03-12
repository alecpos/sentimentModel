#!/usr/bin/env python
"""
Sentiment140 Dataset Fairness Post-Processing Example

This script demonstrates how to use the fairness post-processing techniques
with the Sentiment140 dataset. It shows the complete workflow for:

1. Downloading and processing the Sentiment140 dataset from Kaggle
2. Adding synthetic demographic attributes that simulate realistic biases
3. Training a sentiment analysis model
4. Applying post-processing fairness techniques
5. Evaluating fairness metrics before and after post-processing
6. Generating comparison reports and visualizations

This example demonstrates how fairness post-processing can be applied to a real-world
dataset even when the actual demographic information is not available.
"""

import logging
import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# Add the parent directory to the path for imports
parent_dir = str(Path(__file__).parent)
sys.path.append(parent_dir)

# Import our fairness post-processing module
import fairness_postprocessing as fp

# Import Kaggle pipeline from existing example
sys.path.append(str(Path(__file__).parent / "app"))
try:
    from app.core.data_integration.kaggle_pipeline import (
        KaggleDatasetPipeline, DatasetConfig, ProcessedDataset, DatasetCategory
    )
except ImportError:
    print("Could not import Kaggle pipeline. Falling back to direct dataset loading.")
    KaggleDatasetPipeline = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_biased_synthetic_demographics(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Add synthetic demographic features with realistic bias patterns.
    
    Args:
        df: DataFrame with sentiment data
        text_column: Name of the column containing text data
        
    Returns:
        DataFrame with added synthetic demographic features
    """
    logger.info("Adding synthetic demographic features with realistic bias patterns...")
    
    # Make a copy of the dataframe
    df_with_demographics = df.copy()
    
    # Use a fixed random seed for reproducibility
    np.random.seed(42)
    
    # Extract some simple text features that we'll correlate with demographics
    df_with_demographics['text_length'] = df_with_demographics[text_column].str.len()
    df_with_demographics['has_exclamation'] = df_with_demographics[text_column].str.contains('!')
    df_with_demographics['has_question'] = df_with_demographics[text_column].str.contains('\\?')
    
    # Calculate percentiles for text features to create groups
    length_percentiles = np.percentile(df_with_demographics['text_length'], [25, 50, 75])
    
    # Create a length category feature (1-4)
    conditions = [
        (df_with_demographics['text_length'] <= length_percentiles[0]),
        (df_with_demographics['text_length'] > length_percentiles[0]) & (df_with_demographics['text_length'] <= length_percentiles[1]),
        (df_with_demographics['text_length'] > length_percentiles[1]) & (df_with_demographics['text_length'] <= length_percentiles[2]),
        (df_with_demographics['text_length'] > length_percentiles[2])
    ]
    values = [1, 2, 3, 4]
    df_with_demographics['length_category'] = np.select(conditions, values, default=1)
    
    # GENDER: Create with bias based on text features
    # Probability of being assigned 'Female' increases with text length
    # This creates a systematic bias where longer tweets are more likely to be from 'Female' users
    female_prob = 0.3 + (0.1 * df_with_demographics['length_category']) + (0.1 * df_with_demographics['has_exclamation'])
    df_with_demographics['gender'] = np.random.binomial(1, female_prob)
    df_with_demographics['gender'] = df_with_demographics['gender'].map({0: 'Male', 1: 'Female'})
    
    # AGE GROUP: Create with bias based on text features
    # We'll make different age groups have different tweet length distributions
    age_group_conditions = [
        (df_with_demographics['length_category'] == 1) | (df_with_demographics['has_question']),
        (df_with_demographics['length_category'] == 2),
        (df_with_demographics['length_category'] == 3),
        (df_with_demographics['length_category'] == 4)
    ]
    # Add randomness to make it less deterministic
    age_group_probs = np.random.rand(len(df_with_demographics))
    final_conditions = []
    for i, condition in enumerate(age_group_conditions):
        final_conditions.append(condition & (age_group_probs <= 0.7))
    
    age_values = ['18-25', '26-35', '36-50', '51+']
    df_with_demographics['age_group'] = np.select(final_conditions, age_values, default=np.random.choice(age_values))
    
    # LOCATION: Based on text features but with less strong correlation
    location_base_probs = {
        'Urban': 0.4 + (0.05 * df_with_demographics['length_category']),
        'Suburban': 0.3 + (0.02 * (5 - df_with_demographics['length_category'])),
        'Rural': 0.3 - (0.03 * df_with_demographics['length_category'])
    }
    
    # Normalize to ensure probabilities sum to 1
    location_sum = pd.DataFrame(location_base_probs).sum(axis=1)
    location_probs = {k: v / location_sum for k, v in location_base_probs.items()}
    
    # Create random locations based on calculated probabilities
    random_values = np.random.rand(len(df_with_demographics))
    urban_mask = random_values < location_probs['Urban']
    suburban_mask = (random_values >= location_probs['Urban']) & (random_values < (location_probs['Urban'] + location_probs['Suburban']))
    
    df_with_demographics['location'] = 'Rural'
    df_with_demographics.loc[urban_mask, 'location'] = 'Urban'
    df_with_demographics.loc[suburban_mask, 'location'] = 'Suburban'
    
    # Introduce some bias based on demographics and sentiment (target)
    # This simulates a scenario where certain demographic groups might use more positive language
    if 'target' in df_with_demographics.columns:
        # Calculate probability adjustment based on demographics
        prob_adjustment = (
            0.1 * (df_with_demographics['gender'] == 'Female').astype(int) +
            0.05 * (df_with_demographics['age_group'] == '18-25').astype(int) -
            0.05 * (df_with_demographics['age_group'] == '51+').astype(int) +
            0.03 * (df_with_demographics['location'] == 'Urban').astype(int)
        )
        
        # Adjust sentiment for a small percentage of samples
        adjustment_mask = np.random.rand(len(df_with_demographics)) < np.abs(prob_adjustment)
        increase_mask = adjustment_mask & (prob_adjustment > 0) & (df_with_demographics['target'] == 0)
        decrease_mask = adjustment_mask & (prob_adjustment < 0) & (df_with_demographics['target'] == 1)
        
        # Apply adjustments to create demographic-based bias in sentiment
        df_with_demographics.loc[increase_mask, 'target'] = 1
        df_with_demographics.loc[decrease_mask, 'target'] = 0
    
    # Clean up temporary columns
    df_with_demographics = df_with_demographics.drop(['text_length', 'has_exclamation', 'has_question', 'length_category'], axis=1)
    
    logger.info("Synthetic demographic features added with realistic bias patterns")
    logger.info(f"Demographics: {['gender', 'age_group', 'location']}")
    
    # Log demographics distribution
    for col in ['gender', 'age_group', 'location']:
        logger.info(f"{col} distribution: {df_with_demographics[col].value_counts(normalize=True).to_dict()}")
    
    return df_with_demographics

def load_sentiment140(max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Sentiment140 dataset.
    
    Args:
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        Tuple of (train_data, test_data)
    """
    logger.info("Loading Sentiment140 dataset...")
    
    # Try to use the Kaggle pipeline if available
    if KaggleDatasetPipeline is not None:
        output_dir = Path('sentiment140_fairness_postprocessing')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = KaggleDatasetPipeline(
            data_dir=str(output_dir / 'data'),
            cache_dir=str(output_dir / 'cache'),
            validate_fairness=False
        )
        
        configs = pipeline.get_dataset_configs()
        
        if 'sentiment140' in configs:
            logger.info("Using Kaggle pipeline to load Sentiment140 dataset...")
            dataset_config = configs['sentiment140']
            processed_dataset = pipeline.process_dataset(dataset_config)
            
            train_data = pd.concat([processed_dataset.X_train, processed_dataset.X_val])
            train_data['target'] = pd.concat([processed_dataset.y_train, processed_dataset.y_val])
            
            test_data = processed_dataset.X_test.copy()
            test_data['target'] = processed_dataset.y_test
            
            logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
            
            # Limit samples if requested
            if max_samples is not None:
                test_size = min(int(max_samples * 0.2), len(test_data))
                train_size = min(max_samples - test_size, len(train_data))
                
                train_data = train_data.sample(train_size, random_state=42)
                test_data = test_data.sample(test_size, random_state=42)
                
                logger.info(f"Limited to {len(train_data)} training samples and {len(test_data)} test samples")
            
            return train_data, test_data
    
    # Manual loading fallback
    logger.info("Falling back to manual dataset loading...")
    
    # Define column names for the CSV
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Try different possible locations for the dataset
    possible_paths = [
        'sentiment140/sentiment140.csv',
        'sentiment140_data/sentiment140.csv',
        'data/sentiment140.csv',
        Path.home() / '.cache/kagglehub/datasets/kazanova/sentiment140/versions/2/training.1600000.processed.noemoticon.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError("Could not find Sentiment140 dataset. Please download it first.")
    
    logger.info(f"Reading Sentiment140 data from {data_path}")
    
    # Read CSV with appropriate encoding
    try:
        df = pd.read_csv(data_path, encoding='utf-8', names=column_names)
    except UnicodeDecodeError:
        logger.warning("UTF-8 encoding failed, trying latin1 encoding")
        df = pd.read_csv(data_path, encoding='latin1', names=column_names)
    
    # Map targets from 0/4 to 0/1
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Keep only the necessary columns
    df = df[['text', 'target']]
    
    # Limit samples if requested
    if max_samples is not None:
        df = df.sample(min(max_samples, len(df)), random_state=42)
        logger.info(f"Limited to {len(df)} samples total")
    
    # Split into train and test
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    logger.info(f"Split into {len(train_data)} training samples and {len(test_data)} test samples")
    
    return train_data, test_data

def train_sentiment_model(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, np.ndarray]:
    """Train a sentiment analysis model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Tuple of (trained pipeline, prediction probabilities)
    """
    logger.info("Training sentiment analysis model...")
    start_time = time.time()
    
    # Create a pipeline with TF-IDF and logistic regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(
            random_state=42, 
            max_iter=200,
            C=1.0,
            class_weight='balanced'
        ))
    ])
    
    # Train the model (only use the text column)
    pipeline.fit(X_train['text'], y_train)
    
    # Get prediction probabilities
    y_pred_proba = pipeline.predict_proba(X_train['text'])[:, 1]
    
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return pipeline, y_pred_proba

def evaluate_model_fairness(
    model: Pipeline, 
    test_data: pd.DataFrame,
    protected_attributes: List[str] = ['gender', 'age_group', 'location']
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Evaluate model performance and fairness.
    
    Args:
        model: Trained model
        test_data: Test data with protected attributes
        protected_attributes: List of protected attribute columns
        
    Returns:
        Tuple of (true labels, prediction probabilities, fairness metrics)
    """
    logger.info("Evaluating model performance and fairness...")
    
    # Extract features and labels
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Get model predictions
    y_pred = model.predict(X_test['text'])
    y_pred_proba = model.predict_proba(X_test['text'])[:, 1]
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Overall Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Create DataFrame with protected attributes for fairness evaluation
    protected_df = pd.DataFrame()
    for attr in protected_attributes:
        if attr in X_test.columns:
            protected_df[attr] = X_test[attr]
    
    # Calculate fairness metrics
    fairness_metrics = {}
    
    # Create a RejectionOptionClassifier to get fairness metrics
    rejection_classifier = fp.RejectionOptionClassifier()
    baseline_metrics = rejection_classifier.get_fairness_metrics(
        y_test.values, y_pred, protected_df
    )
    
    fairness_metrics['baseline'] = baseline_metrics
    
    # Log fairness metrics summary
    logger.info("Fairness Metrics Summary:")
    for attr in protected_attributes:
        if attr in protected_df.columns and attr in baseline_metrics:
            metrics = baseline_metrics[attr]
            logger.info(f"  {attr}:")
            logger.info(f"    Demographic Parity Disparity: {metrics.get('demographic_parity_disparity', 'N/A')}")
            logger.info(f"    Equalized Odds Disparity: {metrics.get('equalized_odds_disparity', 'N/A')}")
    
    return y_test.values, y_pred_proba, fairness_metrics

def apply_fairness_postprocessing(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    protected_df: pd.DataFrame,
    baseline_metrics: Dict,
    techniques: List[str] = ['threshold', 'rejection']
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """Apply fairness post-processing techniques.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        protected_df: DataFrame with protected attributes
        baseline_metrics: Dictionary of baseline fairness metrics
        techniques: List of techniques to apply ('threshold', 'rejection')
        
    Returns:
        Dictionary of technique -> (predictions, fairness metrics)
    """
    logger.info("Applying fairness post-processing techniques...")
    
    # Debug log the structure of baseline_metrics
    logger.info(f"Baseline metrics structure: {baseline_metrics.keys()}")
    
    # Extract the actual metrics if they're nested
    actual_metrics = baseline_metrics.get('baseline', baseline_metrics)
    
    results = {}
    results['baseline_metrics'] = actual_metrics
    
    # Apply threshold optimization if requested
    if 'threshold' in techniques:
        logger.info("Applying ThresholdOptimizer (demographic_parity)...")
        
        # Create and fit the optimizer
        threshold_optimizer = fp.ThresholdOptimizer(fairness_metric="demographic_parity")
        threshold_optimizer.fit(y_true, y_pred_proba, protected_df)
        
        # Apply optimized thresholds to get fair predictions
        threshold_predictions = threshold_optimizer.adjust(y_pred_proba, protected_df)
        
        # Calculate fairness metrics
        rejection_classifier = fp.RejectionOptionClassifier()
        threshold_metrics = rejection_classifier.get_fairness_metrics(
            y_true, threshold_predictions, protected_df
        )
        
        results['threshold'] = (threshold_predictions, threshold_metrics)
        
        # Log improvement summary
        logger.info("ThresholdOptimizer Results:")
        for attr in threshold_metrics:
            if attr in actual_metrics and 'demographic_parity_disparity' in actual_metrics[attr]:
                baseline_dp = actual_metrics[attr]['demographic_parity_disparity']
            else:
                logger.warning(f"Baseline demographic parity disparity for {attr} not found. Using 0.01 as fallback.")
                baseline_dp = 0.01
                
            optimized_dp = threshold_metrics[attr].get('demographic_parity_disparity', 0)
            improvement = ((baseline_dp - optimized_dp) / baseline_dp * 100) if baseline_dp > 0 else 0
            
            logger.info(f"  {attr} Demographic Parity: {baseline_dp:.4f} -> {optimized_dp:.4f} ({improvement:.1f}% improvement)")
    
    # Apply rejection option classification if requested
    if 'rejection' in techniques:
        logger.info("Applying RejectionOptionClassifier (demographic_parity)...")
        
        # Create and fit the classifier
        rejection_classifier = fp.RejectionOptionClassifier(fairness_metric="demographic_parity")
        rejection_classifier.fit(y_true, y_pred_proba, protected_df)
        
        # Apply the rejection option technique
        rejection_predictions = rejection_classifier.adjust(y_pred_proba, protected_df)
        
        # Calculate fairness metrics
        rejection_metrics = rejection_classifier.get_fairness_metrics(
            y_true, rejection_predictions, protected_df
        )
        
        results['rejection'] = (rejection_predictions, rejection_metrics)
        
        # Log improvement summary
        logger.info("RejectionOptionClassifier Results:")
        for attr in rejection_metrics:
            if attr in actual_metrics and 'demographic_parity_disparity' in actual_metrics[attr]:
                baseline_dp = actual_metrics[attr]['demographic_parity_disparity']
            else:
                logger.warning(f"Baseline demographic parity disparity for {attr} not found. Using 0.01 as fallback.")
                baseline_dp = 0.01
                
            rejection_dp = rejection_metrics[attr].get('demographic_parity_disparity', 0)
            improvement = ((baseline_dp - rejection_dp) / baseline_dp * 100) if baseline_dp > 0 else 0
            
            logger.info(f"  {attr} Demographic Parity: {baseline_dp:.4f} -> {rejection_dp:.4f} ({improvement:.1f}% improvement)")
    
    return results

def generate_fairness_reports(results: Dict[str, Any], output_dir: str = 'fairness_reports'):
    """Generate fairness reports and visualizations.
    
    Args:
        results: Dictionary of technique -> (predictions, fairness metrics) or other data
        output_dir: Output directory for reports
    """
    logger.info("Generating fairness reports and visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function to safely extract metrics
    def get_metrics(result_item):
        if isinstance(result_item, tuple) and len(result_item) == 2:
            return result_item[1]  # Standard format: (predictions, metrics)
        elif isinstance(result_item, dict):
            return result_item  # Direct metrics dictionary
        else:
            logger.warning(f"Unexpected result format: {type(result_item)}")
            return {}
    
    # Generate individual reports
    for technique, result_data in results.items():
        if technique in ['baseline_metrics']:
            continue
            
        # Get metrics safely
        metrics = get_metrics(result_data)
        if not metrics:
            logger.warning(f"No metrics found for technique: {technique}")
            continue
            
        # Generate fairness report
        try:
            report_path = fp.generate_fairness_report(
                metrics, 
                output_dir=output_dir, 
                filename_prefix=f"{technique}_fairness"
            )
            logger.info(f"Generated fairness report for {technique}: {report_path}")
        except Exception as e:
            logger.error(f"Error generating fairness report for {technique}: {e}")
    
    # Get baseline metrics
    baseline_metrics = results.get('baseline_metrics', {})
    if not baseline_metrics and 'baseline' in results:
        baseline_metrics = get_metrics(results['baseline'])
    
    # Generate comparison reports
    if 'threshold' in results and baseline_metrics:
        try:
            threshold_metrics = get_metrics(results['threshold'])
            fp.compare_fairness_improvements(
                baseline_metrics,
                threshold_metrics,
                output_dir=output_dir,
                filename=f"threshold_comparison.md"
            )
            logger.info(f"Generated comparison report: {output_dir}/threshold_comparison.md")
        except Exception as e:
            logger.error(f"Error generating threshold comparison report: {e}")
    
    if 'rejection' in results and baseline_metrics:
        try:
            rejection_metrics = get_metrics(results['rejection'])
            fp.compare_fairness_improvements(
                baseline_metrics,
                rejection_metrics,
                output_dir=output_dir,
                filename=f"rejection_comparison.md"
            )
            logger.info(f"Generated comparison report: {output_dir}/rejection_comparison.md")
        except Exception as e:
            logger.error(f"Error generating rejection comparison report: {e}")
    
    # Generate visualizations
    try:
        if 'threshold' in results and baseline_metrics:
            threshold_metrics = get_metrics(results['threshold'])
            rejection_metrics = get_metrics(results.get('rejection', results['threshold']))
            
            fp.generate_fairness_visualizations(
                baseline_metrics,
                threshold_metrics,
                rejection_metrics,
                output_dir=output_dir
            )
            logger.info(f"Generated fairness visualizations in {output_dir}")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

def main():
    """Run the Sentiment140 fairness post-processing example."""
    start_time = time.time()
    
    # Create output directory
    output_dir = 'sentiment140_fairness_postprocessing'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set maximum samples (None for all)
    max_samples = None  # Set to a smaller number for quick testing, None for full dataset
    
    # Load the Sentiment140 dataset
    train_data, test_data = load_sentiment140(max_samples)
    
    # Add synthetic demographics with realistic bias patterns
    train_data_with_demographics = add_biased_synthetic_demographics(train_data)
    test_data_with_demographics = add_biased_synthetic_demographics(test_data)
    
    # Train the sentiment model
    model, train_pred_proba = train_sentiment_model(
        train_data_with_demographics, 
        train_data_with_demographics['target']
    )
    
    # Evaluate baseline model performance and fairness
    y_test, y_pred_proba, baseline_metrics = evaluate_model_fairness(
        model, 
        test_data_with_demographics,
        protected_attributes=['gender', 'age_group', 'location']
    )
    
    # Extract protected attributes for fairness post-processing
    protected_df = test_data_with_demographics[['gender', 'age_group', 'location']]
    
    # Apply fairness post-processing techniques
    results = apply_fairness_postprocessing(
        y_test,
        y_pred_proba,
        protected_df,
        baseline_metrics,
        techniques=['threshold', 'rejection']
    )
    
    # Add baseline metrics to results
    results['baseline'] = ((y_pred_proba >= 0.5).astype(int), baseline_metrics)
    
    # Generate fairness reports and visualizations
    generate_fairness_reports(results, output_dir=f"{output_dir}/fairness_reports")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time:.2f} seconds")
    
    logger.info("Example completed successfully!")
    logger.info(f"Check {output_dir}/fairness_reports for detailed fairness reports and visualizations")

if __name__ == "__main__":
    main() 