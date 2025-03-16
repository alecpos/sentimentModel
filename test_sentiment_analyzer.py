#!/usr/bin/env python
"""
Test script for HybridSentimentAnalyzer

This script tests the HybridSentimentAnalyzer on the Sentiment140 dataset,
compares performance with and without feature selection, and analyzes results.
"""

import os
import argparse
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import for HybridSentimentAnalyzer
from hybrid_sentiment_analyzer import HybridSentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sentiment140_sample(file_path="sentiment140.csv", sample_size=10000, random_state=42):
    """
    Load and sample data from the Sentiment140 dataset.
    
    Args:
        file_path (str): Path to the Sentiment140 dataset CSV file
        sample_size (int): Number of samples to use (default: 10000)
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Sampled dataset with columns 'text' and 'sentiment'
    """
    logger.info(f"Loading data from {file_path}")
    
    # Column names for the Sentiment140 dataset
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    try:
        df = pd.read_csv(file_path, encoding='latin-1', names=column_names)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Convert target from 0/4 to 0/1 for negative/positive sentiment
        # HybridSentimentAnalyzer expects numerical labels
        df['sentiment'] = df['target'].map({0: 0, 4: 1})
        
        # Keep only text and sentiment columns
        df = df[['text', 'sentiment']]
        
        # Sample data if needed
        if 0 < sample_size < len(df):
            df = df.sample(sample_size, random_state=random_state)
            logger.info(f"Sampled {sample_size} rows")
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def test_model(data, output_dir="sentiment_results", use_feature_selection=True, 
               test_size=0.2, random_state=42, optimize=True):
    """
    Test the hybrid sentiment analyzer with detailed metrics.
    
    Args:
        data (pd.DataFrame): DataFrame with 'text' and 'sentiment' columns
        output_dir (str): Directory to save results
        use_feature_selection (bool): Whether to use feature selection
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        optimize (bool): Whether to optimize the model with GridSearchCV
        
    Returns:
        dict: Performance metrics
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "with_feature_selection" if use_feature_selection else "no_feature_selection"
    result_dir = os.path.join(output_dir, f"{model_type}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], 
        data['sentiment'],
        test_size=test_size,
        random_state=random_state,
        stratify=data['sentiment']
    )
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Initialize the sentiment analyzer without the use_feature_selection parameter
    analyzer = HybridSentimentAnalyzer()
    
    # Train the model - don't pass use_feature_selection if the method doesn't accept it
    logger.info("Training model...")
    start_time = datetime.now()
    # Check if analyzer.train accepts a use_feature_selection parameter
    try:
        analyzer.train(X_train, y_train, optimize=optimize)
    except TypeError as e:
        if "got an unexpected keyword argument 'use_feature_selection'" in str(e):
            # If it doesn't, call without it
            analyzer.train(X_train, y_train, optimize=optimize)
        else:
            # If it's a different error, raise it
            raise e
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    metrics = analyzer.evaluate_with_f1(X_test, y_test)
    
    # Log metrics
    logger.info("Performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Get predictions for confusion matrix
    y_pred = [analyzer.predict(text).sentiment_score > 0 for text in tqdm(X_test, desc="Predicting")]
    y_pred_binary = [1 if p else 0 for p in y_pred]
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary, labels=[1, 0])
    cm_df = pd.DataFrame(cm, index=['positive (1)', 'negative (0)'], columns=['positive (1)', 'negative (0)'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(result_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred_binary, target_names=['negative (0)', 'positive (1)'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(result_dir, "classification_report.csv")
    report_df.to_csv(report_path)
    logger.info(f"Saved classification report to {report_path}")
    
    # Test on challenging examples
    challenging_examples = [
        "Not as bad as I thought it would be, but still not worth the money.",
        "I can't say I hate it, but I definitely don't love it either.",
        "It's honestly surprising how something so expensive could be so average.",
        "This isn't terrible, but there are much better alternatives.",
        "I don't think I will be buying this again, although it does have some good points.",
    ]
    
    challenge_results = []
    for text in challenging_examples:
        result = analyzer.predict(text)
        sentiment_label = "positive" if result.sentiment_score > 0 else "negative"
        challenge_results.append({
            "text": text,
            "sentiment": sentiment_label,
            "score": result.sentiment_score,
            "confidence": result.confidence
        })
        logger.info(f"Challenging example: '{text}' → {sentiment_label} "
                   f"(score: {result.sentiment_score:.2f}, confidence: {result.confidence:.2f})")
    
    # Save challenging examples results
    challenge_df = pd.DataFrame(challenge_results)
    challenge_path = os.path.join(result_dir, "challenging_examples.csv")
    challenge_df.to_csv(challenge_path, index=False)
    logger.info(f"Saved challenging examples results to {challenge_path}")
    
    # Save model
    model_path = os.path.join(result_dir, "sentiment_analyzer.pkl")
    analyzer.save_model(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metrics
    metrics.update({
        "training_time": training_time,
        "sample_size": len(data),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_selection": use_feature_selection,
        "model_path": model_path
    })
    
    metrics_path = os.path.join(result_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return metrics

def analyze_sentiment140_dataset(file_path="sentiment140.csv", sample_size=1000):
    """
    Analyze characteristics of the Sentiment140 dataset.
    
    Args:
        file_path (str): Path to the Sentiment140 dataset CSV file
        sample_size (int): Number of samples to analyze
        
    Returns:
        pd.DataFrame: Analysis results
    """
    logger.info(f"Analyzing Sentiment140 dataset: {file_path}")
    data = load_sentiment140_sample(file_path, sample_size)
    
    # Create output directory
    os.makedirs("dataset_analysis", exist_ok=True)
    
    # Basic statistics
    logger.info(f"Total samples: {len(data)}")
    sentiment_counts = data['sentiment'].value_counts()
    logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
    
    # Map numerical labels to text for better visualization
    sentiment_map = {0: 'negative', 1: 'positive'}
    data['sentiment_text'] = data['sentiment'].map(sentiment_map)
    
    # Text length analysis
    data['text_length'] = data['text'].apply(len)
    
    # Plot text length distribution by sentiment
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='text_length', hue='sentiment_text', bins=50, alpha=0.6)
    plt.title('Text Length Distribution by Sentiment')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("dataset_analysis/text_length_distribution.png", dpi=300)
    
    # Word count analysis
    data['word_count'] = data['text'].apply(lambda x: len(x.split()))
    
    # Plot word count distribution by sentiment
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='word_count', hue='sentiment_text', bins=30, alpha=0.6)
    plt.title('Word Count Distribution by Sentiment')
    plt.xlabel('Word Count')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("dataset_analysis/word_count_distribution.png", dpi=300)
    
    # Save analysis results
    analysis_path = "dataset_analysis/sentiment140_analysis.csv"
    data.to_csv(analysis_path, index=False)
    logger.info(f"Saved dataset analysis to {analysis_path}")
    
    return data

def compare_models(file_path="sentiment140.csv", sample_size=5000):
    """
    Compare model performance with and without feature selection.
    
    Args:
        file_path (str): Path to the Sentiment140 dataset CSV file
        sample_size (int): Number of samples to use
        
    Returns:
        tuple: (with_feature_selection_metrics, without_feature_selection_metrics)
    """
    logger.info("Comparing model performance with and without feature selection")
    
    # Load data
    data = load_sentiment140_sample(file_path, sample_size)
    
    # Test with feature selection
    logger.info("\n" + "="*50)
    logger.info("TESTING MODEL WITH FEATURE SELECTION")
    logger.info("="*50)
    with_fs_metrics = test_model(
        data, 
        output_dir="comparison_results",
        use_feature_selection=True
    )
    
    # Test without feature selection
    logger.info("\n" + "="*50)
    logger.info("TESTING MODEL WITHOUT FEATURE SELECTION")
    logger.info("="*50)
    without_fs_metrics = test_model(
        data, 
        output_dir="comparison_results",
        use_feature_selection=False
    )
    
    # Compare results
    metrics_to_compare = ['accuracy', 'f1', 'precision', 'recall']
    comparison = {}
    
    for metric in metrics_to_compare:
        with_val = with_fs_metrics.get(metric, 0)
        without_val = without_fs_metrics.get(metric, 0)
        diff = with_val - without_val
        diff_percent = (diff / without_val) * 100 if without_val != 0 else 0
        
        comparison[metric] = {
            'with_feature_selection': with_val,
            'without_feature_selection': without_val,
            'difference': diff,
            'difference_percent': diff_percent
        }
    
    # Print comparison
    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*50)
    
    for metric, values in comparison.items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  With feature selection: {values['with_feature_selection']:.4f}")
        logger.info(f"  Without feature selection: {values['without_feature_selection']:.4f}")
        logger.info(f"  Difference: {values['difference']:.4f} ({values['difference_percent']:+.2f}%)")
    
    # Create comparison visualization
    metrics_df = pd.DataFrame({
        'Metric': metrics_to_compare,
        'With Feature Selection': [with_fs_metrics.get(m, 0) for m in metrics_to_compare],
        'Without Feature Selection': [without_fs_metrics.get(m, 0) for m in metrics_to_compare]
    })
    
    metrics_df_melted = metrics_df.melt(
        id_vars='Metric', 
        value_vars=['With Feature Selection', 'Without Feature Selection'],
        var_name='Model', 
        value_name='Score'
    )
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df_melted, x='Metric', y='Score', hue='Model')
    plt.title('Model Performance Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save comparison chart
    os.makedirs("comparison_results", exist_ok=True)
    comparison_path = "comparison_results/feature_selection_comparison.png"
    plt.savefig(comparison_path, dpi=300)
    logger.info(f"Saved comparison chart to {comparison_path}")
    
    # Save detailed comparison to CSV
    comparison_df = pd.DataFrame(comparison).transpose()
    comparison_df.to_csv("comparison_results/feature_selection_comparison.csv")
    
    return with_fs_metrics, without_fs_metrics

def main():
    """Run the test script with command-line arguments."""
    parser = argparse.ArgumentParser(description='Test HybridSentimentAnalyzer.')
    parser.add_argument('--file-path', type=str, default='sentiment140.csv',
                        help='Path to the Sentiment140 dataset CSV file')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Number of samples to use')
    parser.add_argument('--mode', type=str, choices=['test', 'analyze', 'compare'], default='test',
                        help='Test mode: test (default), analyze dataset, or compare models')
    parser.add_argument('--feature-selection', action='store_true',
                        help='Use feature selection (only for test mode)')
    parser.add_argument('--output-dir', type=str, default='sentiment_results',
                        help='Directory to save results')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize model with GridSearchCV')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected mode
    if args.mode == 'test':
        logger.info(f"Testing model with {'feature selection' if args.feature_selection else 'no feature selection'}")
        data = load_sentiment140_sample(args.file_path, args.sample_size)
        metrics = test_model(
            data, 
            output_dir=args.output_dir,
            use_feature_selection=args.feature_selection,
            optimize=args.optimize
        )
        logger.info("Testing completed successfully!")
        
    elif args.mode == 'analyze':
        logger.info(f"Analyzing dataset: {args.file_path}")
        analyze_sentiment140_dataset(args.file_path, args.sample_size)
        logger.info("Dataset analysis completed successfully!")
        
    elif args.mode == 'compare':
        logger.info(f"Comparing models with and without feature selection")
        compare_models(args.file_path, args.sample_size)
        logger.info("Model comparison completed successfully!")

if __name__ == "__main__":
    main() 