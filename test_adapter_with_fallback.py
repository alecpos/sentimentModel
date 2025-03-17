#!/usr/bin/env python
"""
Test script for the Sentiment Adapter for AdScorePredictor with fallback functionality
"""

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_adapter import SentimentAdapterForAdPredictor, train_adapter_on_sentiment140

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_adapter_with_visualization(adapter, test_texts, test_labels):
    """Test the adapter on examples and visualize results"""
    
    # Process examples
    logger.info("Testing adapter on examples...")
    results = []
    predictions = []
    
    for i, text in enumerate(test_texts):
        # Get prediction
        result = adapter.predict_sentiment(text)
        
        # Store result
        results.append(result)
        predictions.append(1 if result['sentiment'] == 'positive' else 0)
        
        # Log sample predictions (only log a few examples)
        if i < 5:
            logger.info(f"Text: {text[:50]}...")
            logger.info(f"Predicted: {result['sentiment']} (Score: {result['score']:.1f})")
            logger.info(f"Actual: {'positive' if test_labels[i] == 1 else 'negative'}")
            logger.info(f"Model used: {result.get('model_used', 'unknown')}")
            logger.info("-" * 40)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    logger.info(f"Adapter Test Metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("\n" + classification_report(test_labels, predictions))
    
    # Count model usage
    model_used_counts = {}
    for result in results:
        model_used = result.get('model_used', 'unknown')
        model_used_counts[model_used] = model_used_counts.get(model_used, 0) + 1
    
    logger.info("Model usage statistics:")
    for model, count in model_used_counts.items():
        logger.info(f"{model}: {count} predictions ({count/len(results)*100:.1f}%)")
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    logger.info("Saved confusion matrix visualization to confusion_matrix.png")
    
    # Visualize score distribution
    scores = [r['score'] for r in results]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=20, kde=True)
    plt.axvline(x=adapter.threshold, color='r', linestyle='--', 
                label=f'Threshold: {adapter.threshold:.1f}')
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('score_distribution.png')
    logger.info("Saved score distribution visualization to score_distribution.png")
    
    return results, accuracy, f1

def main():
    parser = argparse.ArgumentParser(description='Test Sentiment Adapter with fallback functionality')
    parser.add_argument('--sample-size', type=int, default=1000, 
                        help='Number of samples for training/testing')
    parser.add_argument('--test-size', type=int, default=200,
                        help='Number of samples for testing only')
    parser.add_argument('--load-adapter', type=str, default=None,
                        help='Path to load a pre-trained adapter')
    parser.add_argument('--save-adapter', type=str, default='trained_adapter_with_fallback.joblib',
                        help='Path to save the trained adapter')
    
    args = parser.parse_args()
    
    # Either load a pre-trained adapter or train a new one
    if args.load_adapter:
        logger.info(f"Loading pre-trained adapter from {args.load_adapter}")
        adapter = SentimentAdapterForAdPredictor()
        adapter.load(args.load_adapter)
    else:
        logger.info(f"Training new adapter on {args.sample_size} examples")
        adapter = train_adapter_on_sentiment140(
            sample_size=args.sample_size,
            find_threshold=True,
            use_enhanced_preprocessing=True,
            fallback_to_internal_model=True  # Enable fallback functionality
        )
        
        # Save the trained adapter
        if args.save_adapter:
            adapter.save(args.save_adapter)
            logger.info(f"Saved adapter to {args.save_adapter}")
    
    # Load test data from Sentiment140
    logger.info(f"Loading {args.test_size} test examples from Sentiment140")
    try:
        df = pd.read_csv('sentiment140.csv', 
                         encoding='latin-1', 
                         names=['target', 'id', 'date', 'flag', 'user', 'text'])
        
        # Convert target from Twitter format (0=negative, 4=positive) to binary (0=negative, 1=positive)
        df['target'] = df['target'].map({0: 0, 4: 1})
        
        # Sample equal numbers of positive and negative examples
        pos_samples = df[df['target'] == 1].sample(args.test_size // 2, random_state=42)
        neg_samples = df[df['target'] == 0].sample(args.test_size // 2, random_state=42)
        test_df = pd.concat([pos_samples, neg_samples])
        
        # Shuffle the test data
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        test_texts = test_df['text'].values
        test_labels = test_df['target'].values
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # Test adapter on the test examples
    results, accuracy, f1 = test_adapter_with_visualization(adapter, test_texts, test_labels)
    
    # Log adapter information
    logger.info("\nAdapter Configuration:")
    logger.info(f"Threshold: {adapter.threshold}")
    logger.info(f"Enhanced preprocessing: {adapter.use_enhanced_preprocessing}")
    logger.info(f"Using fallback model: {adapter.using_fallback}")
    logger.info(f"AdScorePredictor is fitted: {adapter.is_ad_predictor_fitted}")
    
    # Additional analysis
    if hasattr(adapter, 'metrics') and adapter.metrics:
        logger.info("\nAdapter Metrics:")
        for key, value in adapter.metrics.items():
            logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main() 