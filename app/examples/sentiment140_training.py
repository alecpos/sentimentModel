#!/usr/bin/env python
"""
Sentiment140 Training Script

This script trains a sentiment analysis model using the Sentiment140 dataset.
It uses the AdSentimentAnalyzer class to train, evaluate, and save the model.

Usage:
    python app/examples/sentiment140_training.py

The script will:
1. Load the Sentiment140 dataset
2. Preprocess the data
3. Train a sentiment analysis model
4. Evaluate the model
5. Save the trained model
"""

import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from app.core.data_integration.kaggle_pipeline import KaggleDatasetPipeline
from app.models.ml.prediction.ad_sentiment_analyzer import AdSentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model on Sentiment140')
    parser.add_argument('--output_dir', type=str, default='models/sentiment140',
                      help='Directory to save the trained model')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use (None to use all)')
    parser.add_argument('--model_type', type=str, default='logistic',
                      choices=['logistic', 'random_forest'],
                      help='Type of ML model to use')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    
    return parser.parse_args()

def preprocess_data(df):
    """Preprocess the Sentiment140 dataset."""
    logger.info(f"Preprocessing {len(df)} samples...")
    
    # Ensure the dataset has the expected format
    if 'target' not in df.columns or 'text' not in df.columns:
        raise ValueError("Dataset does not have the expected columns: 'target' and 'text'")
    
    # Convert target from 0/4 to 0/1 if needed
    if df['target'].max() > 1:
        df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
        logger.info("Converted target values from 0/4 to 0/1")
    
    # Remove samples with empty text
    df = df[df['text'].notna() & (df['text'] != '')]
    logger.info(f"Removed samples with empty text, {len(df)} samples remaining")
    
    # Basic text cleaning
    df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs
    df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)     # Remove mentions
    df['text'] = df['text'].str.replace(r'#\w+', '', regex=True)     # Remove hashtags
    df['text'] = df['text'].str.replace(r'\d+', '', regex=True)      # Remove numbers
    
    # Check for and handle any remaining nulls
    if df['text'].isna().any():
        logger.warning(f"Found {df['text'].isna().sum()} null texts after preprocessing")
        df = df.dropna(subset=['text'])
    
    return df

def train_model(args):
    """Train a sentiment analysis model on the Sentiment140 dataset."""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the Kaggle dataset pipeline
    pipeline = KaggleDatasetPipeline()
    
    # Get the Sentiment140 dataset configuration
    configs = pipeline.get_dataset_configs()
    sentiment140_config = configs["sentiment140"]
    
    # Process the dataset
    logger.info("Loading Sentiment140 dataset...")
    processed_dataset = pipeline.process_dataset(sentiment140_config)
    
    # Combine training and validation sets for model training/testing
    X = pd.concat([processed_dataset.X_train, processed_dataset.X_val])
    y = pd.concat([processed_dataset.y_train, processed_dataset.y_val])
    
    # Limit dataset size if specified
    if args.max_samples is not None and len(X) > args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples")
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=args.max_samples, random_state=42, stratify=y
        )
        X = X_sample
        y = y_sample
    
    # Convert y values to integers if they're not already
    y = y.astype(int)
    
    # Initialize the sentiment analyzer
    sentiment_analyzer = AdSentimentAnalyzer()
    
    # If using a random forest classifier, modify the model
    if args.model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        
        sentiment_analyzer.model = Pipeline([
            ('tfidf', sentiment_analyzer.vectorizer),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        logger.info("Using RandomForestClassifier model")
    else:
        logger.info("Using LogisticRegression model")
    
    # Train the model
    logger.info(f"Training on {len(X)} samples...")
    
    # Extract text from dataframe
    texts = X['text'].tolist()
    labels = y.tolist()
    
    # Train the model with the specified test size
    metrics = sentiment_analyzer.train(
        texts=texts,
        labels=labels,
        train_test_split=args.test_size,
        save_path=output_dir / f"sentiment140_{args.model_type}_{datetime.now().strftime('%Y%m%d')}.joblib"
    )
    
    # Log metrics
    logger.info(f"Training complete. Metrics: {metrics}")
    
    # Save metrics to a JSON file
    import json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Test the model on some examples
    test_examples = [
        "I love this product! It's amazing!",
        "This is absolutely terrible, would not recommend.",
        "It's okay, nothing special but it works."
    ]
    
    logger.info("\nModel predictions on test examples:")
    for example in test_examples:
        result = sentiment_analyzer.predict(example)
        logger.info(f"Text: {example}")
        logger.info(f"  Sentiment: {result.sentiment_label} ({result.sentiment_score:.2f})")
        logger.info(f"  Confidence: {result.confidence:.2f}")
        logger.info(f"  Emotions: {result.emotions}")
        logger.info("")
    
    return sentiment_analyzer, metrics

def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Starting sentiment analysis model training with {args.model_type} classifier")
    
    try:
        model, metrics = train_model(args)
        logger.info(f"Training successful. Model saved to {args.output_dir}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return 1
    
    logger.info("Training complete!")
    return 0

if __name__ == "__main__":
    exit(main()) 