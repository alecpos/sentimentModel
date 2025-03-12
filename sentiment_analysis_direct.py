#!/usr/bin/env python
"""
Sentiment Analysis Direct Training Script

This script trains a sentiment analysis model directly on the Sentiment140 dataset
without using the Kaggle pipeline. It implements a simple text classification model
with scikit-learn and saves the trained model for future use.

Usage:
    python sentiment_analysis_direct.py

The script will:
1. Load the Sentiment140 dataset directly
2. Preprocess the data
3. Train a text classification model
4. Evaluate the model
5. Save the trained model
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import joblib
import time
import re
from typing import Dict, List, Any, Tuple, Optional, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analysis model for text content.
    
    This class provides methods for training, evaluating, and using
    sentiment analysis models on text data.
    """
    
    def __init__(self, model_type='logistic', vectorizer_params=None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type: Type of classifier ('logistic' or 'random_forest')
            vectorizer_params: Optional parameters for the TF-IDF vectorizer
        """
        self.model_type = model_type
        
        # Set default vectorizer parameters if none provided
        if vectorizer_params is None:
            vectorizer_params = {
                'max_features': 5000,
                'ngram_range': (1, 3),
                'min_df': 5
            }
        
        # Initialize the vectorizer
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        
        # Initialize the model based on the specified type
        if model_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000, C=10.0, random_state=42)
        elif model_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create the pipeline
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', classifier)
        ])
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, texts, labels, test_size=0.2):
        """
        Train the sentiment analysis model.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0 for negative, 1 for positive)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must be the same")
        
        # Convert labels to numpy array
        labels_array = np.array(labels)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_array, test_size=test_size, random_state=42, stratify=labels_array
        )
        
        # Train the model
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        return {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "training_time_seconds": training_time
        }
    
    def predict(self, text):
        """
        Predict the sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Get predicted probabilities
        probs = self.model.predict_proba([cleaned_text])[0]
        predicted_class = self.model.predict([cleaned_text])[0]
        confidence = probs[predicted_class]
        
        # Map predicted class to label
        sentiment_label = "positive" if predicted_class == 1 else "negative"
        
        # Map to a score between -1 and 1
        if sentiment_label == "positive":
            sentiment_score = 0.5 + (0.5 * confidence)  # 0.5 to 1.0
        else:
            sentiment_score = -0.5 - (0.5 * confidence)  # -0.5 to -1.0
        
        return {
            "text": text,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "class": int(predicted_class)
        }
    
    def batch_predict(self, texts):
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of prediction results
        """
        return [self.predict(text) for text in texts]
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a pre-trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a sentiment analysis model on Sentiment140')
    parser.add_argument('--dataset_path', type=str, 
                      default='sentiment140/training.1600000.processed.noemoticon.csv',
                      help='Path to the Sentiment140 dataset CSV file')
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

def load_sentiment140(file_path, max_samples=None):
    """
    Load the Sentiment140 dataset.
    
    Args:
        file_path: Path to the dataset file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        DataFrame with the dataset
    """
    logger.info(f"Loading Sentiment140 dataset from {file_path}")
    
    # Define column names for the dataset (it doesn't have a header)
    column_names = ['target', 'id', 'date', 'query', 'user', 'text']
    
    try:
        # Try loading with UTF-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8', header=None, names=column_names)
    except UnicodeDecodeError:
        # Fall back to latin1 encoding
        logger.info("UTF-8 encoding failed, trying latin1 encoding")
        df = pd.read_csv(file_path, encoding='latin1', header=None, names=column_names)
    
    # Convert target from 0/4 to 0/1
    df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
    
    # Get class distribution
    class_dist = df['target'].value_counts()
    logger.info(f"Class distribution: {class_dist.to_dict()}")
    
    # Limit to max_samples if specified
    if max_samples is not None and len(df) > max_samples:
        # Ensure balanced sampling
        df_balanced = pd.DataFrame()
        for target in df['target'].unique():
            df_target = df[df['target'] == target]
            samples_per_class = max_samples // len(df['target'].unique())
            df_balanced = pd.concat([df_balanced, df_target.sample(samples_per_class, random_state=42)])
        
        df = df_balanced
        logger.info(f"Limited to {len(df)} samples ({samples_per_class} per class)")
    
    return df

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load and preprocess the dataset
        df = load_sentiment140(args.dataset_path, args.max_samples)
        
        # Clean text data
        logger.info("Preprocessing text data...")
        analyzer = SentimentAnalyzer(model_type=args.model_type)
        df['cleaned_text'] = df['text'].apply(analyzer.preprocess_text)
        
        # Train the model
        metrics = analyzer.train(
            texts=df['cleaned_text'].tolist(),
            labels=df['target'].tolist(),
            test_size=args.test_size
        )
        
        # Save the model
        model_path = os.path.join(
            args.output_dir, 
            f"sentiment140_{args.model_type}_{datetime.now().strftime('%Y%m%d')}.joblib"
        )
        analyzer.save_model(model_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Test on some examples
        test_examples = [
            "I love this product! It's amazing!",
            "This is absolutely terrible, would not recommend.",
            "It's okay, nothing special but it works."
        ]
        
        logger.info("\nModel predictions on test examples:")
        for example in test_examples:
            result = analyzer.predict(example)
            logger.info(f"Text: {example}")
            logger.info(f"  Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.2f})")
            logger.info(f"  Confidence: {result['confidence']:.2f}")
            logger.info("")
        
        logger.info(f"Training successful. Model saved to {model_path}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 