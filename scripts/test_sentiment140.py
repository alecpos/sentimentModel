#!/usr/bin/env python
"""Test enhanced ensemble model on Sentiment140 dataset."""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.ml.prediction.enhanced_ensemble import (
    EnhancedBaggingEnsemble,
    EnhancedStackingEnsemble,
    visualize_ensemble_performance
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sentiment140(max_samples=None):
    """Load and preprocess Sentiment140 dataset."""
    logger.info("Loading Sentiment140 dataset...")
    
    # Load dataset
    df = pd.read_csv(
        'data/sentiment140.csv',
        encoding='latin-1',
        names=['target', 'id', 'date', 'flag', 'user', 'text']
    )
    
    # Convert target from 0/4 to 0/1
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Subsample if specified
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    logger.info(f"Loaded {len(df)} samples")
    return df

def preprocess_text(texts):
    """Convert texts to TF-IDF features."""
    logger.info("Converting texts to TF-IDF features...")
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.7
    )
    
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def main():
    """Main function to test ensemble models."""
    try:
        # Load data
        df = load_sentiment140(max_samples=100000)  # Using 100k samples for faster testing
        
        # Preprocess text
        features, vectorizer = preprocess_text(df['text'])
        labels = df['target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Test Bagging Ensemble
        logger.info("Testing Bagging Ensemble...")
        base_estimator = DecisionTreeClassifier(random_state=42)
        bagging = EnhancedBaggingEnsemble(
            base_estimator=base_estimator,
            n_estimators=10,
            random_state=42
        )
        
        bagging.fit(X_train, y_train)
        bagging_preds = bagging.predict(X_test)
        bagging_acc = accuracy_score(y_test, bagging_preds)
        logger.info(f"Bagging Accuracy: {bagging_acc:.4f}")
        
        # Save bagging performance visualization
        fig = visualize_ensemble_performance(bagging)
        fig.savefig('results/bagging_performance.png')
        
        # Test Stacking Ensemble
        logger.info("Testing Stacking Ensemble...")
        base_estimators = [
            DecisionTreeClassifier(random_state=42),
            RandomForestClassifier(n_estimators=100, random_state=42),
            LogisticRegression(random_state=42)
        ]
        meta_learner = LogisticRegression(random_state=42)
        
        stacking = EnhancedStackingEnsemble(
            base_estimators=base_estimators,
            meta_learner=meta_learner,
            use_proba=True,
            n_splits=5,
            random_state=42
        )
        
        # Split training data further for stacking
        X_train_stack, X_val, y_train_stack, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        
        stacking.fit(X_train_stack, y_train_stack, X_val, y_val)
        stacking_preds = stacking.predict(X_test)
        stacking_acc = accuracy_score(y_test, stacking_preds)
        logger.info(f"Stacking Accuracy: {stacking_acc:.4f}")
        
        # Save stacking performance visualization
        fig = visualize_ensemble_performance(stacking)
        fig.savefig('results/stacking_performance.png')
        
        # Save detailed results
        results = {
            'bagging': {
                'accuracy': bagging_acc,
                'classification_report': classification_report(y_test, bagging_preds)
            },
            'stacking': {
                'accuracy': stacking_acc,
                'classification_report': classification_report(y_test, stacking_preds)
            }
        }
        
        # Save results to file
        with open('results/ensemble_results.txt', 'w') as f:
            f.write("Enhanced Ensemble Results on Sentiment140\n")
            f.write("=======================================\n\n")
            
            f.write("Bagging Ensemble\n")
            f.write("--------------\n")
            f.write(f"Accuracy: {results['bagging']['accuracy']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(results['bagging']['classification_report'])
            f.write("\n\n")
            
            f.write("Stacking Ensemble\n")
            f.write("---------------\n")
            f.write(f"Accuracy: {results['stacking']['accuracy']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(results['stacking']['classification_report'])
        
        logger.info("Results saved to results/ensemble_results.txt")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 