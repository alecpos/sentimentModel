#!/usr/bin/env python
"""
Script to train the sentiment adapter on a larger dataset with progress tracking.
"""

import argparse
import logging
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sentiment_adapter import SentimentAdapterForAdPredictor
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("large_dataset_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_on_large_dataset(sample_size=10000, test_size=0.2, batch_size=1000):
    """
    Train the sentiment adapter on a large dataset with progress tracking.
    
    Args:
        sample_size: Number of examples to use for training/testing
        test_size: Fraction of data to use for testing
        batch_size: Number of examples to process in each batch
    """
    start_time = time.time()
    logger.info(f"Starting training on {sample_size} examples (test_size={test_size}, batch_size={batch_size})")
    
    # Load sentiment140 data
    try:
        logger.info("Loading Sentiment140 dataset...")
        df = pd.read_csv('sentiment140.csv', 
                         encoding='latin-1', 
                         names=['target', 'id', 'date', 'flag', 'user', 'text'])
        logger.info(f"Dataset loaded with {len(df)} examples")
    except FileNotFoundError:
        logger.error("sentiment140.csv not found.")
        return
    
    # Convert Twitter sentiment (0=negative, 4=positive) to binary (0=negative, 1=positive)
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Sample balanced data
    logger.info(f"Sampling {sample_size} examples (balanced)...")
    pos_samples = df[df['target'] == 1].sample(sample_size // 2, random_state=42)
    neg_samples = df[df['target'] == 0].sample(sample_size // 2, random_state=42)
    sampled_df = pd.concat([pos_samples, neg_samples])
    
    # Shuffle the data
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into training and testing sets
    logger.info("Splitting into training and testing sets...")
    X = sampled_df['text'].values
    y = sampled_df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")
    
    # Create adapter with fallback
    logger.info("Creating sentiment adapter with fallback...")
    adapter = SentimentAdapterForAdPredictor(
        fallback_to_internal_model=True,
        use_enhanced_preprocessing=True
    )
    
    # Force using fallback
    adapter.is_ad_predictor_fitted = False
    adapter.using_fallback = True
    
    # Initialize internal model
    adapter._initialize_internal_model()
    
    # Process training data in batches
    logger.info("Processing training data in batches...")
    all_features = []
    
    # Use tqdm for progress tracking
    for i in tqdm(range(0, len(X_train), batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, len(X_train))
        batch_texts = X_train[i:batch_end]
        
        batch_features = []
        for text in batch_texts:
            # Preprocess the text
            processed_text = adapter._preprocess_for_sentiment(text)
            # Extract features
            features = adapter._extract_sentiment_features(processed_text)
            batch_features.append(features)
        
        all_features.extend(batch_features)
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(X_train) + batch_size - 1)//batch_size} " +
                   f"({batch_end}/{len(X_train)} examples)")
    
    # Train the internal model
    logger.info("Training internal model...")
    adapter._train_internal_model(all_features, y_train)
    
    # Save the trained adapter
    model_filename = f"adapter_trained_{sample_size}_examples.joblib"
    adapter.save(model_filename)
    logger.info(f"Adapter saved to {model_filename}")
    
    # Evaluate on test data
    logger.info("Evaluating on test data...")
    evaluate_adapter(adapter, X_test, y_test)
    
    # Log training time
    training_time = time.time() - start_time
    logger.info(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

def evaluate_adapter(adapter, X_test, y_test):
    """
    Evaluate the adapter on test data.
    
    Args:
        adapter: Trained sentiment adapter
        X_test: Test text data
        y_test: True labels
    """
    # Get predictions
    predictions = []
    confidence_scores = []
    
    logger.info("Making predictions on test data...")
    for i, text in enumerate(tqdm(X_test, desc="Testing")):
        result = adapter.predict_sentiment(text)
        label = 1 if result['sentiment'] == 'positive' else 0
        predictions.append(label)
        confidence_scores.append(result['confidence'])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    
    # Detailed classification report
    report = classification_report(y_test, predictions)
    logger.info(f"Classification Report:\n{report}")
    
    # Create visualizations
    create_evaluation_charts(y_test, predictions, confidence_scores)

def create_evaluation_charts(y_true, y_pred, confidence_scores):
    """
    Create evaluation visualizations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence_scores: Confidence scores for predictions
    """
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=['Predicted Negative', 'Predicted Positive'],
               yticklabels=['True Negative', 'True Positive'])
    ax1.set_title('Confusion Matrix')
    
    # Confidence distribution by class
    confidence_by_class = {
        'correct': [confidence_scores[i] for i in range(len(y_true)) if y_true[i] == y_pred[i]],
        'incorrect': [confidence_scores[i] for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    }
    
    ax2.hist(confidence_by_class['correct'], alpha=0.5, bins=20, label='Correct Predictions', color='green')
    ax2.hist(confidence_by_class['incorrect'], alpha=0.5, bins=20, label='Incorrect Predictions', color='red')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('large_dataset_evaluation.png')
    logger.info("Saved evaluation charts to large_dataset_evaluation.png")
    
    # Create ROC curve
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import roc_curve, auc
    
    # Convert confidence scores to probability of positive class
    # For negative predictions, confidence is for negative class, so we need 1-confidence for positive class
    proba_positive = []
    for i in range(len(y_pred)):
        if y_pred[i] == 0:  # If prediction is negative
            proba_positive.append(1 - confidence_scores[i])
        else:  # If prediction is positive
            proba_positive.append(confidence_scores[i])
    
    fpr, tpr, _ = roc_curve(y_true, proba_positive)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    logger.info("Saved ROC curve to roc_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment adapter on a large dataset')
    parser.add_argument('--sample-size', type=int, default=10000, 
                        help='Number of examples to sample (default: 10000)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of examples to process in each batch (default: 1000)')
    
    args = parser.parse_args()
    
    train_on_large_dataset(args.sample_size, args.test_size, args.batch_size) 