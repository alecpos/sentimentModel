#!/usr/bin/env python
"""
Script to train and evaluate the internal fallback model in the sentiment adapter.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sentiment_adapter import SentimentAdapterForAdPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_internal_model(sample_size=500, test_size=0.3):
    """
    Train and evaluate the internal fallback model.
    
    Args:
        sample_size: Number of examples to use for training/testing
        test_size: Fraction of data to use for testing
    """
    logger.info(f"Loading Sentiment140 data with sample size {sample_size}...")
    
    # Load sentiment140 data
    try:
        df = pd.read_csv('sentiment140.csv', 
                         encoding='latin-1', 
                         names=['target', 'id', 'date', 'flag', 'user', 'text'])
    except FileNotFoundError:
        logger.error("sentiment140.csv not found.")
        return
    
    # Convert Twitter sentiment (0=negative, 4=positive) to binary (0=negative, 1=positive)
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Sample balanced data
    pos_samples = df[df['target'] == 1].sample(sample_size // 2, random_state=42)
    neg_samples = df[df['target'] == 0].sample(sample_size // 2, random_state=42)
    sampled_df = pd.concat([pos_samples, neg_samples])
    
    # Shuffle the data
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into training and testing sets
    X = sampled_df['text'].values
    y = sampled_df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")
    
    # 1. Create adapter without fallback (just uses AdScorePredictor)
    # The AdScorePredictor is known to not be fitted, so this should perform poorly
    logger.info("Creating adapter without fallback...")
    no_fallback_adapter = SentimentAdapterForAdPredictor(
        fallback_to_internal_model=False,
        use_enhanced_preprocessing=True
    )
    
    # 2. Create adapter with fallback and train its internal model
    logger.info("Creating adapter with fallback and training internal model...")
    fallback_adapter = SentimentAdapterForAdPredictor(
        fallback_to_internal_model=True,
        use_enhanced_preprocessing=True
    )
    
    # Manually trigger internal model training
    feature_dicts = []
    for text in X_train:
        # Preprocess the text
        processed_text = fallback_adapter._preprocess_for_sentiment(text)
        # Extract features
        features = fallback_adapter._extract_sentiment_features(processed_text)
        feature_dicts.append(features)
    
    # Force using fallback
    fallback_adapter.is_ad_predictor_fitted = False
    fallback_adapter.using_fallback = True
    
    # Initialize and train internal model
    fallback_adapter._initialize_internal_model()
    fallback_adapter._train_internal_model(feature_dicts, y_train)
    
    # Save the trained adapter
    fallback_adapter.save("internal_model_trained.joblib")
    logger.info("Adapter with trained internal model saved to internal_model_trained.joblib")
    
    # Evaluate both adapters
    logger.info("Evaluating adapters on test data...")
    evaluate_adapters(no_fallback_adapter, fallback_adapter, X_test, y_test)

def evaluate_adapters(no_fallback_adapter, fallback_adapter, X_test, y_test):
    """
    Evaluate and compare the performance of both adapters.
    
    Args:
        no_fallback_adapter: Adapter without fallback
        fallback_adapter: Adapter with fallback
        X_test: Test text data
        y_test: True labels
    """
    # Get predictions
    no_fallback_preds = []
    fallback_preds = []
    
    logger.info("\n=== SAMPLE PREDICTIONS ===\n")
    logger.info(f"{'TEXT':<40} | {'ACTUAL':<8} | {'NO_FALLBACK':<11} | {'WITH_FALLBACK':<13}")
    logger.info("-" * 80)
    
    # Show first 10 examples
    for i, text in enumerate(X_test[:10]):
        # No fallback predictions
        no_fallback_result = no_fallback_adapter.predict_sentiment(text)
        no_fallback_label = 1 if no_fallback_result['sentiment'] == 'positive' else 0
        no_fallback_preds.append(no_fallback_label)
        
        # With fallback predictions
        fallback_result = fallback_adapter.predict_sentiment(text)
        fallback_label = 1 if fallback_result['sentiment'] == 'positive' else 0
        fallback_preds.append(fallback_label)
        
        # Print the result
        actual = "positive" if y_test[i] == 1 else "negative"
        no_fallback = f"{no_fallback_result['sentiment']} ({no_fallback_result['model_used']})"
        fallback = f"{fallback_result['sentiment']} ({fallback_result['model_used']})"
        
        logger.info(f"{text[:37] + '...' if len(text) > 40 else text:<40} | {actual:<8} | {no_fallback:<18} | {fallback:<18}")
    
    # Continue with remaining examples (without printing)
    for i, text in enumerate(X_test[10:], 10):
        # No fallback predictions
        no_fallback_result = no_fallback_adapter.predict_sentiment(text)
        no_fallback_label = 1 if no_fallback_result['sentiment'] == 'positive' else 0
        no_fallback_preds.append(no_fallback_label)
        
        # With fallback predictions
        fallback_result = fallback_adapter.predict_sentiment(text)
        fallback_label = 1 if fallback_result['sentiment'] == 'positive' else 0
        fallback_preds.append(fallback_label)
    
    # Calculate metrics
    no_fallback_accuracy = accuracy_score(y_test, no_fallback_preds)
    no_fallback_f1 = f1_score(y_test, no_fallback_preds)
    
    fallback_accuracy = accuracy_score(y_test, fallback_preds)
    fallback_f1 = f1_score(y_test, fallback_preds)
    
    logger.info("\n=== PERFORMANCE METRICS ===\n")
    logger.info(f"{'Metric':<15} | {'Without Fallback':<20} | {'With Fallback':<20}")
    logger.info("-" * 60)
    logger.info(f"{'Accuracy':<15} | {no_fallback_accuracy:<20.4f} | {fallback_accuracy:<20.4f}")
    logger.info(f"{'F1 Score':<15} | {no_fallback_f1:<20.4f} | {fallback_f1:<20.4f}")
    
    logger.info("\nDetailed classification report (without fallback):")
    logger.info(classification_report(y_test, no_fallback_preds))
    
    logger.info("\nDetailed classification report (with fallback):")
    logger.info(classification_report(y_test, fallback_preds))
    
    # Create visualization
    create_comparison_chart(
        y_test, no_fallback_preds, fallback_preds,
        no_fallback_accuracy, fallback_accuracy,
        no_fallback_f1, fallback_f1
    )

def create_comparison_chart(y_test, no_fallback_preds, fallback_preds, 
                           no_fallback_accuracy, fallback_accuracy,
                           no_fallback_f1, fallback_f1):
    """
    Create visualization comparing the two models.
    """
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Metrics for plotting
    metrics = ['Accuracy', 'F1 Score']
    no_fallback_values = [no_fallback_accuracy, no_fallback_f1]
    fallback_values = [fallback_accuracy, fallback_f1]
    
    # Colors
    colors = ['#4CAF50', '#2196F3']
    
    # Bar chart for metrics
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, no_fallback_values, width, label='Without Fallback', color='#FF9800')
    ax1.bar(x + width/2, fallback_values, width, label='With Fallback', color='#4CAF50')
    
    # Add values on bars
    for i, v in enumerate(no_fallback_values):
        ax1.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
        
    for i, v in enumerate(fallback_values):
        ax1.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
    
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    
    # Confusion matrix visualization
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # First confusion matrix (no fallback)
    cm1 = confusion_matrix(y_test, no_fallback_preds)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='YlOrBr', ax=ax2,
               xticklabels=['Predicted Negative', 'Predicted Positive'],
               yticklabels=['True Negative', 'True Positive'])
    ax2.set_title('Confusion Matrix - Without Fallback')
    
    plt.tight_layout()
    plt.savefig('internal_model_comparison.png')
    logger.info("Saved performance comparison chart to internal_model_comparison.png")
    
    # Create a second figure for the second confusion matrix
    plt.figure(figsize=(8, 6))
    cm2 = confusion_matrix(y_test, fallback_preds)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
               xticklabels=['Predicted Negative', 'Predicted Positive'],
               yticklabels=['True Negative', 'True Positive'])
    plt.title('Confusion Matrix - With Fallback')
    plt.tight_layout()
    plt.savefig('fallback_confusion_matrix.png')
    logger.info("Saved fallback confusion matrix to fallback_confusion_matrix.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate the internal fallback model')
    parser.add_argument('--sample-size', type=int, default=500, 
                        help='Number of examples to sample (default: 500)')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Proportion of data to use for testing (default: 0.3)')
    
    args = parser.parse_args()
    
    train_internal_model(args.sample_size, args.test_size) 