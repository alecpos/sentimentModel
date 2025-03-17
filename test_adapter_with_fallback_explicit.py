#!/usr/bin/env python
"""
Test script for comparing AdScorePredictor vs internal fallback model functionality
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentiment_adapter import SentimentAdapterForAdPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test examples with known sentiment
test_examples = [
    # Positive examples
    {"text": "I love this product, it's amazing!", "sentiment": 1},
    {"text": "Best purchase I've ever made, totally worth it.", "sentiment": 1},
    {"text": "This service exceeded my expectations.", "sentiment": 1},
    {"text": "Great features and excellent customer support.", "sentiment": 1},
    {"text": "I'm really happy with my decision to buy this.", "sentiment": 1},
    
    # Negative examples
    {"text": "This is the worst product I've ever used.", "sentiment": 0},
    {"text": "Terrible experience, would not recommend.", "sentiment": 0},
    {"text": "Waste of money, don't buy this.", "sentiment": 0},
    {"text": "Customer service was awful and unhelpful.", "sentiment": 0},
    {"text": "Very disappointed with the quality.", "sentiment": 0},
    
    # Complex/nuanced examples
    {"text": "I don't hate this product.", "sentiment": 1},  # Double negative
    {"text": "Not the worst experience I've had.", "sentiment": 1},  # Understated positive
    {"text": "This product is growing on me.", "sentiment": 1},  # Gradual positive
    {"text": "I'm starting to question my decision to buy this.", "sentiment": 0},  # Subtle negative
    {"text": "The interface is great but the performance is terrible.", "sentiment": 0},  # Mixed but net negative
    {"text": "While the customer service was awful, the product itself is good.", "sentiment": 1},  # Mixed but net positive
    {"text": "This is slightly better than the last version.", "sentiment": 1},  # Comparative positive
    {"text": "Not as good as I expected, but still usable.", "sentiment": 0},  # Disappointed but functional
    {"text": "Oh great, another update that breaks everything.", "sentiment": 0},  # Sarcasm
    {"text": "Just what I needed, more problems to fix.", "sentiment": 0},  # Sarcasm
]

def test_adapter_configurations():
    """Test different adapter configurations and compare results"""
    
    # Extract test data
    texts = [example["text"] for example in test_examples]
    labels = [example["sentiment"] for example in test_examples]
    
    # Configuration 1: Without fallback (just using AdScorePredictor)
    logger.info("Testing adapter without fallback...")
    adapter_no_fallback = SentimentAdapterForAdPredictor(
        fallback_to_internal_model=False,
        use_enhanced_preprocessing=True,
        threshold=50.0
    )
    
    # Configuration 2: With fallback to internal model
    logger.info("Testing adapter with fallback...")
    adapter_with_fallback = SentimentAdapterForAdPredictor(
        fallback_to_internal_model=True,
        use_enhanced_preprocessing=True,
        threshold=50.0
    )
    
    # Test both configurations
    results_no_fallback = []
    results_with_fallback = []
    
    logger.info("\n=== TEST RESULTS ===\n")
    logger.info(f"{'TEXT':<40} | {'ACTUAL':<8} | {'NO_FALLBACK':<11} | {'WITH_FALLBACK':<13}")
    logger.info("-" * 80)
    
    for i, text in enumerate(texts):
        # Without fallback
        pred_no_fallback = adapter_no_fallback.predict_sentiment(text)
        no_fallback_label = 1 if pred_no_fallback['sentiment'] == 'positive' else 0
        results_no_fallback.append(no_fallback_label)
        
        # With fallback
        pred_with_fallback = adapter_with_fallback.predict_sentiment(text)
        with_fallback_label = 1 if pred_with_fallback['sentiment'] == 'positive' else 0
        results_with_fallback.append(with_fallback_label)
        
        # Print result
        actual = "positive" if labels[i] == 1 else "negative"
        no_fallback = f"{pred_no_fallback['sentiment']} ({pred_no_fallback['model_used']})"
        with_fallback = f"{pred_with_fallback['sentiment']} ({pred_with_fallback['model_used']})"
        
        logger.info(f"{text[:37] + '...' if len(text) > 40 else text:<40} | {actual:<8} | {no_fallback:<18} | {with_fallback:<18}")
    
    # Calculate metrics
    accuracy_no_fallback = accuracy_score(labels, results_no_fallback)
    f1_no_fallback = f1_score(labels, results_no_fallback)
    
    accuracy_with_fallback = accuracy_score(labels, results_with_fallback)
    f1_with_fallback = f1_score(labels, results_with_fallback)
    
    logger.info("\n=== PERFORMANCE METRICS ===\n")
    logger.info(f"Without fallback: Accuracy = {accuracy_no_fallback:.4f}, F1 = {f1_no_fallback:.4f}")
    logger.info(f"With fallback:    Accuracy = {accuracy_with_fallback:.4f}, F1 = {accuracy_with_fallback:.4f}")
    
    logger.info("\nDetailed classification report (without fallback):")
    logger.info(classification_report(labels, results_no_fallback))
    
    logger.info("\nDetailed classification report (with fallback):")
    logger.info(classification_report(labels, results_with_fallback))
    
    # Show adapter settings
    logger.info("\n=== ADAPTER CONFIGURATIONS ===\n")
    logger.info(f"No fallback adapter:")
    logger.info(f"  - Is using fallback: {adapter_no_fallback.using_fallback}")
    logger.info(f"  - AdPredictor fitted: {adapter_no_fallback.is_ad_predictor_fitted}")
    logger.info(f"  - Enhanced preprocessing: {adapter_no_fallback.use_enhanced_preprocessing}")
    logger.info(f"  - Threshold: {adapter_no_fallback.threshold}")
    
    logger.info(f"\nWith fallback adapter:")
    logger.info(f"  - Is using fallback: {adapter_with_fallback.using_fallback}")
    logger.info(f"  - AdPredictor fitted: {adapter_with_fallback.is_ad_predictor_fitted}")
    logger.info(f"  - Enhanced preprocessing: {adapter_with_fallback.use_enhanced_preprocessing}")
    logger.info(f"  - Threshold: {adapter_with_fallback.threshold}")
    
    # Plot comparison
    plot_comparison(labels, results_no_fallback, results_with_fallback)
    
def plot_comparison(actual, pred_no_fallback, pred_with_fallback):
    """Create comparative visualization of results"""
    
    # Calculate correctly/incorrectly classified instances
    correct_no_fallback = sum(a == p for a, p in zip(actual, pred_no_fallback))
    correct_with_fallback = sum(a == p for a, p in zip(actual, pred_with_fallback))
    
    incorrect_no_fallback = len(actual) - correct_no_fallback
    incorrect_with_fallback = len(actual) - correct_with_fallback
    
    # Data for bar chart
    labels = ['Without Fallback', 'With Fallback']
    correct = [correct_no_fallback, correct_with_fallback]
    incorrect = [incorrect_no_fallback, incorrect_with_fallback]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create stacked bar chart
    ax.bar(labels, correct, label='Correct Predictions', color='#4CAF50')
    ax.bar(labels, incorrect, bottom=correct, label='Incorrect Predictions', color='#F44336')
    
    # Add values on bars
    for i in range(len(labels)):
        # Correct predictions
        ax.text(i, correct[i]/2, f"{correct[i]}", ha='center', va='center', color='white', fontweight='bold')
        # Incorrect predictions
        ax.text(i, correct[i] + incorrect[i]/2, f"{incorrect[i]}", ha='center', va='center', color='white', fontweight='bold')
        # Accuracy percentage
        accuracy = 100 * correct[i] / (correct[i] + incorrect[i])
        ax.text(i, correct[i] + incorrect[i] + 0.5, f"{accuracy:.1f}%", ha='center', va='bottom')
    
    # Add labels and title
    ax.set_ylabel('Number of Examples')
    ax.set_title('Performance Comparison: Fallback vs No Fallback')
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('fallback_comparison.png')
    logger.info("Saved performance comparison to fallback_comparison.png")

if __name__ == "__main__":
    test_adapter_configurations() 