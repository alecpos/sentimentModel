#!/usr/bin/env python
"""
Simple test script for the Sentiment Adapter for AdScorePredictor

This script tests the adapter on challenging sentiment examples and prints the results.
"""

import logging
from sentiment_adapter import SentimentAdapterForAdPredictor, train_adapter_on_sentiment140

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_adapter_on_examples():
    """Test the adapter on challenging examples"""
    
    # Create an adapter
    # Option 1: Train a new adapter (using more data for better performance)
    logger.info("Training adapter on 10,000 examples from sentiment140 dataset...")
    adapter = train_adapter_on_sentiment140(sample_size=10000, find_threshold=True)
    
    # Option 2: Load a pre-trained adapter if available
    # adapter = SentimentAdapterForAdPredictor()
    # adapter.load("path/to/adapter.joblib")  # Uncomment if you have a saved adapter
    
    # Challenging examples for sentiment analysis
    examples = [
        # Negation examples
        "I don't hate this product.",  # Positive through double negation
        "Not the worst experience I've had.",  # Positive through double negation
        "I can't say I'm disappointed.",  # Positive through complex negation
        
        # Subtle expressions
        "This product is growing on me.",  # Positive but subtle
        "I'm starting to question my decision to buy this.",  # Negative but subtle
        "Let's just say I've had better experiences.",  # Negative but understated
        
        # Mixed sentiment
        "The interface is great but the performance is terrible.",  # Mixed (overall negative)
        "While the customer service was awful, the product itself is good.",  # Mixed (overall positive)
        
        # Comparative sentiment
        "This is slightly better than the last version.",  # Mild positive
        "Not as good as I expected, but still usable.",  # Mild negative
        
        # Sarcasm (challenging)
        "Oh great, another update that breaks everything.",  # Negative (sarcastic)
        "Just what I needed, more problems to fix.",  # Negative (sarcastic)
        
        # Questions
        "Why would anyone buy this product?",  # Negative (rhetorical)
        "Have you tried turning it off and on again?",  # Neutral (common IT advice)
        
        # Emoticons and slang
        "This product is lit! 🔥",  # Positive (slang)
        "The service was meh :/",  # Negative (slang + emoticon)
        
        # Challenging examples from original test
        "Going to miss Pastor's sermon on Faith...",  # Negative
        "I don't feel good",  # Negative
        "what is with msn making me click the wrong people" # Negative complaint
    ]
    
    # Process examples
    print("\n=== SENTIMENT ANALYSIS RESULTS ===\n")
    print(f"{'TEXT':<50} | {'SENTIMENT':<10} | {'SCORE':<6} | {'CONFIDENCE':<10} | {'MODEL':<10}")
    print("-" * 100)
    
    for text in examples:
        # Get prediction from adapter
        result = adapter.predict_sentiment(text)
        
        # Print results
        score = f"{result['score']:.1f}"
        confidence = f"{result['confidence']:.2f}"
        model_used = result.get('model_used', 'unknown')
        print(f"{text[:47] + '...' if len(text) > 50 else text:<50} | {result['sentiment']:<10} | {score:<6} | {confidence:<10} | {model_used:<10}")
    
    # Print adapter information
    print("\n=== ADAPTER INFORMATION ===\n")
    print(f"Threshold: {adapter.threshold}")
    print(f"Calibrated: {adapter.is_calibrated}")
    print(f"Enhanced preprocessing: {adapter.use_enhanced_preprocessing}")
    print(f"Using internal model fallback: {adapter.using_fallback}")
    print(f"AdScorePredictor is fitted: {adapter.is_ad_predictor_fitted}")
    
    if hasattr(adapter, 'metrics') and adapter.metrics:
        print("\n=== CALIBRATION METRICS ===\n")
        for key, value in adapter.metrics.items():
            print(f"{key}: {value}")
    
    # Save the trained adapter
    adapter.save("trained_sentiment_adapter.joblib")
    logger.info("Adapter saved to trained_sentiment_adapter.joblib")

if __name__ == "__main__":
    test_adapter_on_examples() 