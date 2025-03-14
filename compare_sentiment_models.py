#!/usr/bin/env python
"""
Compare HybridSentimentAnalyzer with AdScorePredictor

This script compares the performance of the HybridSentimentAnalyzer with the
AdScorePredictor on challenging sentiment examples.
"""

import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import for HybridSentimentAnalyzer
from hybrid_sentiment_analyzer import HybridSentimentAnalyzer

# Import for AdScorePredictor (adjust import path as needed)
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models():
    """Load both sentiment analysis models."""
    logger.info("Loading HybridSentimentAnalyzer...")
    hybrid_analyzer = HybridSentimentAnalyzer()
    
    logger.info("Loading AdScorePredictor...")
    ad_score_predictor = AdScorePredictor()
    
    return hybrid_analyzer, ad_score_predictor

def compare_on_challenging_examples(output_dir="model_comparison"):
    """Compare both models on challenging sentiment examples."""
    hybrid_analyzer, ad_score_predictor = load_models()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"comparison_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Define challenging examples focused on nuanced sentiment
    challenging_examples = [
        "Not as bad as I thought it would be, but still not worth the money.",
        "I can't say I hate it, but I definitely don't love it either.",
        "It's honestly surprising how something so expensive could be so average.",
        "This isn't terrible, but there are much better alternatives.",
        "I don't think I will be buying this again, although it does have some good points.",
        "The product is fine, but I expected more given the price",
        "It's not terrible, just mediocre",
        "I don't hate it, but I wouldn't recommend it either",
        "This is absolutely not the best purchase I've made",
        "The service wasn't as bad as I expected",
        "While it has a few good features, overall I'm disappointed.",
        "I'm on the fence about recommending this, it has pros and cons.",
        "For the price, I expected better quality."
    ]
    
    # Collect results
    results = []
    
    for text in challenging_examples:
        # Get predictions
        hybrid_result = hybrid_analyzer.predict(text)
        
        try:
            ad_score_result = ad_score_predictor.predict_sentiment(text)
            ad_score_label = "positive" if ad_score_result > 0 else "negative"
            ad_score_confidence = abs(ad_score_result)
        except Exception as e:
            logger.error(f"Error with AdScorePredictor: {e}")
            ad_score_result = 0
            ad_score_label = "error"
            ad_score_confidence = 0
        
        # Log results
        logger.info(f"\nText: {text}")
        logger.info(f"Hybrid: {hybrid_result.sentiment_label} ({hybrid_result.sentiment_score:.2f}, confidence: {hybrid_result.confidence:.2f})")
        logger.info(f"AdScore: {ad_score_label} ({ad_score_result:.2f}, confidence: {ad_score_confidence:.2f})")
        
        # Store result for comparison
        results.append({
            "text": text,
            "hybrid_label": hybrid_result.sentiment_label,
            "hybrid_score": hybrid_result.sentiment_score,
            "hybrid_confidence": hybrid_result.confidence,
            "adscore_label": ad_score_label,
            "adscore_score": ad_score_result,
            "adscore_confidence": ad_score_confidence,
            "agreement": hybrid_result.sentiment_label == ad_score_label
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(result_dir, "comparison_results.json")
    results_df.to_json(results_path, orient='records', indent=2)
    logger.info(f"Saved comparison results to {results_path}")
    
    # Calculate agreement rate
    agreement_rate = results_df['agreement'].mean() * 100
    logger.info(f"Agreement rate between models: {agreement_rate:.1f}%")
    
    # Generate markdown report
    report_path = os.path.join(result_dir, "comparison_report.md")
    with open(report_path, 'w') as f:
        f.write("# Sentiment Model Comparison Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Agreement\n\n")
        f.write(f"The models agree on {agreement_rate:.1f}% of the examples.\n\n")
        
        f.write("## Example Predictions\n\n")
        for i, row in enumerate(results):
            f.write(f"### Example {i+1}\n\n")
            f.write(f"**Text**: {row['text']}\n\n")
            f.write("**HybridSentimentAnalyzer**:\n")
            f.write(f"- Label: {row['hybrid_label']}\n")
            f.write(f"- Score: {row['hybrid_score']:.2f}\n")
            f.write(f"- Confidence: {row['hybrid_confidence']:.2f}\n\n")
            f.write("**AdScorePredictor**:\n")
            f.write(f"- Label: {row['adscore_label']}\n")
            f.write(f"- Score: {row['adscore_score']:.2f}\n")
            f.write(f"- Confidence: {row['adscore_confidence']:.2f}\n\n")
            f.write(f"**Agreement**: {'Yes' if row['agreement'] else 'No'}\n\n")
            f.write("---\n\n")
    
    logger.info(f"Generated comparison report at {report_path}")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot sentiment scores for both models
    results_df['example_index'] = range(len(results_df))
    
    plt.subplot(2, 1, 1)
    plt.title('Sentiment Scores by Example')
    plt.plot(results_df['example_index'], results_df['hybrid_score'], 'bo-', label='Hybrid')
    plt.plot(results_df['example_index'], results_df['adscore_score'], 'ro-', label='AdScore')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Example Index')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot confidence for both models
    plt.subplot(2, 1, 2)
    plt.title('Confidence by Example')
    plt.plot(results_df['example_index'], results_df['hybrid_confidence'], 'bo-', label='Hybrid')
    plt.plot(results_df['example_index'], results_df['adscore_confidence'], 'ro-', label='AdScore')
    plt.ylim(0, 1.1)
    plt.xlabel('Example Index')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    vis_path = os.path.join(result_dir, "comparison_plot.png")
    plt.savefig(vis_path, dpi=300)
    logger.info(f"Saved visualization to {vis_path}")
    
    return results_df

if __name__ == "__main__":
    os.makedirs("model_comparison", exist_ok=True)
    comparison_results = compare_on_challenging_examples()
    
    # Print summary
    print("\nModel Comparison Summary:")
    print(f"Total examples: {len(comparison_results)}")
    print(f"Agreement rate: {comparison_results['agreement'].mean() * 100:.1f}%")
    
    # Analyze strengths of each model
    hybrid_wins = comparison_results[comparison_results['hybrid_confidence'] > comparison_results['adscore_confidence']]
    adscore_wins = comparison_results[comparison_results['adscore_confidence'] > comparison_results['hybrid_confidence']]
    
    print(f"\nHybridSentimentAnalyzer more confident on {len(hybrid_wins)} examples")
    print(f"AdScorePredictor more confident on {len(adscore_wins)} examples")
    
    print("\nComparison complete! Check the 'model_comparison' directory for detailed results.") 