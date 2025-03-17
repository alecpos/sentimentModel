#!/usr/bin/env python
"""
Test script for SentimentAdapterForAdPredictor

This script tests the adapter's ability to improve AdScorePredictor's performance 
on sentiment analysis tasks using Sentiment140 data.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Import the adapter
from sentiment_adapter import (
    SentimentAdapterForAdPredictor, 
    load_sentiment140_sample,
    train_adapter_on_sentiment140
)

# Import the original AdScorePredictor for comparison
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_adapter_performance(sample_size=500, test_size=0.2):
    """
    Test the performance of the SentimentAdapterForAdPredictor compared to 
    raw AdScorePredictor on Sentiment140 data.
    
    Args:
        sample_size: Number of samples to use
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary with performance metrics
    """
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"adapter_test_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    texts, labels = load_sentiment140_sample(
        file_path="sentiment140.csv", 
        sample_size=sample_size
    )
    
    if not texts:
        logger.error("No data loaded, cannot run test")
        return None
    
    # Split into training and test sets
    from sklearn.model_selection import train_test_split
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    logger.info(f"Training set size: {len(train_texts)}")
    logger.info(f"Test set size: {len(test_texts)}")
    
    # Create and train the adapter
    logger.info("Training SentimentAdapterForAdPredictor...")
    
    adapter = SentimentAdapterForAdPredictor(
        use_enhanced_preprocessing=True,
        calibrate_scores=True
    )
    
    # Train the adapter using the training set
    adapter.train_calibration(train_texts, train_labels)
    
    # Find optimal threshold
    adapter.find_optimal_threshold(train_texts, train_labels)
    
    # Test the adapter on the test set
    logger.info("Testing SentimentAdapterForAdPredictor on test set...")
    
    adapter_results = []
    for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        result = adapter.predict_sentiment(text)
        
        predicted_sentiment = 1 if result['sentiment'] == 'positive' else 0
        
        adapter_results.append({
            'text': text,
            'actual': 'positive' if true_label == 1 else 'negative',
            'predicted': result['sentiment'],
            'actual_value': true_label,
            'predicted_value': predicted_sentiment,
            'score': result['score'],
            'confidence': result['confidence']
        })
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(test_texts)} test samples")
    
    # Now test with raw AdScorePredictor for comparison
    logger.info("Testing raw AdScorePredictor on test set...")
    
    raw_predictor = AdScorePredictor()
    raw_results = []
    
    for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        try:
            prediction = raw_predictor.predict({'text': text, 'id': str(i)})
            
            # Map to sentiment
            predicted_sentiment = 1 if prediction['score'] > 50 else 0
            sentiment_label = "positive" if predicted_sentiment == 1 else "negative"
            
            raw_results.append({
                'text': text,
                'actual': 'positive' if true_label == 1 else 'negative',
                'predicted': sentiment_label,
                'actual_value': true_label,
                'predicted_value': predicted_sentiment,
                'score': prediction['score'],
                'confidence': prediction['confidence']
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_texts)} test samples with raw predictor")
                
        except Exception as e:
            logger.warning(f"Error with raw predictor on sample {i}: {e}")
            # Use a default neutral prediction
            raw_results.append({
                'text': text,
                'actual': 'positive' if true_label == 1 else 'negative',
                'predicted': 'positive',  # Default prediction
                'actual_value': true_label,
                'predicted_value': 1,
                'score': 50.0,
                'confidence': 0.5
            })
    
    # Calculate metrics for adapter
    adapter_df = pd.DataFrame(adapter_results)
    adapter_df.to_csv(f"{output_dir}/adapter_results.csv", index=False)
    
    y_true = adapter_df['actual_value'].values
    y_pred = adapter_df['predicted_value'].values
    
    adapter_accuracy = accuracy_score(y_true, y_pred)
    adapter_f1 = f1_score(y_true, y_pred)
    adapter_precision = precision_score(y_true, y_pred)
    adapter_recall = recall_score(y_true, y_pred)
    
    # Calculate metrics for raw predictor
    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(f"{output_dir}/raw_predictor_results.csv", index=False)
    
    raw_y_true = raw_df['actual_value'].values
    raw_y_pred = raw_df['predicted_value'].values
    
    raw_accuracy = accuracy_score(raw_y_true, raw_y_pred)
    raw_f1 = f1_score(raw_y_true, raw_y_pred)
    raw_precision = precision_score(raw_y_true, raw_y_pred)
    raw_recall = recall_score(raw_y_true, raw_y_pred)
    
    # Log results
    logger.info("=== Performance Comparison ===")
    logger.info(f"Adapter Accuracy: {adapter_accuracy:.4f}")
    logger.info(f"Raw Predictor Accuracy: {raw_accuracy:.4f}")
    logger.info(f"Adapter F1 Score: {adapter_f1:.4f}")
    logger.info(f"Raw Predictor F1 Score: {raw_f1:.4f}")
    
    # Generate visual comparison
    metrics = {
        'Accuracy': [adapter_accuracy, raw_accuracy],
        'F1 Score': [adapter_f1, raw_f1],
        'Precision': [adapter_precision, raw_precision],
        'Recall': [adapter_recall, raw_recall]
    }
    
    for metric_name, values in metrics.items():
        plt.figure(figsize=(8, 5))
        plt.bar(['Adapter', 'Raw Predictor'], values, color=['blue', 'orange'])
        plt.title(f'Comparison of {metric_name}')
        plt.ylabel(metric_name)
        plt.ylim(0, 1)
        
        # Add values on top of the bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric_name.lower().replace(' ', '_')}_comparison.png")
        plt.close()
    
    # Save confusion matrices
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'positive'], 
                yticklabels=['negative', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Adapter Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adapter_confusion_matrix.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    raw_cm = confusion_matrix(raw_y_true, raw_y_pred)
    sns.heatmap(raw_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'positive'], 
                yticklabels=['negative', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Raw Predictor Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/raw_predictor_confusion_matrix.png")
    plt.close()
    
    # Save classification reports
    adapter_report = classification_report(y_true, y_pred, 
                                          target_names=['negative', 'positive'], 
                                          output_dict=True)
    pd.DataFrame(adapter_report).transpose().to_csv(f"{output_dir}/adapter_classification_report.csv")
    
    raw_report = classification_report(raw_y_true, raw_y_pred, 
                                      target_names=['negative', 'positive'], 
                                      output_dict=True)
    pd.DataFrame(raw_report).transpose().to_csv(f"{output_dir}/raw_predictor_classification_report.csv")
    
    # Save threshold and calibration information
    with open(f"{output_dir}/adapter_config.txt", "w") as f:
        f.write(f"Optimal threshold: {adapter.threshold}\n")
        f.write(f"Using enhanced preprocessing: {adapter.use_enhanced_preprocessing}\n")
        f.write(f"Is calibrated: {adapter.is_calibrated}\n")
        
        # Add stats if available
        if hasattr(adapter, 'stats'):
            f.write("\nTraining Statistics:\n")
            for key, value in adapter.stats.items():
                f.write(f"{key}: {value}\n")
    
    # Return performance metrics
    return {
        'adapter': {
            'accuracy': adapter_accuracy,
            'f1_score': adapter_f1,
            'precision': adapter_precision,
            'recall': adapter_recall
        },
        'raw_predictor': {
            'accuracy': raw_accuracy,
            'f1_score': raw_f1,
            'precision': raw_precision,
            'recall': raw_recall
        },
        'improvement': {
            'accuracy': adapter_accuracy - raw_accuracy,
            'f1_score': adapter_f1 - raw_f1,
            'precision': adapter_precision - raw_precision,
            'recall': adapter_recall - raw_recall
        },
        'output_dir': output_dir
    }

def test_challenging_examples(adapter_path=None):
    """
    Test the adapter on a set of challenging examples.
    
    Args:
        adapter_path: Path to a saved adapter (creates a new one if None)
    """
    # Create or load adapter
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Loading adapter from {adapter_path}")
        adapter = SentimentAdapterForAdPredictor()
        adapter.load(adapter_path)
    else:
        logger.info("Training a new adapter on a small sample")
        adapter = train_adapter_on_sentiment140(sample_size=500)
    
    # Define challenging examples
    challenging_examples = [
        # Negation
        "I don't feel good",  # Negative
        "Not a bad product",  # Positive (negated negative)
        
        # Sarcasm
        "Just what I needed, another problem to fix",  # Negative
        "Oh sure, because waiting in line for hours is SO much fun",  # Negative
        
        # Mixed sentiment
        "The product is good but the service was terrible",  # Mixed/Negative
        "It's not perfect, but it gets the job done",  # Mixed/Positive
        
        # Subtle expressions
        "Going to miss Pastor's sermon on Faith...",  # Negative (missing something)
        "I might consider buying this again if they improve it",  # Conditional positive
        
        # Questions
        "Why would anyone buy this?",  # Negative
        "Has anyone tried the new feature?",  # Neutral/Question
        
        # Comparative
        "This is better than the last version",  # Positive
        "Their competitors offer much more value",  # Negative
    ]
    
    # Set up raw predictor for comparison
    raw_predictor = AdScorePredictor()
    
    # Test examples
    results = []
    
    for text in challenging_examples:
        # Get adapter prediction
        adapter_result = adapter.predict_sentiment(text)
        
        # Get raw predictor prediction
        try:
            raw_prediction = raw_predictor.predict({'text': text, 'id': 'challenge'})
            raw_sentiment = "positive" if raw_prediction['score'] > 50 else "negative"
            raw_score = raw_prediction['score']
            raw_confidence = raw_prediction['confidence']
        except Exception as e:
            logger.warning(f"Error with raw predictor: {e}")
            raw_sentiment = "unknown"
            raw_score = 0.0
            raw_confidence = 0.0
        
        # Add to results
        results.append({
            'text': text,
            'adapter_sentiment': adapter_result['sentiment'],
            'adapter_score': adapter_result['score'],
            'adapter_confidence': adapter_result['confidence'],
            'raw_sentiment': raw_sentiment,
            'raw_score': raw_score,
            'raw_confidence': raw_confidence,
            'same_prediction': adapter_result['sentiment'] == raw_sentiment
        })
    
    # Convert to dataframe and display
    results_df = pd.DataFrame(results)
    
    logger.info("\n=== Challenging Examples Test ===\n")
    for i, row in results_df.iterrows():
        logger.info(f"Text: {row['text']}")
        logger.info(f"Adapter: {row['adapter_sentiment']} (Score: {row['adapter_score']:.2f}, Confidence: {row['adapter_confidence']:.2f})")
        logger.info(f"Raw: {row['raw_sentiment']} (Score: {row['raw_score']:.2f}, Confidence: {row['raw_confidence']:.2f})")
        logger.info(f"Same prediction: {row['same_prediction']}")
        logger.info("-" * 50)
    
    # Calculate agreement rate
    agreement_rate = results_df['same_prediction'].mean()
    logger.info(f"Agreement rate: {agreement_rate:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("challenging_examples", exist_ok=True)
    results_df.to_csv(f"challenging_examples/results_{timestamp}.csv", index=False)
    
    return results_df

def generate_summary_report(metrics, output_path="adapter_summary_report.md"):
    """
    Generate a summary report of the adapter's performance.
    
    Args:
        metrics: Dictionary with performance metrics
        output_path: Path to save the report
    """
    adapter_metrics = metrics['adapter']
    raw_metrics = metrics['raw_predictor']
    improvement = metrics['improvement']
    
    report = [
        "# SentimentAdapterForAdPredictor Performance Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Performance Comparison",
        "",
        "| Metric | Adapter | Raw Predictor | Improvement |",
        "|--------|---------|---------------|-------------|",
        f"| Accuracy | {adapter_metrics['accuracy']:.4f} | {raw_metrics['accuracy']:.4f} | {improvement['accuracy']:.4f} |",
        f"| F1 Score | {adapter_metrics['f1_score']:.4f} | {raw_metrics['f1_score']:.4f} | {improvement['f1_score']:.4f} |",
        f"| Precision | {adapter_metrics['precision']:.4f} | {raw_metrics['precision']:.4f} | {improvement['precision']:.4f} |",
        f"| Recall | {adapter_metrics['recall']:.4f} | {raw_metrics['recall']:.4f} | {improvement['recall']:.4f} |",
        "",
        "## Conclusion",
    ]
    
    # Add conclusion based on metrics
    avg_improvement = sum(improvement.values()) / len(improvement)
    
    if avg_improvement > 0.2:
        report.append(
            "The SentimentAdapterForAdPredictor shows **significant improvement** over the raw "
            "AdScorePredictor for sentiment analysis tasks. The adapter's calibration and "
            "preprocessing significantly enhance the model's ability to correctly identify both "
            "positive and negative sentiments."
        )
    elif avg_improvement > 0.1:
        report.append(
            "The SentimentAdapterForAdPredictor shows **moderate improvement** over the raw "
            "AdScorePredictor for sentiment analysis tasks. The adapter's calibration and "
            "preprocessing help better align the model's outputs with sentiment classifications."
        )
    elif avg_improvement > 0:
        report.append(
            "The SentimentAdapterForAdPredictor shows **slight improvement** over the raw "
            "AdScorePredictor for sentiment analysis tasks. While the improvement is positive, "
            "further enhancements to the adapter may be beneficial."
        )
    else:
        report.append(
            "The SentimentAdapterForAdPredictor does not show improvement over the raw "
            "AdScorePredictor for sentiment analysis tasks. This could indicate issues with "
            "the adapter implementation or that the raw predictor is already optimized for "
            "the specific test data used."
        )
    
    # Add recommendations
    report.extend([
        "",
        "## Recommendations",
        "",
        "Based on these results, consider the following next steps:",
        "",
        "1. **Feature Engineering**: Explore additional sentiment-specific features to improve classification",
        "2. **Preprocessing Enhancements**: Further optimize text preprocessing for sentiment analysis",
        "3. **Threshold Tuning**: Experiment with different threshold values for different use cases",
        "4. **Larger Training Sample**: Train the adapter on a larger dataset to improve calibration",
        "5. **Domain Adaptation**: Fine-tune the adapter for specific domains if needed"
    ])
    
    # Write the report
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    
    logger.info(f"Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SentimentAdapterForAdPredictor')
    parser.add_argument('--sample-size', type=int, default=500, 
                        help='Number of samples to use for performance test')
    parser.add_argument('--test-size', type=float, default=0.2, 
                        help='Proportion of data for testing')
    parser.add_argument('--mode', choices=['performance', 'challenging', 'both'], 
                        default='both', help='Test mode')
    parser.add_argument('--adapter-path', type=str, default=None,
                        help='Path to a saved adapter (for challenging examples test)')
    
    args = parser.parse_args()
    
    try:
        if args.mode in ['performance', 'both']:
            # Test overall performance
            logger.info("Testing adapter performance...")
            metrics = test_adapter_performance(
                sample_size=args.sample_size,
                test_size=args.test_size
            )
            
            if metrics:
                # Generate summary report
                generate_summary_report(metrics, 
                                       output_path=f"{metrics['output_dir']}/summary_report.md")
                
                logger.info(f"Performance test completed. Results saved to: {metrics['output_dir']}")
                
                # Save adapter if it performed well
                if metrics['improvement']['accuracy'] > 0.1:
                    adapter = train_adapter_on_sentiment140(
                        sample_size=args.sample_size
                    )
                    adapter.save(f"{metrics['output_dir']}/trained_adapter.joblib")
                    logger.info(f"Adapter saved to {metrics['output_dir']}/trained_adapter.joblib")
        
        if args.mode in ['challenging', 'both']:
            # Test challenging examples
            logger.info("Testing adapter on challenging examples...")
            test_challenging_examples(adapter_path=args.adapter_path)
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc()) 