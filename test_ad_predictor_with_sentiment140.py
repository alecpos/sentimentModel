#!/usr/bin/env python
"""
Test script for evaluating AdScorePredictor with Sentiment140 data

This script tests if the AdScorePredictor class can be used with Sentiment140 data
for sentiment analysis tasks, focusing on direct prediction without prior training.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import the model to test
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sentiment140_sample(file_path="sentiment140.csv", sample_size=100, random_state=42):
    """
    Load and sample data from the Sentiment140 dataset.
    
    Args:
        file_path (str): Path to the Sentiment140 dataset CSV file
        sample_size (int): Number of samples to use (default: 100)
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Sampled dataset with columns 'text' and 'sentiment'
    """
    logger.info(f"Loading data from {file_path}")
    
    # Column names for the Sentiment140 dataset
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    try:
        df = pd.read_csv(file_path, encoding='latin-1', names=column_names)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Convert target from 0/4 to 0/1 for negative/positive sentiment
        df['sentiment'] = df['target'].map({0: 0, 4: 1})
        
        # Keep only text and sentiment columns to simplify
        df = df[['text', 'sentiment']]
        
        # Sample data if needed
        if 0 < sample_size < len(df):
            df = df.sample(sample_size, random_state=random_state)
            logger.info(f"Sampled {sample_size} rows")
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def test_ad_predictor_direct(sample_size=100):
    """
    Test the AdScorePredictor directly on Sentiment140 data without training
    
    This approach uses the model's direct prediction capabilities without
    attempting to train it on Sentiment140 data.
    
    Args:
        sample_size: Number of samples to use
        
    Returns:
        dict: Performance metrics
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"ad_predictor_direct_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_sentiment140_sample(sample_size=sample_size)
    
    logger.info("Initializing AdScorePredictor...")
    predictor = AdScorePredictor()
    
    # Test the model directly without training
    logger.info("Testing AdScorePredictor with direct prediction...")
    test_results = []
    
    for i, row in data.iterrows():
        text = row['text']
        true_sentiment = row['sentiment']
        
        # Predict using simple text input
        try:
            prediction = predictor.predict({'text': text, 'id': str(i)})
            
            # Map the score to sentiment (assuming score > 50 is positive)
            predicted_sentiment = 1 if prediction['score'] > 50 else 0
            sentiment_label = "positive" if predicted_sentiment == 1 else "negative"
            true_label = "positive" if true_sentiment == 1 else "negative"
            
            test_results.append({
                'text': text,
                'actual': true_label,
                'predicted': sentiment_label,
                'actual_value': true_sentiment,
                'predicted_value': predicted_sentiment,
                'score': prediction['score'],
                'confidence': prediction['confidence']
            })
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{sample_size} samples")
                
        except Exception as e:
            logger.error(f"Error predicting sample {i}: {e}")
    
    # Save results to CSV
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(f"{output_dir}/prediction_results.csv", index=False)
    
    # Calculate metrics
    if len(test_results) > 0:
        y_true = results_df['actual_value'].values
        y_pred = results_df['predicted_value'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        logger.info(f"AdScorePredictor Direct Test Results:")
        logger.info(f"Number of samples: {len(test_results)}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Save classification report
        clf_report = classification_report(y_true, y_pred, 
                                          target_names=['negative', 'positive'], 
                                          output_dict=True)
        pd.DataFrame(clf_report).transpose().to_csv(f"{output_dir}/classification_report.csv")
        
        # Create a confusion matrix visualization
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['negative', 'positive'], 
                        yticklabels=['negative', 'positive'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/confusion_matrix.png")
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
        
        # Calculate metrics by sentiment class
        pos_accuracy = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
        neg_accuracy = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
        
        logger.info(f"Positive class accuracy: {pos_accuracy:.4f}")
        logger.info(f"Negative class accuracy: {neg_accuracy:.4f}")
        
        # Analyze high confidence errors
        analyze_errors(results_df, output_dir)
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'positive_accuracy': pos_accuracy,
            'negative_accuracy': neg_accuracy,
            'samples_tested': len(test_results),
            'output_dir': output_dir
        }
    else:
        logger.error("No valid test results were collected")
        return {
            'error': "No valid test results",
            'samples_tested': 0
        }

def analyze_errors(results_df, output_dir):
    """
    Analyze prediction errors to understand model limitations
    
    Args:
        results_df: DataFrame with test results
        output_dir: Directory to save analysis
    """
    # Filter for errors
    errors = results_df[results_df['actual'] != results_df['predicted']]
    
    logger.info(f"Total errors: {len(errors)} out of {len(results_df)} samples ({len(errors)/len(results_df)*100:.2f}%)")
    
    # Save errors to CSV
    errors.to_csv(f"{output_dir}/prediction_errors.csv", index=False)
    
    # Group errors by sentiment
    error_types = errors.groupby(['actual', 'predicted']).size().reset_index(name='count')
    logger.info("Error types:")
    logger.info(error_types)
    
    # Convert confidence to numeric if it's not already
    if 'confidence' in errors.columns:
        errors['confidence'] = pd.to_numeric(errors['confidence'])
        
        # Find challenging examples - look at high confidence errors
        high_confidence_errors = errors[errors['confidence'] > 0.8].sort_values('confidence', ascending=False)
        logger.info(f"High confidence errors: {len(high_confidence_errors)}")
        
        if len(high_confidence_errors) > 0:
            logger.info("Top high confidence errors:")
            for _, row in high_confidence_errors.head(5).iterrows():
                logger.info(f"Text: {row['text']}")
                logger.info(f"Actual: {row['actual']}, Predicted: {row['predicted']}, Confidence: {row['confidence']:.4f}")
                logger.info("-" * 50)
            
            # Save high confidence errors
            high_confidence_errors.to_csv(f"{output_dir}/high_confidence_errors.csv", index=False)

def generate_report(metrics, output_dir):
    """
    Generate a summary report of the test results
    
    Args:
        metrics: Dictionary with test metrics
        output_dir: Directory to save report
    """
    report = [
        "# AdScorePredictor Sentiment140 Compatibility Test Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Metrics",
        f"- Samples tested: {metrics.get('samples_tested', 'N/A')}",
        f"- Overall accuracy: {metrics.get('accuracy', 'N/A'):.4f}",
        f"- Overall F1 score: {metrics.get('f1_score', 'N/A'):.4f}",
        f"- Positive class accuracy: {metrics.get('positive_accuracy', 'N/A'):.4f}",
        f"- Negative class accuracy: {metrics.get('negative_accuracy', 'N/A'):.4f}",
        "",
        "## Conclusion",
    ]
    
    # Add conclusion based on metrics
    if 'accuracy' in metrics:
        if metrics['accuracy'] > 0.7:
            report.append(
                "The AdScorePredictor shows good compatibility with Sentiment140 data, "
                "achieving an accuracy above 70%. It appears suitable for sentiment analysis tasks."
            )
        elif metrics['accuracy'] > 0.6:
            report.append(
                "The AdScorePredictor shows moderate compatibility with Sentiment140 data, "
                "achieving an accuracy between 60-70%. It may be suitable for sentiment analysis "
                "with some modifications or preprocessing."
            )
        else:
            report.append(
                "The AdScorePredictor shows poor compatibility with Sentiment140 data, "
                "achieving an accuracy below 60%. Significant modifications would be needed "
                "to make it suitable for sentiment analysis tasks."
            )
    else:
        report.append("Unable to determine compatibility due to errors during testing.")
    
    # Write the report
    with open(f"{output_dir}/compatibility_report.md", "w") as f:
        f.write("\n".join(report))
    
    logger.info(f"Report saved to {output_dir}/compatibility_report.md")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test AdScorePredictor with Sentiment140 data')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to use')
    
    args = parser.parse_args()
    
    try:
        results = test_ad_predictor_direct(sample_size=args.sample_size)
        
        if 'error' in results:
            logger.error(f"Testing failed: {results['error']}")
        else:
            logger.info(f"Testing completed successfully. Results saved to: {results['output_dir']}")
            generate_report(results, results['output_dir'])
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc()) 