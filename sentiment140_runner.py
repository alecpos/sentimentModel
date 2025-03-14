#!/usr/bin/env python
"""
Sentiment140 Runner for Hybrid Sentiment Analyzer

This script demonstrates how to:
1. Load and preprocess the Sentiment140 dataset
2. Train the Hybrid Sentiment Analyzer using sentiment140 data
3. Evaluate model performance on this dataset
4. Fix the f1 score access issue in the original code

Usage:
    python sentiment140_runner.py --data_path path/to/sentiment140.csv --sample_size -1
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from typing import Dict, List, Any, Tuple, Optional

# Import our hybrid sentiment analyzer
from hybrid_sentiment_analyzer import HybridSentimentAnalyzer, analyze_sentiment_batch, preprocess_text

import logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Hybrid Sentiment Analyzer on Sentiment140 dataset")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="sentiment140.csv",
        help="Path to sentiment140 dataset CSV file"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=-1,
        help="Number of samples to use from dataset (use -1 for all)"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="sentiment_results",
        help="Directory to save model and results"
    )
    parser.add_argument(
        "--use_xgboost", 
        action="store_true",
        default=True,
        help="Use XGBoost classifier (default: True)"
    )
    
    return parser.parse_args()


def load_sentiment140(data_path: str, sample_size: int = -1) -> pd.DataFrame:
    """
    Load the Sentiment140 dataset.
    
    Args:
        data_path: Path to the CSV file
        sample_size: Number of samples to load (-1 for all)
        
    Returns:
        DataFrame with the dataset
    """
    # Check if file exists
    if not os.path.exists(data_path):
        # If not, try to download it
        logger.info(f"Dataset not found at {data_path}, attempting to download...")
        try:
            from urllib import request
            url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
            # Create directory if needed
            os.makedirs(os.path.dirname(data_path) or '.', exist_ok=True)
            # Download and extract
            import zipfile
            from io import BytesIO
            
            logger.info(f"Downloading from {url}")
            response = request.urlopen(url)
            zipdata = BytesIO(response.read())
            
            with zipfile.ZipFile(zipdata) as zip_ref:
                # Extract the training file
                for file in zip_ref.namelist():
                    if "training" in file.lower() and file.endswith(".csv"):
                        logger.info(f"Extracting {file}")
                        with zip_ref.open(file) as source, open(data_path, 'wb') as target:
                            target.write(source.read())
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise FileNotFoundError(f"Dataset not found at {data_path} and download failed")

    # Define column names for Sentiment140
    columns = ["sentiment", "id", "date", "query", "user", "text"]
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, encoding="latin-1", names=columns)
    
    # Convert sentiment from Twitter format (0=negative, 4=positive) to binary (0=negative, 1=positive)
    df["sentiment"] = df["sentiment"].replace(4, 1)
    
    # Sample if requested
    if sample_size > 0 and sample_size < len(df):
        logger.info(f"Sampling {sample_size} examples from dataset")
        df = df.sample(n=sample_size, random_state=42)
    
    logger.info(f"Loaded {len(df)} examples")
    return df


def train_model(
    df: pd.DataFrame, 
    test_size: float = 0.2,
    use_xgboost: bool = True
) -> Tuple[HybridSentimentAnalyzer, Dict[str, float], pd.DataFrame]:
    """
    Train the hybrid sentiment analyzer on the given dataset.
    
    Args:
        df: DataFrame with text and sentiment columns
        test_size: Proportion to use for testing
        use_xgboost: Whether to use XGBoost
        
    Returns:
        Tuple of (trained analyzer, metrics dictionary, test dataframe)
    """
    # Create analyzer
    analyzer = HybridSentimentAnalyzer(use_xgboost=use_xgboost)
    
    # Prepare training data
    texts = df["text"].tolist()
    labels = df["sentiment"].tolist()
    
    # Train model and get metrics
    start_time = time.time()
    metrics, X_test, y_test, y_pred = analyzer.train(texts, labels, test_size=test_size)
    total_time = time.time() - start_time
    
    # Convert test data back to dataframe for analysis
    test_df = pd.DataFrame({
        "text": X_test,
        "true_sentiment": y_test,
        "predicted_sentiment": y_pred
    })
    
    # Add total training time to metrics
    metrics["total_time_seconds"] = total_time
    
    # Calculate and print the F1 score (fixing the unused f1 score issue)
    f1 = metrics["f1_score"]
    logger.info(f"F1 Score: {f1:.4f}")
    
    return analyzer, metrics, test_df


def visualize_results(metrics: Dict[str, float], test_df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualize the model results.
    
    Args:
        metrics: Dictionary of performance metrics
        test_df: DataFrame with test results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(test_df["true_sentiment"], test_df["predicted_sentiment"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Metrics summary
    plt.figure(figsize=(10, 6))
    metrics_to_plot = {k: v for k, v in metrics.items() 
                      if isinstance(v, (int, float)) and not isinstance(v, bool)
                      and k not in ["train_samples", "test_samples", "training_time_seconds", "total_time_seconds", "cv_scores"]}
    
    bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    plt.ylim(0, 1.0)
    plt.title("Model Performance Metrics")
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    # Create examples of correct and incorrect predictions
    examples = []
    
    # Find examples of correct predictions
    correct = test_df[test_df["true_sentiment"] == test_df["predicted_sentiment"]].sample(min(5, len(test_df)))
    for _, row in correct.iterrows():
        examples.append({
            "text": row["text"],
            "true_sentiment": "Positive" if row["true_sentiment"] == 1 else "Negative",
            "predicted_sentiment": "Positive" if row["predicted_sentiment"] == 1 else "Negative",
            "status": "Correct"
        })
    
    # Find examples of incorrect predictions
    incorrect = test_df[test_df["true_sentiment"] != test_df["predicted_sentiment"]].sample(min(5, len(test_df)))
    for _, row in incorrect.iterrows():
        examples.append({
            "text": row["text"],
            "true_sentiment": "Positive" if row["true_sentiment"] == 1 else "Negative",
            "predicted_sentiment": "Positive" if row["predicted_sentiment"] == 1 else "Negative",
            "status": "Incorrect"
        })
    
    # Save examples to CSV
    pd.DataFrame(examples).to_csv(os.path.join(output_dir, "prediction_examples.csv"), index=False)


def main():
    """Main function to run the sentiment analysis on Sentiment140."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    args = parse_args()
    
    # Load dataset
    df = load_sentiment140(args.data_path, args.sample_size)
    
    # Train model
    analyzer, metrics, test_df = train_model(
        df, 
        test_size=args.test_size,
        use_xgboost=args.use_xgboost
    )
    
    # Visualize results
    visualize_results(metrics, test_df, args.output_dir)
    
    # Save model
    model_path = os.path.join(args.output_dir, "hybrid_sentiment_model.joblib")
    analyzer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")  # Using the F1 score
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    
    if "mean_cv_accuracy" in metrics:
        logger.info(f"Mean CV Accuracy: {metrics['mean_cv_accuracy']:.4f}")
    
    logger.info(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("="*50)
    
    # Try a few examples
    logger.info("\nTrying a few new examples:")
    test_texts = [
        "I absolutely love the new features in this update!",
        "The customer service was terrible and the product broke after one day.",
        "It's okay, but I expected more for the price.",
        "Best purchase I've made all year! Highly recommend!",
        "Don't waste your money on this."
    ]
    
    results = analyze_sentiment_batch(test_texts, analyzer)
    
    for i, result in enumerate(results):
        logger.info(f"\nText: {test_texts[i]}")
        logger.info(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.2f})")
        logger.info(f"Confidence: {result['confidence']:.2f}")
        
        # Show top 3 feature importances if available
        if result['feature_importance']:
            top_features = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info("Top features:")
            for feature, importance in top_features:
                logger.info(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    main() 