#!/usr/bin/env python
"""
Enhanced Sentiment Analysis Training Script

This script demonstrates:
1. Training a transformer-based sentiment model with fairness evaluation
2. Applying bias mitigation techniques
3. Generating fairness reports and visualizations

Usage:
    python run_enhanced_training.py
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import json
import torch

# Import our modules
from fairness_evaluation import evaluate_fairness, plot_fairness_summary, generate_fairness_report
from bias_mitigation import apply_bias_mitigation_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run enhanced sentiment analysis training')
    
    # Dataset options
    parser.add_argument('--dataset_path', type=str, 
                      default=None,
                      help='Path to the Sentiment140 dataset CSV file')
    parser.add_argument('--max_samples', type=int, default=80000,
                      help='Maximum number of samples to use (None to use all)')
    parser.add_argument('--output_dir', type=str, default='enhanced_sentiment_results',
                      help='Directory to save results')
    
    # Model options
    parser.add_argument('--model_type', type=str, default='distilbert',
                      choices=['bert', 'roberta', 'distilbert', 'xlnet', 'logistic'],
                      help='Type of model to use')
    parser.add_argument('--train_mode', type=str, default='transformer',
                      choices=['transformer', 'traditional', 'both'],
                      help='Type of training to run')
    
    # Training options
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2,
                      help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate for training')
    parser.add_argument('--max_iter', type=int, default=5000,
                      help='Maximum iterations for logistic regression')
    
    # Fairness options
    parser.add_argument('--fairness_evaluation', action='store_true',
                      help='Perform fairness evaluation after training')
    parser.add_argument('--bias_mitigation', action='store_true',
                      help='Apply bias mitigation techniques during training')
    parser.add_argument('--comparative_analysis', action='store_true',
                      help='Compare performance with and without bias mitigation')
    
    return parser.parse_args()

def run_baseline_and_enhanced_training(args):
    """
    Run both baseline and enhanced model training for comparison.
    
    Args:
        args: Command-line arguments
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a progress tracking file
    progress_file = output_dir / "training_progress.log"
    with open(progress_file, 'w') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: train_mode={args.train_mode}, model_type={args.model_type}, max_samples={args.max_samples}\n")
    
    def update_progress(message):
        """Helper function to update progress in both log and file"""
        logger.info(message)
        with open(progress_file, 'a') as f:
            f.write(f"{time.strftime('%H:%M:%S')} - {message}\n")
    
    # Check if dataset path is provided
    if args.dataset_path is None:
        try:
            import kagglehub
            logger.info("No dataset path provided, downloading Sentiment140 from Kaggle...")
            dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
            csv_files = list(Path(dataset_path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the downloaded dataset")
            
            args.dataset_path = str(csv_files[0])
            logger.info(f"Using dataset at: {args.dataset_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            logger.error("Please provide a dataset path using --dataset_path")
            return 1
    
    # Log the start time
    start_time = time.time()
    logger.info(f"Starting enhanced sentiment analysis training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Configure output subdirectories
    baseline_dir = output_dir / "baseline"
    enhanced_dir = output_dir / "enhanced"
    comparison_dir = output_dir / "comparison"
    
    for directory in [baseline_dir, enhanced_dir, comparison_dir]:
        directory.mkdir(exist_ok=True)
    
    # Import here to avoid circular imports
    if args.train_mode in ['transformer', 'both']:
        update_progress("Importing transformer modules...")
        from transformer_sentiment_analysis import (
            load_sentiment140, 
            TransformerSentimentAnalyzer
        )
    
    if args.train_mode in ['traditional', 'both']:
        update_progress("Importing traditional ML modules...")
        from enhanced_sentiment_analysis import (
            load_sentiment140 as load_sentiment140_traditional,
            EnhancedSentimentAnalyzer
        )
    
    # Step 1: Load the dataset
    update_progress(f"Loading dataset from {args.dataset_path}...")
    update_progress(f"This may take a while for {args.max_samples} samples...")
    
    # Create a status file to track major steps
    status_file = output_dir / "status.json"
    status = {
        "started_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "config": vars(args),
        "steps": {
            "data_loading": {"status": "in_progress", "started_at": time.strftime('%H:%M:%S')}
        }
    }
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    if args.train_mode == 'transformer':
        df = load_sentiment140(args.dataset_path, args.max_samples)
        update_progress(f"Dataset loaded: {len(df)} samples")
        
        # Update status
        status["steps"]["data_loading"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
        status["steps"]["transformer_training"] = {"status": "in_progress", "started_at": time.strftime('%H:%M:%S')}
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Train transformer model
        update_progress("Starting transformer model training...")
        transformer_results = run_transformer_training(df, args, enhanced_dir)
        
        # Update status
        status["steps"]["transformer_training"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
    elif args.train_mode == 'traditional':
        df = load_sentiment140_traditional(args.dataset_path, args.max_samples)
        update_progress(f"Dataset loaded: {len(df)} samples")
        
        # Update status
        status["steps"]["data_loading"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
        status["steps"]["traditional_training"] = {"status": "in_progress", "started_at": time.strftime('%H:%M:%S')}
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Train traditional model with enhancements
        update_progress("Starting traditional model training...")
        traditional_results = run_traditional_training(df, args, enhanced_dir)
        
        # Update status
        status["steps"]["traditional_training"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
    elif args.train_mode == 'both':
        # For transformer model
        update_progress("Loading dataset for transformer model...")
        df_transformer = load_sentiment140(args.dataset_path, args.max_samples)
        update_progress(f"Transformer dataset loaded: {len(df_transformer)} samples")
        
        # Update status
        status["steps"]["data_loading"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
        status["steps"]["transformer_training"] = {"status": "in_progress", "started_at": time.strftime('%H:%M:%S')}
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        update_progress("Starting transformer model training...")
        transformer_results = run_transformer_training(df_transformer, args, enhanced_dir)
        
        # Update status
        status["steps"]["transformer_training"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
        status["steps"]["traditional_loading"] = {"status": "in_progress", "started_at": time.strftime('%H:%M:%S')}
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        # For traditional model
        update_progress("Loading dataset for traditional model...")
        df_traditional = load_sentiment140_traditional(args.dataset_path, args.max_samples)
        update_progress(f"Traditional dataset loaded: {len(df_traditional)} samples")
        
        # Update status
        status["steps"]["traditional_loading"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
        status["steps"]["traditional_training"] = {"status": "in_progress", "started_at": time.strftime('%H:%M:%S')}
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        update_progress("Starting traditional model training...")
        traditional_results = run_traditional_training(df_traditional, args, baseline_dir)
        
        # Update status
        status["steps"]["traditional_training"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Compare results
        if args.comparative_analysis:
            update_progress("Starting comparative analysis...")
            status["steps"]["comparative_analysis"] = {"status": "in_progress", "started_at": time.strftime('%H:%M:%S')}
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
            compare_results(transformer_results, traditional_results, comparison_dir)
            
            # Update status
            status["steps"]["comparative_analysis"] = {"status": "completed", "completed_at": time.strftime('%H:%M:%S')}
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
    
    # Log completion and timing
    elapsed_time = time.time() - start_time
    update_progress(f"Training completed in {elapsed_time:.2f} seconds")
    update_progress(f"Results saved to {output_dir}")
    
    # Final status update
    status["completed_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
    status["total_time_seconds"] = elapsed_time
    status["overall_status"] = "completed"
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    return 0

def run_transformer_training(df, args, output_dir):
    """
    Run transformer-based model training.
    
    Args:
        df: DataFrame with preprocessed data
        args: Command-line arguments
        output_dir: Output directory
    
    Returns:
        Dictionary with training results
    """
    from transformer_sentiment_analysis import TransformerSentimentAnalyzer
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Running transformer-based training with {args.model_type} model...")
    
    # Create output directories
    models_dir = output_dir / "models"
    fairness_dir = output_dir / "fairness"
    plots_dir = output_dir / "plots"
    
    for directory in [models_dir, fairness_dir, plots_dir]:
        directory.mkdir(exist_ok=True)
    
    # Apply bias mitigation if requested
    if args.bias_mitigation:
        logger.info("Applying bias mitigation techniques...")
        
        # Add synthetic demographics for fairness evaluation
        logger.info("Adding synthetic demographic data...")
        from enhanced_sentiment_analysis import add_synthetic_demographics
        df = add_synthetic_demographics(df, random_seed=42)
        
        protected_cols = ['gender', 'age_group', 'location']
        df = apply_bias_mitigation_pipeline(
            df, 
            text_col='text', 
            protected_cols=protected_cols,
            label_col='target',
            methods=['reweight', 'gender_neutral'],
            return_weights=False
        )
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['target']
    )
    
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['target']
    )
    
    logger.info(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")
    
    # Initialize model
    analyzer = TransformerSentimentAnalyzer(model_type=args.model_type)
    
    # Train model
    logger.info("Training transformer model...")
    train_metrics = analyzer.train(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['target'].tolist(),
        val_texts=val_df['text'].tolist(),
        val_labels=val_df['target'].tolist(),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=str(models_dir)
    )
    
    # Evaluate model
    logger.info("Evaluating model on test data...")
    eval_metrics, all_preds, all_labels = analyzer.evaluate(
        test_texts=test_df['text'].tolist(),
        test_labels=test_df['target'].tolist(),
        batch_size=args.batch_size
    )
    
    # Save metrics
    with open(models_dir / "evaluation_metrics.json", 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    # Perform fairness evaluation if requested
    if args.fairness_evaluation:
        logger.info("Performing fairness evaluation...")
        
        # Get test indices
        test_indices = list(range(len(df)))[int(len(df) * 0.8):]
        test_df = df.iloc[test_indices].copy()
        test_df['predictions'] = all_preds
        
        # Make sure demographic columns exist in test data
        demographics_cols = ['gender', 'age_group', 'location']
        for col in demographics_cols:
            if col not in test_df.columns:
                logger.warning(f"Demographic column {col} not found in test data. Adding synthetic demographics.")
                from enhanced_sentiment_analysis import add_synthetic_demographics
                test_df = add_synthetic_demographics(test_df, random_seed=42)
                break
        
        # Evaluate fairness
        fairness_results = evaluate_fairness(
            test_df, 
            predictions_col='predictions',
            demographics_cols=demographics_cols,
            label_col='target',
            output_dir=str(fairness_dir)
        )
        
        # Generate fairness plot
        plot_fairness_summary(
            fairness_results,
            save_path=str(plots_dir / "fairness_summary.png")
        )
        
        # Generate fairness report
        fairness_report = generate_fairness_report(
            fairness_results,
            output_path=str(fairness_dir / "fairness_report.md")
        )
    
    # Return results
    results = {
        "model_type": args.model_type,
        "training_metrics": train_metrics,
        "evaluation_metrics": eval_metrics,
        "fairness_results": fairness_results if args.fairness_evaluation else None
    }
    
    return results

def run_traditional_training(df, args, output_dir):
    """
    Run traditional ML model training with enhancements.
    
    Args:
        df: DataFrame with preprocessed data
        args: Command-line arguments
        output_dir: Output directory
    
    Returns:
        Dictionary with training results
    """
    from enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
    
    logger.info(f"Running enhanced traditional training with {args.model_type} model...")
    
    # Create output directories
    models_dir = output_dir / "models"
    fairness_dir = output_dir / "fairness"
    plots_dir = output_dir / "plots"
    logs_dir = output_dir / "logs"
    
    for directory in [models_dir, fairness_dir, plots_dir, logs_dir]:
        directory.mkdir(exist_ok=True)
    
    # Create a training status file to track progress in more detail
    status_file = logs_dir / "training_status.json"
    training_status = {
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "model_type": args.model_type,
        "phases": {
            "initialization": {"status": "in_progress", "started_at": time.strftime('%H:%M:%S')}
        }
    }
    with open(status_file, 'w') as f:
        json.dump(training_status, f, indent=2)
    
    # Helper function to update status
    def update_training_status(phase, status, message=None, metrics=None):
        if phase not in training_status["phases"]:
            training_status["phases"][phase] = {}
        
        if status == "in_progress":
            training_status["phases"][phase]["status"] = status
            training_status["phases"][phase]["started_at"] = time.strftime('%H:%M:%S')
        else:  # completed or failed
            training_status["phases"][phase]["status"] = status
            training_status["phases"][phase]["completed_at"] = time.strftime('%H:%M:%S')
        
        if message:
            training_status["phases"][phase]["message"] = message
        
        if metrics:
            training_status["phases"][phase]["metrics"] = metrics
        
        # Write status to file
        with open(status_file, 'w') as f:
            json.dump(training_status, f, indent=2)
        
        # Log message if provided
        if message:
            logger.info(f"[{phase}] {message}")
    
    # Initialize model
    use_advanced_features = True
    update_training_status("initialization", "in_progress", "Initializing sentiment analyzer...")
    analyzer = EnhancedSentimentAnalyzer(
        model_type=args.model_type if args.model_type != 'logistic' else 'logistic',
        use_advanced_features=use_advanced_features
    )
    update_training_status("initialization", "completed", "Sentiment analyzer initialized")
    
    # Apply bias mitigation if requested
    sample_weights = None
    if args.bias_mitigation:
        update_training_status("bias_mitigation", "in_progress", "Applying bias mitigation techniques...")
        
        # Add synthetic demographics for fairness evaluation
        logger.info("Adding synthetic demographic data...")
        from enhanced_sentiment_analysis import add_synthetic_demographics
        df = add_synthetic_demographics(df, random_seed=42)
        
        protected_cols = ['gender', 'age_group', 'location']
        update_training_status("bias_mitigation", "in_progress", "Applying reweighting and gender-neutral preprocessing...")
        
        # Create a mini-progress file for bias mitigation
        bias_progress_file = logs_dir / "bias_mitigation_progress.txt"
        with open(bias_progress_file, 'w') as f:
            f.write(f"Starting bias mitigation at {time.strftime('%H:%M:%S')}\n")
            f.write(f"Protected columns: {protected_cols}\n")
        
        try:
            weighted_df = apply_bias_mitigation_pipeline(
                df, 
                text_col='text', 
                protected_cols=protected_cols,
                label_col='target',
                methods=['reweight', 'gender_neutral'],
                return_weights=True
            )
            sample_weights = weighted_df['sample_weight'].values
            
            # Update bias mitigation progress
            with open(bias_progress_file, 'a') as f:
                f.write(f"Bias mitigation completed at {time.strftime('%H:%M:%S')}\n")
                f.write(f"Applied methods: reweight, gender_neutral\n")
                f.write(f"Generated weights for {len(sample_weights)} samples\n")
            
            update_training_status("bias_mitigation", "completed", 
                                  f"Successfully applied bias mitigation to {len(df)} samples",
                                  {"methods": ["reweight", "gender_neutral"]})
        except Exception as e:
            error_msg = f"Error in bias mitigation: {str(e)}"
            with open(bias_progress_file, 'a') as f:
                f.write(f"ERROR: {error_msg}\n")
            update_training_status("bias_mitigation", "failed", error_msg)
    
    # Train model
    update_training_status("model_training", "in_progress", "Training enhanced traditional model...")
    train_start = time.time()
    
    # Create a mini-progress file for training
    training_progress_file = logs_dir / "model_training_progress.txt"
    with open(training_progress_file, 'w') as f:
        f.write(f"Starting model training at {time.strftime('%H:%M:%S')}\n")
        f.write(f"Model type: {args.model_type}, Advanced features: {use_advanced_features}\n")
        f.write(f"Training on {len(df)} samples\n")
    
    try:
        metrics, X_test, y_test, y_pred = analyzer.train(
            texts=df['text'].tolist(),
            labels=df['target'].tolist(),
            test_size=0.2,
            perform_cv=True
        )
        
        train_time = time.time() - train_start
        
        # Update training progress
        with open(training_progress_file, 'a') as f:
            f.write(f"Training completed at {time.strftime('%H:%M:%S')} (took {train_time:.2f} seconds)\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}\n")
        
        update_training_status("model_training", "completed", 
                              f"Model training completed in {train_time:.2f} seconds",
                              metrics)
        
        # Save model
        update_training_status("model_saving", "in_progress", "Saving trained model...")
        model_path = models_dir / f"sentiment140_{args.model_type}.joblib"
        analyzer.save_model(model_path)
        update_training_status("model_saving", "completed", f"Model saved to {model_path}")
        
        # Save metrics
        metrics_file = models_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Regularly update the training status file with progress - checkpoint status
        training_status["current_phase"] = "Completed model training"
        training_status["metrics"] = metrics
        training_status["training_time"] = train_time
        with open(status_file, 'w') as f:
            json.dump(training_status, f, indent=2)
        
    except Exception as e:
        error_msg = f"Error in model training: {str(e)}"
        with open(training_progress_file, 'a') as f:
            f.write(f"ERROR: {error_msg}\n")
        update_training_status("model_training", "failed", error_msg)
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}
    
    # Perform fairness evaluation if requested
    fairness_results = None
    if args.fairness_evaluation:
        update_training_status("fairness_evaluation", "in_progress", "Performing fairness evaluation...")
        
        # Create a mini-progress file for fairness evaluation
        fairness_progress_file = logs_dir / "fairness_evaluation_progress.txt"
        with open(fairness_progress_file, 'w') as f:
            f.write(f"Starting fairness evaluation at {time.strftime('%H:%M:%S')}\n")
        
        try:
            # Get test indices
            test_indices = list(range(len(df)))[int(len(df) * 0.8):]
            test_df = df.iloc[test_indices].copy()
            test_df['predictions'] = y_pred
            
            # Make sure demographic columns exist in test data
            demographics_cols = ['gender', 'age_group', 'location']
            for col in demographics_cols:
                if col not in test_df.columns:
                    logger.warning(f"Demographic column {col} not found in test data. Adding synthetic demographics.")
                    from enhanced_sentiment_analysis import add_synthetic_demographics
                    test_df = add_synthetic_demographics(test_df, random_seed=42)
                    break
            
            with open(fairness_progress_file, 'a') as f:
                f.write(f"Evaluating fairness across demographics: {demographics_cols}\n")
                f.write(f"Test set size: {len(test_df)} samples\n")
            
            # Evaluate fairness
            fairness_results = evaluate_fairness(
                test_df, 
                predictions_col='predictions',
                demographics_cols=demographics_cols,
                label_col='target',
                output_dir=str(fairness_dir)
            )
            
            # Generate fairness plot
            plot_fairness_summary(
                fairness_results,
                save_path=str(plots_dir / "fairness_summary.png")
            )
            
            # Generate fairness report
            fairness_report = generate_fairness_report(
                fairness_results,
                output_path=str(fairness_dir / "fairness_report.md")
            )
            
            with open(fairness_progress_file, 'a') as f:
                f.write(f"Fairness evaluation completed at {time.strftime('%H:%M:%S')}\n")
                f.write(f"Problematic groups found: {len(fairness_results.get('problematic_groups', []))}\n")
                f.write(f"Report saved to: {fairness_dir}/fairness_report.md\n")
            
            update_training_status("fairness_evaluation", "completed", 
                                  "Fairness evaluation completed successfully",
                                  {"problematic_groups": len(fairness_results.get('problematic_groups', []))})
            
        except Exception as e:
            error_msg = f"Error in fairness evaluation: {str(e)}"
            with open(fairness_progress_file, 'a') as f:
                f.write(f"ERROR: {error_msg}\n")
            update_training_status("fairness_evaluation", "failed", error_msg)
            logger.error(error_msg, exc_info=True)
    
    # Finalize training status
    training_status["end_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
    training_status["overall_status"] = "completed"
    with open(status_file, 'w') as f:
        json.dump(training_status, f, indent=2)
    
    # Return results
    results = {
        "model_type": args.model_type,
        "advanced_features": use_advanced_features,
        "metrics": metrics,
        "fairness_results": fairness_results
    }
    
    return results

def compare_results(transformer_results, traditional_results, output_dir):
    """
    Compare results between transformer and traditional models.
    
    Args:
        transformer_results: Results from transformer model
        traditional_results: Results from traditional model
        output_dir: Output directory
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    logger.info("Generating comparative analysis...")
    
    # Create comparison DataFrame
    comparison = {
        "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "Training Time (s)"],
        "Transformer": [
            transformer_results["evaluation_metrics"]["accuracy"],
            transformer_results["evaluation_metrics"]["f1_score"],
            transformer_results["evaluation_metrics"]["precision"],
            transformer_results["evaluation_metrics"]["recall"],
            transformer_results["training_metrics"]["training_time_seconds"]
        ],
        "Traditional": [
            traditional_results["metrics"]["accuracy"],
            traditional_results["metrics"]["f1_score"],
            traditional_results["metrics"]["precision"],
            traditional_results["metrics"]["recall"],
            traditional_results["metrics"].get("training_time_seconds", 0)
        ]
    }
    
    df_comparison = pd.DataFrame(comparison)
    
    # Save comparison to CSV
    df_comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    # Plot performance metrics
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    x = range(len(metrics))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], 
           [comparison["Transformer"][i] for i in range(4)], 
           width, label='Transformer')
    
    plt.bar([i + width/2 for i in x], 
           [comparison["Traditional"][i] for i in range(4)], 
           width, label='Traditional')
    
    plt.ylabel('Score')
    plt.title('Performance Comparison: Transformer vs Traditional')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png", dpi=300)
    
    # Compare fairness results if available
    if (transformer_results.get("fairness_results") and 
        traditional_results.get("fairness_results")):
        
        try:
            # Extract problematic groups count
            t_count = len(transformer_results["fairness_results"]["problematic_groups"])
            trad_count = len(traditional_results["fairness_results"]["problematic_groups"])
            
            # Create fairness comparison bar chart
            plt.figure(figsize=(8, 5))
            plt.bar(['Transformer', 'Traditional'], [t_count, trad_count])
            plt.title('Fairness Comparison: Number of Problematic Groups')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(output_dir / "fairness_comparison.png", dpi=300)
            
            # Create a comprehensive comparison report
            with open(output_dir / "comparison_report.md", 'w') as f:
                f.write("# Model Comparison Report\n\n")
                
                f.write("## Performance Metrics\n\n")
                f.write(df_comparison.to_markdown(index=False))
                f.write("\n\n")
                
                f.write("## Fairness Comparison\n\n")
                f.write(f"- Transformer model problematic groups: {t_count}\n")
                f.write(f"- Traditional model problematic groups: {trad_count}\n\n")
                
                f.write("## Conclusion\n\n")
                
                # Generate conclusion based on results
                if (transformer_results["evaluation_metrics"]["accuracy"] > 
                    traditional_results["metrics"]["accuracy"]):
                    f.write("The transformer model demonstrates superior performance ")
                else:
                    f.write("The traditional model demonstrates comparable performance ")
                
                f.write("in terms of accuracy and F1 score. ")
                
                if t_count < trad_count:
                    f.write("Additionally, the transformer model exhibits fewer fairness concerns ")
                    f.write("across demographic groups, suggesting it may be less biased overall.")
                else:
                    f.write("However, fairness analysis indicates that both models show similar ")
                    f.write("levels of demographic bias that should be addressed.")
                
        except Exception as e:
            logger.error(f"Error creating fairness comparison: {str(e)}")
    
    logger.info(f"Comparison results saved to {output_dir}")

def main():
    """Main function."""
    args = parse_args()
    return run_baseline_and_enhanced_training(args)

if __name__ == "__main__":
    exit(main()) 