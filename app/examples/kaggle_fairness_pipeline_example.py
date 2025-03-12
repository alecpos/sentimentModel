#!/usr/bin/env python
"""
Kaggle Dataset Integration with Fairness Evaluation Example

This script demonstrates how to use the Kaggle dataset integration pipeline
together with fairness evaluation and mitigation techniques. It shows the
complete workflow for:

1. Configuring and downloading datasets from Kaggle
2. Processing and validating datasets
3. Evaluating fairness metrics across protected attributes
4. Applying fairness mitigation techniques
5. Training models on the mitigated data
6. Generating fairness visualizations and reports

This is an example implementation for the WITHIN Ad Score & Account Health Predictor system.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.data_integration.kaggle_pipeline import (
    KaggleDatasetPipeline, DatasetConfig, ProcessedDataset, DatasetCategory
)
from app.core.fairness import (
    FairnessEvaluator, FairnessResults, FairnessMetric, FairnessThreshold,
    Reweighing, FairDataTransformer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Kaggle Fairness Pipeline Example")
    
    parser.add_argument(
        "--dataset", type=str, default="customer_conversion",
        choices=["customer_conversion", "ctr_optimization"],
        help="Dataset to use for the example"
    )
    
    parser.add_argument(
        "--mitigation", type=str, default=None,
        choices=["reweighing", "fair_transform", None],
        help="Fairness mitigation technique to apply"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="kaggle_fairness_example",
        help="Directory to save results and visualizations"
    )
    
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Fairness threshold for evaluation"
    )
    
    return parser.parse_args()

def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, 
               sample_weights: Optional[np.ndarray] = None) -> Tuple[Any, Dict[str, float]]:
    """Train a model and evaluate its performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        sample_weights: Optional sample weights for training
        
    Returns:
        Tuple of (trained model, performance metrics)
    """
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    
    # Log results
    logger.info(f"Model performance metrics: {metrics}")
    logger.info(f"\nClassification report:\n{classification_report(y_test, y_pred)}")
    
    return model, metrics

def evaluate_fairness(X_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, 
                     protected_attributes: List[str], output_dir: Path, threshold: float) -> Dict[str, FairnessResults]:
    """Evaluate fairness metrics for the model predictions.
    
    Args:
        X_test: Test features
        y_test: Test labels
        y_pred: Model predictions
        protected_attributes: List of protected attribute columns
        output_dir: Directory to save fairness results
        threshold: Fairness threshold
        
    Returns:
        Dictionary of fairness results by protected attribute
    """
    # Create combined DataFrame for fairness evaluation
    eval_df = X_test.copy()
    eval_df["target"] = y_test
    eval_df["prediction"] = y_pred
    
    # Initialize fairness evaluator
    evaluator = FairnessEvaluator(
        output_dir=str(output_dir / "fairness"),
        threshold=threshold,
        save_visualizations=True
    )
    
    # Evaluate fairness for each protected attribute
    fairness_results = {}
    for attr in protected_attributes:
        if attr in eval_df.columns:
            logger.info(f"Evaluating fairness for protected attribute: {attr}")
            
            # Run evaluation
            results = evaluator.evaluate(
                df=eval_df,
                protected_attribute=attr,
                target_column="target",
                prediction_column="prediction"
            )
            
            # Store results
            fairness_results[attr] = results
            
            # Log summary
            logger.info(f"\nFairness Evaluation Summary for {attr}:")
            logger.info(results.summary())
    
    return fairness_results

def apply_mitigation(
    processed_dataset: ProcessedDataset,
    mitigation_technique: str,
    protected_attributes: List[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """Apply fairness mitigation to the dataset.
    
    Args:
        processed_dataset: Processed dataset
        mitigation_technique: Name of the mitigation technique to apply
        protected_attributes: List of protected attributes
        
    Returns:
        Tuple of (mitigated X_train, y_train, X_test, y_test, sample_weights)
    """
    # Extract data
    X_train = processed_dataset.X_train
    y_train = processed_dataset.y_train
    X_test = processed_dataset.X_test
    y_test = processed_dataset.y_test
    
    sample_weights = None
    
    # No mitigation
    if mitigation_technique is None:
        logger.info("No fairness mitigation applied")
        return X_train, y_train, X_test, y_test, sample_weights
    
    # Choose primary protected attribute for mitigation
    # In a real application, you might want to apply mitigation for each attribute
    # or use an intersectional approach
    protected_attr = protected_attributes[0] if protected_attributes else None
    
    if protected_attr is None or protected_attr not in X_train.columns:
        logger.warning("No valid protected attribute for mitigation")
        return X_train, y_train, X_test, y_test, sample_weights
    
    logger.info(f"Applying {mitigation_technique} for protected attribute: {protected_attr}")
    
    # Apply reweighing
    if mitigation_technique == "reweighing":
        reweigher = Reweighing(protected_attribute=protected_attr)
        reweigher.fit(X_train, y_train)
        
        # Get sample weights for training
        sample_weights = reweigher.get_sample_weights(X_train, y_train)
        
        # Log weight statistics
        logger.info(f"Reweighing sample weight stats: min={sample_weights.min():.4f}, "
                   f"max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
        
    # Apply fair data transformation
    elif mitigation_technique == "fair_transform":
        transformer = FairDataTransformer(
            protected_attribute=protected_attr,
            method="decorrelation"
        )
        
        # Fit and transform the data
        X_train, y_train = transformer.fit_transform(X_train, y_train)
        X_test, y_test = transformer.transform(X_test, y_test)
        
        logger.info(f"Applied fair data transformation to remove bias for {protected_attr}")
    
    return X_train, y_train, X_test, y_test, sample_weights

def main() -> None:
    """Run the Kaggle fairness pipeline example."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the Kaggle dataset pipeline
    pipeline = KaggleDatasetPipeline(
        data_dir=str(output_dir / "data"),
        cache_dir=str(output_dir / "cache"),
        validate_fairness=True
    )
    
    # Get dataset configs
    configs = pipeline.get_dataset_configs()
    
    # Select the dataset
    if args.dataset not in configs:
        logger.error(f"Dataset {args.dataset} not found in available configs")
        return
    
    dataset_config = configs[args.dataset]
    
    # Process the dataset
    logger.info(f"Processing dataset: {dataset_config.dataset_slug}")
    processed_dataset = pipeline.process_dataset(dataset_config)
    
    # Log dataset information
    logger.info(f"Dataset loaded: {processed_dataset.metadata.dataset_name}")
    logger.info(f"Training set: {len(processed_dataset.X_train)} samples")
    logger.info(f"Validation set: {len(processed_dataset.X_val)} samples")
    logger.info(f"Test set: {len(processed_dataset.X_test)} samples")
    
    # Get protected attributes
    protected_attributes = dataset_config.protected_attributes
    logger.info(f"Protected attributes: {protected_attributes}")
    
    # Apply fairness mitigation
    X_train, y_train, X_test, y_test, sample_weights = apply_mitigation(
        processed_dataset=processed_dataset,
        mitigation_technique=args.mitigation,
        protected_attributes=protected_attributes
    )
    
    # Train model
    logger.info("Training model...")
    model, metrics = train_model(X_train, y_train, X_test, y_test, sample_weights)
    
    # Get model predictions
    y_pred = model.predict(X_test)
    
    # Evaluate fairness
    logger.info("Evaluating fairness...")
    fairness_results = evaluate_fairness(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        protected_attributes=protected_attributes,
        output_dir=output_dir,
        threshold=args.threshold
    )
    
    # Save fairness results as JSON
    for attr, results in fairness_results.items():
        results_path = output_dir / "fairness" / f"{attr}_results.json"
        
        with open(results_path, "w") as f:
            import json
            json.dump(results.to_dict(), f, indent=2)
            
        logger.info(f"Saved fairness results to {results_path}")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main() 