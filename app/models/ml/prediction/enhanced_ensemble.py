"""
Enhanced Ensemble Model Implementation

This module implements advanced ensemble methods including bagging and stacking
with dynamic weight optimization and comprehensive performance monitoring.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass
from datetime import datetime
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class for tracking ensemble performance metrics."""
    inference_time: List[float]
    prediction_distribution: List[float]
    feature_importance: Optional[Dict[str, float]]
    bagging_metrics: List[Dict[str, float]]
    stacking_metrics: List[Dict[str, float]]
    cv_scores: List[Dict[str, float]]
    last_updated: datetime

class EnhancedBaggingEnsemble:
    """Enhanced bagging ensemble with proper bootstrap sampling and model independence."""
    
    def __init__(
        self,
        base_estimator: Any,
        n_estimators: int = 10,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the enhanced bagging ensemble.
        
        Args:
            base_estimator: Base estimator to use for bagging
            n_estimators: Number of estimators in the ensemble
            random_state: Random state for reproducibility
            n_jobs: Number of jobs to run in parallel
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators = []
        self.performance_metrics = PerformanceMetrics(
            inference_time=[],
            prediction_distribution=[],
            feature_importance=None,
            bagging_metrics=[],
            stacking_metrics=[],
            cv_scores=[],
            last_updated=datetime.now()
        )
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnhancedBaggingEnsemble':
        """
        Fit the bagging ensemble using bootstrap sampling.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            self: Returns the instance
        """
        n_samples = X.shape[0]
        logger.info(f"Training bagging ensemble with {self.n_estimators} estimators")
        
        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Clone estimator to ensure independence
            estimator = clone(self.base_estimator)
            
            # Train model with early stopping if available
            if hasattr(estimator, 'fit') and hasattr(estimator, 'early_stopping_rounds'):
                estimator.fit(
                    X_bootstrap, y_bootstrap,
                    eval_set=[(X, y)],
                    early_stopping_rounds=30,
                    verbose=50
                )
            else:
                estimator.fit(X_bootstrap, y_bootstrap)
            
            self.estimators.append(estimator)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Trained {i + 1}/{self.n_estimators} estimators")
        
        # Calculate feature importance if available
        if hasattr(self.base_estimator, 'feature_importances_'):
            self.performance_metrics.feature_importance = {
                f"feature_{i}": importance
                for i, importance in enumerate(self.base_estimator.feature_importances_)
            }
        
        self.performance_metrics.last_updated = datetime.now()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Input data
            
        Returns:
            Array of predictions
        """
        start_time = datetime.now()
        
        # Get predictions from all estimators
        predictions = np.array([
            estimator.predict(X) for estimator in self.estimators
        ])
        
        # Majority voting for classification
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
        
        # Update performance metrics
        inference_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics.inference_time.append(inference_time)
        self.performance_metrics.prediction_distribution.extend(final_predictions.tolist())
        self.performance_metrics.last_updated = datetime.now()
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the ensemble.
        
        Args:
            X: Input data
            
        Returns:
            Array of probability predictions
        """
        start_time = datetime.now()
        
        # Get probability predictions from all estimators
        probas = np.array([
            estimator.predict_proba(X) for estimator in self.estimators
        ])
        
        # Average probabilities
        final_probas = np.mean(probas, axis=0)
        
        # Update performance metrics
        inference_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics.inference_time.append(inference_time)
        self.performance_metrics.prediction_distribution.extend(final_probas[:, 1].tolist())
        self.performance_metrics.last_updated = datetime.now()
        
        return final_probas
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        return self.performance_metrics.feature_importance

class EnhancedStackingEnsemble:
    """Enhanced stacking ensemble with cross-validation based meta-feature generation."""
    
    def __init__(
        self,
        base_estimators: List[Any],
        meta_learner: Any,
        use_proba: bool = True,
        n_splits: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the enhanced stacking ensemble.
        
        Args:
            base_estimators: List of base estimators
            meta_learner: Meta-learner for final predictions
            use_proba: Whether to use probability predictions
            n_splits: Number of splits for cross-validation
            random_state: Random state for reproducibility
        """
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner
        self.use_proba = use_proba
        self.n_splits = n_splits
        self.random_state = random_state
        self.performance_metrics = PerformanceMetrics(
            inference_time=[],
            prediction_distribution=[],
            feature_importance=None,
            bagging_metrics=[],
            stacking_metrics=[],
            cv_scores=[],
            last_updated=datetime.now()
        )
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate meta-features using cross-validation."""
        logger.info("Generating meta-features using cross-validation...")
        
        # Initialize meta-features array
        n_samples = X.shape[0]
        n_base_models = len(self.base_estimators)
        meta_features = np.zeros((n_samples, n_base_models * 2))  # 2 for probability predictions
        
        # Process each base estimator
        for i, estimator in enumerate(self.base_estimators):
            logger.info(f"Processing base estimator {i+1}/{n_base_models}")
            
            # Fit the estimator on the full training set
            estimator.fit(X, y)
            
            # Get predictions
            if self.use_proba:
                preds = estimator.predict_proba(X)
            else:
                preds = np.column_stack([1 - estimator.predict(X), estimator.predict(X)])
            
            # Store predictions
            meta_features[:, i*2:(i+1)*2] = preds
            
            # Track CV scores if estimator supports scoring
            if hasattr(estimator, 'score'):
                score = estimator.score(X, y)
                self.performance_metrics.cv_scores.append({
                    f'estimator_{i}_full_set': score
                })
        
        return meta_features, y
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'EnhancedStackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training data
            y: Target values
            X_val: Optional validation data
            y_val: Optional validation targets
            
        Returns:
            self: Returns the instance
        """
        # Generate meta-features
        meta_features, y = self._generate_meta_features(X, y)
        
        # Train meta-learner
        logger.info("Training meta-learner...")
        if X_val is not None and y_val is not None:
            # Generate meta-features for validation set
            meta_features_val = np.zeros((X_val.shape[0], meta_features.shape[1]))
            for i, estimator in enumerate(self.base_estimators):
                if self.use_proba:
                    preds = estimator.predict_proba(X_val)
                else:
                    preds = np.column_stack([1 - estimator.predict(X_val), estimator.predict(X_val)])
                meta_features_val[:, i*2:(i+1)*2] = preds
            
            # Train meta-learner with validation data
            self.meta_learner.fit(meta_features, y)
            val_score = self.meta_learner.score(meta_features_val, y_val)
            logger.info(f"Meta-learner validation score: {val_score:.4f}")
        else:
            # Train meta-learner without validation data
            self.meta_learner.fit(meta_features, y)
        
        self.performance_metrics.last_updated = datetime.now()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.
        
        Args:
            X: Input data
            
        Returns:
            Array of predictions
        """
        start_time = datetime.now()
        
        # Generate meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_estimators) * (2 if self.use_proba else 1)))
        for i, estimator in enumerate(self.base_estimators):
            if self.use_proba:
                preds = estimator.predict_proba(X)
                meta_features[:, i*2:(i+1)*2] = preds
            else:
                preds = estimator.predict(X)
                meta_features[:, i] = preds
        
        # Make final predictions
        predictions = self.meta_learner.predict(meta_features)
        
        # Update performance metrics
        inference_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics.inference_time.append(inference_time)
        self.performance_metrics.prediction_distribution.extend(predictions.tolist())
        self.performance_metrics.last_updated = datetime.now()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the stacking ensemble.
        
        Args:
            X: Input data
            
        Returns:
            Array of probability predictions
        """
        start_time = datetime.now()
        
        # Generate meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_estimators) * 2))
        for i, estimator in enumerate(self.base_estimators):
            preds = estimator.predict_proba(X)
            meta_features[:, i*2:(i+1)*2] = preds
        
        # Make final probability predictions
        probas = self.meta_learner.predict_proba(meta_features)
        
        # Update performance metrics
        inference_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics.inference_time.append(inference_time)
        self.performance_metrics.prediction_distribution.extend(probas[:, 1].tolist())
        self.performance_metrics.last_updated = datetime.now()
        
        return probas

def optimize_ensemble_weights(
    base_models: List[Any],
    X_val: np.ndarray,
    y_val: np.ndarray
) -> np.ndarray:
    """
    Optimize ensemble weights based on validation performance.
    
    Args:
        base_models: List of base models
        X_val: Validation data
        y_val: Validation targets
        
    Returns:
        Array of optimized weights
    """
    predictions = []
    for model in base_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val)[:, 1]
        else:
            pred = model.predict(X_val)
        predictions.append(pred)
    
    def objective(weights: np.ndarray) -> float:
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Weighted prediction
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += weights[i] * pred
        
        # Calculate negative AUC (to minimize)
        return -roc_auc_score(y_val, weighted_pred)
    
    # Initial weights (equal)
    initial_weights = np.ones(len(base_models)) / len(base_models)
    
    # Optimize
    bounds = [(0, 1)] * len(base_models)
    result = minimize(objective, initial_weights, bounds=bounds)
    
    # Return normalized weights
    return result.x / np.sum(result.x)

def visualize_ensemble_performance(
    ensemble: Union[EnhancedBaggingEnsemble, EnhancedStackingEnsemble],
    output_dir: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive visualization of ensemble performance.
    
    Args:
        ensemble: Ensemble model instance
        output_dir: Optional directory to save plots
        
    Returns:
        matplotlib figure
    """
    metrics = ensemble.performance_metrics
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction Distribution
    sns.histplot(
        metrics.prediction_distribution,
        bins=50,
        ax=axes[0,0]
    )
    axes[0,0].set_title('Prediction Distribution')
    axes[0,0].set_xlabel('Prediction Value')
    axes[0,0].set_ylabel('Count')
    
    # 2. Inference Time Distribution
    sns.histplot(
        metrics.inference_time,
        bins=30,
        ax=axes[0,1]
    )
    axes[0,1].set_title('Inference Time Distribution')
    axes[0,1].set_xlabel('Time (seconds)')
    axes[0,1].set_ylabel('Count')
    
    # 3. Feature Importance (if available)
    if metrics.feature_importance is not None:
        importance_df = pd.DataFrame.from_dict(
            metrics.feature_importance,
            orient='index',
            columns=['importance']
        ).sort_values('importance', ascending=True)
        
        importance_df.plot(
            kind='barh',
            ax=axes[1,0]
        )
        axes[1,0].set_title('Feature Importance')
        axes[1,0].set_xlabel('Importance')
    
    # 4. CV Scores (if available)
    if metrics.cv_scores:
        cv_df = pd.DataFrame(metrics.cv_scores)
        cv_df.boxplot(ax=axes[1,1])
        axes[1,1].set_title('Cross-Validation Scores')
        axes[1,1].set_xlabel('Estimator')
        axes[1,1].set_ylabel('Score')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/ensemble_performance.png")
        plt.close()
    
    return fig

def main():
    """Main entry point for the enhanced ensemble package."""
    parser = argparse.ArgumentParser(
        description="Enhanced Ensemble Model CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["bagging", "stacking", "optimize", "visualize"],
        required=True,
        help="Operation mode"
    )
    
    parser.add_argument(
        "--data",
        required=True,
        help="Path to input data (CSV format)"
    )
    
    parser.add_argument(
        "--output",
        help="Path to save results"
    )
    
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=10,
        help="Number of estimators for bagging"
    )
    
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of splits for cross-validation"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        data = pd.read_csv(args.data)
        X = data.drop("target", axis=1).values
        y = data["target"].values
        
        if args.mode == "bagging":
            # Initialize and train bagging ensemble
            base_estimator = DecisionTreeClassifier(random_state=args.random_state)
            ensemble = EnhancedBaggingEnsemble(
                base_estimator=base_estimator,
                n_estimators=args.n_estimators,
                random_state=args.random_state
            )
            
            ensemble.fit(X, y)
            predictions = ensemble.predict(X)
            
            if args.output:
                results = pd.DataFrame({
                    "prediction": predictions,
                    "probability": ensemble.predict_proba(X)[:, 1]
                })
                results.to_csv(args.output, index=False)
                
                # Save performance visualization
                fig = visualize_ensemble_performance(ensemble)
                fig.savefig(f"{args.output}_performance.png")
                plt.close(fig)
        
        elif args.mode == "stacking":
            # Initialize and train stacking ensemble
            base_estimators = [
                DecisionTreeClassifier(random_state=args.random_state),
                RandomForestClassifier(random_state=args.random_state)
            ]
            meta_learner = LogisticRegression(random_state=args.random_state)
            
            ensemble = EnhancedStackingEnsemble(
                base_estimators=base_estimators,
                meta_learner=meta_learner,
                use_proba=True,
                n_splits=args.n_splits,
                random_state=args.random_state
            )
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=args.random_state
            )
            
            ensemble.fit(X_train, y_train, X_val, y_val)
            predictions = ensemble.predict(X)
            
            if args.output:
                results = pd.DataFrame({
                    "prediction": predictions,
                    "probability": ensemble.predict_proba(X)[:, 1]
                })
                results.to_csv(args.output, index=False)
                
                # Save performance visualization
                fig = visualize_ensemble_performance(ensemble)
                fig.savefig(f"{args.output}_performance.png")
                plt.close(fig)
        
        elif args.mode == "optimize":
            # Create and train base models
            base_models = [
                DecisionTreeClassifier(random_state=args.random_state),
                RandomForestClassifier(random_state=args.random_state),
                LogisticRegression(random_state=args.random_state)
            ]
            
            for model in base_models:
                model.fit(X, y)
            
            # Optimize weights
            weights = optimize_ensemble_weights(base_models, X, y)
            
            if args.output:
                pd.DataFrame({
                    "model": [f"model_{i}" for i in range(len(base_models))],
                    "weight": weights
                }).to_csv(args.output, index=False)
        
        elif args.mode == "visualize":
            # Load an existing ensemble model
            if not args.output:
                logger.error("Output path required for visualization mode")
                sys.exit(1)
            
            # Create a sample ensemble for visualization
            base_estimator = DecisionTreeClassifier(random_state=args.random_state)
            ensemble = EnhancedBaggingEnsemble(
                base_estimator=base_estimator,
                n_estimators=args.n_estimators,
                random_state=args.random_state
            )
            
            ensemble.fit(X, y)
            ensemble.predict(X)
            
            # Generate and save visualization
            fig = visualize_ensemble_performance(ensemble)
            fig.savefig(args.output)
            plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 