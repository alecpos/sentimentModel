#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Ad Score Predictor

This is a simplified version of the AdScorePredictor for demonstration purposes.
It uses a random forest regressor to predict ad scores based on numerical features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Union, Any, Optional, Tuple


class SimpleAdScorePredictor:
    """A simplified predictor for ad performance scores using a random forest model.
    
    This class implements a basic version of the ad score prediction functionality
    for demonstration and testing purposes.
    
    Attributes:
        model: The trained random forest model
        scaler: Standardizer for input features
        feature_names: Names of the features used for prediction
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """Initialize the SimpleAdScorePredictor.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random seed for reproducibility
        
        Returns:
            None
            
        Raises:
            None
            
        Examples:
            >>> predictor = SimpleAdScorePredictor(n_estimators=100)
            >>> predictor.fit(X_train, y_train)
            >>> scores = predictor.predict(X_test)
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.numeric_cols = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on the provided data.
        
        Args:
            X: DataFrame containing features
            y: Series containing target ad scores
            
        Returns:
            None
            
        Raises:
            ValueError: If input data is not in the expected format
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
            
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Select only numeric columns for this simplified model
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if 'ad_text' in self.feature_names:
            print("Note: Text features are ignored in this simplified model")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X[self.numeric_cols])
        
        # Train the model
        self.model.fit(X_scaled, y)
        print(f"Model trained on {X.shape[0]} samples with {len(self.numeric_cols)} numeric features")
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions for the input data.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            Dictionary with prediction results including 'score' and 'confidence'
            
        Raises:
            ValueError: If model has not been trained or input format is invalid
        """
        if self.model is None or not hasattr(self.model, 'predict'):
            raise ValueError("Model has not been trained. Call fit() first.")
            
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        # Select only numeric columns that were used during training
        numeric_cols = [col for col in self.numeric_cols if col in X.columns]
        
        # Standardize features
        X_scaled = self.scaler.transform(X[numeric_cols])
        
        # Generate predictions
        predictions = self.model.predict(X_scaled)
        
        # Estimate confidence using prediction variance from trees
        tree_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
        confidence = 1.0 - np.std(tree_preds, axis=0) / np.mean(tree_preds, axis=0)
        
        return {
            'score': predictions,
            'confidence': confidence
        }
    
    def feature_importance(self) -> pd.DataFrame:
        """Return the feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and their importance scores
            
        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Make sure we have the same number of feature names and importance values
        importances = self.model.feature_importances_
        
        if len(self.numeric_cols) != len(importances):
            raise ValueError(f"Feature name count ({len(self.numeric_cols)}) doesn't match importance count ({len(importances)})")
        
        # Create a DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': self.numeric_cols,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df 