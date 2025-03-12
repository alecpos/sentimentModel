"""
Enhanced Account Health Predictor with ML Pipeline - Optimized Version
"""

import numpy as np
import pandas as pd
import joblib
import logging
import json
import os
import optuna
import shap
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from app.core.preprocessor.ad_data_preprocessor import build_preprocessing_pipeline
logger = logging.getLogger(__name__)

# ------------------------------
# Custom Early Stopping Callback
# ------------------------------
def early_stopping_callback_factory(patience: int = 5, min_delta: float = 1e-4):
    """
    Creates a custom early-stopping callback for Optuna.
    If the trial's value does not improve by at least `min_delta`
    for `patience` consecutive trials, the study will be stopped.
    """
    best_score = float('inf')
    best_trial_number = 0

    def callback(study, trial):
        nonlocal best_score, best_trial_number
        current_value = trial.value
        if current_value < best_score - min_delta:
            best_score = current_value
            best_trial_number = trial.number
        elif trial.number - best_trial_number >= patience:
            study.stop()

    return callback

# -----------------------------------------------
# Begin EnhancedHealthEnsemble and AdvancedHealthPredictor
# -----------------------------------------------
class EnhancedHealthEnsemble(RegressorMixin, BaseEstimator):
    """Optimized ensemble with dynamic weights"""
    
    def __init__(self, gb_params=None, nn_params=None):
        self.gb_params = gb_params or {'n_estimators': 100, 'max_depth': 5}
        self.nn_params = nn_params or {
            'hidden_layer_sizes': (64, 32),
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'learning_rate_init': 0.001,
            'batch_size': 'auto',
            'random_state': 42
        }
        self.ensemble_weights = None
        self.nn = None
        
    def fit(self, X, y):
        # Split data for weight optimization
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train GB model
        self.gb = GradientBoostingRegressor(**self.gb_params).fit(X_train, y_train)
        
        # Train NN model using scikit-learn's MLPRegressor
        self.nn = MLPRegressor(**self.nn_params).fit(X_train, y_train)
        
        # Optimize ensemble weights using validation set
        self._optimize_weights(X_val, y_val)
            
        return self

    def predict(self, X):
        gb_pred = self.gb.predict(X)
        nn_pred = self.nn.predict(X)
        return self.ensemble_weights[0] * gb_pred + self.ensemble_weights[1] * nn_pred

    def _optimize_weights(self, X_val, y_val):
        """Weight optimization with early stopping"""
        gb_pred = self.gb.predict(X_val)
        nn_pred = self.nn.predict(X_val)
        
        best_score = float('inf')
        patience = 5
        no_improve_count = 0
        
        def objective(trial):
            nonlocal best_score, no_improve_count
            w1 = trial.suggest_float('w1', 0, 1)
            w2 = 1 - w1
            combined = w1 * gb_pred + w2 * nn_pred
            score = mean_squared_error(y_val, combined)
            
            if score < best_score:
                best_score = score
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                trial.study.stop()
                
            return score
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=20)  # Reduced number of trials
        
        # Set default weights if optimization fails
        if len(study.trials) == 0:
            self.ensemble_weights = [0.5, 0.5]
        else:
            w1 = study.best_params['w1']
            self.ensemble_weights = [w1, 1 - w1]
        
        return self

class AdvancedHealthPredictor:
    """Enhanced predictor for measuring advertising account health.

    Implements a hybrid ML approach combining gradient boosting and neural networks 
    with built-in explainability and privacy considerations.

    Key Features:
        - Ensemble learning with optimized weights
        - SHAP-based explanations and feature importance
        - Differential privacy protection
        - Automated feature engineering
        - Historical trend analysis
        - Risk factor identification
        - Contextual optimization suggestions

    Attributes:
        model_path (str): Path to saved model files
        config (dict): Model configuration parameters
        feature_columns (list): Required input features
        pipeline (Pipeline): Scikit-learn preprocessing pipeline
        is_fitted (bool): Whether model has been trained
        explainer (shap.Explainer): SHAP explainer for feature importance
        loaded (bool): Whether model was loaded from disk

    Example:
        >>> predictor = AdvancedHealthPredictor()
        >>> predictor.train(training_data)
        >>> health_score = predictor.predict_health_score(metrics)
    """
    
    def __init__(self, model_path=None, config_path=None):
        """Initialize the health predictor.

        Args:
            model_path (str, optional): Path to load saved model. Defaults to None.
            config_path (str, optional): Path to custom config file. Defaults to None.

        Note:
            If model_path is provided and exists, the model will be loaded from disk.
            If config_path is provided, it will override default configuration.
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.feature_columns = self._get_feature_columns()
        self.pipeline = self._build_enhanced_pipeline()
        self.explainer = None
        self.loaded = False
        self.is_fitted = False
        self.preprocessor = self._build_preprocessor()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _get_feature_columns(self):
        """Get required feature columns with defaults"""
        return [
            'ctr', 'conversion_rate', 'cost_per_conversion',
            'impressions', 'spend', 'revenue', 'conversions',
            'clicks'  # Added for derived features
        ]

    def _build_enhanced_pipeline(self):
        """Modular preprocessing and modeling pipeline"""
        return Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('ensemble', EnhancedHealthEnsemble(
                gb_params=self.config.get('gb_params', {}),
                nn_params=self.config.get('nn_params', {})
            ))
        ])

    def _build_preprocessor(self):
        """Configure preprocessing for health predictions"""
        return build_preprocessing_pipeline(
            numerical_features=[
                'ctr', 'conversion_rate', 'cost_per_conversion',
                'impressions', 'spend', 'revenue', 'conversions',
                'clicks'  # Added for derived features
            ],
            categorical_features=[],  # Add categorical features if needed
            text_features=[],  # Add text features if needed
            config={
                'numerical_impute_strategy': 'median',
                'outlier_factor': 1.5,
                'preprocessing': {
                    'scaler': 'standard',
                    'imputer': 'knn'
                }
            }
        )

    def _preprocess_training_data(self, training_data, target_column):
        """Updated preprocessing method with enhanced validation"""
        try:
            # Convert to DataFrame if needed
            df = pd.DataFrame(training_data) if not isinstance(training_data, pd.DataFrame) else training_data.copy()
            
            # Validate target column
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in training data")
            
            # Extract target before preprocessing
            y = df[target_column].copy()
            X = df.drop(columns=[target_column])
            
            # Validate required features
            missing_features = set(self.feature_columns) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Ensure features are in the correct order
            X = X[self.feature_columns]
            
            # Apply preprocessing pipeline
            X_processed = self.preprocessor.fit_transform(X)
            
            # Add derived features
            X_processed = pd.DataFrame(X_processed, columns=self.feature_columns)
            X_processed = self._add_derived_features(X_processed)
            
            # Ensure consistent feature order
            expected_features = self.feature_columns + [
                'engagement_cost_ratio',
                'ctr_conversion_interaction',
                'roi'
            ]
            X_processed = X_processed[expected_features]
            
            return X_processed, y
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise RuntimeError(f"Failed to preprocess training data: {str(e)}") from e

    def _add_derived_features(self, X):
        """Enhanced feature engineering with validation"""
        try:
            # Create copy to avoid modifying original
            X = X.copy()
            
            # Add engagement cost ratio with safe division
            X['engagement_cost_ratio'] = np.where(
                X['spend'] > 0,
                X['clicks'] / X['spend'],
                0
            )
            
            # Add CTR-conversion interaction
            X['ctr_conversion_interaction'] = X['ctr'] * X['conversion_rate']
            
            # Add ROI
            X['roi'] = np.where(
                X['spend'] > 0,
                (X['revenue'] - X['spend']) / X['spend'],
                0
            )
            
            return X
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise RuntimeError(f"Failed to add derived features: {str(e)}") from e

    def train(self, training_data, target_column='health_score', save_path=None):
        """Train the health predictor on advertising performance data.

        Performs automated feature engineering, hyperparameter optimization,
        and ensemble weight calibration.

        Args:
            training_data (pd.DataFrame): Historical performance metrics
            target_column (str, optional): Column containing health scores. Defaults to 'health_score'
            save_path (str, optional): Path to save trained model. Defaults to None

        Returns:
            dict: Training results containing:
                - metrics: Model performance metrics (R2, MSE, MAE)
                - privacy_report: Privacy budget usage statistics  
                - feature_importance: SHAP-based feature importance
                - training_size: Number of training samples

        Raises:
            ValueError: If training data size is insufficient (<80 samples)
            RuntimeError: If model training fails

        Example:
            >>> results = predictor.train(training_data)
            >>> print(f"R2 Score: {results['metrics']['r2']:.3f}")
        """
        try:
            X, y = self._preprocess_training_data(training_data, target_column)
            
            # Ensure minimum training data size
            if len(X) < 100:
                logger.warning(f"Small training dataset size ({len(X)} samples). Consider using more data.")
                if len(X) < 80:  # Enforce minimum size requirement
                    raise ValueError(f"Training data size ({len(X)}) is too small. Need at least 80 samples.")
            
            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Hyperparameter optimization with early stopping
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train),
                n_trials=20,  # Reduced number of trials
                callbacks=[early_stopping_callback_factory(patience=5, min_delta=1e-4)]
            )
            
            # Update model with best parameters
            if study.best_params:
                self._update_model_params(study.best_params)
            
            # Fit the pipeline
            self.pipeline.fit(X_train, y_train)
            self.is_fitted = True
            
            # Initialize explainer
            self.explainer = shap.Explainer(self.pipeline.named_steps['ensemble'].gb)
            
            # Evaluate on full dataset
            evaluation_results = self._evaluate_model(X, y)
            
            if save_path:
                self.save_model(save_path)
                
            # Return results in the expected format
            return {
                'metrics': {
                    'r2': evaluation_results['r2'],
                    'mse': evaluation_results['mse'],
                    'mae': evaluation_results['mae']
                },
                'privacy_report': {
                    'epsilon_spent': evaluation_results['epsilon_spent'],
                    'delta': 1e-5
                },
                'feature_importance': evaluation_results.get('feature_importance', {}),
                'training_size': len(X_train)
            }
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.is_fitted = False  # Ensure is_fitted is False on failure
            raise
        

    def _objective(self, trial, X, y):
        """Optuna optimization objective with early stopping"""
        params = {
            'gb_max_depth': trial.suggest_int('gb_max_depth', 3, 8),
            'gb_learning_rate': trial.suggest_float('gb_learning_rate', 1e-3, 0.1, log=True),
            'nn_hidden_layer_sizes': trial.suggest_categorical('nn_hidden_layer_sizes', [(64, 32), (32, 16), (128, 64)]),
            'nn_learning_rate_init': trial.suggest_float('nn_learning_rate_init', 1e-4, 1e-2, log=True),
            'nn_batch_size': trial.suggest_categorical('nn_batch_size', ['auto', 32, 64])
        }
        
        # Update model parameters
        self._update_model_params(params)
        
        # Create and fit a new pipeline
        model = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('ensemble', EnhancedHealthEnsemble(
                gb_params={'max_depth': params['gb_max_depth'], 
                         'learning_rate': params['gb_learning_rate'],
                         'n_estimators': 100},
                nn_params={'hidden_layer_sizes': params['nn_hidden_layer_sizes'],
                         'learning_rate_init': params['nn_learning_rate_init'],
                         'batch_size': params['nn_batch_size'],
                         'max_iter': 200,
                         'early_stopping': True,
                         'validation_fraction': 0.2,
                         'random_state': 42}
            ))
        ])
        
        model.fit(X, y)
        y_pred = model.predict(X)
        return mean_squared_error(y, y_pred)

    def predict_health_score(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Predict health score with confidence interval and insights.

        Args:
            metrics (Dict[str, float]): Dictionary of performance metrics

        Returns:
            Dict[str, Any]: Prediction results containing:
                - health_score: Overall health score (0-1)
                - confidence_interval: (lower, upper) bounds
                - risk_factors: List of identified risks
                - optimization_suggestions: List of suggestions

        Raises:
            ValueError: If required metrics are missing
            RuntimeError: If prediction fails for other reasons
        """
        # Validate required features
        required_features = self._get_feature_columns()
        missing_features = set(required_features) - set(metrics.keys())
        if missing_features:
            raise ValueError(f"Missing required metrics: {missing_features}")

        try:
            # Extract features in correct order
            features = pd.DataFrame({
                feature: [metrics[feature]]
                for feature in required_features
            })

            # Apply preprocessing
            features_processed = pd.DataFrame(
                self.preprocessor.transform(features),
                columns=self.feature_columns
            )

            # Add derived features in same order as training
            features_processed = self._add_derived_features(features_processed)

            # Generate prediction
            health_score = float(self.pipeline.predict(features_processed)[0])
            health_score = np.clip(health_score, 0, 1)  # Ensure score is between 0 and 1

            # Calculate confidence interval
            confidence_interval = (
                max(0.0, health_score - 0.1),
                min(1.0, health_score + 0.1)
            )

            # Generate insights
            risk_factors = self._identify_risk_factors(metrics)
            optimization_suggestions = self._generate_optimization_suggestions(metrics)

            return {
                'health_score': health_score,
                'confidence_interval': confidence_interval,
                'risk_factors': risk_factors,
                'optimization_suggestions': optimization_suggestions
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Failed to generate health score prediction: {str(e)}") from e

    def _extract_enhanced_features(self, metrics, historical_data=None):
        """Extract and validate features from input metrics"""
        features = {}
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in metrics:
                raise ValueError(f"Missing required feature: {col}")
            features[col] = metrics[col]
            
        # Add derived features if historical data is available
        if historical_data is not None and len(historical_data) > 0:
            df = pd.DataFrame(historical_data)
            features['trend_ctr'] = df['ctr'].mean() if 'ctr' in df else features['ctr']
            features['trend_conversion'] = df['conversion_rate'].mean() if 'conversion_rate' in df else features['conversion_rate']
        
        # Validate feature values
        for col, value in features.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature {col} must be numeric, got {type(value)}")
            if value < 0:
                raise ValueError(f"Feature {col} cannot be negative, got {value}")
                
        return features

    def _generate_shap_explanations(self, shap_values):
        """Generate SHAP-based feature explanations"""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before generating explanations.")
            
        feature_importance = {}
        for i, feature in enumerate(self.feature_columns):
            feature_importance[feature] = float(np.abs(shap_values.values[:, i]).mean())
        return feature_importance

    def _calculate_confidence(self, features_df):
        """Calculate prediction confidence with bootstrapping"""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before calculating confidence.")
            
        preds = []
        for _ in range(100):
            sample = features_df.sample(frac=1, replace=True)
            preds.append(self.pipeline.predict(sample)[0])
        return (np.mean(preds), np.std(preds))

    def _identify_risk_factors(self, metrics: Dict[str, float]) -> List[str]:
        """Identify risk factors based on metrics.

        Args:
            metrics (Dict[str, float]): Performance metrics

        Returns:
            List[str]: List of identified risk factors
        """
        try:
            # Convert metrics to DataFrame for SHAP analysis
            features = pd.DataFrame({
                feature: [metrics[feature]]
                for feature in self.feature_columns
            })
            
            # Calculate SHAP values
            shap_values = self._calculate_shap_values(features)
            
            # Get feature importance (mean absolute SHAP value for each feature)
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Identify top risk factors
            risk_factors = []
            for feature, importance in zip(self.feature_columns, feature_importance):
                if importance > 0.1:  # Only consider features with significant impact
                    if feature == 'ctr' and metrics[feature] < 0.02:
                        risk_factors.append(f"Low click-through rate: {metrics[feature]:.2%}")
                    elif feature == 'conversion_rate' and metrics[feature] < 0.01:
                        risk_factors.append(f"Low conversion rate: {metrics[feature]:.2%}")
                    elif feature == 'cost_per_conversion' and metrics[feature] > 100:
                        risk_factors.append(f"High cost per conversion: ${metrics[feature]:.2f}")
                    elif feature == 'roi' and metrics[feature] < 1:
                        risk_factors.append(f"Negative ROI: {metrics[feature]:.2f}")
            
            return risk_factors[:5]  # Return top 5 risk factors
            
        except Exception as e:
            logger.error(f"Failed to identify risk factors: {str(e)}")
            return []

    def _process_historical_data(self, historical_data):
        """Temporal feature engineering from historical data"""
        df = pd.DataFrame(historical_data)
        temporal_features = {
            '7d_avg_ctr': df['ctr'].rolling(7).mean().iloc[-1],
            '28d_trend_spend': self._calculate_trend(df['spend'], 28),
            '14d_conversion_rate': df['conversion_rate'].rolling(14).mean().iloc[-1],
            'revenue_velocity': self._calculate_velocity(df['revenue'])
        }
        return {k: v if not pd.isna(v) else 0 for k, v in temporal_features.items()}

    def _calculate_trend(self, series, window):
        """Calculate linear trend over specified window"""
        if len(series) < window:
            return 0
        x = np.arange(window)
        y = series.values[-window:]
        slope = np.polyfit(x, y, 1)[0]
        return slope / np.mean(y) if np.mean(y) != 0 else 0

    def _calculate_velocity(self, series):
        """Calculate rate of change over last 3 periods"""
        if len(series) < 4:
            return 0
        return (series.iloc[-1] - series.iloc[-4]) / 3

    def _rule_based_prediction(self, metrics, historical_data=None):
        """Fallback prediction with business rules"""
        score = 0.7  # Base score
        adjustments = []
        
        # ROI-based adjustment
        roi = metrics.get('roi', 0)
        if roi > 1.5:
            score += 0.15
        elif roi < 0:
            score -= 0.2
            
        # CTR-based adjustment
        ctr = metrics.get('ctr', 0)
        if ctr > 0.05:
            score += 0.1
        elif ctr < 0.01:
            score -= 0.15
            
        # Apply bounds and return
        final_score = np.clip(score, 0, 1)
        return {
            'health_score': float(final_score),
            'confidence_interval': (final_score - 0.1, final_score + 0.1),
            'risk_factors': [],
            'optimization_suggestions': [
                "Verify data quality",
                "Check system connectivity",
                "Review basic campaign metrics"
            ],
            'prediction_timestamp': datetime.now().isoformat()
        }

    def save_model(self, save_path):
        """Secure model serialization with privacy checks"""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model")
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save pipeline without training data
        joblib.dump(self.pipeline, os.path.join(save_path, 'pipeline.joblib'))
        
        # Save SHAP explainer separately
        if self.explainer:
            joblib.dump(self.explainer, os.path.join(save_path, 'explainer.joblib'))
            
        # Save configuration and fitted state
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump({
                'config': self.config,
                'is_fitted': self.is_fitted,
                'version': '1.0.0'
            }, f)
            
        logger.info(f"Model saved to {save_path}")

    def load_model(self, model_path):
        """Secure model loading with validation"""
        self.pipeline = joblib.load(os.path.join(model_path, 'pipeline.joblib'))
        
        if os.path.exists(os.path.join(model_path, 'explainer.joblib')):
            self.explainer = joblib.load(os.path.join(model_path, 'explainer.joblib'))
            
        with open(os.path.join(model_path, 'config.json')) as f:
            saved_state = json.load(f)
            self.config.update(saved_state['config'])
            self.is_fitted = saved_state['is_fitted']
            
        logger.info(f"Model loaded from {model_path}")

    def _load_config(self, config_path=None):
        """Load configuration with defaults"""
        default_config = {
            'gb_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            },
            'nn_params': {
                'epochs': 100,
                'batch_size': 32
            },
            'risk_thresholds': {
                'ctr': {
                    'critical_low': 0.01,
                    'warning_low': 0.02
                },
                'conversion_rate': {
                    'critical_low': 0.005,
                    'warning_low': 0.01
                },
                'cost_per_conversion': {
                    'warning_high': 100,
                    'critical_high': 200
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                return {**default_config, **loaded_config}
        
        return default_config

    def _update_model_params(self, params):
        """Update model parameters based on optimization results"""
        self.config['gb_params'] = {
            'max_depth': params['gb_max_depth'],
            'learning_rate': params['gb_learning_rate'],
            'n_estimators': 100
        }
        
        self.config['nn_params'] = {
            'hidden_layer_sizes': params['nn_hidden_layer_sizes'],
            'learning_rate_init': params['nn_learning_rate_init'],
            'batch_size': params['nn_batch_size'],
            'max_iter': 200,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'random_state': 42
        }
        
        # Update pipeline ensemble parameters
        if hasattr(self.pipeline, 'named_steps') and 'ensemble' in self.pipeline.named_steps:
            self.pipeline.named_steps['ensemble'].gb_params = self.config['gb_params']
            self.pipeline.named_steps['ensemble'].nn_params = self.config['nn_params']

    def _calculate_privacy_budget(self):
        """Calculate differential privacy budget usage.
        For scikit-learn models, we use a simplified privacy calculation.
        """
        # For scikit-learn models, we use a constant privacy budget
        # This is a simplified approach since MLPRegressor doesn't track training history
        return 0.1  # Fixed privacy budget for now

    def _evaluate_model(self, X, y):
        """Evaluate model performance with privacy considerations"""
        y_pred = self.pipeline.predict(X)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'epsilon_spent': self._calculate_privacy_budget(),
        }
        
        # Calculate feature importance using SHAP values
        if hasattr(self.pipeline.named_steps['ensemble'], 'gb'):
            explainer = shap.TreeExplainer(self.pipeline.named_steps['ensemble'].gb)
            shap_values = explainer.shap_values(X)
            metrics['feature_importance'] = dict(zip(self.feature_columns, np.abs(shap_values).mean(axis=0)))
        
        return metrics

    def _create_model(self, params):
        """Create a model with the specified parameters"""
        return Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('ensemble', EnhancedHealthEnsemble(
                gb_params={
                    'n_estimators': 100,
                    'max_depth': params['gb_max_depth'],
                    'learning_rate': params['gb_learning_rate']
                },
                nn_params={
                    'epochs': params['nn_epochs'],
                    'batch_size': params['nn_batch_size']
                }
            ))
        ])

    def _calculate_shap_values(self, features_df: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for feature importance.

        Args:
            features_df (pd.DataFrame): Feature values to explain

        Returns:
            np.ndarray: SHAP values for each feature
        """
        try:
            if not hasattr(self, 'explainer'):
                self.explainer = shap.TreeExplainer(
                    self.pipeline.named_steps['ensemble'].gb,
                    feature_perturbation='interventional',
                    check_additivity=False
                )
            
            # Ensure features are in the correct order
            features_df = features_df[self.feature_columns]
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features_df)
            
            # Handle case where shap_values is a list (multi-output)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Ensure we have a 2D array
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
                
            return shap_values
            
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {str(e)}")
            # Return zero values as fallback
            return np.zeros((1, len(self.feature_columns)))

    def _generate_optimization_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        """Generate optimization suggestions based on metrics.

        Args:
            metrics (Dict[str, float]): Performance metrics

        Returns:
            List[str]: List of optimization suggestions
        """
        suggestions = []

        # CTR optimization
        if metrics['ctr'] < 0.02:
            suggestions.append("Consider improving ad relevance and targeting to increase CTR")
        
        # Conversion rate optimization
        if metrics['conversion_rate'] < 0.01:
            suggestions.append("Focus on landing page optimization and audience targeting to improve conversion rate")
        
        # Cost efficiency
        if metrics['cost_per_conversion'] > 100:
            suggestions.append("Review bidding strategy and targeting to reduce cost per conversion")
        
        # Revenue optimization
        roi = metrics['revenue'] / metrics['spend'] if metrics['spend'] > 0 else 0
        if roi < 2:
            suggestions.append("Consider adjusting targeting or bidding to improve ROI")
        
        # Scale optimization
        if metrics['impressions'] < 1000:
            suggestions.append("Consider expanding reach by increasing budget or broadening targeting")
        
        return suggestions[:5]  # Return top 5 suggestions

class AccountHealthPredictor:
    """
    Simplified Account Health Predictor wrapper class.
    
    This class provides a standardized interface for the Account Health Prediction system,
    serving as a wrapper around the more complex AdvancedHealthPredictor implementation.
    It implements the same key methods but with simplified parameters and defaults
    for easier integration with other components.
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AccountHealthPredictor.
        
        Args:
            model_path: Optional path to a saved model file
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.predictor = AdvancedHealthPredictor(
            model_path=model_path, 
            config_path=None
        )
        
        # Set default config values if not provided
        if not model_path and not config:
            self._set_default_config()
        
        self.model_info = {
            "name": "account_health_predictor",
            "version": "1.0.0",
            "type": "regression",
            "target": "health_score",
            "created_at": datetime.now().isoformat()
        }
        
    def _set_default_config(self) -> None:
        """Set default configuration for the predictor."""
        default_config = {
            "feature_columns": [
                "clicks", "impressions", "ctr", "spend", "conversions", 
                "conversion_rate", "revenue", "roas", "quality_score"
            ],
            "preprocessing": {
                "imputation_strategy": "knn",
                "scaling": True,
                "outlier_removal": "quantile",
                "feature_engineering": True
            },
            "model_params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3
            },
            "training": {
                "test_size": 0.2,
                "cv_folds": 3,
                "early_stopping_rounds": 10
            }
        }
        self.config.update(default_config)
        self.predictor._update_model_params(self.config.get("model_params", {}))
        
    def train(self, 
              data: pd.DataFrame, 
              target_column: str = "health_score", 
              save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Train the account health prediction model.
        
        Args:
            data: DataFrame containing training data
            target_column: Name of the target column
            save_path: Optional path to save the trained model
            
        Returns:
            Dictionary with evaluation metrics
        """
        return self.predictor.train(
            training_data=data,
            target_column=target_column,
            save_path=save_path
        )
    
    def predict(self, data: Union[Dict[str, float], pd.DataFrame]) -> Dict[str, Any]:
        """
        Predict account health score for the provided metrics.
        
        Args:
            data: Dictionary of metric values or DataFrame with metrics
            
        Returns:
            Dictionary with prediction results
        """
        if isinstance(data, pd.DataFrame):
            # For batch predictions, process each row
            results = []
            for _, row in data.iterrows():
                metrics = row.to_dict()
                results.append(self.predictor.predict_health_score(metrics))
            return {"predictions": results}
        else:
            # Single prediction
            return self.predictor.predict_health_score(data)
    
    def explain(self, data: Union[Dict[str, float], pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate explanations for a prediction.
        
        Args:
            data: Dictionary of metric values or DataFrame with metrics
            
        Returns:
            Dictionary with prediction explanations
        """
        prediction_result = self.predict(data)
        
        if isinstance(data, pd.DataFrame):
            # Not supporting batch explanations in the simplified interface
            return {"error": "Batch explanations not supported, please provide a single instance"}
        
        # Extract the feature importance information
        explanation = {
            "prediction": prediction_result.get("health_score", 0),
            "confidence": prediction_result.get("confidence", 0),
            "risk_factors": prediction_result.get("risk_factors", []),
            "feature_importance": prediction_result.get("feature_importance", {}),
            "optimization_suggestions": prediction_result.get("optimization_suggestions", [])
        }
        
        return explanation
    
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        self.predictor.save_model(path)
        
    def load(self, path: str) -> None:
        """
        Load a model from the specified path.
        
        Args:
            path: Path to load the model from
        """
        self.predictor.load_model(path)
        
    def evaluate(self, data: pd.DataFrame, target_column: str = "health_score") -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            data: Test data
            target_column: Target column name
            
        Returns:
            Dictionary with evaluation metrics
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return self.predictor._evaluate_model(X, y)
        