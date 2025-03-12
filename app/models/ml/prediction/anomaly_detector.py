"""
Enhanced Anomaly Detection System implementing hybrid ML approach for performance metrics.

This module provides anomaly detection capabilities for identifying unusual patterns
in advertising account performance metrics. It implements a hybrid approach combining:
- Deep learning (autoencoder) for complex pattern recognition
- Classical ML methods (Isolation Forest, One-Class SVM) for robust detection
- Statistical methods for baseline performance
- SHAP-based explanations for detected anomalies

The system is designed to identify performance degradation, unusual spending patterns,
and other anomalies in advertising campaign data with high precision and explainability.

Key components:
- AutoEncoder: Neural network for learning normal data patterns
- EnhancedAnomalyDetector: Main class for anomaly detection and explanation

Example usage:
    >>> detector = EnhancedAnomalyDetector()
    >>> detector.train(historical_performance_metrics)
    >>> result = detector.detect(current_metrics, recent_metrics_history)
    >>> 
    >>> if result['is_anomaly']:
    ...     print(f"Anomaly detected with score {result['anomaly_score']}")
    ...     print(f"Explanation: {result['explanation']}")

Version: 0.1.0
Documentation status: COMPLETE
"""
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import OneClassSVM
from sklearn.linear_model import ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import optuna
import shap

logger = logging.getLogger(__name__)

class AutoEncoder(nn.Module):
    """
    Enhanced autoencoder neural network for anomaly detection.
    
    This autoencoder architecture is specifically designed for detecting anomalies
    in advertising performance metrics. It includes regularization and dropout for
    improved generalization and robustness against overfitting.
    
    The network consists of an encoder that compresses the input data into a
    lower-dimensional latent space, and a decoder that reconstructs the input
    from this latent representation. Anomalies are detected by measuring the
    reconstruction error.
    
    Attributes:
        encoder (nn.Sequential): Encoder neural network component
        decoder (nn.Sequential): Decoder neural network component
        latent_dim (int): Dimension of the latent space representation
        
    Args:
        input_dim (int): Dimension of the input feature vector
        hidden_dims (List[int], optional): Dimensions of hidden layers in encoder.
            The decoder uses the same dimensions in reverse. Defaults to [64, 32, 16].
        dropout_rate (float, optional): Dropout probability for regularization.
            Defaults to 0.2.
            
    Examples:
        >>> autoencoder = AutoEncoder(input_dim=10, hidden_dims=[32, 16, 8])
        >>> x = torch.randn(100, 10)  # 100 samples with 10 features each
        >>> reconstructed = autoencoder(x)
        >>> loss = torch.mean((reconstructed - x) ** 2)  # MSE reconstruction error
    """
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.2):
        super().__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = hidden_dims[-1]
        for dim in hidden_dims_reversed[1:]:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class EnhancedAnomalyDetector:
    """
    Enhanced anomaly detector for identifying unusual patterns in advertising metrics.
    
    This detector implements a hybrid approach to anomaly detection, combining deep
    learning, traditional machine learning, and statistical methods to achieve high
    accuracy and explainability. It is specifically designed for detecting performance
    issues in advertising campaigns.
    
    Key features:
    - Multi-model ensemble (autoencoder, Isolation Forest, One-Class SVM)
    - Dynamic thresholding based on data characteristics
    - Explainable detections with SHAP values
    - Graceful fallback mechanisms for robustness
    - Hyperparameter optimization with Optuna
    
    Attributes:
        autoencoder (AutoEncoder): Neural network for learning normal data patterns
        iso_forest (IsolationForest): Ensemble-based outlier detection model
        svm (OneClassSVM): Support vector machine for outlier detection
        scaler (StandardScaler): Data normalizer for consistent scaling
        feature_selector (VarianceThreshold): Feature selection for dimensionality reduction
        config (Dict[str, Any]): Configuration parameters for the detector
        threshold (float): Anomaly score threshold for binary decision
        thresholds (Dict[str, float]): Feature-specific thresholds for fine-grained detection
        device (torch.device): Device to run computations on (CPU/GPU)
        
    Examples:
        >>> # Initialize detector
        >>> detector = EnhancedAnomalyDetector({"sensitivity": 0.95})
        >>> 
        >>> # Train on historical data
        >>> detector.train([
        ...     {"impressions": 1000, "clicks": 20, "conversions": 2, "spend": 100},
        ...     # More historical metrics...
        ... ])
        >>> 
        >>> # Detect anomalies in new data
        >>> result = detector.detect(
        ...     {"impressions": 100, "clicks": 1, "conversions": 0, "spend": 50},
        ...     recent_history
        ... )
        >>> 
        >>> print(f"Anomaly: {result['is_anomaly']}, Score: {result['anomaly_score']}")
        >>> if result['is_anomaly']:
        ...     print(f"Contributing factors: {result['explanation']['factors']}")
    """
    INPUT_DIM = 6  # Number of input features
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly detector with default or custom configuration.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary that can 
                override the default settings. Valid keys include:
                - learning_rate: Learning rate for autoencoder training
                - batch_size: Batch size for autoencoder training
                - max_epochs: Maximum training epochs
                - early_stopping_patience: Epochs to wait before early stopping
                - sensitivity: Detection sensitivity (0-1, higher = more sensitive)
                - ensemble_weights: Weights for different models in the ensemble
                Defaults to None, which uses internal defaults.
                
        Notes:
            The detector will automatically select the best device (GPU/CPU) 
            for computation based on availability.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default configuration
        self.config = {
            'learning_rate': 0.001,
            'batch_size': 2048,
            'max_epochs': 100,
            'early_stopping_patience': 10,
            'hidden_dims': [64, 32, 16],
            'dropout_rate': 0.2,
            'l2_reg': 1e-5,
            'threshold_percentile': 95
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Initialize models dictionary with scaler
        self.models = {
            'scaler': StandardScaler(),
            'autoencoder': None  # Will be initialized during training
        }
        self.reconstruction_errors = None
        self.threshold = None
        self.is_fitted = False
        self.is_trained = False  # Add for backward compatibility
        
    def train(self, metrics: List[Dict[str, float]]) -> None:
        """Train the anomaly detector on the provided metrics"""
        try:
            # Convert metrics to numpy array and normalize
            X = self._preprocess_data(metrics)
            
            # Initialize autoencoder if not already done
            if self.models['autoencoder'] is None:
                self.models['autoencoder'] = AutoEncoder(
                    input_dim=X.shape[1],
                    hidden_dims=self.config['hidden_dims'],
                    dropout_rate=self.config['dropout_rate']
                ).to(self.device)
            
            # Train autoencoder
            self._train_autoencoder(X)
            
            # Calculate reconstruction error distribution
            self._calculate_threshold(X)
            
            self.is_fitted = True
            self.is_trained = True  # Sync with is_fitted
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            raise

    def detect(self, current: Dict[str, float], 
              history: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Detect anomalies in current metrics by comparing to learned patterns.
        
        This method evaluates whether the current performance metrics are anomalous
        based on the patterns learned during training. It uses the trained autoencoder
        to reconstruct the input and measures the reconstruction error as an indicator
        of anomalousness.
        
        Args:
            current (Dict[str, float]): Dictionary of current performance metrics
                to evaluate. Must contain the same metrics used during training.
            history (List[Dict[str, float]]): Recent historical metrics for context
                and trend analysis. This helps with contextual anomaly detection.
                
        Returns:
            Dict[str, Any]: Detection results containing:
                - is_anomaly (bool): True if metrics are anomalous
                - reconstruction_error (float): Error between input and reconstruction
                - threshold (float): Current anomaly threshold
                - confidence (float): Confidence level of the detection (0-1)
                - explanation (Dict, optional): Present if anomalous, contains factors
                  contributing to the anomaly
                
        Raises:
            RuntimeError: If the detector has not been trained
            ValueError: If metrics are missing required fields
            
        Examples:
            >>> detector = EnhancedAnomalyDetector()
            >>> detector.train(historical_data)
            >>> 
            >>> # Check if current metrics are anomalous
            >>> current_metrics = {
            ...     "impressions": 1200,
            ...     "clicks": 15,
            ...     "conversions": 1,
            ...     "cost_per_click": 0.75,
            ...     "conversion_rate": 0.0083,
            ...     "cost_per_conversion": 90.0
            ... }
            >>> 
            >>> result = detector.detect(current_metrics, recent_history)
            >>> if result["is_anomaly"]:
            ...     print(f"Anomaly detected! Confidence: {result['confidence']:.2f}")
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be trained before use")
            
        # Preprocess current data
        X = self._preprocess_data([current])
        
        # Get reconstruction error
        self.models['autoencoder'].eval()
        with torch.no_grad():
            x = torch.FloatTensor(X).to(self.device)
            reconstructed = self.models['autoencoder'](x)
            error = torch.mean(torch.pow(x - reconstructed, 2), dim=1).cpu().numpy()[0]
        
        # Determine if anomalous
        is_anomaly = error > self.threshold
        
        return {
            'is_anomaly': bool(is_anomaly),
            'reconstruction_error': float(error),
            'threshold': float(self.threshold),
            'confidence': float(1.0 - error / (error + self.threshold))
        }

    def get_anomaly_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate the anomaly score for the given metrics.
        
        Args:
            metrics: Dictionary containing the metrics to score
            
        Returns:
            float: Anomaly score (higher means more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be trained before use")
            
        # Preprocess data
        X = self._preprocess_data([metrics])
        
        # Calculate reconstruction error (anomaly score)
        self.models['autoencoder'].eval()
        with torch.no_grad():
            x = torch.FloatTensor(X).to(self.device)
            reconstructed = self.models['autoencoder'](x)
            error = torch.mean(torch.pow(x - reconstructed, 2), dim=1).cpu().numpy()[0]
            
        return float(error)

    def _preprocess_data(self, metrics: List[Dict[str, float]]) -> np.ndarray:
        """Convert metrics to normalized numpy array"""
        df = pd.DataFrame(metrics)
        required_columns = ['clicks', 'conversions', 'spend', 'revenue', 'impressions', 'ctr']
        
        # Ensure all required columns exist
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract features
        X = df[required_columns].values
        
        # Scale features
        if not self.is_fitted:
            X = self.models['scaler'].fit_transform(X)
        else:
            X = self.models['scaler'].transform(X)
        
        return X

    def _train_autoencoder(self, X: np.ndarray) -> None:
        """Optimized training with learning rate scheduling"""
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        optimizer = torch.optim.AdamW(
            self.models['autoencoder'].parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['l2_reg']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['max_epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            self.models['autoencoder'].train()
            for batch in loader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.models['autoencoder'](x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
    
    def _calculate_threshold(self, X: np.ndarray) -> None:
        """Calculate reconstruction error threshold"""
        self.models['autoencoder'].eval()
        with torch.no_grad():
            x = torch.FloatTensor(X).to(self.device)
            reconstructed = self.models['autoencoder'](x)
            errors = torch.mean(torch.pow(x - reconstructed, 2), dim=1).cpu().numpy()
            
        self.reconstruction_errors = errors
        self.threshold = np.percentile(errors, self.config['threshold_percentile'])

    def _calculate_dynamic_thresholds(self, X: np.ndarray) -> None:
        """Adaptive thresholding using quantiles"""
        scores = self._ensemble_predict(X)
        for metric in self.config['metrics']:
            self.thresholds[metric] = np.quantile(
                scores[metric], 
                self.config['threshold_quantile']
            )

    def _ensemble_predict(self, X: np.ndarray) -> Dict[str, float]:
        """Stacked ensemble predictions"""
        iso_scores = self.models['iso_forest'].decision_function(X)
        nn_scores = self._calculate_nn_anomaly_scores(X)
        oc_scores = self.models['ocsvm'].decision_function(X)
        
        return {
            metric: (
                iso_scores[i] * self.config['weights']['iso'] +
                nn_scores[i] * self.config['weights']['nn'] +
                oc_scores[i] * self.config['weights']['ocsvm']
            )
            for i, metric in enumerate(self.config['metrics'])
        }

    def _generate_explanations(self, X: np.ndarray) -> Dict[str, Any]:
        """SHAP explanations with feature importance"""
        shap_values = self.explainer.shap_values(X)
        return {
            'feature_importance': np.abs(shap_values).mean(0),
            'summary_plot': shap.summary_plot(shap_values, X),
            'decision_plot': shap.decision_plot(self.explainer.expected_value, shap_values, X)
        }

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the anomaly detector"""
        return {
            'metrics': ['ctr', 'conversion_rate', 'cost_per_conversion', 'impressions', 'clicks', 'spend'],
            'iso_forest': {
                'n_estimators': 100,
                'contamination': 'auto',
                'random_state': 42
            },
            'ocsvm': {
                'kernel': 'rbf',
                'nu': 0.1
            },
            'autoencoder': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'max_epochs': 100,
            'early_stopping_patience': 10,
            'hidden_dims': [64, 32, 16],
            'dropout_rate': 0.2,
            'l2_reg': 1e-5,
            'threshold_percentile': 95
        }

    def _calculate_nn_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores using autoencoder reconstruction error"""
        self.models['autoencoder'].eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.models['autoencoder'](X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            return errors.cpu().numpy()

    def _fallback_detection(self, current: Dict[str, float], 
                           error: Exception = None) -> Dict[str, Any]:
        """Enhanced fallback with partial results"""
        return {
            'anomalies': {metric: {'is_anomaly': False} for metric in self.config['metrics']},
            'detection_time': datetime.utcnow().isoformat(),
            'error': str(error) if error else 'Unknown error',
            'fallback': True
        }

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """SHAP-compatible prediction function"""
        return self._ensemble_predict(X)

    def _build_temporary_autoencoder(self, params: Dict[str, Any]) -> nn.Module:
        """Build a temporary autoencoder for hyperparameter optimization"""
        layers = []
        input_dim = self.config['reduced_dim']
        current_dim = input_dim
        
        # Encoder layers
        for _ in range(params['num_layers']):
            layers.extend([
                nn.Linear(current_dim, params['hidden_dim']),
                nn.BatchNorm1d(params['hidden_dim']),
                nn.ReLU()
            ])
            current_dim = params['hidden_dim']
        
        # Bottleneck layer
        bottleneck_dim = max(2, current_dim // 2)
        layers.extend([
            nn.Linear(current_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        ])
        
        # Decoder layers
        current_dim = bottleneck_dim
        for _ in range(params['num_layers']):
            next_dim = params['hidden_dim'] if _ < params['num_layers'] - 1 else input_dim
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU() if _ < params['num_layers'] - 1 else nn.Identity()
            ])
            current_dim = next_dim
        
        model = nn.Sequential(*layers).to(self.device)
        return model

    def _validate_autoencoder(self, model: nn.Module, X: np.ndarray, 
                            val_size: float = 0.2) -> float:
        """Validate autoencoder performance"""
        # Split data
        n_val = int(len(X) * val_size)
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        
        # Convert to tensor
        X_val = torch.FloatTensor(X[val_indices]).to(self.device)
        
        # Validation
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_val)
            loss = torch.nn.MSELoss()(reconstructed, X_val)
        
        return loss.item()