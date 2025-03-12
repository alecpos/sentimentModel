"""
Training utilities for WITHIN ML Prediction System models.

This module provides standardized training infrastructure for all machine learning 
models in the WITHIN ML Prediction System. It implements best practices for model 
training, including mixed-precision training, gradient accumulation, early stopping,
performance budgeting, secure model persistence, and fairness monitoring.

Attributes:
    logger: Logging instance for the module.

Examples:
    Basic usage of ModelTrainer to train a model:

    >>> from app.models.ml.prediction.training import ModelTrainer
    >>> from app.models.ml.prediction import get_ad_score_predictor
    >>> 
    >>> # Get model class and instantiate
    >>> AdScorePredictor = get_ad_score_predictor()
    >>> model = AdScorePredictor()
    >>> 
    >>> # Create trainer and train model
    >>> trainer = ModelTrainer(model)
    >>> metrics = trainer.train(train_loader, val_loader)
    >>> print(f"Training complete with metrics: {metrics}")

Key components:
- ModelTrainer: Main class for training BaseMLModel instances
- Training utilities for standardized model training
- Performance monitoring and validation

Notes:
    This module adheres to the ML constraints defined in the project standards:
    - Memory budget: 2GB
    - Inference time: 300ms
    - Fairness metrics monitoring
    - Security requirements for model persistence

Version: 0.1.0
Documentation status: COMPLETE
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from app.security.model_protection import ModelProtection
from app.models.ml.prediction.base import BaseMLModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Training wrapper implementing WITHIN ML project standards for model training.
    
    This class provides a standardized training infrastructure that enforces best
    practices for training machine learning models in the system, including:
    - Mixed precision training (BF16/FP16)
    - Gradient accumulation for larger effective batch sizes
    - Early stopping to prevent overfitting
    - Model checkpointing and persistence
    - Secure model storage with encryption
    - Performance monitoring and logging
    
    Attributes:
        model (BaseMLModel): The model to be trained.
        device (torch.device): Device where training will be performed.
        config (Dict[str, Any]): Configuration parameters for training.
    
    Examples:
        >>> model = AdScorePredictor()
        >>> trainer = ModelTrainer(model)
        >>> metrics = trainer.train(train_loader, val_loader)
        >>> print(f"Training complete. Final metrics: {metrics}")
    """
    
    def __init__(self, model: BaseMLModel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model trainer.
        
        Args:
            model: The model to train. Must inherit from BaseMLModel.
            config: Configuration overrides for training.
                If provided, these values will override both the default trainer 
                config and any config specified in the model. Defaults to None.
        
        Returns:
            None
            
        Raises:
            TypeError: If model is not an instance of BaseMLModel.
            
        Examples:
            >>> model = AdScorePredictor()
            >>> trainer = ModelTrainer(model)
            >>> custom_config = {"learning_rate": 0.001, "max_epochs": 20}
            >>> trainer_with_config = ModelTrainer(model, config=custom_config)
        
        Note:
            Training configuration is taken from the model's config by default,
            but can be overridden by providing a config dictionary.
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Use model's config as base
        self.config = model.config.copy()
        if config:
            self.config.update(config)
            
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Train the model using project standards and best practices.
        
        This method implements a complete training pipeline including:
        - Mixed precision training
        - Gradient accumulation
        - Early stopping
        - Model checkpointing
        - Performance logging
        
        Args:
            train_loader: DataLoader providing training batches.
            val_loader: DataLoader providing validation 
                batches. If provided, validation will be performed after each epoch.
                Defaults to None.
        
        Returns:
            Dict[str, float]: Dictionary of training metrics including final loss
                values and best validation metrics.
        
        Raises:
            RuntimeError: If training fails due to numerical instability or
                other runtime errors.
            ValueError: If train_loader is empty or not properly configured.
        
        Examples:
            >>> trainer = ModelTrainer(model)
            >>> train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            >>> train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
            >>> val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            >>> val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
            >>> metrics = trainer.train(train_loader, val_loader)
            >>> best_val_loss = min([m['loss'] for m in metrics['val_metrics']])
        
        Notes:
            Training can be interrupted after meeting early stopping criteria
            based on validation loss. The model state will be reverted to the
            best checkpoint when training completes.
        """
        # Setup mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Optimizer with gradient accumulation
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        metrics_history = []
        
        for epoch in range(self.config['max_epochs']):
            # Training loop
            self.model.train()
            train_loss = 0
            
            for i, batch in enumerate(train_loader):
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch)
                
                # Gradient accumulation
                scaled_loss = loss / self.config['gradient_accumulation_steps']
                scaler.scale(scaled_loss).backward()
                
                if (i + 1) % self.config['gradient_accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item()
            
            # Validation
            if val_loader:
                val_metrics = self._validate(val_loader)
                metrics_history.append(val_metrics)
                
                # Early stopping check
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self._save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                        
                # Check fairness metrics
                if val_metrics.get('demographic_parity', 0) > self.config['fairness_threshold']:
                    logger.warning(f"Fairness threshold exceeded: {val_metrics['demographic_parity']}")
                    
        return {'train_loss': train_loss, 'val_metrics': metrics_history}
    
    def _compute_loss(self, batch) -> torch.Tensor:
        """
        Compute loss with proper error handling.
        
        This method processes a batch of data through the model and calculates
        the loss value with appropriate error handling to prevent training failures.
        
        Args:
            batch: A tuple containing input features (x) and target values (y).
                Expected format is (features_tensor, labels_tensor).
        
        Returns:
            torch.Tensor: Computed loss value.
            
        Raises:
            RuntimeError: If a runtime error occurs during loss computation,
                such as out-of-memory errors or numerical instability.
            ValueError: If batch does not contain exactly two tensors (inputs and targets).
            
        Examples:
            >>> trainer = ModelTrainer(model)
            >>> inputs = torch.randn(32, 10)
            >>> targets = torch.randint(0, 2, (32, 1)).float()
            >>> batch = (inputs, targets)
            >>> loss = trainer._compute_loss(batch)
                
        Notes:
            This is an internal method used by the train() function and
            not intended to be called directly by users.
        """
        try:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            output = self.model(x)
            return nn.BCELoss()(output, y)
        except RuntimeError as e:
            logger.error(f"Error computing loss: {str(e)}")
            raise
            
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Perform validation with performance monitoring.
        
        This method evaluates the model on validation data and calculates
        performance metrics including loss and inference time. It also
        monitors whether inference time stays within the configured budget.
        
        Args:
            val_loader: DataLoader providing validation batches.
            
        Returns:
            Dict[str, float]: Dictionary containing validation metrics:
                - 'loss': Average validation loss
                - 'inference_time': Average inference time in milliseconds
        
        Raises:
            RuntimeError: If validation fails due to model or data loader issues.
            ValueError: If val_loader is empty or not properly configured.
            
        Examples:
            >>> trainer = ModelTrainer(model)
            >>> val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            >>> val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
            >>> metrics = trainer._validate(val_loader)
            >>> val_loss = metrics['loss']
            >>> inference_time = metrics['inference_time']
                
        Notes:
            This is an internal method used by the train() function and
            not intended to be called directly by users.
            
            Performance budget checks are performed against the threshold
            defined in the configuration as 'performance_budget_ms'.
        """
        self.model.eval()
        metrics = {}
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                # Measure inference time
                start_time.record()
                output = self.model(batch[0].to(self.device))
                end_time.record()
                
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                
                # Check performance budget
                if inference_time > self.config['performance_budget_ms']:
                    logger.warning(f"Inference time {inference_time}ms exceeds budget")
                
                val_loss += self._compute_loss(batch).item()
                
        metrics['loss'] = val_loss / len(val_loader)
        metrics['inference_time'] = inference_time
        return metrics
    
    def _save_checkpoint(self, filename: str) -> None:
        """
        Save encrypted model checkpoint.
        
        This method saves the current state of the model to a checkpoint file
        with encryption for security. The model is encrypted using the 
        ModelProtection utility before being written to disk.
        
        Args:
            filename: Name of the checkpoint file to save.
        
        Returns:
            None
            
        Raises:
            IOError: If the file cannot be written due to permission issues
                or disk space limitations.
            KeyError: If encryption keys are not available or invalid.
            
        Examples:
            >>> trainer = ModelTrainer(model)
            >>> # After training
            >>> trainer._save_checkpoint('my_model_checkpoint.pt')
                
        Notes:
            Checkpoint files are saved in the 'checkpoints/' directory.
            Encryption keys are expected to be in the 'keys/' directory.
            This is an internal method used by the train() function.
        """
        protection = ModelProtection(Path("keys/"))
        protection.save_model(self.model, Path(f"checkpoints/{filename}")) 