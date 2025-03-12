from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class BaseMLModel(nn.Module):
    """
    Base class for all machine learning models in the WITHIN ML Prediction System.
    
    This class provides standardized interfaces and common functionality for
    model initialization, training, evaluation, and inference. It enforces
    project-wide standards for ML models including hardware acceleration,
    mixed precision training, gradient accumulation, and performance monitoring.
    
    All prediction models should inherit from this base class to ensure
    consistent behavior and compatibility with the ML system infrastructure.
    
    Attributes:
        device (torch.device): The device (CPU/GPU) where the model runs.
        config (Dict[str, Any]): Configuration parameters for model training and inference.
        
    Examples:
        >>> class MyModel(BaseMLModel):
        ...     def __init__(self, config=None):
        ...         super().__init__(config)
        ...         self.linear = nn.Linear(10, 1)
        ...     
        ...     def forward(self, x):
        ...         return self.linear(x)
        >>> 
        >>> model = MyModel()
        >>> model.to(model.device)
        >>> # Train and use the model
        
    Notes:
        This class inherits from torch.nn.Module and follows PyTorch conventions.
        It automatically enables advanced CUDA optimizations when available.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base ML model with configuration.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary with parameters
                for model training and inference. Values provided will override defaults.
                Defaults to None.
                
        Notes:
            Default configuration includes:
            - learning_rate: 0.001
            - batch_size: 2048
            - max_epochs: 100
            - early_stopping_patience: 10
            - gradient_accumulation_steps: 1
            - weight_decay: 1e-5
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default configuration
        self.config = {
            'learning_rate': 0.001,
            'batch_size': 2048,
            'max_epochs': 100,
            'early_stopping_patience': 10,
            'gradient_accumulation_steps': 1,
            'weight_decay': 1e-5
        }
        if config:
            self.config.update(config)
            
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high') 