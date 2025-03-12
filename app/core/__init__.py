"""
Core Module for WITHIN

This module provides core functionality for the WITHIN
Ad Score & Account Health Predictor system, including
data integration, validation, and fairness evaluation.
"""

from .database import BaseModel
from .errors import (
    BaseMLError,
    ValidationError,
    PredictionError,
    ModelNotFoundError,
    ModelLoadError,
    DataError,
    SecurityError,
    ConfigurationError
)

# Import submodules for easy access
from . import db
from . import validation
from . import preprocessor
from . import search
from . import events
from . import feedback
from . import data_lake
from . import ml

__all__ = [
    # Base model
    "BaseModel",
    
    # Error types
    "BaseMLError",
    "ValidationError",
    "PredictionError",
    "ModelNotFoundError",
    "ModelLoadError",
    "DataError",
    "SecurityError",
    "ConfigurationError",
    
    # Submodules
    "db",
    "validation",
    "preprocessor",
    "search",
    "events",
    "feedback",
    "data_lake",
    "ml"
]
