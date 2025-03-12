"""
Security components for the WITHIN ML Prediction System.

This package contains security mechanisms for protecting ML models, 
ensuring data privacy, and providing authentication and authorization.
These components work together to create a secure environment for 
machine learning operations.
"""

from .model_protection import ModelProtection

__version__ = "0.1.0"  # Pre-release version

__all__ = [
    "ModelProtection"
] 