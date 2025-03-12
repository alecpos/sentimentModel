"""
Production validation module for machine learning models.

This module provides tools and utilities for validating ML models
in production environments, including shadow deployments, A/B testing,
canary releases, and model monitoring.

Key components include:
- Shadow deployment for risk-free testing
- A/B testing framework for model comparison
- Canary deployment for staged rollouts
- Golden set validation for regression testing
- Performance comparison utilities

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

from app.models.ml.validation.shadow_deployment import (
    ShadowDeployment,
    ABTestDeployment,
    CanaryDeployment
)

from app.models.ml.validation.ab_test_manager import (
    ABTestManager
)

__all__ = [
    # Deployment strategies
    'ShadowDeployment',
    'ABTestDeployment', 
    'CanaryDeployment',
    
    # Test management
    'ABTestManager'
]
