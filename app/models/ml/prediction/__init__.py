"""
Enhanced Ensemble Model Package

This package provides advanced ensemble methods for machine learning,
including enhanced bagging and stacking implementations with comprehensive
performance monitoring and visualization capabilities.
"""

from .enhanced_ensemble import (
    EnhancedBaggingEnsemble,
    EnhancedStackingEnsemble,
    optimize_ensemble_weights,
    visualize_ensemble_performance,
    PerformanceMetrics
)

__version__ = "0.1.0"
__all__ = [
    "EnhancedBaggingEnsemble",
    "EnhancedStackingEnsemble",
    "optimize_ensemble_weights",
    "visualize_ensemble_performance",
    "PerformanceMetrics"
]
