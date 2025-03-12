"""
Monitoring module for machine learning models.

This module provides tools for monitoring ML models in production,
including drift detection, performance monitoring, and alerting.

Key components include:
- Data drift detection for input features
- Concept drift detection for model predictions
- Feature distribution monitoring
- Feature correlation monitoring
- Performance degradation alerting system

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

from app.models.ml.monitoring.drift_detector import (
    DriftDetector,
    DataDriftDetector,
    ConceptDriftDetector
)

from app.models.ml.monitoring.prediction_drift_detector import (
    PredictionDriftDetector
)

from app.models.ml.monitoring.feature_monitor import (
    FeatureDistributionMonitor,
    FeatureCorrelationMonitor
)

from app.models.ml.monitoring.concept_drift_detector import (
    ConceptDriftDetector as ConceptDriftDetectorEnhanced
)

from app.models.ml.monitoring.alert_manager import (
    AlertManager
)

__all__ = [
    # Drift detection
    'DriftDetector',
    'DataDriftDetector',
    'ConceptDriftDetector',
    'PredictionDriftDetector',
    'ConceptDriftDetectorEnhanced',
    
    # Feature monitoring
    'FeatureDistributionMonitor',
    'FeatureCorrelationMonitor',
    
    # Alerting
    'AlertManager'
]
