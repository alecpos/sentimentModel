"""
Feedback management components for the WITHIN ML Prediction System.

This module provides functionality for collecting, storing, and analyzing user feedback
on model predictions. It enables continuous improvement of the ML models through
human feedback loops and corrections.

Key functionality includes:
- Storing user feedback on predictions
- Analyzing feedback patterns
- Incorporating feedback into model training
- Tracking feedback metrics over time
"""

__version__ = "0.1.0"  # Pre-release version

# Package-level constants
FEEDBACK_TYPES = ["correction", "confirmation", "rejection", "suggestion"]
FEEDBACK_PRIORITIES = ["low", "medium", "high", "critical"]

# When implementations are added, they will be imported and exported here 