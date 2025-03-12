"""
Robustness module for machine learning models.

This module provides tools for assessing and enhancing the robustness
of machine learning models against adversarial attacks, distribution shifts,
and other perturbations.

Key components include:
- Adversarial attack implementations and defenses
- Certification methods for model robustness
- Gradient masking detection
- Noise resilience assessment tools
- Input perturbation testing utilities

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

from app.models.ml.robustness.certification import (
    RandomizedSmoothingCertifier,
    detect_gradient_masking
)
from app.models.ml.robustness.attacks import (
    AutoAttack,
    BoundaryAttack
)

__all__ = [
    # Certification
    'RandomizedSmoothingCertifier',
    'detect_gradient_masking',
    
    # Attacks
    'AutoAttack',
    'BoundaryAttack'
]
