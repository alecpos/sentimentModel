"""
Fairness module for machine learning models.

This module provides tools for assessing and mitigating bias and
unfairness in machine learning models and datasets.

Key components include:
- Fairness evaluation metrics and assessment tools
- Bias detection and quantification
- Mitigation strategies for biased models
- Counterfactual generation and analysis
- Model auditing for fairness compliance

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

from app.models.ml.fairness.evaluator import (
    FairnessEvaluator,
    CounterfactualFairnessEvaluator as EvaluatorCFE
)
from app.models.ml.fairness.model_auditor import (
    FairnessAuditor,
    BiasDetector
)
from app.models.ml.fairness.mitigation import (
    AdversarialDebiasing,
    FairnessConstraint,
    ReweighingMitigation
)
from app.models.ml.fairness.counterfactual import (
    CounterfactualFairnessEvaluator,
    CounterfactualGenerator,
    CounterfactualAuditor
)

__all__ = [
    # Fairness evaluation
    'FairnessEvaluator',
    'EvaluatorCFE',
    
    # Auditing
    'FairnessAuditor',
    'BiasDetector',
    
    # Mitigation
    'AdversarialDebiasing',
    'FairnessConstraint',
    'ReweighingMitigation',
    
    # Counterfactual analysis
    'CounterfactualFairnessEvaluator',
    'CounterfactualGenerator',
    'CounterfactualAuditor'
]
