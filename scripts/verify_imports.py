#!/usr/bin/env python
"""
Script to verify that all the stub modules can be imported correctly.

This script attempts to import all the classes from the ML modules
that were recently added, to confirm they are correctly accessible.
"""
import sys
import importlib
from typing import List, Dict, Any, Optional, Tuple


def print_status(message: str, success: bool) -> None:
    """Print a status message with appropriate formatting."""
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"  # Green or Red
    reset = "\033[0m"
    print(f"{color}{status} {message}{reset}")


def import_and_verify(module_path: str, class_names: List[str]) -> Tuple[bool, Optional[Exception]]:
    """
    Import a module and verify that the specified classes exist.
    
    Args:
        module_path: Dotted path to the module
        class_names: List of class names to verify
        
    Returns:
        Tuple of (success, exception)
    """
    try:
        module = importlib.import_module(module_path)
        for class_name in class_names:
            if not hasattr(module, class_name):
                return False, ImportError(f"Class {class_name} not found in {module_path}")
        return True, None
    except Exception as e:
        return False, e


def main():
    """Main function to verify all imports."""
    print("\nVerifying imports for ML modules...\n")
    
    modules_to_verify = [
        # Robustness module
        ("app.models.ml.robustness.certification", ["RandomizedSmoothingCertifier", "detect_gradient_masking"]),
        ("app.models.ml.robustness.attacks", ["AutoAttack", "BoundaryAttack"]),
        
        # Monitoring module
        ("app.models.ml.monitoring.drift_detector", ["DriftDetector", "DataDriftDetector", "ConceptDriftDetector"]),
        
        # Fairness module
        ("app.models.ml.fairness.evaluator", ["FairnessEvaluator", "AdversarialDebiasing", "CounterfactualFairnessEvaluator"]),
        ("app.models.ml.fairness.model_auditor", ["FairnessAuditor", "BiasDetector"]),
        
        # Validation module
        ("app.models.ml.validation.shadow_deployment", ["ShadowDeployment", "ABTestDeployment", "CanaryDeployment"]),
    ]
    
    # Add the parent directory to sys.path
    sys.path.insert(0, ".")
    
    all_success = True
    results = []
    
    for module_path, class_names in modules_to_verify:
        success, error = import_and_verify(module_path, class_names)
        all_success = all_success and success
        
        if success:
            message = f"Successfully imported {module_path} with classes: {', '.join(class_names)}"
            print_status(message, True)
        else:
            message = f"Failed to import {module_path}: {error}"
            print_status(message, False)
        
        results.append({
            "module": module_path,
            "classes": class_names,
            "success": success,
            "error": str(error) if error else None
        })
    
    # Print summary
    print("\nImport Verification Summary:")
    print("-" * 50)
    success_count = sum(1 for r in results if r["success"])
    print(f"Total modules checked: {len(results)}")
    print(f"Successfully imported: {success_count}")
    print(f"Failed imports: {len(results) - success_count}")
    
    # Return appropriate exit code
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main()) 