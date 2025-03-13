# Docstring Verification Guidelines

## Purpose

This document provides guidelines for verifying docstrings in the WITHIN codebase, particularly when using automated generation tools. These guidelines help ensure that:

1. High-quality existing docstrings are preserved
2. Generated docstrings accurately reflect the function/class behavior
3. Documentation meets the project's quality and consistency standards

## Verification Process

### 1. Pre-Generation Assessment

Before using docstring generation tools, perform these checks:

- [ ] Identify files with high-quality existing docstrings (e.g., `validation.py`)
- [ ] Create a "protected docstrings" list for functions/classes with exemplary documentation
- [ ] Run the docstring generator with `--apply=false` first to preview changes

### 2. Post-Generation Verification

After generating docstrings, verify each generated docstring against these criteria:

#### Quality Checklist

- [ ] **Accuracy**: Does the docstring correctly describe what the function/class does?
- [ ] **Completeness**: Are all parameters, return values, and exceptions documented?
- [ ] **Clarity**: Is the language clear and precise?
- [ ] **Consistency**: Does it follow the project's docstring style guide (Google format)?
- [ ] **Specificity**: Does it provide specific details rather than generic descriptions?
- [ ] **Context**: Does it explain the role of the function in the larger system?

#### Technical Verification

- [ ] Parameter names match function signature
- [ ] Parameter types are correctly identified
- [ ] Return types are accurate
- [ ] Examples (if included) are syntactically correct
- [ ] Any constraints or edge cases are documented

### 3. Preserving Existing Docstrings

Some existing docstrings in the codebase are high-quality and should be preserved. Use these guidelines to identify them:

#### Indicators of High-Quality Existing Docstrings

- Detailed descriptions beyond just the function name
- Complete parameter documentation with types and descriptions
- Meaningful return value documentation
- Examples of usage
- Notes on edge cases or performance considerations
- Domain-specific explanations

**Example of a high-quality docstring (from `validation.py`):**

```python
def validate_dataset(df: pd.DataFrame, schema: DataValidationSchema) -> Dict[str, Any]:
    """Validate a dataframe against a schema.
    
    Args:
        df: Dataframe to validate
        schema: Validation schema
        
    Returns:
        Dictionary with validation results
    """
```

### 4. Manual Improvements

For generated docstrings that don't meet quality standards:

- [ ] Add domain-specific details
- [ ] Improve parameter descriptions with more context
- [ ] Add examples for complex functions
- [ ] Document constraints, assumptions and edge cases
- [ ] Add cross-references to related functions/modules

## Common Docstring Issues to Fix

### 1. Generic Descriptions

**Bad:**
```python
"""Ad predictor n n that inherits from BaseMLModel."""
```

**Better:**
```python
"""Neural network-based ad score predictor model.

This class implements a deep learning approach to ad score prediction,
supporting multimodal inputs, differential privacy, and quantum noise
for improved model robustness."""
```

### 2. Missing Parameter Context

**Bad:**
```python
"""
Args:
    input_dim: Description of input_dim.
"""
```

**Better:**
```python
"""
Args:
    input_dim: Dimension of the input feature vector. For text-only models, 
        typically 256; for multimodal models, varies based on enabled features.
"""
```

### 3. Missing Return Value Detail

**Bad:**
```python
"""
Returns:
    Dict[str, Any]: Description of return value.
"""
```

**Better:**
```python
"""
Returns:
    Dict[str, Any]: Dictionary containing:
        - 'scores': Array of predicted ad scores (0-100)
        - 'confidence': Prediction confidence levels
        - 'calibrated': Whether scores have been calibrated
"""
```

## Implementation Examples

### Example 1: HierarchicalCalibrator

Generated docstring:
```python
"""Hierarchical calibrator that inherits from nn.Module.

Detailed description of the class's purpose and behavior.

Attributes:
    Placeholder for class attributes.
"""
```

Improved docstring:
```python
"""Hierarchical calibrator for multi-level prediction calibration.

This class implements a hierarchical approach to prediction calibration,
using splines at different granularity levels to adjust raw model outputs.
It employs monotonicity constraints to ensure calibration preserves
prediction rankings.

Attributes:
    num_spline_points: Number of control points in each calibration spline
    splines: Dictionary of calibration splines at different hierarchy levels
    monotonicity_weight: Weight of the monotonicity constraint in training
"""
```

### Example 2: AdPredictorNN

Generated docstring:
```python
"""Ad predictor n n that inherits from BaseMLModel.

Detailed description of the class's purpose and behavior.

Attributes:
    Placeholder for class attributes.
"""
```

Improved docstring:
```python
"""Neural network model for ad performance prediction.

This model predicts ad effectiveness scores using a deep neural network
architecture with optional quantum noise layers for improved generalization.
The model includes fairness-aware training capabilities and supports
differential privacy for sensitive data.

Attributes:
    input_dim: Dimension of input features
    hidden_dims: List of hidden layer dimensions
    enable_quantum_noise: Whether quantum noise layer is enabled
    layers: Sequential neural network layers
    dropout_rate: Adaptive dropout rate based on training dynamics
"""
```

## Verification Workflow

1. Run the docstring generator in preview mode:
   ```bash
   ./scripts/mac_docstring_generator.py app/models/ml/prediction/ad_score_predictor.py
   ```

2. Review the generated docstrings against this verification guide

3. Apply only to files where the quality is acceptable:
   ```bash
   ./scripts/mac_docstring_generator.py app/models/ml/prediction/ad_score_predictor.py --apply
   ```

4. Manually improve key docstrings for critical components

5. Maintain a list of "protected files" where docstrings should not be auto-generated

## Protected Files List

The following files contain high-quality docstrings that should not be overwritten:

- `app/core/validation.py`
- Add others here...

## Regular Maintenance

Schedule regular verification and improvement of docstrings:

1. Before major releases
2. When new modules or classes are added
3. During refactoring efforts

Document verification date and responsible developer to maintain accountability. 