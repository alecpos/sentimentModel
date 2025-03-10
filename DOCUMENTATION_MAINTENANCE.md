# Documentation Maintenance Guide

This guide provides instructions for maintaining the documentation for the WITHIN ML Prediction System.

## Documentation Structure

The project documentation is organized as follows:

- **Root README.md**: Project overview, installation, and high-level usage
- **Directory-level READMEs**: Documentation for specific components
- **Docstrings**: In-code documentation for classes and functions
- **API documentation**: Documentation for public API endpoints
- **Tutorials and examples**: In the code examples and notebooks

## Documentation Standards

### README Files

- Use Markdown format for all README files
- Include a clear title and brief description
- Structure with headings and subheadings
- Use code blocks with language specifiers
- Include usage examples where appropriate
- Link to related documentation

### Docstrings

- Use Google-style docstring format
- Document all parameters, return values, and exceptions
- Include type information
- Provide examples for complex functions
- Document edge cases and limitations

Example:

```python
def process_features(data: Dict[str, Any], normalize: bool = True) -> np.ndarray:
    """Process input features for model prediction.
    
    Extracts relevant features from input data and applies preprocessing steps
    including normalization if specified.
    
    Args:
        data: Dictionary containing input features. Must include 'text_features',
            'numeric_features', and 'categorical_features'.
        normalize: Whether to normalize numerical features. Defaults to True.
        
    Returns:
        Processed feature array ready for model input.
        
    Raises:
        ValueError: If required features are missing or of incorrect type.
        
    Example:
        >>> data = {
        ...     'text_features': ['Sample text'],
        ...     'numeric_features': [0.1, 0.5, 0.3],
        ...     'categorical_features': ['category1']
        ... }
        >>> processed = process_features(data)
    """
    # Implementation...
```

## Documentation Update Process

### When to Update Documentation

Documentation should be updated in the following situations:

1. When adding new features or functionality
2. When modifying existing behavior
3. When fixing bugs that affect user-facing behavior
4. When changing API signatures or return values
5. When improving or optimizing implementation details users should be aware of
6. When changing dependencies or requirements

### Linking Documentation Updates to Code Changes

Documentation updates should be part of the same commit or pull request as the code changes they describe. This ensures that documentation stays in sync with the code.

## Generating Documentation

### API Documentation

The API documentation can be generated using auto-generation tools compatible with FastAPI. To generate the API documentation:

```bash
# Install documentation tools if not already installed
pip install mkdocs mkdocs-material

# Generate documentation
mkdocs build
```

The generated documentation will be available in the `site` directory.

## Documentation Templates

### New Module Template

When creating a new module, use this template for the README.md:

```markdown
# Module Name

Brief description of the module's purpose.

## Features

- Feature 1: Brief description
- Feature 2: Brief description
- ...

## Usage

```python
# Example code showing basic usage
```

## API Reference

### Class/Function Name

Description of class/function.

**Parameters:**
- param1: Description
- param2: Description

**Returns:**
- Description of return value

**Raises:**
- ExceptionType: When and why it's raised

## Configuration

Description of configuration options.

## Dependencies

List of dependencies and requirements.
```

### New ML Model Template

When adding a new ML model, use this template for documentation:

```markdown
# Model Name

Brief description of the model and its purpose.

## Features

- Feature 1: Brief description
- Feature 2: Brief description
- ...

## Training Data Requirements

Description of required training data format and characteristics.

## Model Architecture

Description of the model architecture.

## Performance Characteristics

- Accuracy: X%
- Inference time: X ms
- Memory usage: X MB

## Usage

```python
# Example code showing how to use the model
```

## Limitations

Description of known limitations or constraints.
```

## Documentation Best Practices

1. **Keep it up to date**: Documentation should always reflect the current state of the code
2. **Be clear and concise**: Use simple language and avoid jargon
3. **Use examples**: Provide concrete examples for complex concepts
4. **Highlight edge cases**: Document known edge cases and limitations
5. **Link related content**: Cross-reference related documentation
6. **Use visuals**: Include diagrams, charts, or screenshots where helpful
7. **Focus on the user**: Write documentation from the user's perspective
8. **Version documentation**: Update version numbers when significant changes occur

## Documentation Review Process

Before submitting documentation changes:

1. Check for technical accuracy
2. Verify all code examples work as expected
3. Ensure consistent terminology throughout the documentation
4. Spell-check all content
5. Verify links to other documentation or external resources
6. Have at least one peer review the changes

## Automated Documentation Checks

Documentation quality is verified using automated checks:

1. Spell checking using `codespell`
2. Link checking using `linkchecker`
3. Markdown linting using `markdownlint`
4. Code example validation using `doctest`

To run these checks:

```bash
# Install tools if not already installed
pip install codespell linkchecker pytest

# Run checks
codespell *.md app/**/*.md
linkchecker -r 2 site/
pytest --doctest-modules app/
``` 