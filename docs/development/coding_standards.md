# Coding Standards

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document outlines the coding standards, best practices, and conventions to be followed in the WITHIN Ad Score & Account Health Predictor system. Adhering to these standards ensures code quality, maintainability, and consistency across the codebase.

## Table of Contents

- [General Principles](#general-principles)
- [Python Standards](#python-standards)
- [ML-Specific Standards](#ml-specific-standards)
- [API Development Standards](#api-development-standards)
- [Database Standards](#database-standards)
- [Testing Standards](#testing-standards)
- [Documentation Standards](#documentation-standards)
- [Code Review](#code-review)
- [Version Control](#version-control)

## General Principles

### Code Quality

- **Readability**: Write code that is easy to read and understand
- **Simplicity**: Prefer simple solutions over complex ones
- **Modularity**: Design modular code that follows the Single Responsibility Principle
- **Reusability**: Create reusable components rather than duplicating code
- **Maintainability**: Write code that is easy to maintain and extend
- **Performance**: Optimize code when necessary, but prioritize readability for non-critical paths

### Naming Conventions

- Use descriptive and meaningful names for variables, functions, classes, and modules
- Choose names that reveal intent and avoid abbreviations except for widely accepted ones
- Maintain consistency in naming across the codebase

### Code Organization

- Organize code logically within files, modules, and packages
- Group related functionality together
- Separate different layers of the application (data access, business logic, presentation)
- Maintain a clear and consistent project structure

## Python Standards

### Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following specific requirements:

- **Line Length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
- **Whitespace**: Follow PEP 8 whitespace guidelines
- **Comments**: Use inline comments sparingly; prefer docstrings for documentation
- **Quote Styles**: Prefer double quotes for strings; use single quotes for strings containing double quotes

### Type Annotations

Strict type annotations are required for all code:

```python
def calculate_score(features: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate weighted score based on features and weights."""
    return sum(weights.get(feature, 0) * value for feature, value in features.items())
```

Use typing module for complex type annotations:

```python
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# For complex types
FeatureVector = Dict[str, float]
ModelResult = Tuple[float, Dict[str, float]]

def predict(
    features: FeatureVector, 
    thresholds: Optional[Dict[str, float]] = None
) -> ModelResult:
    """Predict result based on features."""
    # Implementation
    pass
```

### Docstrings

Use Google style docstrings for all modules, classes, and functions:

```python
def calculate_metric(
    predictions: List[float], 
    actual: List[float], 
    weights: Optional[List[float]] = None
) -> float:
    """Calculate weighted metric between predictions and actual values.
    
    Args:
        predictions: List of predicted values
        actual: List of actual values
        weights: Optional list of weights for each sample. If None,
            equal weights are assigned.
    
    Returns:
        Weighted metric value
        
    Raises:
        ValueError: If predictions and actual have different lengths
    
    Example:
        >>> calculate_metric([0.1, 0.2, 0.3], [0.0, 0.0, 1.0])
        0.7666
    """
    # Implementation
    pass
```

### Error Handling

- Use specific exception types rather than catching or raising generic exceptions
- Provide informative error messages
- Use context managers (`with` statements) for resource management
- Handle errors at the appropriate level, avoid excessive try/except blocks

```python
def load_model(model_path: str) -> Model:
    """Load model from disk.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model
        
    Raises:
        FileNotFoundError: If model file does not exist
        ModelLoadError: If model cannot be loaded due to compatibility or corruption
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except (pickle.PickleError, AttributeError) as e:
        raise ModelLoadError(f"Failed to load model: {e}")
```

### Code Formatting and Linting

Use the following tools to enforce code style:

- **Black**: Code formatter with a line length of 100
- **isort**: Import sorter (compatible with Black)
- **Flake8**: Linter for style guide enforcement
- **Mypy**: Static type checker
- **Pylint**: Static code analyzer

Configuration for these tools is provided in the project root files:

- `pyproject.toml` for Black and isort
- `.flake8` for Flake8
- `mypy.ini` for Mypy
- `.pylintrc` for Pylint

## ML-Specific Standards

### Model Implementation

- Implement all models as classes that inherit from appropriate base classes
- Separate model definition from training, evaluation, and inference logic
- Encapsulate preprocessing in the model pipeline
- Ensure models are serializable with proper versioning
- Include metadata with all models (training date, version, performance metrics)

Example model class structure:

```python
class AdScoreModel:
    """Ad score prediction model."""
    
    def __init__(
        self, 
        feature_processors: Dict[str, Processor],
        model: Any,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize model.
        
        Args:
            feature_processors: Dictionary mapping feature names to processors
            model: Underlying prediction model
            version: Model version
            metadata: Additional model metadata
        """
        self.feature_processors = feature_processors
        self.model = model
        self.version = version
        self.metadata = metadata or {}
        
    def preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for model input.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Processed features as numpy array
        """
        # Implementation
        pass
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict ad score from features.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Prediction result dictionary with score and additional info
        """
        # Implementation
        pass
    
    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Explain prediction for features.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Explanation dictionary
        """
        # Implementation
        pass
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        # Implementation
        pass
    
    @classmethod
    def load(cls, path: str) -> "AdScoreModel":
        """Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded model
        """
        # Implementation
        pass
```

### Feature Engineering

- Document all features, including their purpose, data type, and expected range
- Implement feature transformations as reusable components
- Include validation for feature inputs
- Use feature stores or caching for expensive computations
- Track feature importance and impact

```python
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    description: str
    data_type: str
    required: bool = True
    default_value: Any = None
    validation_func: Optional[Callable[[Any], bool]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categorical_values: Optional[List[str]] = None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate feature value.
        
        Args:
            value: Feature value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"Feature '{self.name}' is required"
            return True, None
            
        # Validate data type
        if self.data_type == "numeric" and not isinstance(value, (int, float)):
            return False, f"Feature '{self.name}' must be numeric"
        
        # Validate range
        if self.data_type == "numeric" and self.min_value is not None and value < self.min_value:
            return False, f"Feature '{self.name}' must be >= {self.min_value}"
        
        # Validate max value
        if self.data_type == "numeric" and self.max_value is not None and value > self.max_value:
            return False, f"Feature '{self.name}' must be <= {self.max_value}"
        
        # Validate categorical values
        if self.categorical_values and value not in self.categorical_values:
            return False, f"Feature '{self.name}' must be one of {self.categorical_values}"
        
        # Custom validation
        if self.validation_func and not self.validation_func(value):
            return False, f"Feature '{self.name}' failed custom validation"
            
        return True, None
```

### Model Training

- Track all experiments with parameters, metrics, and artifacts
- Use configuration files or objects for model hyperparameters
- Implement reproducible training with fixed random seeds
- Include evaluation metrics and validation procedures
- Maintain separate training, validation, and test sets

```python
def train_model(
    config: Dict[str, Any],
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    experiment_tracking: bool = True
) -> Tuple[AdScoreModel, Dict[str, Any]]:
    """Train ad score model with given configuration.
    
    Args:
        config: Training configuration
        train_data: Training data
        validation_data: Validation data
        experiment_tracking: Whether to track experiment
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Set random seeds for reproducibility
    np.random.seed(config.get("random_seed", 42))
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(config.get("random_seed", 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.get("random_seed", 42))
    
    # Initialize experiment tracking
    if experiment_tracking:
        experiment = init_experiment(config)
    
    # Prepare data
    X_train, y_train = prepare_data(train_data, config["features"], config["target"])
    X_val, y_val = prepare_data(validation_data, config["features"], config["target"])
    
    # Initialize and train model
    model_type = config["model_type"]
    if model_type == "gradient_boosting":
        model = train_gradient_boosting(X_train, y_train, X_val, y_val, config)
    elif model_type == "neural_network":
        model = train_neural_network(X_train, y_train, X_val, y_val, config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Evaluate model
    metrics = evaluate_model(model, X_val, y_val)
    
    # Log metrics
    if experiment_tracking:
        log_metrics(experiment, metrics)
    
    # Create and return model
    ad_score_model = AdScoreModel(
        feature_processors=create_feature_processors(config["features"]),
        model=model,
        version=config["version"],
        metadata={
            "training_date": datetime.now().isoformat(),
            "training_config": config,
            "metrics": metrics,
            "feature_importance": get_feature_importance(model, config["features"])
        }
    )
    
    return ad_score_model, metrics
```

### Model Evaluation

- Use appropriate metrics for each model type
- Implement both offline and online evaluation
- Include fairness and bias assessments
- Benchmark against baseline models
- Use cross-validation where appropriate

```python
def evaluate_model(
    model: Any, 
    X: np.ndarray, 
    y: np.ndarray, 
    groups: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Evaluate model performance.
    
    Args:
        model: Model to evaluate
        X: Feature data
        y: Target data
        groups: Optional grouping data for fairness evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Basic prediction metrics
    y_pred = model.predict(X)
    metrics = {
        "rmse": mean_squared_error(y, y_pred, squared=False),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred)
    }
    
    # Classification metrics if applicable
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)
        if y_pred_proba.shape[1] == 2:  # Binary classification
            metrics.update({
                "auc": roc_auc_score(y, y_pred_proba[:, 1]),
                "log_loss": log_loss(y, y_pred_proba),
                "precision": precision_score(y, (y_pred_proba[:, 1] > 0.5).astype(int)),
                "recall": recall_score(y, (y_pred_proba[:, 1] > 0.5).astype(int)),
                "f1": f1_score(y, (y_pred_proba[:, 1] > 0.5).astype(int)),
            })
    
    # Group fairness metrics if groups are provided
    if groups is not None:
        group_metrics = {}
        for group in np.unique(groups):
            mask = groups == group
            if np.sum(mask) > 0:
                group_metrics[f"group_{group}_rmse"] = mean_squared_error(
                    y[mask], y_pred[mask], squared=False
                )
        metrics["group_metrics"] = group_metrics
        
        # Calculate fairness metrics
        metrics["max_group_delta"] = max([
            abs(v - metrics["rmse"]) for v in group_metrics.values()
        ])
    
    return metrics
```

## API Development Standards

### API Design

- Follow RESTful design principles
- Use versioned endpoints (e.g., `/api/v1/resource`)
- Use nouns for resources, not verbs
- Use appropriate HTTP methods (GET, POST, PUT, DELETE)
- Follow consistent URL patterns
- Implement pagination for collection endpoints
- Use appropriate response status codes

### Request and Response Formats

- Use JSON for request and response bodies
- Use snake_case for JSON field names
- Include appropriate content-type headers
- Implement consistent error responses
- Include request IDs for traceability
- Use envelope structure for responses when appropriate

Example response format:

```json
{
  "data": {
    "id": "12345",
    "name": "Example Resource",
    "created_at": "2023-10-01T12:34:56Z",
    "properties": {
      "property1": "value1",
      "property2": "value2"
    }
  },
  "meta": {
    "request_id": "req_abcdefg",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

Example error format:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested resource was not found",
    "details": {
      "resource_id": "12345",
      "resource_type": "example"
    }
  },
  "meta": {
    "request_id": "req_abcdefg",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

### API Implementation

- Use FastAPI for API development
- Implement input validation using Pydantic models
- Document all endpoints with OpenAPI/Swagger
- Include proper authentication and authorization
- Implement appropriate rate limiting
- Add logging for requests and responses

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional

# Pydantic models for request and response
class AdScorePredictionRequest(BaseModel):
    """Request model for ad score prediction."""
    
    ad_content: str = Field(..., min_length=1, max_length=1000, description="Ad content text")
    platform: str = Field(..., description="Advertising platform")
    
    class Config:
        schema_extra = {
            "example": {
                "ad_content": "Limited time offer: 20% off all products",
                "platform": "facebook"
            }
        }

class AdScorePredictionResponse(BaseModel):
    """Response model for ad score prediction."""
    
    score: float = Field(..., ge=0, le=100, description="Predicted ad score")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    components: dict = Field(..., description="Score components")
    
    class Config:
        schema_extra = {
            "example": {
                "score": 85.4,
                "confidence": 0.92,
                "components": {
                    "engagement": 87.2,
                    "conversion": 82.9,
                    "audience_match": 86.1
                }
            }
        }

# API router
router = APIRouter()

@router.post(
    "/predict", 
    response_model=AdScorePredictionResponse,
    summary="Predict ad score",
    description="Predict the effectiveness score for ad content",
    response_description="The predicted ad score"
)
async def predict_ad_score(
    request: AdScorePredictionRequest,
    version: Optional[str] = Query("latest", description="Model version")
):
    """Predict ad score endpoint."""
    try:
        # Get predictor for specified version
        predictor = get_predictor(version)
        
        # Make prediction
        result = predictor.predict(
            ad_content=request.ad_content,
            platform=request.platform
        )
        
        # Return response
        return {
            "score": result["score"],
            "confidence": result["confidence"],
            "components": result["components"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_exception(e)
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Database Standards

### Database Schema

- Use snake_case for table and column names
- Include appropriate indexes for query performance
- Implement foreign key constraints for data integrity
- Use explicit primary keys for all tables
- Include created_at and updated_at timestamps for all tables
- Document table schemas and relationships

### Data Access

- Use SQLAlchemy for database access
- Implement data access layers separate from business logic
- Use connection pooling for database connections
- Handle database transactions explicitly
- Use asynchronous access where appropriate

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AdMetrics(Base):
    """Table for storing ad metrics."""
    
    __tablename__ = "ad_metrics"
    
    id = Column(Integer, primary_key=True)
    ad_id = Column(String(50), nullable=False, index=True)
    platform = Column(String(50), nullable=False, index=True)
    impressions = Column(Integer, nullable=False, default=0)
    clicks = Column(Integer, nullable=False, default=0)
    conversions = Column(Integer, nullable=False, default=0)
    ctr = Column(Float, nullable=True)
    conversion_rate = Column(Float, nullable=True)
    score = Column(Float, nullable=True)
    date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<AdMetrics(ad_id='{self.ad_id}', date='{self.date}', score={self.score})>"
```

## Testing Standards

### Test Types

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **API Tests**: Test API endpoints and responses
- **Functional Tests**: Test complete application workflows
- **Performance Tests**: Test application performance and scalability
- **ML-Specific Tests**: Test model training, evaluation, and predictions

### Test Implementation

- Use pytest for all tests
- Organize tests to mirror the project structure
- Name test files with `test_` prefix
- Name test functions with `test_` prefix
- Use fixtures for common test setup
- Implement parametrized tests for multiple test cases
- Include proper assertions with informative messages

```python
import pytest
import numpy as np
from app.models.ml.prediction import AdScorePredictor

@pytest.fixture
def sample_features():
    """Fixture for sample features."""
    return {
        "ad_content": "Limited time offer: 20% off all products",
        "platform": "facebook",
        "target_audience": ["shoppers", "deal_seekers"]
    }

@pytest.fixture
def predictor():
    """Fixture for ad score predictor."""
    return AdScorePredictor(version="test")

def test_prediction_range(predictor, sample_features):
    """Test that prediction score is within expected range."""
    result = predictor.predict(sample_features)
    assert "score" in result, "Prediction result should contain score"
    assert 0 <= result["score"] <= 100, f"Score should be between 0 and 100, got {result['score']}"
    assert "confidence" in result, "Prediction result should contain confidence"
    assert 0 <= result["confidence"] <= 1, f"Confidence should be between 0 and 1, got {result['confidence']}"

@pytest.mark.parametrize("platform", ["facebook", "google", "tiktok"])
def test_different_platforms(predictor, sample_features, platform):
    """Test prediction works for different platforms."""
    features = sample_features.copy()
    features["platform"] = platform
    result = predictor.predict(features)
    assert "score" in result, f"Prediction for platform {platform} failed"

def test_empty_ad_content(predictor, sample_features):
    """Test that empty ad content raises appropriate error."""
    features = sample_features.copy()
    features["ad_content"] = ""
    with pytest.raises(ValueError, match="Ad content cannot be empty"):
        predictor.predict(features)
```

### Test Coverage

- Aim for at least 90% test coverage
- Include edge cases and error cases
- Use mocks for external dependencies
- Use property-based testing for complex logic
- Use snapshot testing for stable interfaces

## Documentation Standards

### Code Documentation

- Include docstrings for all modules, classes, and functions
- Document parameters, return values, and exceptions
- Add inline comments for complex logic
- Maintain up-to-date README files
- Document API endpoints with OpenAPI/Swagger
- Include examples for complex functions

### Project Documentation

- Maintain comprehensive documentation in `/docs` directory
- Use Markdown for documentation files
- Include diagrams for complex systems
- Document architecture and design decisions
- Provide getting started guides and tutorials
- Document troubleshooting and common issues

## Code Review

### Review Process

1. Create a pull request with a clear description
2. Ensure all automated checks pass (tests, linting, type checking)
3. Request review from appropriate team members
4. Address all review comments
5. Obtain approval from required reviewers
6. Merge the pull request

### Review Checklist

- Code follows the established coding standards
- Tests are comprehensive and pass
- Documentation is clear and up-to-date
- Error handling is appropriate
- Performance considerations are addressed
- Security implications are considered
- No unnecessary complexity
- Code is maintainable and extensible

## Version Control

### Git Workflow

- Use feature branches for all changes
- Keep branches short-lived and focused
- Use meaningful branch names (e.g., `feature/add-sentiment-analysis`)
- Make small, focused commits
- Write clear and informative commit messages
- Rebase or merge from main branch regularly
- Squash commits before merging to main

### Commit Messages

Follow the conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without functionality changes
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks, dependency updates, etc.

Example:

```
feat(prediction): add sentiment analysis to ad score prediction

Add sentiment analysis component to the ad score prediction to improve accuracy.
The sentiment scores are now included in the score components.

Closes #123
```

### Versioning

Follow semantic versioning (SemVer) for all releases:

- **Major version**: Incompatible API changes
- **Minor version**: Add functionality in a backward-compatible manner
- **Patch version**: Backward-compatible bug fixes

Use git tags for releases:

```bash
# Create a new release tag
git tag -a v1.2.3 -m "Release v1.2.3"

# Push tags to remote
git push --tags
```

## Additional Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Effective Python](https://effectivepython.com/)
- [Clean Code in Python](https://www.packtpub.com/product/clean-code-in-python/9781788835831)
- [Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) 