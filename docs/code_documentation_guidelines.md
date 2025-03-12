# Code Documentation Guidelines

This document outlines the standards for documenting Python code within the WITHIN ML Prediction System.

## Docstring Standards

We follow the Google docstring format for all Python code. This format is readable, comprehensive, and well-supported by documentation generation tools.

### Module Docstrings

Every Python module should have a module-level docstring that describes its purpose and contents:

```python
"""
Module for [description of module purpose].

This module contains [classes/functions] that [description of functionality].
Key components include [list of key components].

DOCUMENTATION STATUS: [COMPLETE|PARTIAL|INCOMPLETE]
"""
```

### Class Docstrings

Every class should have a docstring describing its purpose, key features, and usage:

```python
class ClassName:
    """[Brief one-line description of the class].
    
    [Extended description of the class's purpose and functionality.]
    
    Key features:
        - [Feature 1]: [Description]
        - [Feature 2]: [Description]
    
    Attributes:
        attr1 (type): [Description of attribute]
        attr2 (type): [Description of attribute]
    
    Example:
        >>> instance = ClassName(param="value")
        >>> result = instance.method()
    """
```

### Method Docstrings

Methods should be documented with descriptions, parameters, return values, and exceptions:

```python
def method_name(self, param1, param2=None):
    """[Brief one-line description of the method].
    
    [Extended description of what the method does.]
    
    Args:
        param1 (type): [Description of parameter]
        param2 (type, optional): [Description of parameter]. Defaults to None.
    
    Returns:
        type: [Description of return value]
    
    Raises:
        ExceptionType: [When this exception is raised]
        
    Example:
        >>> instance.method_name("value")
    """
```

### Function Docstrings

Standalone functions should be documented similarly to methods:

```python
def function_name(param1, param2=None):
    """[Brief one-line description of the function].
    
    [Extended description of what the function does.]
    
    Args:
        param1 (type): [Description of parameter]
        param2 (type, optional): [Description of parameter]. Defaults to None.
    
    Returns:
        type: [Description of return value]
    
    Raises:
        ExceptionType: [When this exception is raised]
        
    Example:
        >>> result = function_name("value")
    """
```

## Type Hints

All code should use Python type hints to specify parameter types, return types, and variable types:

```python
from typing import List, Dict, Optional, Union, Tuple, Any, TypeVar

def function_name(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """Function documentation..."""
    result: Dict[str, Any] = {}
    return result

class ClassName:
    def __init__(self, param: str) -> None:
        self.attribute: List[str] = []
        
    def method_name(self, param: Union[str, int]) -> Tuple[bool, str]:
        """Method documentation..."""
        return True, "Success"
```

## Comments

Use comments sparingly and only when necessary to explain complex logic or decisions:

```python
# This algorithm uses a modified version of breadth-first search to handle cycles
def complex_algorithm():
    # Initialize with special case handling for empty inputs
    if not data:
        return default_value
        
    # Implementation follows...
```

## Constants and Configuration

Document constants and configuration variables to explain their purpose and valid values:

```python
# Maximum number of retry attempts for API calls
MAX_RETRIES = 3

# Threshold for considering a prediction anomalous (0.0 to 1.0)
ANOMALY_THRESHOLD = 0.85

# Available model types with their configurations
MODEL_TYPES = {
    "light": {"layers": 2, "units": 64},
    "standard": {"layers": 4, "units": 128},
    "deep": {"layers": 8, "units": 256}
}
```

## Database Models

Document database models with field descriptions and constraints:

```python
class UserModel(BaseModel):
    """Model representing a user in the system.
    
    This model stores user account information and preferences.
    """
    
    # Primary identifier for the user
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # User's email address (must be unique)
    email = Column(String(255), nullable=False, unique=True)
    
    # User's encrypted password
    password_hash = Column(String(255), nullable=False)
    
    # User's preferred timezone
    timezone = Column(String(50), default="UTC")
    
    # When the account was created
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
```

## Validation Code

Document validation logic to explain the business rules being enforced:

```python
@validates('email')
def validate_email(self, key, email):
    """Validate that the email is properly formatted and unique.
    
    Args:
        key: Field name being validated ('email')
        email: Email value to validate
        
    Returns:
        str: Validated email if valid
        
    Raises:
        ValueError: If email format is invalid
    """
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        raise ValueError("Invalid email format")
    return email
```

## Exception Classes

Document custom exception classes to explain when they're raised:

```python
class InsufficientDataError(Exception):
    """Raised when there is not enough data to perform an operation.
    
    This exception indicates that the requested operation requires
    more data points than were provided.
    
    Attributes:
        message: Explanation of the error
        min_required: Minimum number of data points required
        actual: Actual number of data points provided
    """
    
    def __init__(self, message, min_required, actual):
        self.message = message
        self.min_required = min_required
        self.actual = actual
        super().__init__(self.message)
```

## Documentation-First Approach

For complex features, write documentation before implementing the code:

1. Start by documenting the intended functionality
2. Define the interface (parameters, return values)
3. Write examples of how the code will be used
4. Implement the code according to the documentation
5. Update the documentation if the implementation details change

This approach ensures that code is designed with usability in mind and that documentation remains a priority throughout development. 