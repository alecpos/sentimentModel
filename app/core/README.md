# Core Components

This directory contains the core utilities and foundational components used throughout the WITHIN Ad Score & Account Health Predictor system. These components provide essential functionality for configuration management, data handling, authentication, and other shared services.

## Directory Structure

```
core/
├── __init__.py                     # Core package initialization
├── config/                         # Configuration management
│   ├── __init__.py                 # Config package initialization
│   ├── loader.py                   # Configuration loading utilities
│   ├── validators.py               # Configuration validation
│   └── defaults.py                 # Default configuration values
├── database/                       # Database connectivity
│   ├── __init__.py                 # Database package initialization
│   ├── session.py                  # Database session management
│   ├── models.py                   # SQLAlchemy model base classes
│   └── migrations/                 # Database migration scripts
├── auth/                           # Authentication and authorization
│   ├── __init__.py                 # Auth package initialization
│   ├── jwt.py                      # JWT token handling
│   ├── permissions.py              # Permission management
│   └── users.py                    # User management
├── logging/                        # Logging utilities
│   ├── __init__.py                 # Logging package initialization
│   ├── formatters.py               # Custom log formatters
│   ├── handlers.py                 # Custom log handlers
│   └── context.py                  # Context-aware logging
├── cache/                          # Caching mechanisms
│   ├── __init__.py                 # Cache package initialization
│   ├── memory.py                   # In-memory caching
│   ├── redis.py                    # Redis-based caching
│   └── decorators.py               # Caching decorators
├── utils/                          # General utilities
│   ├── __init__.py                 # Utils package initialization
│   ├── date.py                     # Date and time utilities
│   ├── validation.py               # Data validation utilities
│   ├── serialization.py            # Serialization utilities
│   └── profiling.py                # Performance profiling
├── errors/                         # Error handling
│   ├── __init__.py                 # Errors package initialization
│   ├── exceptions.py               # Custom exception classes
│   ├── handlers.py                 # Exception handlers
│   └── codes.py                    # Error codes and messages
├── metrics/                        # Metrics collection
│   ├── __init__.py                 # Metrics package initialization
│   ├── collectors.py               # Metrics collectors
│   ├── exporters.py                # Metrics exporters
│   └── middleware.py               # Metrics middleware
└── constants.py                    # System-wide constants
```

## Core Components

### Configuration Management

The configuration system handles loading, validation, and access to application settings:

```python
from app.core.config import ConfigLoader, settings

# Load configuration
config = ConfigLoader.load(environment="production")

# Access configuration
database_url = settings.database.url
api_port = settings.api.port

# Validate configuration
validation_result = ConfigLoader.validate(config)
if not validation_result.is_valid:
    print(f"Configuration errors: {validation_result.errors}")
```

### Database Connectivity

The database components provide SQLAlchemy integration and session management:

```python
from app.core.database import get_db, Base
from sqlalchemy import Column, String, Integer

# Define a model
class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)

# Use database session
def get_item(item_id: int):
    with get_db() as db:
        return db.query(Item).filter(Item.id == item_id).first()
```

### Authentication and Authorization

The authentication system handles user authentication and permission management:

```python
from app.core.auth import create_access_token, get_current_user, requires_permission

# Create JWT token
token = create_access_token(
    data={"sub": user.username},
    expires_delta=timedelta(minutes=30)
)

# Permission-protected endpoint
@router.get("/protected")
@requires_permission("read:items")
async def protected_route(current_user = Depends(get_current_user)):
    return {"message": "This is protected", "user": current_user.username}
```

### Logging

The logging system provides structured logging with context tracking:

```python
from app.core.logging import get_logger

# Get logger
logger = get_logger(__name__)

# Log with context
with logger.context(request_id="req123", user_id="user456"):
    logger.info("Processing request", extra={"item_id": item_id})
    try:
        result = process_item(item_id)
        logger.info("Request processed successfully", extra={"result": result})
    except Exception as e:
        logger.error("Error processing request", exc_info=e)
```

### Caching

The caching system provides memory and Redis-based caching:

```python
from app.core.cache import Cache, cached

# Use cache directly
cache = Cache.get_instance()
cache.set("key", "value", ttl=3600)
value = cache.get("key")

# Use caching decorator
@cached(ttl=3600, key_prefix="user_data")
def get_user_data(user_id: str):
    # Expensive operation to fetch user data
    return fetch_user_data_from_database(user_id)
```

### Error Handling

The error handling system provides custom exceptions and error handling:

```python
from app.core.errors import ApplicationError, ErrorCodes, handle_exception

# Custom exceptions
class ItemNotFoundError(ApplicationError):
    """Raised when an item is not found."""
    code = ErrorCodes.ITEM_NOT_FOUND
    message = "Item not found"
    status_code = 404

# Error handling
try:
    item = get_item(item_id)
    if not item:
        raise ItemNotFoundError(f"Item with ID {item_id} not found")
except Exception as e:
    response = handle_exception(e)
    return response
```

### Metrics Collection

The metrics system collects and exports performance and business metrics:

```python
from app.core.metrics import metrics, Timer

# Increment counter
metrics.counter("api_requests_total").inc(1)

# Record histogram
metrics.histogram("response_time_seconds").observe(response_time)

# Use timer context manager
with Timer("process_time", {"endpoint": "/items"}):
    result = process_items()
```

## Utility Functions

### Date and Time Utilities

```python
from app.core.utils.date import parse_date_range, format_timestamp

# Parse date range string
start_date, end_date = parse_date_range("last_30_days")

# Format timestamp
formatted_date = format_timestamp(timestamp, format="%Y-%m-%d %H:%M:%S")
```

### Validation Utilities

```python
from app.core.utils.validation import validate_schema, ValidationError

# Define schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number", "minimum": 0},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "email"]
}

# Validate data
try:
    validate_schema(data, schema)
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Serialization Utilities

```python
from app.core.utils.serialization import to_json, from_json

# Serialize to JSON
json_data = to_json(complex_object)

# Deserialize from JSON
object_data = from_json(json_data, target_class=TargetClass)
```

## Base Classes

### Model Base Class

```python
from app.core.database import Base
from sqlalchemy import Column, Integer, DateTime
from datetime import datetime

class TimestampedModel(Base):
    """Base model with created and updated timestamps."""
    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
```

### Service Base Class

```python
from app.core.database import get_db
from app.core.logging import get_logger

class BaseService:
    """Base class for services."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.db = get_db()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.db.rollback()
        else:
            self.db.commit()
```

## Constants

System-wide constants are defined in `constants.py`:

```python
# Status codes
class Status:
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"

# Ad platforms
class Platform:
    FACEBOOK = "facebook"
    GOOGLE = "google"
    AMAZON = "amazon"
    TIKTOK = "tiktok"

# Score ranges
class ScoreRange:
    MIN = 0
    MAX = 100
    POOR = (0, 40)
    FAIR = (40, 60)
    GOOD = (60, 80)
    EXCELLENT = (80, 100)
```

## Development Guidelines

When enhancing or adding core components:

1. **Maintain Backward Compatibility**: Core components are widely used, so changes should be backward compatible
2. **Comprehensive Testing**: Core components need extensive unit testing
3. **Clear Documentation**: Document public interfaces thoroughly
4. **Error Handling**: Implement robust error handling with clear error messages
5. **Performance Consideration**: Core components are used frequently, so performance is critical
6. **Thread Safety**: Ensure thread safety for components used in concurrent contexts
7. **Configuration Options**: Make behavior configurable where appropriate
8. **Minimal Dependencies**: Keep dependencies between core components minimal

# WITHIN Core Module

This directory contains core functionality for the WITHIN Ad Score & Account Health Predictor system, including data validation, fairness evaluation, and data integration capabilities.

## Modules

### Data Integration

The `data_integration` module provides tools for integrating external datasets, particularly from Kaggle, into the WITHIN system. It includes:

- `KaggleDatasetPipeline`: A robust pipeline for downloading, validating, and processing Kaggle datasets with fairness evaluation.
- `DatasetConfig`: Configuration class for dataset specifications.
- `ProcessedDataset`: A processed dataset ready for model training.

See [data_integration/README.md](data_integration/README.md) for detailed documentation.

### Validation

The `validation.py` module provides data validation capabilities:

- `DataValidationSchema`: Schema class for validating datasets.
- `validate_dataset`: Function to validate a dataset against a schema.

### Fairness

The `fairness.py` module provides fairness evaluation and mitigation:

- `FairnessEvaluator`: Evaluates fairness metrics across protected attributes.
- `FairnessResults`: Encapsulates fairness evaluation results.
- `FairnessMitigation`: Base class for fairness mitigation techniques.
- `Reweighing`: Mitigation technique that reweighs samples to ensure fairness.
- `FairDataTransformer`: Mitigation technique that transforms features to remove bias.

## Usage

### Integrating Kaggle Datasets

```python
from app.core.data_integration import KaggleDatasetPipeline

# Initialize the pipeline
pipeline = KaggleDatasetPipeline(
    data_dir="data/kaggle",
    cache_dir="data/cache",
    validate_fairness=True
)

# Get predefined dataset configurations
configs = pipeline.get_dataset_configs()

# Process a specific dataset
processed_dataset = pipeline.process_dataset(configs["customer_conversion"])

# Access the processed data
X_train = processed_dataset.X_train
y_train = processed_dataset.y_train
```

### Evaluating Fairness

```python
from app.core.fairness import FairnessEvaluator

# Initialize fairness evaluator
evaluator = FairnessEvaluator(output_dir="fairness_results")

# Evaluate fairness for a protected attribute
results = evaluator.evaluate(
    df=df,
    protected_attribute="gender",
    target_column="converted",
    prediction_column="prediction"
)

# Access fairness metrics
overall_metrics = results.overall_metrics
group_metrics = results.group_metrics
```

### Mitigating Bias

```python
from app.core.fairness import Reweighing, FairDataTransformer

# Mitigate bias using reweighing
reweigher = Reweighing(protected_attribute="gender")
reweigher.fit(X_train, y_train)
sample_weights = reweigher.get_sample_weights(X_train, y_train)

# Train a model with fairness-aware sample weights
model.fit(X_train, y_train, sample_weight=sample_weights)

# Or transform the features to remove bias
transformer = FairDataTransformer(protected_attribute="gender")
X_train_fair, y_train = transformer.fit_transform(X_train, y_train)
```

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- kaggle
- matplotlib

## Development

When enhancing this module, please ensure:

1. All new features include comprehensive type annotations
2. Documentation is updated to reflect changes
3. New components include appropriate tests
4. Fairness considerations are addressed in any data processing functions 