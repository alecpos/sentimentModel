# Error Handling Patterns

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document outlines the standardized error handling patterns implemented throughout the WITHIN ML platform. Proper error handling is critical in ML systems to ensure robustness, provide meaningful feedback, and maintain operational reliability. This document details the error handling architecture, exception hierarchy, logging practices, and recovery mechanisms used across the codebase.

## Table of Contents

1. [Error Handling Philosophy](#error-handling-philosophy)
2. [Exception Hierarchy](#exception-hierarchy)
3. [Error Boundaries](#error-boundaries)
4. [Fallback Mechanisms](#fallback-mechanisms)
5. [Logging and Monitoring](#logging-and-monitoring)
6. [Graceful Degradation](#graceful-degradation)
7. [Error Recovery](#error-recovery)
8. [Implementation Examples](#implementation-examples)

## Error Handling Philosophy

The WITHIN ML platform follows these core principles for error handling:

1. **Explicit is Better than Implicit**: Errors are explicitly caught and handled rather than relying on default behaviors
2. **Early Detection**: Validate inputs and preconditions early to fail fast
3. **Specific Exceptions**: Use specific exception types that clearly communicate the error
4. **Informative Messages**: Error messages provide actionable information for debugging
5. **Clean Recovery**: Ensure the system can recover to a clean state after errors
6. **Consistent Patterns**: Apply consistent error handling patterns throughout the codebase

## Exception Hierarchy

The platform implements a custom exception hierarchy to categorize and handle different error types:

```
MLBaseException
├── DataException
│   ├── DataValidationError
│   ├── DataCorruptionError
│   ├── SchemaError
│   └── DataSourceError
├── ModelException
│   ├── ModelNotFoundError
│   ├── ModelLoadError
│   ├── ModelSerializationError
│   ├── InvalidModelConfigError
│   └── ModelRuntimeError
├── PipelineException
│   ├── PipelineConfigError
│   ├── PipelineExecutionError
│   ├── StepFailedError
│   └── ResourceExhaustedError
├── APIException
│   ├── AuthenticationError
│   ├── AuthorizationError
│   ├── InvalidRequestError
│   ├── RateLimitError
│   └── ServiceUnavailableError
└── InfrastructureException
    ├── StorageError
    ├── NetworkError
    ├── ComputeResourceError
    └── DependencyError
```

### Base Exception

```python
class MLBaseException(Exception):
    """Base exception for all ML platform exceptions"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        """Initialize the base exception
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details for debugging
        """
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        """Convert exception to dictionary representation
        
        Returns:
            Dictionary representation of exception
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "type": self.__class__.__name__
        }
    
    def log(self, logger: logging.Logger):
        """Log the exception
        
        Args:
            logger: Logger to use
        """
        log_data = self.to_dict()
        logger.error(f"{self.__class__.__name__}: {self.message}", extra={"error_data": log_data})
```

### Data Exceptions

```python
class DataException(MLBaseException):
    """Base class for data-related exceptions"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        error_code = error_code or "DATA_ERROR"
        super().__init__(message, error_code, details)


class DataValidationError(DataException):
    """Exception raised when data validation fails"""
    
    def __init__(self, message: str, validation_errors: List[Dict] = None, details: Dict = None):
        error_code = "DATA_VALIDATION_ERROR"
        details = details or {}
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, error_code, details)


class SchemaError(DataException):
    """Exception raised when data doesn't match expected schema"""
    
    def __init__(self, message: str, schema_name: str = None, field_errors: List[Dict] = None, details: Dict = None):
        error_code = "SCHEMA_ERROR"
        details = details or {}
        if schema_name:
            details["schema_name"] = schema_name
        if field_errors:
            details["field_errors"] = field_errors
        super().__init__(message, error_code, details)
```

### Model Exceptions

```python
class ModelException(MLBaseException):
    """Base class for model-related exceptions"""
    
    def __init__(self, message: str, model_id: str = None, model_version: str = None, error_code: str = None, details: Dict = None):
        error_code = error_code or "MODEL_ERROR"
        details = details or {}
        if model_id:
            details["model_id"] = model_id
        if model_version:
            details["model_version"] = model_version
        super().__init__(message, error_code, details)


class ModelNotFoundError(ModelException):
    """Exception raised when a model cannot be found"""
    
    def __init__(self, message: str, model_id: str = None, model_version: str = None, details: Dict = None):
        error_code = "MODEL_NOT_FOUND"
        super().__init__(message, model_id, model_version, error_code, details)


class ModelRuntimeError(ModelException):
    """Exception raised when a model fails during execution"""
    
    def __init__(self, message: str, model_id: str = None, model_version: str = None, input_shape: Tuple = None, details: Dict = None):
        error_code = "MODEL_RUNTIME_ERROR"
        details = details or {}
        if input_shape:
            details["input_shape"] = input_shape
        super().__init__(message, model_id, model_version, error_code, details)
```

## Error Boundaries

The platform implements error boundaries to isolate failures and prevent cascading errors:

### Function-Level Error Handling

```python
@contextlib.contextmanager
def error_boundary(
    error_type: Type[Exception] = Exception,
    fallback_func: Callable = None,
    logger: logging.Logger = None
):
    """Context manager for handling errors within a boundary
    
    Args:
        error_type: Type of exception to catch
        fallback_func: Function to call with the error if caught
        logger: Logger to use for logging the error
    
    Yields:
        None
    """
    try:
        yield
    except error_type as e:
        if logger:
            if isinstance(e, MLBaseException):
                e.log(logger)
            else:
                logger.exception(f"Error in boundary: {str(e)}")
        
        if fallback_func:
            fallback_func(e)
        else:
            raise
```

### Service-Level Error Boundaries

```python
class ModelServiceErrorBoundary:
    """Error boundary for model service calls"""
    
    def __init__(self, logger: logging.Logger, metrics_client: Any = None):
        """Initialize the error boundary
        
        Args:
            logger: Logger for error logging
            metrics_client: Client for recording error metrics
        """
        self.logger = logger
        self.metrics_client = metrics_client
    
    def execute(self, func: Callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """Execute a function within an error boundary
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple containing the result (or None) and the exception (or None)
        """
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            if isinstance(e, MLBaseException):
                e.log(self.logger)
            else:
                self.logger.exception(f"Error in model service: {str(e)}")
            
            if self.metrics_client:
                self.metrics_client.increment("model.service.error", tags={
                    "error_type": e.__class__.__name__,
                    "function": func.__name__
                })
            
            return None, e
```

## Fallback Mechanisms

The platform implements various fallback mechanisms to handle errors gracefully:

### Default Values

```python
def get_feature_with_fallback(feature_store, feature_name: str, entity_id: str, default_value=None):
    """Get feature with fallback to default value
    
    Args:
        feature_store: Feature store to query
        feature_name: Name of feature to retrieve
        entity_id: Entity ID to retrieve feature for
        default_value: Default value to return if feature retrieval fails
        
    Returns:
        Feature value or default value
    """
    try:
        return feature_store.get_feature(feature_name, entity_id)
    except (DataException, StorageError) as e:
        logger.warning(f"Failed to retrieve feature {feature_name} for entity {entity_id}: {str(e)}")
        return default_value
```

### Fallback Models

```python
class FallbackModelChain:
    """Chain of models with fallback capability"""
    
    def __init__(self, primary_model, fallback_models: List, logger: logging.Logger):
        """Initialize the fallback chain
        
        Args:
            primary_model: Primary model to use
            fallback_models: List of fallback models in priority order
            logger: Logger for error logging
        """
        self.primary_model = primary_model
        self.fallback_models = fallback_models
        self.logger = logger
    
    def predict(self, input_data, timeout: float = 1.0) -> Dict:
        """Make prediction with fallback
        
        Args:
            input_data: Input data for prediction
            timeout: Timeout for primary model in seconds
            
        Returns:
            Prediction result
            
        Raises:
            ModelChainExhaustedException: If all models in the chain fail
        """
        # Try primary model with timeout
        try:
            return self._predict_with_timeout(self.primary_model, input_data, timeout)
        except Exception as e:
            self.logger.warning(f"Primary model failed: {str(e)}")
            
            # Try fallback models
            for i, model in enumerate(self.fallback_models):
                try:
                    result = model.predict(input_data)
                    result["used_fallback"] = True
                    result["fallback_level"] = i + 1
                    return result
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback model {i+1} failed: {str(fallback_error)}")
            
            # All models failed
            raise ModelChainExhaustedException("All models in fallback chain failed")
    
    def _predict_with_timeout(self, model, input_data, timeout: float):
        """Make prediction with timeout
        
        Args:
            model: Model to use
            input_data: Input data for prediction
            timeout: Timeout in seconds
            
        Returns:
            Prediction result
            
        Raises:
            TimeoutError: If prediction times out
        """
        # Implementation of prediction with timeout
        # ...
```

## Logging and Monitoring

The platform implements comprehensive logging and monitoring for errors:

### Structured Error Logging

```python
class StructuredErrorLogger:
    """Logger for structured error logging"""
    
    def __init__(self, app_name: str, log_level: int = logging.INFO):
        """Initialize structured logger
        
        Args:
            app_name: Name of application
            log_level: Logging level
        """
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(log_level)
        
        # Configure structured JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_json_formatter())
        self.logger.addHandler(handler)
    
    def log_error(self, error: Exception, context: Dict = None):
        """Log error with structured context
        
        Args:
            error: Exception to log
            context: Additional context information
        """
        context = context or {}
        
        if isinstance(error, MLBaseException):
            error_data = error.to_dict()
        else:
            error_data = {
                "error_code": "UNHANDLED_ERROR",
                "message": str(error),
                "type": error.__class__.__name__
            }
        
        # Add stack trace for non-MLBaseExceptions
        if not isinstance(error, MLBaseException):
            import traceback
            error_data["stack_trace"] = traceback.format_exc()
        
        # Merge context and error data
        log_data = {**context, "error": error_data}
        
        # Log with appropriate level
        if isinstance(error, (DataValidationError, InvalidRequestError)):
            self.logger.warning(error_data["message"], extra={"data": log_data})
        else:
            self.logger.error(error_data["message"], extra={"data": log_data})
    
    def _get_json_formatter(self):
        """Get JSON formatter for structured logging
        
        Returns:
            JSON formatter instance
        """
        # Implementation of JSON formatter
        # ...
```

### Error Metrics

```python
class ErrorMetricsRecorder:
    """Records error metrics for monitoring"""
    
    def __init__(self, metrics_client):
        """Initialize metrics recorder
        
        Args:
            metrics_client: Client for recording metrics
        """
        self.metrics_client = metrics_client
    
    def record_error(self, error: Exception, context: Dict = None):
        """Record error metrics
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        context = context or {}
        
        # Record error count
        tags = {
            "error_type": error.__class__.__name__,
            "component": context.get("component", "unknown")
        }
        
        # Add specific tags for different error types
        if isinstance(error, ModelException):
            tags["model_id"] = getattr(error, "details", {}).get("model_id", "unknown")
            
        if isinstance(error, DataException):
            tags["data_source"] = context.get("data_source", "unknown")
        
        # Record count metric
        self.metrics_client.increment("ml.errors", tags=tags)
        
        # Record latency if available
        if "duration_ms" in context:
            self.metrics_client.histogram(
                "ml.error.latency", 
                context["duration_ms"],
                tags=tags
            )
```

## Graceful Degradation

The platform implements graceful degradation strategies to maintain service quality during failures:

### Feature Degradation

```python
class FeatureDegradationManager:
    """Manages graceful degradation of features"""
    
    def __init__(self, config: Dict[str, Dict], logger: logging.Logger):
        """Initialize degradation manager
        
        Args:
            config: Configuration mapping features to degradation strategies
            logger: Logger for degradation events
        """
        self.config = config
        self.logger = logger
        self.degraded_features = set()
    
    def execute_with_degradation(self, 
                              feature_name: str, 
                              func: Callable, 
                              *args, 
                              **kwargs) -> Any:
        """Execute function with degradation capability
        
        Args:
            feature_name: Name of feature being executed
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result or degraded result
        """
        # Skip execution if already degraded
        if feature_name in self.degraded_features:
            return self._get_degraded_response(feature_name)
        
        # Get degradation config
        feature_config = self.config.get(feature_name, {})
        max_failures = feature_config.get("max_failures", 3)
        failure_window = feature_config.get("failure_window_seconds", 60)
        
        # Check recent failures
        recent_failures = self._get_recent_failures(feature_name, failure_window)
        if len(recent_failures) >= max_failures:
            self._degrade_feature(feature_name)
            return self._get_degraded_response(feature_name)
        
        # Try to execute
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Record failure
            self._record_failure(feature_name, e)
            
            # Check if should degrade after this failure
            recent_failures = self._get_recent_failures(feature_name, failure_window)
            if len(recent_failures) >= max_failures:
                self._degrade_feature(feature_name)
            
            # Re-raise if no degradation strategy
            if "degraded_response" not in feature_config:
                raise
            
            return self._get_degraded_response(feature_name)
    
    def _get_degraded_response(self, feature_name: str) -> Any:
        """Get degraded response for a feature
        
        Args:
            feature_name: Name of feature
            
        Returns:
            Degraded response
        """
        feature_config = self.config.get(feature_name, {})
        return feature_config.get("degraded_response")
    
    def _record_failure(self, feature_name: str, error: Exception):
        """Record feature failure
        
        Args:
            feature_name: Name of feature
            error: Exception that occurred
        """
        # Implementation of failure recording
        # ...
    
    def _get_recent_failures(self, feature_name: str, window_seconds: int) -> List[Dict]:
        """Get recent failures for a feature
        
        Args:
            feature_name: Name of feature
            window_seconds: Time window in seconds
            
        Returns:
            List of recent failures
        """
        # Implementation of failure retrieval
        # ...
    
    def _degrade_feature(self, feature_name: str):
        """Degrade a feature
        
        Args:
            feature_name: Name of feature to degrade
        """
        self.degraded_features.add(feature_name)
        self.logger.warning(f"Degrading feature: {feature_name}")
```

### Circuit Breaker

```python
class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""
    
    def __init__(self, 
                name: str, 
                failure_threshold: int = 5,
                reset_timeout: int = 60,
                half_open_timeout: int = 30,
                excluded_exceptions: List[Type[Exception]] = None,
                logger: logging.Logger = None):
        """Initialize circuit breaker
        
        Args:
            name: Name of circuit breaker
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting reset from open to half-open
            half_open_timeout: Seconds before reset from half-open to closed if successful
            excluded_exceptions: Exception types that don't count as failures
            logger: Logger for circuit events
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.excluded_exceptions = excluded_exceptions or []
        self.logger = logger or logging.getLogger(__name__)
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_state_change_time = time.time()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails and circuit is not open
        """
        current_time = time.time()
        
        # Handle OPEN state
        if self.state == "OPEN":
            if current_time - self.last_state_change_time >= self.reset_timeout:
                self._transition_to("HALF_OPEN")
            else:
                raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        # Execute with protection
        try:
            result = func(*args, **kwargs)
            
            # On success in HALF_OPEN, transition to CLOSED
            if self.state == "HALF_OPEN":
                if current_time - self.last_state_change_time >= self.half_open_timeout:
                    self._transition_to("CLOSED")
            
            # Reset failure count on success in CLOSED
            if self.state == "CLOSED" and self.failure_count > 0:
                self.failure_count = 0
                self.logger.info(f"Circuit {self.name}: Reset failure count")
            
            return result
            
        except Exception as e:
            # Don't count excluded exceptions as failures
            if any(isinstance(e, exc_type) for exc_type in self.excluded_exceptions):
                raise
            
            # Record failure
            self.failure_count += 1
            self.last_failure_time = current_time
            
            # Check for state transition
            if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
                self._transition_to("OPEN")
            
            # If failed in HALF_OPEN, go back to OPEN
            if self.state == "HALF_OPEN":
                self._transition_to("OPEN")
            
            # Re-raise the exception
            raise
    
    def _transition_to(self, new_state: str):
        """Transition circuit breaker to a new state
        
        Args:
            new_state: New state to transition to
        """
        old_state = self.state
        self.state = new_state
        self.last_state_change_time = time.time()
        
        if old_state != new_state:
            self.logger.warning(f"Circuit {self.name}: State changed from {old_state} to {new_state}")
        
        if new_state == "CLOSED":
            self.failure_count = 0
```

## Error Recovery

The platform implements various recovery mechanisms to handle errors:

### Retry Mechanism

```python
def retry(max_attempts: int = 3, 
          retry_delay: float = 1.0, 
          backoff_factor: float = 2.0,
          jitter: float = 0.1,
          retryable_exceptions: List[Type[Exception]] = None):
    """Decorator for retrying functions
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for increasing delay between retries
        jitter: Random jitter factor to add to delay (0-1)
        retryable_exceptions: List of exceptions that trigger retry
        
    Returns:
        Decorated function
    """
    retryable_exceptions = retryable_exceptions or [Exception]
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Only retry for specified exception types
                    if not any(isinstance(e, exc) for exc in retryable_exceptions):
                        raise
                    
                    last_exception = e
                    attempt += 1
                    
                    if attempt >= max_attempts:
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = retry_delay * (backoff_factor ** (attempt - 1))
                    if jitter > 0:
                        delay += random.uniform(0, jitter * delay)
                    
                    logger.warning(f"Retry {attempt}/{max_attempts} after error: {str(e)}, "
                                  f"retrying in {delay:.2f}s")
                    time.sleep(delay)
            
            # If we got here, all retries failed
            logger.error(f"All {max_attempts} retry attempts failed")
            if last_exception:
                raise last_exception
        
        return wrapper
    
    return decorator
```

### Checkpoint Recovery

```python
class CheckpointRecovery:
    """Recovery from checkpoints for long-running operations"""
    
    def __init__(self, 
                checkpoint_dir: str,
                operation_id: str,
                logger: logging.Logger):
        """Initialize checkpoint recovery
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            operation_id: ID of operation for checkpoint naming
            logger: Logger for recovery events
        """
        self.checkpoint_dir = checkpoint_dir
        self.operation_id = operation_id
        self.logger = logger
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, state: Dict):
        """Save checkpoint state
        
        Args:
            state: State to checkpoint
        """
        checkpoint_path = self._get_checkpoint_path()
        
        # Add metadata
        state_with_meta = {
            "timestamp": time.time(),
            "operation_id": self.operation_id,
            "version": "1.0",
            "state": state
        }
        
        # Save to temp file first, then rename for atomicity
        temp_path = f"{checkpoint_path}.tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(state_with_meta, f)
            
            # Rename for atomic update
            os.rename(temp_path, checkpoint_path)
            self.logger.debug(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint state if exists
        
        Returns:
            State dictionary or None if no checkpoint
        """
        checkpoint_path = self._get_checkpoint_path()
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint_data.get("state")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def clear_checkpoint(self):
        """Clear the checkpoint file"""
        checkpoint_path = self._get_checkpoint_path()
        
        if os.path.exists(checkpoint_path):
            try:
                os.unlink(checkpoint_path)
                self.logger.debug(f"Cleared checkpoint: {checkpoint_path}")
            except Exception as e:
                self.logger.error(f"Failed to clear checkpoint: {str(e)}")
    
    def _get_checkpoint_path(self) -> str:
        """Get path to checkpoint file
        
        Returns:
            Checkpoint file path
        """
        return os.path.join(self.checkpoint_dir, f"{self.operation_id}.checkpoint.json")
```

## Implementation Examples

Here are concrete examples of error handling patterns in action within the platform:

### Data Validation with Error Handling

```python
def preprocess_ad_data(ad_data: Dict) -> Dict[str, np.ndarray]:
    """Preprocess advertisement data for model input
    
    Args:
        ad_data: Raw advertisement data
        
    Returns:
        Preprocessed features
        
    Raises:
        DataValidationError: If input data is invalid
        SchemaError: If input schema doesn't match
    """
    try:
        # Validate schema
        validator = AdDataValidator()
        validation_result = validator.validate(ad_data)
        
        if not validation_result.is_valid:
            raise DataValidationError(
                "Invalid ad data format", 
                validation_errors=validation_result.errors
            )
        
        # Transform textual features
        text_features = extract_text_features(ad_data)
        
        # Transform visual features if available
        visual_features = None
        if "image_url" in ad_data and ad_data["image_url"]:
            try:
                visual_features = extract_visual_features(ad_data["image_url"])
            except (NetworkError, ResourceExhaustedError) as e:
                # Log but continue with None for visual features
                logger.warning(f"Failed to extract visual features: {str(e)}")
                
        # Transform categorical features
        categorical_features = extract_categorical_features(ad_data)
        
        # Transform numerical features
        numerical_features = extract_numerical_features(ad_data)
        
        return {
            "text": text_features,
            "visual": visual_features,
            "categorical": categorical_features,
            "numerical": numerical_features
        }
    except DataException:
        # Re-raise DataExceptions as is
        raise
    except Exception as e:
        # Wrap other exceptions
        raise DataException(
            f"Failed to preprocess ad data: {str(e)}",
            details={"original_error": str(e), "error_type": e.__class__.__name__}
        ) from e
```

### Model Inference with Error Handling

```python
class AdScorePredictor:
    """Predictor for ad scores"""
    
    def __init__(self, model_path: str, fallback_model_path: str = None):
        """Initialize predictor
        
        Args:
            model_path: Path to primary model
            fallback_model_path: Path to fallback model
        """
        self.logger = logging.getLogger(__name__)
        self.error_metrics = ErrorMetricsRecorder(metrics_client)
        
        try:
            self.model = self._load_model(model_path)
            self.fallback_model = None
            if fallback_model_path:
                self.fallback_model = self._load_model(fallback_model_path)
        except Exception as e:
            self.logger.error(f"Failed to initialize AdScorePredictor: {str(e)}")
            raise
    
    def predict(self, ad_data: Dict) -> Dict:
        """Predict ad score
        
        Args:
            ad_data: Advertisement data
            
        Returns:
            Prediction result with score and explanations
            
        Raises:
            DataValidationError: If input data is invalid
            ModelRuntimeError: If model fails during prediction
        """
        start_time = time.time()
        used_fallback = False
        
        try:
            # Preprocess data
            try:
                features = preprocess_ad_data(ad_data)
            except DataException as e:
                e.log(self.logger)
                self.error_metrics.record_error(e, {"component": "preprocessing"})
                raise
            
            # Make prediction with primary model
            try:
                prediction = self.model.predict(features)
            except Exception as e:
                # Try fallback if available
                if self.fallback_model:
                    self.logger.warning(f"Primary model failed, using fallback: {str(e)}")
                    used_fallback = True
                    prediction = self.fallback_model.predict(features)
                else:
                    # No fallback, re-raise
                    raise ModelRuntimeError(
                        f"Model prediction failed: {str(e)}",
                        model_id=getattr(self.model, "model_id", None),
                        details={"input_keys": list(features.keys())}
                    ) from e
            
            # Generate explanation if needed
            explanation = None
            if ad_data.get("explain", False):
                try:
                    explanation = self._generate_explanation(features, prediction)
                except Exception as explain_error:
                    # Log but continue without explanation
                    self.logger.warning(f"Failed to generate explanation: {str(explain_error)}")
            
            # Return prediction result
            result = {
                "score": float(prediction),
                "used_fallback": used_fallback,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
            
            if explanation:
                result["explanation"] = explanation
                
            return result
            
        except MLBaseException:
            # Log timing for errors too
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(f"Prediction failed in {duration_ms}ms")
            raise
        except Exception as e:
            # Wrap unexpected errors
            duration_ms = int((time.time() - start_time) * 1000)
            error = ModelRuntimeError(
                f"Unexpected error in prediction: {str(e)}",
                model_id=getattr(self.model, "model_id", None),
                details={"duration_ms": duration_ms}
            )
            error.log(self.logger)
            self.error_metrics.record_error(error, {"component": "prediction", "duration_ms": duration_ms})
            raise error from e
    
    def _load_model(self, model_path: str):
        """Load model from path
        
        Args:
            model_path: Path to model
            
        Returns:
            Loaded model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Implementation of model loading
            pass
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {model_path}: {str(e)}") from e
    
    def _generate_explanation(self, features: Dict, prediction: float) -> Dict:
        """Generate explanation for prediction
        
        Args:
            features: Input features
            prediction: Model prediction
            
        Returns:
            Explanation dictionary
        """
        # Implementation of explanation generation
        pass
```

### API Endpoint with Error Handling

```python
@app.post("/api/v1/predict/ad_score")
async def predict_ad_score(ad_data: AdDataModel, request: Request):
    """API endpoint for ad score prediction
    
    Args:
        ad_data: Advertisement data
        request: FastAPI request
        
    Returns:
        Prediction result
    """
    request_id = str(uuid.uuid4())
    logger = get_request_logger(request_id)
    start_time = time.time()
    
    logger.info(f"Received ad score prediction request: {request_id}")
    
    try:
        # Get predictor instance
        predictor = get_predictor_instance()
        
        # Make prediction
        result = predictor.predict(ad_data.dict())
        
        # Add request metadata
        result["request_id"] = request_id
        result["processed_at"] = datetime.utcnow().isoformat()
        
        # Log success
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Prediction successful in {duration_ms}ms")
        
        return result
        
    except DataValidationError as e:
        # Client error - return 400
        e.log(logger)
        
        return JSONResponse(
            status_code=400,
            content={"error": e.to_dict(), "request_id": request_id}
        )
        
    except (ModelNotFoundError, ModelLoadError) as e:
        # Service configuration error - return 503
        e.log(logger)
        
        return JSONResponse(
            status_code=503,
            content={"error": e.to_dict(), "request_id": request_id}
        )
        
    except ModelRuntimeError as e:
        # Model execution error - return 500
        e.log(logger)
        
        return JSONResponse(
            status_code=500,
            content={"error": e.to_dict(), "request_id": request_id}
        )
        
    except Exception as e:
        # Unexpected error - return 500
        error = APIException(
            f"Unexpected error: {str(e)}",
            error_code="INTERNAL_SERVER_ERROR",
            details={"request_id": request_id}
        )
        error.log(logger)
        
        return JSONResponse(
            status_code=500,
            content={"error": error.to_dict(), "request_id": request_id}
        )
    finally:
        # Always log request completion
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Request {request_id} completed in {duration_ms}ms")
```

## Related Documentation

For comprehensive understanding of the WITHIN ML error handling ecosystem, refer to these related documents:

- [Test Strategy and Coverage](test_strategy.md) - Testing approach for error conditions
- [Production Validation](production_validation.md) - Monitoring and validating errors in production
- [Inference API Documentation](inference_api.md) - API error handling specifications
- [Training Pipeline Documentation](training_pipeline.md) - Error handling in model training
- [Implementation Roadmap](implementation_roadmap.md) - Planned improvements to error handling

## Conclusion

This document outlines the standardized error handling patterns implemented across the WITHIN ML platform. By following these consistent patterns, the platform achieves robust error handling, clear error reporting, and graceful degradation in the face of failures.

The error handling architecture is designed to:

1. Provide clear, actionable error information
2. Isolate failures to prevent cascading errors
3. Enable graceful degradation of functionality
4. Support comprehensive monitoring and debugging
5. Facilitate rapid recovery from failures

These patterns should be consistently applied throughout the codebase to maintain robustness and reliability across the platform. 