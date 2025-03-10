# Configuration Guide

This directory contains the configuration files for the WITHIN Ad Score & Account Health Predictor system. These configuration files control system behavior, model parameters, API connections, and other settings.

## Directory Structure

```
config/
├── settings.yaml                  # Main application settings
├── settings.example.yaml          # Example settings with documentation
├── environments/                  # Environment-specific settings
│   ├── development.yaml           # Development environment settings
│   ├── testing.yaml               # Testing environment settings
│   ├── staging.yaml               # Staging environment settings
│   └── production.yaml            # Production environment settings
├── secrets/                       # Secret configuration (not in version control)
│   ├── .gitignore                 # Ignores all files except itself and README
│   ├── README.md                  # Instructions for secrets management
│   └── secrets.example.yaml       # Example secrets template
├── models/                        # Model-specific configuration
│   ├── ad_score_model.yaml        # Ad score model configuration
│   ├── account_health_model.yaml  # Account health model configuration
│   └── feature_config.yaml        # Feature configuration
├── logging/                       # Logging configuration
│   ├── logging.yaml               # Main logging configuration
│   └── log_rotation.yaml          # Log rotation settings
└── api/                           # API configuration
    ├── rate_limits.yaml           # API rate limiting configuration
    ├── cors.yaml                  # CORS configuration
    └── api_docs.yaml              # API documentation configuration
```

## Main Settings

The `settings.yaml` file is the primary configuration file:

```yaml
# Application settings
app:
  name: "WITHIN Ad Score & Account Health Predictor"
  version: "1.0.0"
  environment: "development"  # development, testing, staging, production
  debug: true
  temporary_directory: "/tmp/within"
  max_workers: 4

# Database settings
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  name: "within_db"
  user: "within_user"
  use_ssl: true
  pool_size: 10
  max_overflow: 20
  timeout: 30

# API settings
api:
  host: "0.0.0.0"
  port: 8000
  base_path: "/api/v1"
  workers: 4
  cors_settings: "config/api/cors.yaml"
  rate_limit_settings: "config/api/rate_limits.yaml"
  docs_settings: "config/api/api_docs.yaml"

# Model settings
models:
  default_ad_score_model: "gradient_boosting_v1"
  default_account_health_model: "ensemble_v1"
  model_configs_path: "config/models"
  model_registry_path: "models/registry"
  cache_predictions: true
  prediction_cache_ttl: 3600  # seconds
  
# Monitoring settings
monitoring:
  enable_performance_monitoring: true
  enable_drift_detection: true
  enable_fairness_monitoring: true
  metrics_storage_days: 90
  alert_on_drift: true
  alert_on_performance_degradation: true
  alert_channels:
    - email
    - slack

# Logging settings
logging:
  config_file: "config/logging/logging.yaml"
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  enable_request_logging: true
  enable_performance_logging: true
  
# Feature settings
features:
  enable_nlp_analysis: true
  enable_cross_platform_analysis: true
  enable_advanced_visualizations: true
```

## Environment-Specific Settings

Environment-specific settings override the main settings for different environments:

### Development Settings (`environments/development.yaml`)

```yaml
app:
  debug: true
  
database:
  host: "localhost"
  
api:
  cors:
    allow_origins: ["*"]
    
monitoring:
  metrics_storage_days: 30
  
logging:
  log_level: "DEBUG"
```

### Production Settings (`environments/production.yaml`)

```yaml
app:
  debug: false
  max_workers: 8
  
database:
  pool_size: 20
  max_overflow: 30
  
api:
  workers: 8
  
monitoring:
  metrics_storage_days: 365
  
logging:
  log_level: "INFO"
  enable_performance_logging: true
```

## Secret Management

Secrets are stored in the `secrets/` directory which is excluded from version control. The `secrets.example.yaml` provides a template:

```yaml
# Database credentials
database:
  password: "your_db_password_here"
  
# API keys for ad platforms
api_keys:
  facebook:
    app_id: "your_app_id"
    app_secret: "your_app_secret"
    access_token: "your_access_token"
  google:
    client_id: "your_client_id"
    client_secret: "your_client_secret"
    refresh_token: "your_refresh_token"
  amazon:
    client_id: "your_client_id"
    client_secret: "your_client_secret"
    
# JWT secret
auth:
  jwt_secret: "your_jwt_secret"
  
# Encryption keys
encryption:
  key: "your_encryption_key"
```

In production, secrets are typically managed via environment variables or a secrets management service like AWS Secrets Manager, HashiCorp Vault, or Kubernetes Secrets.

## Model Configuration

Model-specific configuration is stored in the `models/` directory:

### Ad Score Model Configuration (`models/ad_score_model.yaml`)

```yaml
# Model version and metadata
version: "1.0.0"
name: "Ad Score Gradient Boosting Model"
description: "Predicts ad effectiveness on a scale of 0-100"
author: "ML Engineering Team"
creation_date: "2023-01-15"

# Model parameters
parameters:
  model_type: "gradient_boosting"
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  
# Feature configuration
features:
  text_features:
    enabled: true
    embedding_model: "ad-optimized-bert"
    max_length: 512
  numerical_features:
    enabled: true
    normalization: "standard_scaler"
  categorical_features:
    enabled: true
    encoding: "one_hot"
    max_categories: 10
    
# Training configuration
training:
  validation_split: 0.2
  early_stopping_rounds: 10
  eval_metric: "rmse"
  
# Prediction configuration
prediction:
  batch_size: 64
  calibration: "isotonic"
  threshold: 0.5
  
# Explainability
explainability:
  method: "shap"
  background_samples: 100
```

## API Configuration

API-specific configuration:

### Rate Limits (`api/rate_limits.yaml`)

```yaml
# Default rate limit
default:
  limit: 100
  period: "minute"
  
# Endpoint-specific rate limits
endpoints:
  "/ad-score/predict":
    limit: 100
    period: "minute"
  "/ad-score/batch-predict":
    limit: 10
    period: "minute"
  "/account-health/score":
    limit: 30
    period: "minute"
  "/analytics/":
    limit: 30
    period: "minute"
    
# Client-specific rate limits
clients:
  premium:
    limit: 1000
    period: "minute"
  standard:
    limit: 100
    period: "minute"
```

### CORS Configuration (`api/cors.yaml`)

```yaml
# CORS settings
allow_origins:
  - "https://app.within.co"
  - "https://dashboard.within.co"
allow_methods:
  - "GET"
  - "POST"
  - "PUT"
  - "DELETE"
allow_headers:
  - "Authorization"
  - "Content-Type"
  - "X-Request-ID"
allow_credentials: true
max_age: 3600
```

## Logging Configuration

Logging configuration is defined in `logging/logging.yaml`:

```yaml
# Root logger
root:
  level: INFO
  handlers:
    - console
    - file

# Loggers
loggers:
  app:
    level: INFO
    handlers:
      - app_file
    propagate: false
  app.api:
    level: INFO
    handlers:
      - api_file
    propagate: false
  app.models:
    level: INFO
    handlers:
      - model_file
    propagate: false

# Handlers
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 10
  app_file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: logs/application.log
    maxBytes: 10485760  # 10MB
    backupCount: 10
  api_file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: logs/api.log
    maxBytes: 10485760  # 10MB
    backupCount: 10
  model_file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: logs/models.log
    maxBytes: 10485760  # 10MB
    backupCount: 10

# Formatters
formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
  detailed:
    format: "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
```

## Configuration Loading

The application loads configuration using the following hierarchy:

1. Default configuration from `settings.yaml`
2. Environment-specific overrides from `environments/{env}.yaml`
3. Secret values from `secrets/secrets.yaml` or environment variables
4. Command-line arguments and environment variables

Example configuration loading:

```python
from app.core.config import ConfigLoader

# Load configuration
config = ConfigLoader.load(
    base_config_path="config/settings.yaml",
    environment="production",  # or from environment variable
    secrets_path="config/secrets/secrets.yaml",
    override_from_env=True
)

# Access configuration values
db_config = config.database
api_config = config.api
model_config = config.models.ad_score_model
```

## Environment Variables

The application supports configuration via environment variables which override file-based configuration:

- `WITHIN_ENV`: Application environment (development, testing, staging, production)
- `WITHIN_DEBUG`: Enable debug mode (true/false)
- `WITHIN_DB_HOST`: Database host
- `WITHIN_DB_PORT`: Database port
- `WITHIN_DB_NAME`: Database name
- `WITHIN_DB_USER`: Database user
- `WITHIN_DB_PASSWORD`: Database password
- `WITHIN_API_PORT`: API port
- `WITHIN_LOG_LEVEL`: Logging level

Environment variables can be set in a `.env` file in the project root or in the system environment.

## Feature Flags

The application implements feature flags for enabling/disabling specific features:

```yaml
# Feature flags
features:
  enable_nlp_analysis: true
  enable_cross_platform_analysis: true
  enable_advanced_visualizations: true
  enable_account_health_predictions: true
  enable_real_time_predictions: true
  enable_batch_predictions: true
  enable_model_explanations: true
  enable_fairness_analysis: true
```

Feature flags can be checked in code:

```python
from app.core.config import features

if features.is_enabled("enable_nlp_analysis"):
    # Perform NLP analysis
    pass
```

## Development Guidelines

When modifying configuration:

1. **Document New Options**: Add new configuration options to `settings.example.yaml` with comments
2. **Default Values**: Provide sensible default values for all configuration options
3. **Environment-Specific Settings**: Place environment-specific settings in the appropriate environment file
4. **Secret Management**: Never commit secrets to version control
5. **Configuration Validation**: Add validation for configuration values
6. **Documentation**: Update this README when adding significant configuration options
7. **Backward Compatibility**: Maintain backward compatibility when possible 