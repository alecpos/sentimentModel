# Application Structure

This directory contains the main application code for the WITHIN ML Prediction System. The application follows a modular architecture with clear separation of concerns between data access, processing, ML prediction, and API layers.

## Directory Structure

### api/
API endpoints and routes organized by version.

- **v1/**: API version 1
  - **endpoints/**: Individual API endpoint handlers
  - **routes/**: API route definitions

### core/
Core functionality and infrastructure components.

- **config/**: Application configuration utilities
- **data_lake/**: Data lake access and management
- **db/**: Database connection and session management
- **events/**: Event handling system
- **feedback/**: User feedback processing
- **ml/**: Core ML infrastructure
- **preprocessor/**: Data preprocessing utilities
- **search/**: Search functionality
- **validation/**: Data validation utilities

### models/
Data and ML models.

- **database/**: Database models with SQLAlchemy
- **domain/**: Domain entities and business logic
- **ml/**: Machine learning models
  - **prediction/**: Prediction models for various tasks

### schemas/
Pydantic schemas for request/response validation and data transfer.

### services/
Business logic services that coordinate between different application layers.

### utils/
Utility functions and helpers used across the application.

## Key Components

### Ad Score Predictor
The `AdScorePredictor` class in `models/ml/prediction/ad_score_predictor.py` is the main predictor model, combining:
- Neural network models
- XGBoost ensemble
- Multi-modal feature extraction (text, images, numerical data)
- Model calibration
- Fairness evaluation

### Anomaly Detector
The `AnomalyDetector` in `models/ml/prediction/anomaly_detector.py` identifies unusual patterns in advertising data.

### Account Health Predictor
The `AccountHealthPredictor` in `models/ml/prediction/account_health_predictor.py` evaluates the overall health of advertising accounts.

## Architecture Design

The application follows a layered architecture:

1. **API Layer**: Handles HTTP requests/responses
2. **Service Layer**: Implements business logic
3. **Model Layer**: Provides data access and ML prediction
4. **Core Layer**: Supplies infrastructure and common utilities

Data flows through the application as follows:
1. Request data is validated using Pydantic schemas
2. Services coordinate data processing and prediction
3. ML models perform predictions with proper validation
4. Validated responses are returned to the client

## Configuration

The application is configured using environment variables through Pydantic Settings. See `app/config.py` for available settings.

## Error Handling

The application implements a consistent error handling approach:
- Custom exception types for domain-specific errors
- Proper context information in error logs
- Graceful degradation for ML components
- Comprehensive error tracking

## Type Safety

The entire application uses strict Python type hints, enforced through static type checking tools. 