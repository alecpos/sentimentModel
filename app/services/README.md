# Services Directory

This directory contains business logic services for the WITHIN ML Prediction System. These services coordinate interactions between different application layers, implementing the core business logic and orchestrating complex operations.

## Directory Structure

- **__init__.py**: Module initialization with service exports
- **reporting.py**: Services for generating analytics reports
- **monitoring/**: Services for monitoring system health and performance
- **domain/**: Domain-specific business services
- **ml/**: Machine learning services

## Key Components

### Reporting Service

Located in `reporting.py`, this service generates analytics reports on ad performance and account health:

- **Report generation**: Creates various report types (daily, weekly, monthly)
- **Data aggregation**: Combines data from multiple sources
- **Visualization**: Generates data visualizations
- **Export formats**: Supports multiple export formats (CSV, JSON, PDF)

### Monitoring Services

Located in the `monitoring/` directory, these services monitor system health:

- **PerformanceMonitor**: Tracks API and model performance
- **AlertingService**: Sends alerts when anomalies are detected
- **ResourceMonitor**: Monitors system resource usage
- **HealthCheckService**: Performs system health checks

### Domain Services

Located in the `domain/` directory, these services implement domain-specific business logic:

- **AdService**: Business logic for ad management
- **AccountService**: Business logic for account management
- **CampaignService**: Business logic for campaign management
- **UserService**: Business logic for user management

### ML Services

Located in the `ml/` directory, these services coordinate ML operations:

- **PredictionService**: Coordinates prediction requests
- **TrainingService**: Manages model training processes
- **EvaluationService**: Evaluates model performance
- **DeploymentService**: Handles model deployment

## Usage Examples

### Using the Reporting Service

```python
from app.services import ReportingService

# Initialize the service
reporting_service = ReportingService()

# Generate an ad performance report
report = reporting_service.generate_ad_performance_report(
    account_id="123456",
    start_date="2025-01-01",
    end_date="2025-01-31",
    format="pdf"
)

# Save the report
with open("ad_performance_report.pdf", "wb") as f:
    f.write(report.content)
```

### Using the ML Prediction Service

```python
from app.services.ml import PredictionService
from app.schemas import AdScoreRequest

# Initialize the service
prediction_service = PredictionService()

# Create a prediction request
request = AdScoreRequest(
    ad_id="ad123",
    headline="Limited Time Offer: 20% Off All Products",
    description="Shop our entire collection and save with this exclusive discount.",
    cta="Shop Now",
    platform="facebook",
    industry="retail"
)

# Get a prediction
prediction = prediction_service.predict_ad_score(request)
print(f"Ad score: {prediction.score} (confidence: {prediction.confidence})")
```

## Service Design Principles

The services follow these design principles:

1. **Single Responsibility**: Each service focuses on a specific area of functionality
2. **Dependency Injection**: Services receive dependencies through constructors
3. **Interface Segregation**: Clients only depend on methods they use
4. **Error Handling**: Comprehensive error handling and reporting
5. **Logging**: Detailed logging for debugging and monitoring
6. **Statelessness**: Services avoid maintaining state when possible

## Dependencies

- **Core Application Components**: app.models, app.core, app.utils
- **External Services**: Database, caching, messaging
- **Third-party Libraries**: Data processing and analytics libraries

## Additional Resources

- See `app/api/README.md` for API endpoint information
- See `app/models/README.md` for model implementation details 