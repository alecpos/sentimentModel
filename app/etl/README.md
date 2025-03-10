# ETL Pipelines

This directory contains the Extract-Transform-Load (ETL) pipelines for the WITHIN Ad Score & Account Health Predictor system. These pipelines acquire, process, and store data from various advertising platforms and other sources to power the prediction models.

## Directory Structure

```
etl/
├── __init__.py                     # ETL package initialization
├── sources/                        # Data source connectors
│   ├── __init__.py                 # Sources package initialization
│   ├── facebook.py                 # Facebook Ads API connector
│   ├── google.py                   # Google Ads API connector
│   ├── amazon.py                   # Amazon Advertising API connector
│   ├── tiktok.py                   # TikTok Ads API connector
│   └── internal.py                 # Internal data source connector
├── transforms/                     # Data transformation functions
│   ├── __init__.py                 # Transforms package initialization
│   ├── normalization.py            # Data normalization functions
│   ├── enrichment.py               # Data enrichment functions
│   ├── aggregation.py              # Data aggregation functions
│   └── validation.py               # Data validation functions
├── loaders/                        # Data loading components
│   ├── __init__.py                 # Loaders package initialization
│   ├── database.py                 # Database loader
│   ├── feature_store.py            # Feature store loader
│   └── model_store.py              # Model training data loader
├── pipelines/                      # Orchestrated ETL pipelines
│   ├── __init__.py                 # Pipelines package initialization
│   ├── ad_data_pipeline.py         # Ad data processing pipeline
│   ├── account_data_pipeline.py    # Account data processing pipeline
│   ├── platform_data_pipeline.py   # Cross-platform data pipeline
│   └── training_data_pipeline.py   # Model training data pipeline
├── tasks/                          # Scheduled ETL tasks
│   ├── __init__.py                 # Tasks package initialization
│   ├── daily_tasks.py              # Daily ETL tasks
│   ├── hourly_tasks.py             # Hourly ETL tasks
│   └── weekly_tasks.py             # Weekly ETL tasks
└── utils/                          # ETL utilities
    ├── __init__.py                 # Utils package initialization
    ├── retry.py                    # Retry mechanism for API calls
    ├── batching.py                 # Batch processing utilities
    └── logging.py                  # ETL-specific logging
```

## Core Components

### Data Source Connectors

The ETL system includes connectors for various advertising platforms:

```python
from app.etl.sources import FacebookAdsConnector

# Initialize connector
connector = FacebookAdsConnector(
    account_id="123456789",
    access_token="your_access_token",
    rate_limit=100  # requests per minute
)

# Fetch ad data
ads = connector.get_ads(
    date_range=("2023-01-01", "2023-03-31"),
    fields=["id", "name", "status", "creative", "insights"],
    limit=1000
)

# Fetch account data
account_stats = connector.get_account_stats(
    date_range=("2023-01-01", "2023-03-31"),
    metrics=["spend", "impressions", "clicks", "conversions"]
)
```

### Data Transformation

The transformation components standardize and enrich data from different sources:

```python
from app.etl.transforms import DataNormalizer, DataEnricher

# Initialize normalizer
normalizer = DataNormalizer()

# Normalize platform-specific data
normalized_data = normalizer.normalize_ad_data(
    data=raw_ad_data,
    platform="facebook"
)

# Initialize enricher
enricher = DataEnricher()

# Enrich with additional features
enriched_data = enricher.enrich_ad_data(
    data=normalized_data,
    enrich_with=["sentiment", "topic_classification", "historical_performance"]
)
```

### Data Validation

The validation component ensures data quality and completeness:

```python
from app.etl.transforms import DataValidator

# Initialize validator
validator = DataValidator()

# Define validation schema
ad_data_schema = {
    "ad_id": {"type": "string", "required": True},
    "headline": {"type": "string", "required": True, "max_length": 200},
    "description": {"type": "string", "required": False},
    "platform": {"type": "string", "required": True, "allowed": ["facebook", "google", "amazon", "tiktok"]},
    "impressions": {"type": "integer", "required": True, "min": 0},
    "clicks": {"type": "integer", "required": True, "min": 0},
    "ctr": {"type": "float", "required": False, "min": 0, "max": 1},
    "spend": {"type": "float", "required": True, "min": 0},
    "conversions": {"type": "integer", "required": False, "min": 0},
    "start_date": {"type": "date", "required": True},
    "end_date": {"type": "date", "required": True}
}

# Validate data
validation_result = validator.validate(data=normalized_data, schema=ad_data_schema)

if not validation_result["valid"]:
    for error in validation_result["errors"]:
        print(f"Error in {error['field']}: {error['message']}")
```

### ETL Pipelines

The ETL pipelines orchestrate the entire data processing workflow:

```python
from app.etl.pipelines import AdDataPipeline

# Initialize pipeline
pipeline = AdDataPipeline(
    source_platform="facebook",
    destination="feature_store",
    validate=True,
    enrich=True
)

# Run pipeline
result = pipeline.run(
    account_id="123456789",
    date_range=("2023-01-01", "2023-03-31"),
    batch_size=1000
)

# Get pipeline result
if result["success"]:
    print(f"Processed {result['records_processed']} records")
    print(f"Pipeline took {result['duration_seconds']} seconds")
else:
    print(f"Pipeline failed: {result['error']}")
```

### Scheduled Tasks

ETL tasks can be scheduled to run at specified intervals:

```python
from app.etl.tasks import register_daily_task

# Register a daily ETL task
register_daily_task(
    task_name="facebook_ads_daily",
    pipeline="AdDataPipeline",
    pipeline_args={
        "source_platform": "facebook",
        "destination": "feature_store",
        "validate": True,
        "enrich": True
    },
    account_ids=["123456789", "987654321"],
    run_time="01:00",  # 1:00 AM
    lookback_days=7
)
```

## Data Schemas

### Ad Data Schema

The core ad data schema includes:

| Field | Type | Description |
|-------|------|-------------|
| `ad_id` | string | Unique identifier for the ad |
| `platform` | string | Advertising platform (facebook, google, etc.) |
| `account_id` | string | Account ID the ad belongs to |
| `campaign_id` | string | Campaign ID the ad belongs to |
| `ad_group_id` | string | Ad group ID the ad belongs to |
| `headline` | string | Ad headline text |
| `description` | string | Ad description text |
| `cta` | string | Call to action text |
| `image_url` | string | URL of the ad image (if applicable) |
| `video_url` | string | URL of the ad video (if applicable) |
| `landing_page_url` | string | Landing page URL |
| `start_date` | date | Ad start date |
| `end_date` | date | Ad end date (or null if ongoing) |
| `status` | string | Ad status (active, paused, archived) |
| `impressions` | integer | Number of impressions |
| `clicks` | integer | Number of clicks |
| `ctr` | float | Click-through rate |
| `spend` | float | Amount spent |
| `conversions` | integer | Number of conversions |
| `conversion_rate` | float | Conversion rate |
| `cpa` | float | Cost per acquisition |
| `roas` | float | Return on ad spend |
| `relevance_score` | float | Platform-provided relevance score |
| `quality_score` | float | Platform-provided quality score |

### Account Data Schema

The account data schema includes:

| Field | Type | Description |
|-------|------|-------------|
| `account_id` | string | Unique identifier for the account |
| `platform` | string | Advertising platform |
| `account_name` | string | Account name |
| `currency` | string | Account currency |
| `timezone` | string | Account timezone |
| `date` | date | Stats date |
| `daily_spend` | float | Daily spend amount |
| `daily_impressions` | integer | Daily impressions |
| `daily_clicks` | integer | Daily clicks |
| `daily_conversions` | integer | Daily conversions |
| `daily_revenue` | float | Daily revenue (if available) |
| `daily_roas` | float | Daily ROAS |
| `campaign_count` | integer | Number of active campaigns |
| `ad_count` | integer | Number of active ads |

## Cross-Platform Normalization

The ETL system normalizes data across advertising platforms using standardized schemas and field mappings:

```python
from app.etl.transforms import CrossPlatformNormalizer

# Initialize normalizer
normalizer = CrossPlatformNormalizer()

# Normalize data from different platforms
normalized_facebook = normalizer.normalize(facebook_data, platform="facebook")
normalized_google = normalizer.normalize(google_data, platform="google")

# Combine into a unified dataset
unified_data = normalizer.combine([normalized_facebook, normalized_google])
```

## Data Enrichment

The ETL pipelines can enrich ad data with additional features:

1. **Sentiment Analysis**: Analyzes ad text for sentiment
2. **Topic Classification**: Categorizes ads by topic
3. **Entity Extraction**: Extracts entities from ad text
4. **Image Analysis**: Analyzes ad images for features
5. **Historical Performance**: Adds historical performance metrics
6. **Audience Insights**: Adds audience demographic information
7. **Competitive Analysis**: Adds competitive benchmarking data

## Error Handling

ETL pipelines implement robust error handling:

```python
from app.etl.utils import retry_with_backoff

@retry_with_backoff(
    max_retries=3,
    backoff_factor=2,
    errors=(ConnectionError, TimeoutError)
)
def fetch_api_data(api_endpoint, params):
    """Fetch data from API with retry logic."""
    response = requests.get(api_endpoint, params=params)
    response.raise_for_status()
    return response.json()
```

Pipeline error handling includes:

1. **Retry Logic**: Automatic retries for transient failures
2. **Circuit Breakers**: Prevents repeated failures from overloading external systems
3. **Error Logging**: Comprehensive error logging for diagnostics
4. **Partial Success**: Can proceed with partial data when some records fail
5. **Recovery Points**: Allows resuming from the last successful batch

## Data Quality Monitoring

The ETL system includes data quality monitoring:

```python
from app.etl.transforms import DataQualityMonitor

# Initialize monitor
monitor = DataQualityMonitor()

# Check data quality
quality_report = monitor.check_quality(
    data=processed_data,
    metrics=["completeness", "accuracy", "consistency"]
)

# Log quality issues
if quality_report["issues"]:
    for issue in quality_report["issues"]:
        print(f"Quality issue: {issue['type']} in {issue['field']}, {issue['description']}")
```

Data quality dimensions monitored include:

1. **Completeness**: Missing or null values
2. **Accuracy**: Values within expected ranges
3. **Consistency**: Logical relationships between fields
4. **Timeliness**: Data freshness
5. **Uniqueness**: Duplicate detection
6. **Validity**: Adherence to business rules

## Development Guidelines

When enhancing or adding ETL components:

1. **Follow ETL Pattern**: Maintain clear separation between extract, transform, and load steps
2. **Implement Idempotent Processing**: Ensure pipelines can be safely re-run
3. **Include Data Validation**: Validate data at each stage
4. **Provide Clear Logging**: Log processing steps for traceability
5. **Gracefully Handle Errors**: Implement robust error handling
6. **Use Circuit Breakers**: Protect external dependencies
7. **Maintain Transaction Logs**: Enable recovery from failures
8. **Alert on Quality Issues**: Set up monitoring for data quality problems
9. **Batch Processing**: Implement appropriate batching for large datasets
10. **Apply Parallelization**: Use parallel processing where appropriate
11. **Cache Reference Data**: Optimize performance with caching
12. **Monitor Memory Usage**: Be mindful of memory constraints for large operations 