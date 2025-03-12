# Data Lake Components

This directory contains data lake components for the WITHIN ML Prediction System.

## Purpose

The data lake system provides capabilities for:
- Storing and organizing large volumes of structured and unstructured data
- Supporting ML training and inference data pipelines
- Maintaining historical data for analysis and model retraining
- Providing efficient access patterns for different data consumption needs
- Managing metadata for data discovery and governance

## Key Components

### Data Ingestion

Components for loading data into the data lake:
- Batch ingestion from various sources
- Streaming data capture
- Schema evolution handling
- Data transformation during ingestion
- Quality validation during write operations

### Data Organization

Components for organizing data in the lake:
- Partitioning strategies for efficient access
- Data layout optimization
- Time-based and feature-based partitioning
- Metadata tagging and categorization
- Data lifecycle management

### Data Access

Components for retrieving data from the lake:
- Query interfaces for data retrieval
- Predicate pushdown for efficient filtering
- Column pruning for optimized reads
- Format conversion for consumer needs
- Access pattern optimization

### Data Governance

Components for managing data quality and compliance:
- Data cataloging and discovery
- Schema management and evolution
- Data lineage tracking
- Access control and audit logging
- Retention policy enforcement

## Usage Example

```python
from app.core.data_lake import DataLakeClient, StorageFormat, PartitionScheme

# Initialize the data lake client
client = DataLakeClient(
    storage_location="s3://within-ml-data-lake",
    default_format=StorageFormat.PARQUET
)

# Write data to the lake
client.write_dataframe(
    df=campaign_performance_df,
    table_name="campaign_performance",
    partition_by=PartitionScheme.YEAR_MONTH_DAY,
    partition_columns=["event_date"],
    compression="snappy"
)

# Read data from the lake with filtering
recent_high_performing_campaigns = client.read_dataframe(
    table_name="campaign_performance",
    filters=[
        ("event_date", ">=", "2023-01-01"),
        ("performance_score", ">", 0.8)
    ],
    columns=["campaign_id", "platform", "performance_score", "spend", "roi"]
)

# Get metadata about a table
table_info = client.get_table_metadata("campaign_performance")
print(f"Table size: {table_info.size_bytes / (1024**3):.2f} GB")
print(f"Row count: {table_info.row_count}")
print(f"Last updated: {table_info.last_modified}")
```

## Integration Points

- **ETL Pipeline**: Sources and sinks for data processing
- **ML Training**: Provides training and validation datasets
- **Feature Store**: Supplies raw data for feature computation
- **Analytics**: Supports ad-hoc analysis and reporting

## Dependencies

- Cloud storage SDKs (S3, GCS, ADLS)
- Data format libraries (Parquet, ORC, Avro)
- SQL and query engines
- Metadata management tools
- Data compression libraries 