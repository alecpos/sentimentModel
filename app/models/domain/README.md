# Domain Models

This directory contains domain-specific models for the WITHIN ML Prediction System that represent key business entities and data structures.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The domain models system provides capabilities for:
- Representing core domain entities and business concepts
- Defining the structure of the data lake components
- Managing metadata in the data catalog
- Supporting domain-driven design principles
- Facilitating data access patterns across the system

## Key Components

### Base

`Base` is the SQLAlchemy declarative base class imported from the core database module. It provides:
- Common foundation for all SQLAlchemy ORM models
- Metadata configuration for the SQLAlchemy tables
- Session binding utilities
- Schema generation capabilities
- Consistent model definition patterns

All domain-specific database models inherit from this base class to ensure consistent database interaction.

### DataLakeModel

`DataLakeModel` provides a structured interface to data storage in the data lake. It includes:
- Storage location and path management
- Data format specifications (Parquet, CSV, JSON)
- Partition key definitions for efficient data access
- Metadata properties (owner, description, creation date)
- Methods for data retrieval with filtering capabilities
- Storage efficiency optimization
- Data versioning and history tracking

### DataCatalogModel

`DataCatalogModel` manages metadata about datasets in the system. It includes:
- Dataset descriptions and schema definitions
- Data lineage tracking and provenance information
- Feature definitions and data type specifications
- Data quality metrics and validation rules
- Inter-dataset relationships and dependencies
- Tagging and categorization functionality
- Update frequency and freshness tracking

## Usage Example

```python
from app.models.domain import DataLakeModel, DataCatalogModel

# Create data lake entry for campaign data
data_lake_entry = DataLakeModel(
    name="campaign_performance_2023",
    path="s3://within-data-lake/campaigns/2023/",
    format="parquet",
    partition_keys=["date", "campaign_id"],
    owner="data_engineering",
    description="Daily campaign performance metrics for 2023"
)

# Create catalog entry for the dataset
catalog_entry = DataCatalogModel(
    name="Campaign Performance 2023",
    source_id=data_lake_entry.id,
    schema={
        "campaign_id": "string",
        "date": "date",
        "impressions": "integer",
        "clicks": "integer",
        "conversions": "integer",
        "spend": "float"
    },
    tags=["campaign", "performance", "2023"],
    update_frequency="daily",
    last_updated="2023-12-31",
    quality_score=0.98
)

# Access data efficiently using the models
performance_data = data_lake_entry.get_data(
    filters={"date": "2023-09-01", "campaign_id": "C12345"}
)
```

## Integration Points

- **ETL Pipeline**: Uses domain models for data transformation
- **ML Feature Engineering**: Extracts features using domain models
- **API Layer**: Domain models form the basis of API schemas
- **Reporting System**: Domain models drive reporting structures
- **Data Validation**: Domain models define validation constraints

## Dependencies

- **SQLAlchemy**: ORM for database operations
- **Pydantic**: Data validation for models
- **Core DB**: Base model and database utilities
- **Cloud Storage**: Storage access implementations
- **Metadata Store**: Catalog storage backend

## Future Extensions

- Enhanced versioning for data catalog entries
- Data lineage graph representations
- Automated data quality scoring
- Cross-domain entity relationships
- Semantic tagging and categorization 