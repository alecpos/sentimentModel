"""
Data lake components for the WITHIN ML Prediction System.

This module provides interfaces and utilities for interacting with the data lake
storage system. It includes components for data ingestion, retrieval, and management
across various storage formats and locations.

Key functionality includes:
- Data ingestion and storage operations
- Data retrieval and querying capabilities
- Metadata management and cataloging
- Data partitioning and organization
- Storage format handling (Parquet, ORC, JSON, etc.)
"""

__version__ = "0.1.0"  # Pre-release version

# Package-level constants
DEFAULT_STORAGE_FORMAT = "parquet"
DEFAULT_PARTITION_SCHEME = "year_month_day"
COMPRESSION_METHODS = ["snappy", "gzip", "zstd", "none"]
DEFAULT_COMPRESSION = "snappy"

# When implementations are added, they will be imported and exported here 