# /Users/alecposner/WITHIN/app/models/__init__.py
"""
Domain models for the WITHIN ML Prediction System.

This module provides domain-specific models that represent business entities
and data structures used throughout the application. These models define
the core domain objects and their relationships.

Key components include:
- Data Lake models for data storage and retrieval
- Data Catalog models for metadata management
- Domain-specific business entities
- Core data structures and representations

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

from app.core.database import Base
from .data_lake_model import DataLakeModel
from .data_catalog_model import DataCatalogModel

__all__ = [
    # Base model
    "Base",
    
    # Domain models
    "DataLakeModel",
    "DataCatalogModel"
]