"""
Domain services for the WITHIN ML Prediction System.

This module provides services that implement domain-specific business logic,
managing domain entities and coordinating operations across the system.
It includes services for data catalog management, data lake operations,
and other domain-specific functionalities.

Key functionality includes:
- Data catalog management
- Data lake operations
- Domain entity operations
- Business rule implementation
- Domain-specific workflows

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

# Export services
from .data_catalog_service import add_to_data_catalog
from .data_lake_service import save_to_data_lake

__all__ = [
    "add_to_data_catalog",
    "save_to_data_lake"
] 