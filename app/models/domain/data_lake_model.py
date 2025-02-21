from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    JSON,
    DateTime,
    LargeBinary,
    CheckConstraint
)
from sqlalchemy.orm import validates, relationship

from app.core.database import Base

class DataLakeModel(Base):
    """Model for storing data lake entries with metadata."""

    __tablename__ = "data_lake"

    # Store UUID as a 36-char string for broader DB compatibility
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(String(500), nullable=True)
    data = Column(LargeBinary, nullable=False)  # Remove size constraint from column definition
    meta_info = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Add relationship to DataCatalogModel
    catalog_entries = relationship("DataCatalogModel", back_populates="data_lake_entry")

    # Example constraints
    __table_args__ = (
        CheckConstraint('length(name) > 0', name='check_name_not_empty'),
        # 10MB limit on 'data'
        CheckConstraint('length(data) <= 10485760', name='data_size_check'),  # 10MB in bytes
    )

    # Validate fields
    @validates('name')
    def validate_name(self, key, name):
        """Validate name field."""
        if not name or len(name) == 0 or len(name) > 255:
            raise AssertionError("Name must be non-empty and <= 255 characters")
        return name

    @validates('meta_info')
    def validate_meta_info(self, key, meta):
        """Comprehensive metadata validation"""
        # Ensure meta_info is a dictionary
        if not isinstance(meta, dict):
            raise TypeError("meta_info must be a dictionary")
        
        # Define allowed values
        VALID_LEVELS = {"campaign", "user", "environment"}
        VALID_DATA_LAYERS = {"raw", "processed", "curated"}
        
        # Required fields validation
        if not meta.get("level"):
            raise ValueError("meta_info must contain 'level'")
        if not meta.get("data_layer"):
            raise ValueError("meta_info must contain 'data_layer'")
        
        # Validate values when present
        if meta["level"] not in VALID_LEVELS:
            raise ValueError(f"Invalid level. Must be one of {VALID_LEVELS}")
        
        if meta["data_layer"] not in VALID_DATA_LAYERS:
            raise ValueError(f"Invalid data layer. Must be one of {VALID_DATA_LAYERS}")
        
        return meta

    @validates('data')
    def validate_data(self, key, data):
        """Data validation"""
        if not data:
            raise AssertionError("Data cannot be empty")
            
        # Prevent modifications to raw data
        if (self.id is not None and  # Only check for existing entries
            hasattr(self, 'meta_info') and 
            isinstance(self.meta_info, dict) and
            self.meta_info.get('data_layer') == 'raw'):
            raise ValueError("Cannot modify data in raw layer after creation")
            
        return data  # Let the DB constraint handle the size limit