# app/core/db/mixins.py

from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON
from uuid import uuid4

class TimestampMixin:
    """Mixin for timestamp columns"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class UUIDMixin:
    """Mixin for UUID primary key"""
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))

class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    deleted_at = Column(DateTime, nullable=True)
    
    def soft_delete(self):
        self.deleted_at = datetime.utcnow()

    @property
    def is_deleted(self):
        return self.deleted_at is not None

class AuditMixin:
    """Mixin for audit fields"""
    created_by = Column(String(36), nullable=True)
    updated_by = Column(String(36), nullable=True)

    def set_created_by(self, user_id):
        if not self.created_by:
            self.created_by = user_id
            self.updated_by = user_id

    def set_updated_by(self, user_id):
        self.updated_by = user_id
        
        
        # app/core/db/mixins.py

class VersioningMixin:
    """Mixin for version history tracking"""
    version = Column(String(20), nullable=False, default="1.0.0")
    version_history = Column(JSON, nullable=False, default=list)

    def save_version(self, change_type: str, description: str, user_id: str = None):
        """Track version history with metadata"""
        major, minor, patch = self.version.split(".")
        if change_type == "major":
            major = str(int(major) + 1)
            minor = "0"
            patch = "0"
        elif change_type == "minor":
            minor = str(int(minor) + 1)
            patch = "0"
        else:
            patch = str(int(patch) + 1)
            
        self.version = f"{major}.{minor}.{patch}"
        if not hasattr(self, 'version_history'):
            self.version_history = []
            
        self.version_history.append({
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "change_type": change_type,
            "description": description,
            "user_id": user_id
        })