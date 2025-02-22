# app/core/db/composite.py

from .models import BaseModel
from .mixins import TimestampMixin, UUIDMixin, AuditMixin

class TimestampedModel(BaseModel, TimestampMixin):
    """Base model with timestamps"""
    __abstract__ = True

class FullModel(BaseModel, UUIDMixin, TimestampMixin, AuditMixin):
    """Complete base model with all common mixins"""
    __abstract__ = True