# app/core/db/models.py

from datetime import datetime
from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import declared_attr
from uuid import uuid4
from .base import Base

class BaseModel(Base):
    """Abstract base model with common functionality"""
    __abstract__ = True

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }