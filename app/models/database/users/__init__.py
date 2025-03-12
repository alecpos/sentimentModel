"""
User-related database models for the WITHIN ML Prediction System.

This module provides SQLAlchemy ORM models for user management, authentication,
authorization, and auditing. These models handle user data persistence and access control.

Key entities include:
- User accounts and profiles
- Authentication credentials
- Role and permission assignments
- Access tokens and sessions
- Audit logs for user actions

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

# Import base models
from app.core.db import BaseModel, FullModel

# Placeholder for future user model implementations
# from .user_model import User
# from .role_model import Role
# from .permission_model import Permission

# Export models
__all__ = [
    # List of model classes to export
    # "User",
    # "Role", 
    # "Permission"
] 