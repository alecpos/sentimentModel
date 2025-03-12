# User Database Models

This directory contains SQLAlchemy database models for user management and authentication in the WITHIN ML Prediction System.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The user models system provides database representations for:
- Managing user accounts and profiles
- Handling authentication and authorization
- Supporting role-based access control
- Tracking user activity and audit logs
- Managing session and token data

## Future Components

This directory is prepared for implementing the following database models:

### User Model

Core user entity representation:
- User profile information
- Account status and settings
- Email and contact details
- Password hashes or auth references
- User preferences

### Role and Permission Models

Access control models:
- Role definitions and hierarchies
- Permission assignments
- Resource access controls
- Feature access flags
- User-role mappings

### Session and Token Models

Authentication state models:
- Session tracking and management
- Access token storage
- Refresh token management
- Device and IP tracking
- Login history

### Audit Log Models

Security and compliance tracking:
- User action logging
- Login attempt tracking
- Security event recording
- Data access audit trails
- Administrative action logs

## Usage Example

Once implemented, user models would be used like this:

```python
from sqlalchemy.orm import Session
from app.models.database.users import User, Role, Permission
from app.core.db import get_db

def authenticate_user(email: str, password: str):
    db = next(get_db())
    
    # Find the user
    user = db.query(User).filter(User.email == email).first()
    
    if not user or not user.verify_password(password):
        return None
    
    # Get user roles and permissions
    roles = db.query(Role).join(User.roles).filter(User.id == user.id).all()
    permissions = db.query(Permission).join(Role.permissions)\
                   .join(User.roles)\
                   .filter(User.id == user.id)\
                   .distinct()\
                   .all()
    
    return {
        "user": user.to_dict(),
        "roles": [role.name for role in roles],
        "permissions": [perm.name for perm in permissions]
    }
```

## Integration Points

- **Authentication System**: Uses these models for user verification
- **API Authorization**: Uses roles and permissions for access control
- **User Management API**: CRUD operations on user data
- **Audit System**: Records user activities for compliance
- **Admin Interface**: Manages users, roles, and permissions

## Dependencies

- SQLAlchemy ORM for database operations
- Core DB components (FullModel for audit trails)
- Password hashing libraries
- JWT or token management libraries
- User validation utilities 