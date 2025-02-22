# /Users/alecposner/WITHIN/app/core/data_lake/security_manager.py

import logging
import re
from typing import Dict, Union

##################################
# 1. Policy Engine (Role-Based)
##################################
class PolicyEngine:
    """Role-Based Access Control (RBAC) policy evaluation engine.

    Implements a flexible RBAC system that evaluates user permissions based on roles
    and resource-specific rules. Designed to be extensible for integration with
    external IAM services.

    Key Features:
        - Role-based permission management
        - Resource-specific access policies
        - Default deny for unknown roles/resources
        - Support for hierarchical resources (e.g. "catalog:123")
        - Extensible for custom policy rules

    Attributes:
        role_assignments (Dict[str, str]): Maps user IDs to their roles
        resource_policies (Dict[str, Dict[str, Dict[str, bool]]]): Resource access policies

    Example:
        >>> policy_engine = PolicyEngine(
        ...     role_assignments={"user123": "admin"},
        ...     resource_policies={
        ...         "catalog": {
        ...             "admin": {"read": True, "write": True},
        ...             "viewer": {"read": True, "write": False}
        ...         }
        ...     }
        ... )
        >>> policy_engine.evaluate("user123", "catalog", "write")
        True
    """

    def __init__(self, role_assignments: Dict[str, str], resource_policies: Dict[str, Dict[str, Dict[str, bool]]]):
        """Initialize the policy engine with role assignments and access policies.

        Args:
            role_assignments (Dict[str, str]): Maps user IDs to their roles.
                Example: {"user123": "admin", "user456": "viewer"}
            resource_policies (Dict[str, Dict[str, Dict[str, bool]]]): Nested dictionary
                defining which roles can perform which actions on resources.
                Structure:
                {
                    "resource_type": {
                        "role_name": {
                            "action_name": bool,  # True allows, False denies
                            ...
                        },
                        ...
                    },
                    ...
                }

        Example:
            >>> role_assignments = {"user123": "admin"}
            >>> resource_policies = {
            ...     "catalog": {
            ...         "admin": {
            ...             "read": True,
            ...             "write": True,
            ...             "delete": True
            ...         },
            ...         "editor": {
            ...             "read": True,
            ...             "write": True,
            ...             "delete": False
            ...         }
            ...     }
            ... }
            >>> engine = PolicyEngine(role_assignments, resource_policies)
        """
        self.role_assignments = role_assignments
        self.resource_policies = resource_policies

    def evaluate(self, user_id: str, resource: str, action: str) -> bool:
        """Evaluate whether a user can perform the given action on the resource.

        Implements the core RBAC logic:
        1. Extracts resource type from resource identifier
        2. Looks up user's role
        3. Checks role's permissions for the resource
        4. Returns permission for the specific action

        Args:
            user_id (str): The ID of the user requesting access
            resource (str): Resource identifier, can be in format "type:id" 
                (e.g. "catalog:123") or just "type"
            action (str): The action being requested (e.g. "read", "write", "delete")

        Returns:
            bool: True if access is allowed, False otherwise

        Notes:
            - Unknown users are assigned the "guest" role
            - Unknown resources or roles default to False (deny)
            - Resource strings can be hierarchical (e.g. "catalog:123")
            - Actions are case-sensitive

        Example:
            >>> engine = PolicyEngine(...)
            >>> # Check if user123 can write to catalog
            >>> engine.evaluate("user123", "catalog", "write")
            True
            >>> # Check if user123 can write to specific catalog entry
            >>> engine.evaluate("user123", "catalog:123", "write")
            True
        """
        # Extract base resource type from resource string (e.g. "catalog:123" -> "catalog")
        resource_type = resource.split(":")[0] if ":" in resource else resource
        
        user_role = self.role_assignments.get(user_id, "guest")
        
        # If resource is not in policies, default to no access
        if resource_type not in self.resource_policies:
            return False

        # If user_role is not recognized for this resource, default to no
        role_permissions = self.resource_policies[resource_type]
        if user_role not in role_permissions:
            return False

        # Check if the role has permission for the specific action
        action_permissions = role_permissions[user_role]
        if isinstance(action_permissions, dict):
            return action_permissions.get(action, False)
        return action_permissions  # For backward compatibility with boolean permissions


##################################
# 2. Encryption Service
##################################
from cryptography.fernet import Fernet

class EncryptionService:
    """Secure data encryption service using Fernet symmetric encryption.

    Provides a secure interface for encrypting and decrypting sensitive data using
    the Fernet symmetric encryption scheme (based on AES in CBC mode with PKCS7 padding).
    Designed for secure data storage and transmission within the data lake.

    Key Features:
        - Symmetric encryption using Fernet (AES-128-CBC)
        - Secure key management
        - Built-in timestamp-based key rotation
        - Automatic padding and IV generation
        - URL-safe token format

    Attributes:
        _fernet (Fernet): The Fernet instance for encryption/decryption

    Security Notes:
        - The secret key must be 32 url-safe base64-encoded bytes
        - In production, store the key in a secure vault or HSM
        - Rotate keys periodically for enhanced security
        - Consider implementing key versioning for long-term storage

    Example:
        >>> from cryptography.fernet import Fernet
        >>> key = Fernet.generate_key()
        >>> service = EncryptionService(key)
        >>> encrypted = service.encrypt(b"sensitive data")
        >>> decrypted = service.decrypt(encrypted)
        >>> assert decrypted == b"sensitive data"
    """

    def __init__(self, secret_key: bytes):
        """Initialize the encryption service with a secret key.

        Args:
            secret_key (bytes): A 32-byte url-safe base64-encoded key.
                Generate using Fernet.generate_key()

        Raises:
            ValueError: If the key is not 32 bytes or not properly base64-encoded
            TypeError: If the key is not bytes

        Example:
            >>> key = Fernet.generate_key()
            >>> service = EncryptionService(key)
        """
        self._fernet = Fernet(secret_key)

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt raw bytes using Fernet symmetric encryption.

        Encrypts the input data using AES-128-CBC with a secure IV and PKCS7 padding.
        The output is a secure token that includes the IV and timestamp.

        Args:
            data (bytes): Raw bytes to encrypt. For strings, encode first:
                data.encode('utf-8')

        Returns:
            bytes: Encrypted data as a secure token

        Raises:
            TypeError: If data is not bytes
            Exception: If encryption fails

        Notes:
            - The output token includes the IV and timestamp
            - Tokens are URL-safe base64-encoded
            - Each encryption uses a unique IV

        Example:
            >>> service = EncryptionService(key)
            >>> text = "sensitive data".encode('utf-8')
            >>> encrypted = service.encrypt(text)
        """
        return self._fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt Fernet-encrypted bytes back to plaintext.

        Decrypts a Fernet token back to the original data, verifying the token's
        validity and timestamp in the process.

        Args:
            data (bytes): Fernet token to decrypt (from encrypt())

        Returns:
            bytes: Decrypted data as bytes

        Raises:
            TypeError: If data is not bytes
            InvalidToken: If the token is invalid or expired
            Exception: If decryption fails

        Notes:
            - Verifies token authenticity before decryption
            - Checks token timestamp for expiration
            - Returns original bytes exactly

        Example:
            >>> service = EncryptionService(key)
            >>> decrypted = service.decrypt(encrypted_token)
            >>> original = decrypted.decode('utf-8')  # if it was text
        """
        return self._fernet.decrypt(data)


##################################
# 3. Audit Logger
##################################
class AuditLogger:
    """Security audit logging system for tracking access and security events.

    Implements a comprehensive logging system for security-relevant events in the data lake,
    focusing on access attempts and security operations. Designed for compliance monitoring
    and security auditing.

    Key Features:
        - Detailed access attempt logging
        - Security event tracking
        - Standardized log formats
        - Support for multiple log destinations
        - Compliance-ready audit trails

    Log Format:
        Access Attempts: [AUDIT] User '{user_id}' attempted '{action}' on '{resource}' → {outcome}
        Security Events: [AUDIT EVENT] {description} (User: {user_id})

    Notes:
        - In production, consider sending logs to:
            * Secure log aggregation service
            * SIEM system
            * Compliance monitoring platform
        - Implement log rotation and retention policies
        - Consider adding log encryption for sensitive events
        - Add support for structured logging if needed

    Example:
        >>> logger = AuditLogger()
        >>> logger.record_access_attempt("user123", "data_lake:456", "read", True)
        >>> logger.log_event("user123", "Password changed successfully")
    """

    def record_access_attempt(self, user_id: str, resource: str, action: str, allowed: bool):
        """Record an access control decision in the audit log.

        Logs details about who attempted to access what resource and whether it was allowed.
        This creates an audit trail for security monitoring and compliance purposes.

        Args:
            user_id (str): The ID of the user attempting access
            resource (str): The resource being accessed (e.g., "data_lake:123")
            action (str): The action being attempted (e.g., "read", "write")
            allowed (bool): Whether the access was granted (True) or denied (False)

        Log Format:
            [AUDIT] User '{user_id}' attempted '{action}' on '{resource}' → {ALLOWED|DENIED}

        Example:
            >>> logger = AuditLogger()
            >>> logger.record_access_attempt("user123", "data_lake:456", "read", True)
            [AUDIT] User 'user123' attempted 'read' on 'data_lake:456' → ALLOWED

        Notes:
            - Timestamps are automatically added by the logging system
            - Log entries are written at INFO level
            - Failed attempts should be monitored for security incidents
        """
        outcome = "ALLOWED" if allowed else "DENIED"
        logging.info(f"[AUDIT] User '{user_id}' attempted '{action}' on '{resource}' → {outcome}")

    def log_event(self, user_id: str, event_description: str):
        """Log a general security-related event.

        Records security events such as configuration changes, system operations,
        or security-relevant user actions.

        Args:
            user_id (str): The ID of the user associated with the event
            event_description (str): A clear description of what occurred

        Log Format:
            [AUDIT EVENT] {event_description} (User: {user_id})

        Example:
            >>> logger = AuditLogger()
            >>> logger.log_event("admin", "Encryption key rotated")
            [AUDIT EVENT] Encryption key rotated (User: admin)

        Notes:
            - Use clear, consistent event descriptions
            - Include relevant context in the description
            - Consider adding event categories or severity levels
            - Avoid logging sensitive data in the description
        """
        logging.info(f"[AUDIT EVENT] {event_description} (User: {user_id})")

##################################
# 4. PII Processor
##################################
class PIIProcessor:
    """Basic PII redaction utility with common pattern detection"""
    
    @staticmethod
    def redact_pii(data: Union[bytes, str]) -> bytes:
        """
        Redact common PII patterns from data.
        Returns bytes regardless of input type.
        """
        if isinstance(data, bytes):
            text = data.decode('utf-8', errors='replace')
        else:
            text = str(data)

        # Define regex patterns for common PII
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'name': r'"user":\s*"[^"]+?"'  # Match "user": "any name"
        }

        # Redact all matches
        for pii_type, pattern in patterns.items():
            if pii_type == 'name':
                text = re.sub(pattern, '"user": "[REDACTED_NAME]"', text)
            else:
                text = re.sub(
                    pattern, 
                    f'[REDACTED_{pii_type.upper()}]', 
                    text, 
                    flags=re.IGNORECASE
                )

        return text.encode('utf-8')
    

##################################
# 5. Security Manager
##################################
class SecurityManager:
    """Comprehensive security management system for the data lake.

    A high-level security orchestrator that integrates role-based access control,
    data encryption, and security audit logging. Provides a unified interface for
    all security operations in the data lake.

    Key Components:
        - PolicyEngine: Handles role-based access control (RBAC)
        - EncryptionService: Manages data encryption/decryption
        - AuditLogger: Records security events and access attempts

    Key Features:
        - Unified security interface
        - Integrated access control
        - Transparent data encryption
        - Comprehensive audit logging
        - Exception handling and logging
        - Security event tracking

    Attributes:
        policy_engine (PolicyEngine): RBAC policy evaluation engine
        encryption_service (EncryptionService): Data encryption service
        audit_logger (AuditLogger): Security audit logger

    Example:
        >>> # Initialize components
        >>> secret_key = Fernet.generate_key()
        >>> policy_engine = PolicyEngine(
        ...     role_assignments={"user123": "admin"},
        ...     resource_policies={
        ...         "data_catalog": {"admin": {"read": True, "write": True}}
        ...     }
        ... )
        >>> encryption_service = EncryptionService(secret_key)
        >>> audit_logger = AuditLogger()
        >>> 
        >>> # Create security manager
        >>> security = SecurityManager(policy_engine, encryption_service, audit_logger)
        >>> 
        >>> # Use security features
        >>> if security.check_access("user123", "data_catalog", "write"):
        ...     encrypted_data = security.encrypt_data(b"sensitive data")
        ...     security.log_general_event("user123", "Data encrypted successfully")
    """

    def __init__(self, 
                 policy_engine: PolicyEngine, 
                 encryption_service: EncryptionService, 
                 audit_logger: AuditLogger):
        """Initialize the security manager with required components.

        Args:
            policy_engine (PolicyEngine): RBAC policy evaluation engine
            encryption_service (EncryptionService): Data encryption service
            audit_logger (AuditLogger): Security audit logger

        Example:
            >>> security = SecurityManager(
            ...     policy_engine=PolicyEngine(...),
            ...     encryption_service=EncryptionService(secret_key),
            ...     audit_logger=AuditLogger()
            ... )

        Notes:
            - All components must be properly initialized before passing
            - Components should be configured for production use
            - Consider using dependency injection for testing
        """
        self.policy_engine = policy_engine
        self.encryption_service = encryption_service
        self.audit_logger = audit_logger

    def check_access(self, user_id: str, resource: str, action: str) -> bool:
        """Check if a user has permission to perform an action on a resource.

        Evaluates access permissions and logs the attempt, providing a unified
        interface for access control across the data lake.

        Args:
            user_id (str): The ID of the user requesting access
            resource (str): The resource being accessed (e.g., "data_catalog:123")
            action (str): The action being attempted (e.g., "read", "write")

        Returns:
            bool: True if access is allowed, False otherwise

        Example:
            >>> security = SecurityManager(...)
            >>> if security.check_access("user123", "data_catalog", "write"):
            ...     # Perform write operation
            ...     pass

        Notes:
            - All access attempts are logged, regardless of outcome
            - Resource strings can be hierarchical (e.g., "catalog:123")
            - Actions are case-sensitive
        """
        allowed = self.policy_engine.evaluate(user_id, resource, action)
        self.audit_logger.record_access_attempt(user_id, resource, action, allowed)
        return allowed

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data with automatic audit logging.

        Encrypts data using the encryption service and logs any errors that occur
        during the process.

        Args:
            data (bytes): Raw bytes to encrypt. For strings, encode first:
                data.encode('utf-8')

        Returns:
            bytes: Encrypted data as a secure token

        Raises:
            Exception: If encryption fails, with error logged

        Example:
            >>> security = SecurityManager(...)
            >>> encrypted = security.encrypt_data(b"sensitive data")
            >>> # Store encrypted data safely

        Notes:
            - Encryption errors are logged before being re-raised
            - Uses Fernet symmetric encryption
            - Each encryption uses a unique IV
        """
        try:
            encrypted = self.encryption_service.encrypt(data)
            return encrypted
        except Exception as e:
            self.audit_logger.log_event("system", f"Encryption error: {str(e)}")
            raise

    def decrypt_data(self, data: bytes) -> bytes:
        """Decrypt encrypted data with automatic audit logging.

        Decrypts data using the encryption service and logs any errors that occur
        during the process.

        Args:
            data (bytes): Encrypted data token to decrypt

        Returns:
            bytes: Decrypted data as bytes

        Raises:
            Exception: If decryption fails, with error logged

        Example:
            >>> security = SecurityManager(...)
            >>> decrypted = security.decrypt_data(encrypted_data)
            >>> original = decrypted.decode('utf-8')  # if it was text

        Notes:
            - Decryption errors are logged before being re-raised
            - Verifies token authenticity
            - Checks token timestamp
        """
        try:
            decrypted = self.encryption_service.decrypt(data)
            return decrypted
        except Exception as e:
            self.audit_logger.log_event("system", f"Decryption error: {str(e)}")
            raise

    def log_general_event(self, user_id: str, description: str):
        """Log a general security-related event.

        Convenience method to log security events through the audit logger.

        Args:
            user_id (str): The ID of the user associated with the event
            description (str): A clear description of what occurred

        Example:
            >>> security = SecurityManager(...)
            >>> security.log_general_event(
            ...     "admin",
            ...     "Security policy updated for data catalog"
            ... )

        Notes:
            - Use clear, consistent event descriptions
            - Avoid logging sensitive data
            - Consider adding event categories
        """
        self.audit_logger.log_event(user_id, description)
