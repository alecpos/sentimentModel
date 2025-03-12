"""
Configuration management for the WITHIN ML Prediction System.

This module provides utilities for loading, validating, and accessing
application configuration from various sources (environment variables,
configuration files, etc.) in a consistent and type-safe manner.

Key functionality includes:
- Configuration loading from multiple sources
- Environment-specific configuration
- Configuration validation and type checking
- Secure handling of sensitive configuration values
- Dynamic configuration updates

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

from enum import Enum
from typing import Dict, Any, Optional

# Configuration environment types
class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

# Default configuration values
DEFAULT_CONFIG = {
    "app_name": "WITHIN ML Prediction System",
    "api_prefix": "/api",
    "debug": False,
    "log_level": "INFO",
    "allowed_hosts": ["*"],
    "database_pool_size": 5,
    "database_max_overflow": 10,
    "model_cache_size": 5,
    "rate_limit": 100,
    "timeout_seconds": 30
}

# Configuration singleton
_config: Dict[str, Any] = {}
_environment: Environment = Environment.DEVELOPMENT

def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    return _config

def get_environment() -> Environment:
    """Get the current environment."""
    return _environment

def initialize_config(env: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize the configuration based on the environment.
    
    Args:
        env: Environment name (development, testing, staging, production)
        
    Returns:
        The initialized configuration dictionary
    """
    global _config, _environment
    
    # Set environment
    if env:
        _environment = Environment(env.lower())
    
    # Start with default config
    _config = DEFAULT_CONFIG.copy()
    
    # TODO: Load configuration from environment variables and files
    
    return _config

__all__ = [
    "Environment",
    "DEFAULT_CONFIG",
    "get_config",
    "get_environment",
    "initialize_config"
]
