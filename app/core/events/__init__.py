"""
Event handling components for the WITHIN ML Prediction System.

This module provides a standardized event system for communicating between
components of the ML system. It enables asynchronous message passing, event-driven
architecture patterns, and integration with external event sources and sinks.

Key functionality includes:
- Event publication and subscription
- Event routing and filtering
- Event persistence and replay
- Event-driven workflow triggers
- Monitoring and tracking of system events
"""

__version__ = "0.1.0"  # Pre-release version

# Package-level constants
EVENT_PRIORITY_LEVELS = ["low", "normal", "high", "critical"]
DEFAULT_EVENT_TTL = 86400  # seconds (1 day)
MAX_RETRY_COUNT = 3

# When implementations are added, they will be imported and exported here 