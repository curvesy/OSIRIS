"""
AURA Common Utilities

Shared utilities and components for AURA Intelligence system.
"""

__version__ = "0.1.0"

from .logging import get_logger, with_correlation_id
from .atomic import AtomicComponent, ComponentError
from .errors import (
    AuraError,
    TDAError,
    AgentError,
    OrchestrationError,
    ConfigurationError,
    ValidationError
)
# Import resilient_operation from error_utils.py
from .error_utils import resilient_operation
from .config import is_feature_enabled, get_config_value

__all__ = [
    # Logging
    "get_logger",
    "with_correlation_id",
    
    # Atomic components
    "AtomicComponent",
    "ComponentError",
    
    # Error handling
    "AuraError",
    "resilient_operation",
    "TDAError",
    "AgentError",
    "OrchestrationError",
    "ConfigurationError",
    "ValidationError",
    
    # Configuration
    "is_feature_enabled",
    "get_config_value",
]