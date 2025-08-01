"""
⚡ AURA Error Handling
Production-grade error handling with recovery strategies.
"""

from .exceptions import (
    AuraError,
    ConfigurationError,
    ValidationError,
    IntegrationError,
    ResourceError,
    SecurityError,
    StateError
)
from .handlers import ErrorHandler, GlobalErrorHandler
from .recovery import RecoveryStrategy, ExponentialBackoff, LinearBackoff
from .circuit_breaker import CircuitBreaker, CircuitState, resilient_operation

__all__ = [
    # Exceptions
    "AuraError",
    "ConfigurationError",
    "ValidationError",
    "IntegrationError",
    "ResourceError",
    "SecurityError",
    "StateError",
    
    # Handlers
    "ErrorHandler",
    "GlobalErrorHandler",
    
    # Recovery
    "RecoveryStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "resilient_operation",
]