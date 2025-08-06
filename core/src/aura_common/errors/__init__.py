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
    StateError,
    TDAError,
    AgentError,
    OrchestrationError
)
from .handlers import ErrorHandler, GlobalErrorHandler
from .recovery import RecoveryStrategy, ExponentialBackoff, LinearBackoff
from .circuit_breaker import CircuitBreaker, CircuitState
# Note: resilient_operation is defined in ../errors.py, not imported here to avoid circular import

__all__ = [
    # Exceptions
    "AuraError",
    "ConfigurationError",
    "ValidationError",
    "IntegrationError",
    "ResourceError",
    "SecurityError",
    "StateError",
    "TDAError",
    "AgentError",
    "OrchestrationError",
    
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

]