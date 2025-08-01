"""
🔧 AURA Common Libraries
Shared utilities and cross-cutting concerns for AURA Intelligence.

This package provides:
- Structured logging with correlation
- Error handling and recovery
- Configuration management
- Cryptographic utilities
- State management patterns
- Validation helpers
"""

from .logging import get_logger, get_correlation_id, with_correlation_id
from .errors import AuraError, CircuitBreaker, resilient_operation
from .config import AuraConfig, ConfigManager, get_config

__version__ = "2.0.0"

__all__ = [
    # Logging
    "get_logger",
    "get_correlation_id", 
    "with_correlation_id",
    
    # Errors
    "AuraError",
    "CircuitBreaker",
    "resilient_operation",
    
    # Config
    "AuraConfig",
    "ConfigManager",
    "get_config",
]