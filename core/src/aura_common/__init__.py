"""
AURA Common - Shared utilities and base classes

This package provides common functionality used across AURA Intelligence:
- Base classes for atomic components
- Logging utilities
- Error handling decorators
- Configuration helpers
"""

__version__ = "1.0.0"
__all__ = [
    "get_logger",
    "AtomicComponent",
    "ComponentError",
    "resilient_operation",
    "is_feature_enabled",
    "with_correlation_id"
]

from .logging import get_logger, with_correlation_id
from .atomic.base import AtomicComponent
from .atomic.base.exceptions import ComponentError
from .errors import resilient_operation
from .config import is_feature_enabled