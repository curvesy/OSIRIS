"""
Atomic components for AURA Intelligence

Provides base classes and utilities for building atomic, composable components.
"""

# Import AtomicComponent from the base subdirectory which has the 3-parameter version
from .base.component import AtomicComponent

# Import ComponentMetadata from atomic_base.py
from .atomic_base import ComponentMetadata

# Import exceptions from the base subdirectory
from .base.exceptions import (
    ComponentError,
    ComponentInitializationError,
    ComponentProcessingError,
    ComponentCleanupError,
    ComponentTimeoutError,
    ComponentValidationError,
    ComponentConfigurationError
)

__all__ = [
    "AtomicComponent", 
    "ComponentMetadata", 
    "ComponentError",
    "ComponentInitializationError",
    "ComponentProcessingError",
    "ComponentCleanupError",
    "ComponentTimeoutError",
    "ComponentValidationError",
    "ComponentConfigurationError"
]