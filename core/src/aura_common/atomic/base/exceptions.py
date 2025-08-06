"""
Exceptions for atomic components
"""

from ...errors import AuraError


class ComponentError(AuraError):
    """Base exception for component errors."""
    pass


class ComponentInitializationError(ComponentError):
    """Raised when component initialization fails."""
    pass


class ComponentProcessingError(ComponentError):
    """Raised when component processing fails."""
    pass


class ComponentConfigurationError(ComponentError):
    """Raised when component configuration is invalid."""
    pass


class ComponentValidationError(ComponentError):
    """Raised when component validation fails."""
    pass


class ComponentCleanupError(ComponentError):
    """Raised when component cleanup fails."""
    pass


class ComponentTimeoutError(ComponentError):
    """Raised when component operation times out."""
    pass