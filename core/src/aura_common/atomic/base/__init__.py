"""Base classes and protocols for atomic components."""

from .component import AtomicComponent, ComponentMetrics
from .protocols import (
    ProcessorProtocol,
    ConnectorProtocol,
    HandlerProtocol,
    ConfigProtocol
)
from .exceptions import (
    ComponentError,
    ConfigurationError,
    ProcessingError,
    ValidationError
)

__all__ = [
    # Base classes
    "AtomicComponent",
    "ComponentMetrics",
    
    # Protocols
    "ProcessorProtocol",
    "ConnectorProtocol",
    "HandlerProtocol",
    "ConfigProtocol",
    
    # Exceptions
    "ComponentError",
    "ConfigurationError",
    "ProcessingError",
    "ValidationError"
]