"""
AURA Common Atomic Components Library

A collection of atomic, single-responsibility components that form
the foundation of the AURA Intelligence platform. Each component:
- Has a single, clear responsibility
- Is under 150 lines of code
- Uses dependency injection
- Is independently testable
- Follows strict interface contracts
"""

from .base.component import AtomicComponent, ComponentMetrics
from .base.protocols import (
    ProcessorProtocol,
    ConnectorProtocol,
    HandlerProtocol,
    ConfigProtocol
)

__version__ = "1.0.0"

__all__ = [
    "AtomicComponent",
    "ComponentMetrics",
    "ProcessorProtocol",
    "ConnectorProtocol", 
    "HandlerProtocol",
    "ConfigProtocol"
]