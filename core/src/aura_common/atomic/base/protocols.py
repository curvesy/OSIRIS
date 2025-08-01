"""
Protocol definitions for atomic components.

These protocols define the contracts that components must follow,
enabling type-safe composition and dependency injection.
"""

from typing import Protocol, TypeVar, Dict, Any, Optional, List
from abc import abstractmethod


class ConfigProtocol(Protocol):
    """Base protocol for component configurations."""
    
    @abstractmethod
    def validate(self) -> None:
        """Validate configuration values."""
        ...
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        ...


class ProcessorProtocol(Protocol):
    """Protocol for data processing components."""
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process input data and return result."""
        ...
    
    @abstractmethod
    async def validate_input(self, data: Any) -> bool:
        """Validate input data before processing."""
        ...


class ConnectorProtocol(Protocol):
    """Protocol for external system connectors."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to external system."""
        ...
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to external system."""
        ...
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connector is connected."""
        ...


class HandlerProtocol(Protocol):
    """Protocol for event/error handlers."""
    
    @abstractmethod
    async def handle(self, event: Any) -> Any:
        """Handle an event or error."""
        ...
    
    @abstractmethod
    def can_handle(self, event: Any) -> bool:
        """Check if handler can process the event."""
        ...


class CacheProtocol(Protocol):
    """Protocol for caching implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        ...
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all values from cache."""
        ...


class QueueProtocol(Protocol):
    """Protocol for queue implementations."""
    
    @abstractmethod
    async def enqueue(self, item: Any) -> None:
        """Add item to queue."""
        ...
    
    @abstractmethod
    async def dequeue(self) -> Optional[Any]:
        """Remove and return item from queue."""
        ...
    
    @abstractmethod
    async def peek(self) -> Optional[Any]:
        """View next item without removing."""
        ...
    
    @abstractmethod
    async def size(self) -> int:
        """Get current queue size."""
        ...


class VectorStoreProtocol(Protocol):
    """Protocol for vector storage implementations."""
    
    @abstractmethod
    async def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Insert or update vectors."""
        ...
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        ...
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        ...


class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""
    
    @abstractmethod
    def increment(self, metric: str, value: float = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        ...
    
    @abstractmethod
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        ...
    
    @abstractmethod
    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        ...
    
    @abstractmethod
    def timer(self, metric: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        ...