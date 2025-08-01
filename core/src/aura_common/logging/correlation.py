"""
🔗 Correlation ID Management
Thread-safe correlation ID tracking across async contexts.
"""

from typing import Optional, TypeVar, Callable, Any
from contextvars import ContextVar
from functools import wraps
import uuid
from collections.abc import Coroutine

# Context variable for correlation ID
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

T = TypeVar('T')


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set correlation ID in current context."""
    _correlation_id.set(correlation_id)


def with_correlation_id(
    correlation_id: Optional[str] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to set correlation ID for function execution.
    
    Args:
        correlation_id: Specific ID to use, or None to generate
        
    Example:
        ```python
        @with_correlation_id()
        async def process_request(request):
            # Correlation ID is automatically set
            logger.info("Processing request")
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            cid = correlation_id or generate_correlation_id()
            token = _correlation_id.set(cid)
            try:
                return await func(*args, **kwargs)
            finally:
                _correlation_id.reset(token)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            cid = correlation_id or generate_correlation_id()
            token = _correlation_id.set(cid)
            try:
                return func(*args, **kwargs)
            finally:
                _correlation_id.reset(token)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


class CorrelationContext:
    """
    Context manager for correlation ID scope.
    
    Example:
        ```python
        with CorrelationContext() as correlation_id:
            logger.info("Starting operation", correlation_id=correlation_id)
            await process_data()
        ```
    """
    
    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize with optional correlation ID."""
        self.correlation_id = correlation_id or generate_correlation_id()
        self._token: Optional[Any] = None
    
    def __enter__(self) -> str:
        """Enter context and set correlation ID."""
        self._token = _correlation_id.set(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, *args: Any) -> None:
        """Exit context and reset correlation ID."""
        if self._token:
            _correlation_id.reset(self._token)
    
    async def __aenter__(self) -> str:
        """Async enter context."""
        self._token = _correlation_id.set(self.correlation_id)
        return self.correlation_id
    
    async def __aexit__(self, *args: Any) -> None:
        """Async exit context."""
        if self._token:
            _correlation_id.reset(self._token)


def propagate_correlation_id(headers: dict[str, str]) -> None:
    """
    Extract and set correlation ID from headers.
    
    Args:
        headers: HTTP headers containing X-Correlation-ID
    """
    correlation_id = headers.get('X-Correlation-ID') or headers.get('x-correlation-id')
    if correlation_id:
        set_correlation_id(correlation_id)


def inject_correlation_id(headers: dict[str, str]) -> dict[str, str]:
    """
    Inject current correlation ID into headers.
    
    Args:
        headers: Headers dict to update
        
    Returns:
        Updated headers dict
    """
    correlation_id = get_correlation_id()
    if correlation_id:
        headers['X-Correlation-ID'] = correlation_id
    return headers