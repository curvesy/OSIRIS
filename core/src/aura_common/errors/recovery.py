"""
⚡ Recovery Strategies
Retry and recovery patterns for resilient operations.
"""

from typing import TypeVar, Callable, Optional, Any, Union
from abc import ABC, abstractmethod
import asyncio
import random
import time
from functools import wraps

T = TypeVar('T')


class RecoveryStrategy(ABC):
    """Base class for recovery strategies."""
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        pass
    
    @abstractmethod
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if we should retry."""
        pass


class ExponentialBackoff(RecoveryStrategy):
    """Exponential backoff with jitter."""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_attempts: int = 5,
        jitter: bool = True
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Check if we should retry."""
        return attempt < self.max_attempts


class LinearBackoff(RecoveryStrategy):
    """Linear backoff strategy."""
    
    def __init__(
        self,
        delay: float = 1.0,
        max_attempts: int = 3
    ):
        self.delay = delay
        self.max_attempts = max_attempts
    
    def get_delay(self, attempt: int) -> float:
        """Get constant delay."""
        return self.delay
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Check if we should retry."""
        return attempt < self.max_attempts


def with_retry(
    strategy: Optional[RecoveryStrategy] = None,
    *,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for automatic retry with recovery strategy.
    
    Args:
        strategy: Recovery strategy to use
        exceptions: Exception types to catch
        on_retry: Callback on each retry
        
    Example:
        ```python
        @with_retry(
            strategy=ExponentialBackoff(base_delay=0.5),
            exceptions=(ConnectionError, TimeoutError)
        )
        async def fetch_data():
            return await api.get_data()
        ```
    """
    if strategy is None:
        strategy = ExponentialBackoff()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            last_error: Optional[Exception] = None
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    
                    if not strategy.should_retry(attempt, e):
                        raise
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    delay = strategy.get_delay(attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            last_error: Optional[Exception] = None
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    
                    if not strategy.should_retry(attempt, e):
                        raise
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    delay = strategy.get_delay(attempt)
                    time.sleep(delay)
                    attempt += 1
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator