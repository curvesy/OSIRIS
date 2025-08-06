"""
⚡ Circuit Breaker Pattern
Prevents cascading failures in distributed systems.
"""

from typing import TypeVar, Callable, Optional, Any, Union
from enum import Enum, auto
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import asyncio
from functools import wraps
import time

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject calls
    HALF_OPEN = auto()   # Testing if recovered


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: list[tuple[datetime, CircuitState]] = field(default_factory=list)
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return 1.0 - self.failure_rate


class CircuitBreaker:
    """
    Circuit breaker implementation with async support.
    
    Example:
        ```python
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=IntegrationError
        )
        
        @breaker
        async def call_external_service():
            return await external_api.call()
        ```
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
        success_threshold: int = 3,
        half_open_calls: int = 1,
        name: Optional[str] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying half-open
            expected_exception: Exception type to catch
            success_threshold: Successes needed to close from half-open
            half_open_calls: Max concurrent calls in half-open state
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.half_open_calls = half_open_calls
        self.name = name or "CircuitBreaker"
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_semaphore = asyncio.Semaphore(half_open_calls)
        self._state_lock = asyncio.Lock()
        self._last_attempt_time = time.time()
    
    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset from open state."""
        return (
            self._state == CircuitState.OPEN and
            time.time() - self._last_attempt_time >= self.recovery_timeout
        )
    
    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._state_lock:
            self._stats.successful_calls += 1
            self._stats.total_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.now(timezone.utc)
            
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    await self._change_state(CircuitState.CLOSED)
    
    async def _record_failure(self, error: Exception) -> None:
        """Record failed call."""
        async with self._state_lock:
            self._stats.failed_calls += 1
            self._stats.total_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = datetime.now(timezone.utc)
            
            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    await self._change_state(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                await self._change_state(CircuitState.OPEN)
    
    async def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit state."""
        if self._state != new_state:
            self._state = new_state
            self._stats.state_changes.append((datetime.now(timezone.utc), new_state))
            self._last_attempt_time = time.time()
            
            # Log state change
            from ..logging import get_logger
            logger = get_logger(__name__)
            logger.info(
                f"Circuit breaker state changed",
                circuit_name=self.name,
                old_state=self._state.name,
                new_state=new_state.name,
                stats=self._stats.__dict__
            )
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for protecting functions."""
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Check if we should attempt reset
            if self._should_attempt_reset():
                async with self._state_lock:
                    if self._state == CircuitState.OPEN:
                        await self._change_state(CircuitState.HALF_OPEN)
            
            # Check circuit state
            if self._state == CircuitState.OPEN:
                raise IntegrationError(
                    f"Circuit breaker is OPEN for {self.name}",
                    service=self.name,
                    details={'stats': self._stats.to_dict()}
                )
            
            # Handle half-open state with limited concurrency
            if self._state == CircuitState.HALF_OPEN:
                if not self._half_open_semaphore.locked():
                    async with self._half_open_semaphore:
                        return await self._execute_call(func, args, kwargs)
                else:
                    raise IntegrationError(
                        f"Circuit breaker is HALF_OPEN with max calls for {self.name}",
                        service=self.name
                    )
            
            # Normal execution for closed state
            return await self._execute_call(func, args, kwargs)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # Simplified sync version - just track basic stats
            try:
                result = func(*args, **kwargs)
                self._stats.successful_calls += 1
                self._stats.total_calls += 1
                return result
            except self.expected_exception as e:
                self._stats.failed_calls += 1
                self._stats.total_calls += 1
                raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    async def _execute_call(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict
    ) -> Any:
        """Execute the protected call."""
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except self.expected_exception as e:
            await self._record_failure(e)
            raise
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_attempt_time = time.time()


# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    **kwargs: Any
) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]


def resilient_operation(
    name: str,
    *,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[Exception] = Exception
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory for resilient operations.
    
    Example:
        ```python
        @resilient_operation(
            "external_api",
            failure_threshold=3,
            recovery_timeout=30
        )
        async def call_api():
            return await external_api.call()
        ```
    """
    breaker = get_circuit_breaker(
        name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )
    return breaker


# Import after class definitions to avoid circular imports
from .exceptions import IntegrationError