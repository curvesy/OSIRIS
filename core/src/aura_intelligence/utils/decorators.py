"""
Decorators for AURA Intelligence.

Provides production-ready decorators for error handling, retries, circuit breakers,
rate limiting, and performance monitoring.
"""

import asyncio
import functools
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""
    failure_threshold: int = Field(default=5, ge=1, description="Number of failures before opening")
    recovery_timeout: int = Field(default=60, ge=1, description="Seconds before attempting recovery")
    expected_exception: type[Exception] = Field(default=Exception, description="Exception type to catch")
    success_threshold: int = Field(default=2, ge=1, description="Successes needed to close circuit")


class RetryConfig(BaseModel):
    """Configuration for retry decorator."""
    max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts")
    delay: float = Field(default=1.0, gt=0, description="Initial delay between retries")
    backoff_factor: float = Field(default=2.0, ge=1.0, description="Backoff multiplier")
    max_delay: float = Field(default=60.0, gt=0, description="Maximum delay between retries")
    exceptions: tuple[type[Exception], ...] = Field(
        default=(Exception,),
        description="Exception types to retry"
    )


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_attempt_time: Optional[datetime] = None
    
    def call_succeeded(self) -> None:
        """Record successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker closed after successful recovery")
    
    def call_failed(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_attempt_call(self) -> bool:
        """Check if call can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
                logger.info("Circuit breaker entering half-open state")
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def __call__(self, func: F) -> F:
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_attempt_call():
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self.call_succeeded()
                return result
            except self.config.expected_exception as e:
                self.call_failed()
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_attempt_call():
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                self.call_succeeded()
                return result
            except self.config.expected_exception as e:
                self.call_failed()
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type[Exception] = Exception,
    success_threshold: int = 2
) -> Callable[[F], F]:
    """
    Circuit breaker decorator.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        expected_exception: Exception type to catch
        success_threshold: Successes needed to close circuit
    
    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def external_api_call():
            ...
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        success_threshold=success_threshold
    )
    breaker = CircuitBreaker(config)
    return breaker


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[F], F]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to retry
    
    Example:
        @retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
        def unstable_operation():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {type(e).__name__}: {str(e)}"
                        )
                        time.sleep(min(current_delay, max_delay))
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            raise last_exception  # type: ignore
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {type(e).__name__}: {str(e)}"
                        )
                        await asyncio.sleep(min(current_delay, max_delay))
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            raise last_exception  # type: ignore
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def rate_limit(calls: int = 10, period: float = 60.0) -> Callable[[F], F]:
    """
    Rate limiting decorator.
    
    Args:
        calls: Maximum number of calls allowed
        period: Time period in seconds
    
    Example:
        @rate_limit(calls=10, period=60.0)  # 10 calls per minute
        def api_endpoint():
            ...
    """
    call_times: dict[str, list[float]] = defaultdict(list)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.time()
            func_name = func.__name__
            
            # Remove old calls outside the period
            call_times[func_name] = [
                t for t in call_times[func_name]
                if now - t < period
            ]
            
            if len(call_times[func_name]) >= calls:
                raise Exception(
                    f"Rate limit exceeded for {func_name}: "
                    f"{calls} calls per {period} seconds"
                )
            
            call_times[func_name].append(now)
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.time()
            func_name = func.__name__
            
            # Remove old calls outside the period
            call_times[func_name] = [
                t for t in call_times[func_name]
                if now - t < period
            ]
            
            if len(call_times[func_name]) >= calls:
                raise Exception(
                    f"Rate limit exceeded for {func_name}: "
                    f"{calls} calls per {period} seconds"
                )
            
            call_times[func_name].append(now)
            return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Timeout decorator for async functions.
    
    Args:
        seconds: Timeout in seconds
    
    Example:
        @timeout(30.0)
        async def slow_operation():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"{func.__name__} timed out after {seconds} seconds"
                )
        
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f"@timeout can only be used on async functions")
        
        return wrapper  # type: ignore
    
    return decorator


def log_performance(threshold_ms: float = 1000.0) -> Callable[[F], F]:
    """
    Log function performance and warn if slow.
    
    Args:
        threshold_ms: Warning threshold in milliseconds
    
    Example:
        @log_performance(threshold_ms=500)
        def process_data():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                log_level = logging.WARNING if duration_ms > threshold_ms else logging.DEBUG
                logger.log(
                    log_level,
                    f"{func.__name__} took {duration_ms:.2f}ms"
                )
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                log_level = logging.WARNING if duration_ms > threshold_ms else logging.DEBUG
                logger.log(
                    log_level,
                    f"{func.__name__} took {duration_ms:.2f}ms"
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def handle_errors(
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
) -> Callable[[F], F]:
    """
    Generic error handling decorator.
    
    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
        reraise: Whether to re-raise the exception
    
    Example:
        @handle_errors(default_return=[], log_errors=True)
        def get_items():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {type(e).__name__}: {str(e)}",
                        exc_info=True
                    )
                if reraise:
                    raise
                return default_return
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {type(e).__name__}: {str(e)}",
                        exc_info=True
                    )
                if reraise:
                    raise
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


@contextmanager
def timer(name: str = "Operation", logger_func: Optional[Callable] = None):
    """
    Context manager for timing operations.
    
    Args:
        name: Name of the operation being timed
        logger_func: Optional logging function (defaults to logger.info)
    
    Example:
        with timer("Data processing"):
            process_large_dataset()
    """
    if logger_func is None:
        logger_func = logger.info
    
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger_func(f"{name} completed in {duration:.2f} seconds")