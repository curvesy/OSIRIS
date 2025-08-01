"""
Retry Policy Implementation for Agent Resilience

Provides configurable retry mechanisms with exponential backoff
and jitter for handling transient failures.
"""

import asyncio
import random
from typing import TypeVar, Callable, Optional, List, Type
from dataclasses import dataclass
from datetime import timedelta
from abc import ABC, abstractmethod

from opentelemetry import trace
import structlog

# Type variable for generic return types
T = TypeVar('T')

tracer = trace.get_tracer(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    
    max_attempts: int = 3
    initial_delay: timedelta = timedelta(seconds=1)
    max_delay: timedelta = timedelta(seconds=60)
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Optional[List[Type[Exception]]] = None
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_delay.total_seconds() <= 0:
            raise ValueError("initial_delay must be positive")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if self.exponential_base <= 1:
            raise ValueError("exponential_base must be > 1")


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry policy."""
        config.validate()
        self.config = config
        self.logger = structlog.get_logger()
    
    @abstractmethod
    def calculate_delay(self, attempt: int) -> timedelta:
        """Calculate delay before next retry attempt."""
        pass
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry after an exception."""
        # Check attempt limit
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception is retryable
        if self.config.retryable_exceptions:
            return any(
                isinstance(exception, exc_type)
                for exc_type in self.config.retryable_exceptions
            )
        
        # Default: retry on any exception
        return True
    
    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function with retry policy.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            Exception: The last exception if all retries fail
        """
        attempt = 0
        last_exception = None
        
        with tracer.start_as_current_span(
            "retry_policy.execute",
            attributes={
                "retry.max_attempts": self.config.max_attempts,
                "retry.policy": type(self).__name__
            }
        ) as span:
            while attempt < self.config.max_attempts:
                attempt += 1
                
                try:
                    # Execute the function
                    result = await func(*args, **kwargs)
                    
                    # Success
                    span.set_attribute("retry.attempts", attempt)
                    span.set_attribute("retry.succeeded", True)
                    
                    if attempt > 1:
                        self.logger.info(
                            "Retry succeeded",
                            attempt=attempt,
                            max_attempts=self.config.max_attempts
                        )
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if not self.should_retry(e, attempt):
                        span.set_attribute("retry.attempts", attempt)
                        span.set_attribute("retry.succeeded", False)
                        span.set_attribute("retry.stopped_reason", "non_retryable")
                        raise
                    
                    # Check if this was the last attempt
                    if attempt >= self.config.max_attempts:
                        span.set_attribute("retry.attempts", attempt)
                        span.set_attribute("retry.succeeded", False)
                        span.set_attribute("retry.stopped_reason", "max_attempts")
                        
                        self.logger.error(
                            "All retry attempts failed",
                            attempts=attempt,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                        raise
                    
                    # Calculate delay
                    delay = self.calculate_delay(attempt)
                    
                    self.logger.warning(
                        "Retry attempt failed, waiting before next attempt",
                        attempt=attempt,
                        max_attempts=self.config.max_attempts,
                        delay_seconds=delay.total_seconds(),
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    
                    # Wait before retry
                    await asyncio.sleep(delay.total_seconds())
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Retry loop exited unexpectedly")


class ExponentialBackoff(RetryPolicy):
    """
    Exponential backoff retry policy with optional jitter.
    
    Delay calculation: min(initial_delay * (base ^ attempt), max_delay)
    With jitter: delay * random(0.5, 1.5)
    """
    
    def calculate_delay(self, attempt: int) -> timedelta:
        """Calculate exponential backoff delay with optional jitter."""
        # Calculate base delay
        delay_seconds = self.config.initial_delay.total_seconds() * (
            self.config.exponential_base ** (attempt - 1)
        )
        
        # Apply max delay cap
        delay_seconds = min(delay_seconds, self.config.max_delay.total_seconds())
        
        # Apply jitter if enabled
        if self.config.jitter:
            # Add random jitter between 50% and 150% of delay
            jitter_factor = random.uniform(0.5, 1.5)
            delay_seconds *= jitter_factor
        
        return timedelta(seconds=delay_seconds)


class LinearBackoff(RetryPolicy):
    """
    Linear backoff retry policy.
    
    Delay calculation: min(initial_delay * attempt, max_delay)
    """
    
    def calculate_delay(self, attempt: int) -> timedelta:
        """Calculate linear backoff delay."""
        delay_seconds = self.config.initial_delay.total_seconds() * attempt
        delay_seconds = min(delay_seconds, self.config.max_delay.total_seconds())
        
        if self.config.jitter:
            jitter_factor = random.uniform(0.8, 1.2)
            delay_seconds *= jitter_factor
        
        return timedelta(seconds=delay_seconds)


class FixedDelay(RetryPolicy):
    """
    Fixed delay retry policy.
    
    Always uses the same delay between retries.
    """
    
    def calculate_delay(self, attempt: int) -> timedelta:
        """Return fixed delay."""
        delay = self.config.initial_delay
        
        if self.config.jitter:
            # Add small jitter even for fixed delay
            jitter_seconds = random.uniform(-0.1, 0.1) * delay.total_seconds()
            delay = timedelta(seconds=delay.total_seconds() + jitter_seconds)
        
        return delay


class RetryWithBackoff:
    """
    Decorator for adding retry logic to async functions.
    
    Usage:
        @RetryWithBackoff(max_attempts=3)
        async def my_function():
            # Function that might fail
            pass
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """Initialize retry decorator."""
        self.config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=timedelta(seconds=initial_delay),
            max_delay=timedelta(seconds=max_delay),
            exponential_base=exponential_base,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions
        )
        self.policy = ExponentialBackoff(self.config)
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap function with retry logic."""
        async def wrapper(*args, **kwargs):
            return await self.policy.execute(func, *args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        
        return wrapper