"""
Retry handler atomic component.

Implements configurable retry logic with various backoff strategies
and retry policies for resilient operations.
"""

from typing import TypeVar, Callable, Optional, Dict, Any, Awaitable
from dataclasses import dataclass
from enum import Enum
import asyncio
import random
from datetime import datetime, timedelta, timezone

from ..base import AtomicComponent
from ..base.exceptions import RetryableError, ComponentError

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry backoff strategies."""
    
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    initial_delay_ms: float = 100
    max_delay_ms: float = 30000
    backoff_factor: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    retry_on: tuple[type[Exception], ...] = (RetryableError, ConnectionError, TimeoutError)
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_delay_ms <= 0:
            raise ValueError("initial_delay_ms must be positive")
        if self.max_delay_ms < self.initial_delay_ms:
            raise ValueError("max_delay_ms must be >= initial_delay_ms")
        if self.backoff_factor <= 0:
            raise ValueError("backoff_factor must be positive")


@dataclass
class RetryAttempt:
    """Details of a single retry attempt."""
    
    attempt_number: int
    delay_ms: float
    error: Optional[Exception]
    timestamp: datetime
    succeeded: bool


@dataclass
class RetryResult:
    """Result of retry operation."""
    
    success: bool
    result: Any
    total_attempts: int
    attempts: list[RetryAttempt]
    total_duration_ms: float
    final_error: Optional[Exception] = None


class RetryHandler(AtomicComponent[Callable[[], Awaitable[T]], RetryResult, RetryConfig]):
    """
    Atomic component for retry logic.
    
    Features:
    - Multiple backoff strategies
    - Configurable retry conditions
    - Detailed attempt tracking
    - Jitter for thundering herd prevention
    """
    
    def _validate_config(self) -> None:
        """Validate retry configuration."""
        self.config.validate()
    
    async def _process(self, operation: Callable[[], Awaitable[T]]) -> RetryResult:
        """
        Execute operation with retry logic.
        
        Args:
            operation: Async function to retry
            
        Returns:
            RetryResult with execution details
        """
        attempts = []
        start_time = datetime.now(timezone.utc)
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt_start = datetime.now(timezone.utc)
            
            try:
                # Execute operation
                result = await operation()
                
                # Success!
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    delay_ms=0,
                    error=None,
                    timestamp=attempt_start,
                    succeeded=True
                )
                attempts.append(attempt)
                
                total_duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                return RetryResult(
                    success=True,
                    result=result,
                    total_attempts=attempt_num,
                    attempts=attempts,
                    total_duration_ms=total_duration
                )
                
            except Exception as e:
                # Check if error is retryable
                if not self._should_retry(e):
                    # Non-retryable error
                    attempt = RetryAttempt(
                        attempt_number=attempt_num,
                        delay_ms=0,
                        error=e,
                        timestamp=attempt_start,
                        succeeded=False
                    )
                    attempts.append(attempt)
                    
                    total_duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    
                    return RetryResult(
                        success=False,
                        result=None,
                        total_attempts=attempt_num,
                        attempts=attempts,
                        total_duration_ms=total_duration,
                        final_error=e
                    )
                
                # Calculate delay for next attempt
                delay_ms = self._calculate_delay(attempt_num) if attempt_num < self.config.max_attempts else 0
                
                # Record attempt
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    delay_ms=delay_ms,
                    error=e,
                    timestamp=attempt_start,
                    succeeded=False
                )
                attempts.append(attempt)
                
                # Log retry
                self.logger.warning(
                    f"Retry attempt {attempt_num}/{self.config.max_attempts} failed",
                    error=str(e),
                    delay_ms=delay_ms
                )
                
                # Wait before next attempt
                if attempt_num < self.config.max_attempts:
                    await asyncio.sleep(delay_ms / 1000)
        
        # All attempts exhausted
        total_duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        return RetryResult(
            success=False,
            result=None,
            total_attempts=self.config.max_attempts,
            attempts=attempts,
            total_duration_ms=total_duration,
            final_error=attempts[-1].error if attempts else None
        )
    
    def _should_retry(self, error: Exception) -> bool:
        """Check if error should trigger retry."""
        return isinstance(error, self.config.retry_on)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.initial_delay_ms
            
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.initial_delay_ms * attempt
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.initial_delay_ms * (self.config.backoff_factor ** (attempt - 1))
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            # Full jitter
            max_delay = self.config.initial_delay_ms * (self.config.backoff_factor ** (attempt - 1))
            delay = random.uniform(0, max_delay)
            
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            # Fibonacci sequence
            a, b = self.config.initial_delay_ms, self.config.initial_delay_ms
            for _ in range(attempt - 1):
                a, b = b, a + b
            delay = a
            
        else:
            delay = self.config.initial_delay_ms
        
        # Cap at max delay
        return min(delay, self.config.max_delay_ms)
    
    @classmethod
    def create_with_exponential_backoff(
        cls,
        max_attempts: int = 3,
        initial_delay_ms: float = 100,
        **kwargs
    ) -> 'RetryHandler':
        """Factory for exponential backoff retry handler."""
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay_ms=initial_delay_ms,
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            **kwargs
        )
        return cls("exponential_retry", config)
    
    def get_retry_stats(self, results: list[RetryResult]) -> Dict[str, Any]:
        """Calculate retry statistics from multiple results."""
        if not results:
            return {
                "total_operations": 0,
                "success_rate": 1.0,
                "avg_attempts": 0,
                "avg_duration_ms": 0
            }
        
        successful = sum(1 for r in results if r.success)
        total_attempts = sum(r.total_attempts for r in results)
        total_duration = sum(r.total_duration_ms for r in results)
        
        return {
            "total_operations": len(results),
            "success_rate": successful / len(results),
            "avg_attempts": total_attempts / len(results),
            "avg_duration_ms": total_duration / len(results),
            "retry_breakdown": {
                "succeeded_first_try": sum(1 for r in results if r.success and r.total_attempts == 1),
                "succeeded_with_retry": sum(1 for r in results if r.success and r.total_attempts > 1),
                "failed_all_attempts": sum(1 for r in results if not r.success)
            }
        }