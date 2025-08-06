"""
Circuit breaker atomic component.

Implements the circuit breaker pattern for fault isolation
and preventing cascading failures in distributed systems.
"""

from typing import TypeVar, Callable, Optional, Dict, Any, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, timezone
import asyncio

from ..base import AtomicComponent
from ..base.exceptions import CircuitBreakerError

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    error_types: tuple[type[Exception], ...] = (Exception,)
    exclude_types: tuple[type[Exception], ...] = ()
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be positive")


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: list[tuple[CircuitState, datetime]] = field(default_factory=list)


@dataclass
class CircuitBreakerResult:
    """Result of circuit breaker operation."""
    
    success: bool
    result: Any
    state: CircuitState
    execution_time_ms: float
    error: Optional[Exception] = None
    rejected: bool = False


class CircuitBreaker(AtomicComponent[Callable[[], Awaitable[T]], CircuitBreakerResult, CircuitBreakerConfig]):
    """
    Atomic component implementing circuit breaker pattern.
    
    Features:
    - Three states: Closed, Open, Half-Open
    - Configurable failure thresholds
    - Automatic recovery testing
    - Detailed statistics
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._half_open_calls = 0
        self._last_state_change = datetime.now(timezone.utc)
    
    def _validate_config(self) -> None:
        """Validate circuit breaker configuration."""
        self.config.validate()
    
    async def _process(self, operation: Callable[[], Awaitable[T]]) -> CircuitBreakerResult:
        """
        Execute operation with circuit breaker protection.
        
        Args:
            operation: Async function to protect
            
        Returns:
            CircuitBreakerResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        
        # Check if we should transition states
        self._check_state_transition()
        
        # Handle based on current state
        if self._state == CircuitState.OPEN:
            # Circuit is open, reject call
            self._stats.rejected_calls += 1
            self._stats.total_calls += 1
            
            self.logger.warning(
                "Circuit breaker is OPEN, rejecting call",
                consecutive_failures=self._stats.consecutive_failures,
                last_failure=self._stats.last_failure_time
            )
            
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN",
                reset_after=(self.config.timeout_seconds - 
                           (datetime.now(timezone.utc) - self._stats.last_failure_time).total_seconds()),
                failure_count=self._stats.consecutive_failures,
                component_name=self.name
            )
        
        elif self._state == CircuitState.HALF_OPEN:
            # Check if we've exceeded half-open call limit
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._stats.rejected_calls += 1
                self._stats.total_calls += 1
                
                return CircuitBreakerResult(
                    success=False,
                    result=None,
                    state=self._state,
                    execution_time_ms=0,
                    rejected=True
                )
            
            self._half_open_calls += 1
        
        # Try to execute operation
        try:
            result = await operation()
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Success
            self._record_success()
            
            return CircuitBreakerResult(
                success=True,
                result=result,
                state=self._state,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Check if this error should trip the breaker
            if self._should_count_failure(e):
                self._record_failure()
            
            return CircuitBreakerResult(
                success=False,
                result=None,
                state=self._state,
                execution_time_ms=execution_time,
                error=e
            )
    
    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if (self._stats.last_failure_time and 
                (datetime.now(timezone.utc) - self._stats.last_failure_time).total_seconds() >= self.config.timeout_seconds):
                self._transition_to(CircuitState.HALF_OPEN)
                
        elif self._state == CircuitState.HALF_OPEN:
            # Check if we should close or re-open
            if self._stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
            elif self._stats.consecutive_failures >= 1:  # Single failure in half-open reopens
                self._transition_to(CircuitState.OPEN)
    
    def _record_success(self) -> None:
        """Record successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        self._stats.last_success_time = datetime.now(timezone.utc)
        
        # Check state transitions
        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
    
    def _record_failure(self) -> None:
        """Record failed call."""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._stats.last_failure_time = datetime.now(timezone.utc)
        
        # Check state transitions
        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now(timezone.utc)
        
        # Reset half-open counter
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        
        # Log transition
        self.logger.info(
            f"Circuit breaker state transition",
            old_state=old_state.value,
            new_state=new_state.value,
            consecutive_failures=self._stats.consecutive_failures,
            consecutive_successes=self._stats.consecutive_successes
        )
        
        # Record in stats
        self._stats.state_changes.append((new_state, datetime.now(timezone.utc)))
    
    def _should_count_failure(self, error: Exception) -> bool:
        """Check if error should count as failure."""
        # Check exclusions first
        if isinstance(error, self.config.exclude_types):
            return False
        
        # Check if error matches configured types
        return isinstance(error, self.config.error_types)
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = (self._stats.successful_calls / self._stats.total_calls 
                       if self._stats.total_calls > 0 else 1.0)
        
        return {
            "state": self._state.value,
            "total_calls": self._stats.total_calls,
            "successful_calls": self._stats.successful_calls,
            "failed_calls": self._stats.failed_calls,
            "rejected_calls": self._stats.rejected_calls,
            "success_rate": success_rate,
            "consecutive_failures": self._stats.consecutive_failures,
            "consecutive_successes": self._stats.consecutive_successes,
            "last_failure": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
            "last_success": self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
            "state_changes": len(self._stats.state_changes)
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._half_open_calls = 0
        self.logger.info("Circuit breaker manually reset")