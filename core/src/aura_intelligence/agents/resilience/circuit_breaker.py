"""
Circuit Breaker Pattern for Agent Resilience

Implements the circuit breaker pattern to prevent cascading failures
in multi-agent systems with full observability.
"""

import asyncio
from enum import Enum
from typing import Optional, Callable, Any, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

# Type variable for generic return types
T = TypeVar('T')

# Get tracer and meter
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Circuit breaker metrics
circuit_state_gauge = meter.create_up_down_counter(
    name="circuit_breaker.state",
    description="Circuit breaker state (1=closed, 0=half-open, -1=open)",
    unit="1"
)

circuit_failures_counter = meter.create_counter(
    name="circuit_breaker.failures",
    description="Number of failures observed by circuit breaker",
    unit="1"
)

circuit_trips_counter = meter.create_counter(
    name="circuit_breaker.trips",
    description="Number of times circuit breaker has tripped",
    unit="1"
)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, circuit_name: str, state: CircuitState):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.state = state


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    name: str
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout: timedelta = timedelta(seconds=60)  # Time before half-open
    failure_rate_threshold: float = 0.5  # Failure rate to open
    min_calls: int = 10                # Min calls for rate calculation
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.failure_rate_threshold < 0 or self.failure_rate_threshold > 1:
            raise ValueError("failure_rate_threshold must be between 0 and 1")


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation with observability.
    
    Features:
    - Configurable failure thresholds
    - Automatic recovery testing
    - Full OpenTelemetry instrumentation
    - Thread-safe async implementation
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker."""
        config.validate()
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.state_changed_at = datetime.now()
        self._lock = asyncio.Lock()
        self.logger = structlog.get_logger().bind(circuit=config.name)
        
        # Update initial state metric
        self._update_state_metric()
    
    def _update_state_metric(self) -> None:
        """Update state metric."""
        state_value = {
            CircuitState.CLOSED: 1,
            CircuitState.HALF_OPEN: 0,
            CircuitState.OPEN: -1
        }[self.state]
        
        circuit_state_gauge.add(
            state_value,
            {"circuit.name": self.config.name, "state": self.state.value}
        )
    
    async def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.state != CircuitState.OPEN:
            return False
        
        time_since_change = datetime.now() - self.state_changed_at
        return time_since_change >= self.config.timeout
    
    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self.stats.last_success_time = datetime.now()
            
            # Check state transitions
            if self.state == CircuitState.HALF_OPEN:
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
    
    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = datetime.now()
            
            # Record failure metric
            circuit_failures_counter.add(
                1,
                {"circuit.name": self.config.name}
            )
            
            # Check state transitions
            if self.state == CircuitState.CLOSED:
                should_open = False
                
                # Check absolute threshold
                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    should_open = True
                
                # Check failure rate
                if (self.stats.total_calls >= self.config.min_calls and
                    self.stats.failure_rate >= self.config.failure_rate_threshold):
                    should_open = True
                
                if should_open:
                    await self._transition_to(CircuitState.OPEN)
                    circuit_trips_counter.add(
                        1,
                        {"circuit.name": self.config.name}
                    )
            
            elif self.state == CircuitState.HALF_OPEN:
                # Single failure in half-open returns to open
                await self._transition_to(CircuitState.OPEN)
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.state_changed_at = datetime.now()
        
        # Reset stats on close
        if new_state == CircuitState.CLOSED:
            self.stats = CircuitBreakerStats()
        
        self._update_state_metric()
        
        self.logger.info(
            "Circuit breaker state transition",
            old_state=old_state.value,
            new_state=new_state.value,
            stats={
                "total_calls": self.stats.total_calls,
                "failure_rate": self.stats.failure_rate,
                "consecutive_failures": self.stats.consecutive_failures
            }
        )
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If func raises an exception
        """
        # Check if we should attempt reset
        if await self._should_attempt_reset():
            async with self._lock:
                if self.state == CircuitState.OPEN:
                    await self._transition_to(CircuitState.HALF_OPEN)
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit breaker '{self.config.name}' is open",
                self.config.name,
                self.state
            )
        
        # Execute with tracing
        with tracer.start_as_current_span(
            f"circuit_breaker.{self.config.name}",
            attributes={
                "circuit.name": self.config.name,
                "circuit.state": self.state.value,
                "circuit.stats.total_calls": self.stats.total_calls,
                "circuit.stats.failure_rate": self.stats.failure_rate
            }
        ) as span:
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Record success
                await self._record_success()
                span.set_status(Status(StatusCode.OK))
                
                return result
                
            except Exception as e:
                # Record failure
                await self._record_failure()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            await self._record_success()
        else:
            await self._record_failure()
        return False  # Don't suppress exceptions
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit statistics."""
        return self.stats
    
    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
    
    async def trip(self) -> None:
        """Manually trip the circuit breaker."""
        async with self._lock:
            await self._transition_to(CircuitState.OPEN)
            circuit_trips_counter.add(
                1,
                {"circuit.name": self.config.name, "manual": True}
            )