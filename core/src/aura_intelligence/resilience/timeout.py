"""
Adaptive Timeout implementation for AURA Intelligence.

Features:
- P99-based timeout calculation
- Deadline propagation across service calls
- Timeout budgeting for complex operations
- Adaptive adjustment based on observed latencies
- Integration with distributed tracing
"""

from typing import Dict, Any, Optional, Callable, TypeVar, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import statistics
from collections import deque, defaultdict
import structlog
import contextvars

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

T = TypeVar('T')

# Context variable for deadline propagation
deadline_context = contextvars.ContextVar('deadline_context', default=None)

# Metrics
timeout_triggered = meter.create_counter(
    name="aura.resilience.timeout.triggered",
    description="Number of timeouts triggered"
)

timeout_duration = meter.create_histogram(
    name="aura.resilience.timeout.duration",
    description="Actual operation duration before timeout",
    unit="ms"
)

deadline_violations = meter.create_counter(
    name="aura.resilience.timeout.deadline_violations",
    description="Number of deadline violations"
)

adaptive_adjustments = meter.create_counter(
    name="aura.resilience.timeout.adjustments",
    description="Number of adaptive timeout adjustments"
)


class TimeoutStrategy(Enum):
    """Strategies for timeout calculation."""
    FIXED = "fixed"
    PERCENTILE = "percentile"
    ADAPTIVE = "adaptive"
    DEADLINE_AWARE = "deadline_aware"


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""
    # Basic settings
    default_timeout_ms: int = 5000
    strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE
    
    # Percentile settings
    percentile: float = 99.0  # P99 by default
    percentile_multiplier: float = 1.5  # P99 * 1.5
    
    # Adaptive settings
    adaptive_enabled: bool = True
    min_timeout_ms: int = 100
    max_timeout_ms: int = 30000
    history_size: int = 1000
    adjustment_threshold: float = 0.2  # 20% change triggers adjustment
    
    # Deadline settings
    deadline_propagation: bool = True
    deadline_buffer_ms: int = 100  # Reserve buffer for response processing
    
    # Timeout budgeting
    budget_enabled: bool = True
    budget_allocation: Dict[str, float] = field(default_factory=lambda: {
        "network": 0.1,      # 10% for network overhead
        "processing": 0.8,   # 80% for actual processing
        "response": 0.1      # 10% for response handling
    })


@dataclass
class DeadlineContext:
    """Context for deadline propagation."""
    absolute_deadline: datetime
    operation_stack: List[str] = field(default_factory=list)
    consumed_time_ms: float = 0.0
    
    @property
    def remaining_ms(self) -> float:
        """Calculate remaining time in milliseconds."""
        remaining = (self.absolute_deadline - datetime.utcnow()).total_seconds() * 1000
        return max(0, remaining)
    
    def has_time_for(self, required_ms: float) -> bool:
        """Check if there's enough time for an operation."""
        return self.remaining_ms > required_ms
    
    def consume(self, duration_ms: float):
        """Consume time from the deadline."""
        self.consumed_time_ms += duration_ms
    
    def push_operation(self, operation: str):
        """Push operation onto stack for tracing."""
        self.operation_stack.append(operation)
    
    def pop_operation(self):
        """Pop operation from stack."""
        if self.operation_stack:
            self.operation_stack.pop()


class LatencyTracker:
    """
    Tracks operation latencies for adaptive timeout calculation.
    
    Uses a sliding window to calculate percentiles.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
        self.operation_latencies: Dict[str, deque] = {}
        self.lock = asyncio.Lock()
        
        # Track timeout effectiveness
        self.timeout_count = 0
        self.successful_count = 0
        
    async def record_latency(
        self,
        operation: str,
        duration_ms: float,
        timed_out: bool = False
    ):
        """Record operation latency."""
        async with self.lock:
            # Global latencies
            self.latencies.append(duration_ms)
            
            # Per-operation latencies
            if operation not in self.operation_latencies:
                self.operation_latencies[operation] = deque(maxlen=self.window_size)
            
            self.operation_latencies[operation].append(duration_ms)
            
            # Track timeout effectiveness
            if timed_out:
                self.timeout_count += 1
            else:
                self.successful_count += 1
    
    async def get_percentile(
        self,
        percentile: float,
        operation: Optional[str] = None
    ) -> float:
        """Calculate percentile latency."""
        async with self.lock:
            if operation and operation in self.operation_latencies:
                data = list(self.operation_latencies[operation])
            else:
                data = list(self.latencies)
            
            if not data:
                return 0.0
            
            return statistics.quantiles(data, n=100)[int(percentile) - 1]
    
    async def get_stats(self, operation: Optional[str] = None) -> Dict[str, float]:
        """Get latency statistics."""
        async with self.lock:
            if operation and operation in self.operation_latencies:
                data = list(self.operation_latencies[operation])
            else:
                data = list(self.latencies)
            
            if not data:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "max": 0.0
                }
            
            sorted_data = sorted(data)
            
            return {
                "count": len(data),
                "mean": statistics.mean(data),
                "p50": sorted_data[len(sorted_data) // 2],
                "p95": sorted_data[int(len(sorted_data) * 0.95)],
                "p99": sorted_data[int(len(sorted_data) * 0.99)],
                "max": max(data)
            }
    
    def get_timeout_effectiveness(self) -> float:
        """Calculate timeout effectiveness ratio."""
        total = self.timeout_count + self.successful_count
        if total == 0:
            return 1.0
        
        # Lower is better (fewer timeouts)
        return 1.0 - (self.timeout_count / total)


class TimeoutCalculator:
    """
    Calculates appropriate timeouts based on strategy and historical data.
    """
    
    def __init__(self, config: TimeoutConfig, latency_tracker: LatencyTracker):
        self.config = config
        self.latency_tracker = latency_tracker
        self.adaptive_timeouts: Dict[str, float] = {}
        self.last_adjustment: Dict[str, datetime] = {}
        
    async def calculate_timeout(
        self,
        operation: str,
        context: Optional[DeadlineContext] = None
    ) -> float:
        """Calculate timeout for operation."""
        # Check deadline context first
        if context and self.config.deadline_propagation:
            available_ms = context.remaining_ms - self.config.deadline_buffer_ms
            if available_ms <= 0:
                raise DeadlineExceededException(
                    f"Deadline already exceeded for {operation}"
                )
            
            # Use deadline-aware calculation
            calculated_timeout = await self._calculate_base_timeout(operation)
            return min(calculated_timeout, available_ms)
        
        # Use configured strategy
        if self.config.strategy == TimeoutStrategy.FIXED:
            return self.config.default_timeout_ms
        
        elif self.config.strategy == TimeoutStrategy.PERCENTILE:
            return await self._calculate_percentile_timeout(operation)
        
        elif self.config.strategy == TimeoutStrategy.ADAPTIVE:
            return await self._calculate_adaptive_timeout(operation)
        
        elif self.config.strategy == TimeoutStrategy.DEADLINE_AWARE:
            # Without context, fall back to adaptive
            return await self._calculate_adaptive_timeout(operation)
        
        return self.config.default_timeout_ms
    
    async def _calculate_base_timeout(self, operation: str) -> float:
        """Calculate base timeout without deadline constraints."""
        if self.config.adaptive_enabled and operation in self.adaptive_timeouts:
            return self.adaptive_timeouts[operation]
        
        return await self._calculate_percentile_timeout(operation)
    
    async def _calculate_percentile_timeout(self, operation: str) -> float:
        """Calculate timeout based on percentile."""
        percentile_value = await self.latency_tracker.get_percentile(
            self.config.percentile,
            operation
        )
        
        if percentile_value == 0:
            # No data, use default
            return self.config.default_timeout_ms
        
        # Apply multiplier
        timeout = percentile_value * self.config.percentile_multiplier
        
        # Clamp to bounds
        return max(
            self.config.min_timeout_ms,
            min(self.config.max_timeout_ms, timeout)
        )
    
    async def _calculate_adaptive_timeout(self, operation: str) -> float:
        """Calculate adaptive timeout with automatic adjustment."""
        # Get current timeout
        current_timeout = self.adaptive_timeouts.get(
            operation,
            await self._calculate_percentile_timeout(operation)
        )
        
        # Check if adjustment needed
        if await self._should_adjust_timeout(operation):
            new_timeout = await self._adjust_timeout(operation, current_timeout)
            self.adaptive_timeouts[operation] = new_timeout
            self.last_adjustment[operation] = datetime.utcnow()
            
            adaptive_adjustments.add(1, {
                "operation": operation,
                "direction": "increase" if new_timeout > current_timeout else "decrease"
            })
            
            logger.info(
                f"Adjusted timeout for {operation}",
                old_timeout=current_timeout,
                new_timeout=new_timeout
            )
            
            return new_timeout
        
        return current_timeout
    
    async def _should_adjust_timeout(self, operation: str) -> bool:
        """Check if timeout should be adjusted."""
        # Don't adjust too frequently
        if operation in self.last_adjustment:
            time_since = datetime.utcnow() - self.last_adjustment[operation]
            if time_since < timedelta(minutes=1):
                return False
        
        # Check timeout effectiveness
        effectiveness = self.latency_tracker.get_timeout_effectiveness()
        
        # Adjust if too many timeouts or too conservative
        return effectiveness < 0.95 or effectiveness > 0.99
    
    async def _adjust_timeout(self, operation: str, current: float) -> float:
        """Adjust timeout based on observed behavior."""
        stats = await self.latency_tracker.get_stats(operation)
        effectiveness = self.latency_tracker.get_timeout_effectiveness()
        
        if effectiveness < 0.95:  # Too many timeouts
            # Increase timeout
            # Use P99.9 if available, otherwise increase by 20%
            if stats["count"] > 100:
                new_timeout = stats["p99"] * 1.5
            else:
                new_timeout = current * 1.2
        else:  # Too conservative
            # Decrease timeout
            # Use P95 as new target
            if stats["count"] > 100:
                new_timeout = stats["p95"] * 1.2
            else:
                new_timeout = current * 0.9
        
        # Clamp to bounds
        return max(
            self.config.min_timeout_ms,
            min(self.config.max_timeout_ms, new_timeout)
        )


class TimeoutBudget:
    """
    Manages timeout budget allocation for complex operations.
    
    Useful for operations that involve multiple sub-operations.
    """
    
    def __init__(self, total_timeout_ms: float, allocation: Dict[str, float]):
        self.total_timeout_ms = total_timeout_ms
        self.allocation = allocation
        self.consumed: Dict[str, float] = defaultdict(float)
        self.start_time = datetime.utcnow()
        
    def get_allocation(self, phase: str) -> float:
        """Get timeout allocation for a phase."""
        if phase not in self.allocation:
            raise ValueError(f"Unknown phase: {phase}")
        
        return self.total_timeout_ms * self.allocation[phase]
    
    def consume(self, phase: str, duration_ms: float):
        """Consume time from a phase's budget."""
        self.consumed[phase] += duration_ms
    
    def remaining_for_phase(self, phase: str) -> float:
        """Get remaining time for a phase."""
        allocated = self.get_allocation(phase)
        consumed = self.consumed.get(phase, 0)
        return max(0, allocated - consumed)
    
    def total_remaining(self) -> float:
        """Get total remaining time."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        return max(0, self.total_timeout_ms - elapsed)
    
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.total_remaining() <= 0


class AdaptiveTimeout:
    """
    Adaptive timeout with all advanced features.
    
    Features:
    - P99-based calculation
    - Deadline propagation
    - Adaptive adjustment
    - Timeout budgeting
    """
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.latency_tracker = LatencyTracker(config.history_size)
        self.timeout_calculator = TimeoutCalculator(config, self.latency_tracker)
        
    async def execute(
        self,
        operation: Callable[..., T],
        *args,
        timeout: Optional[timedelta] = None,
        operation_name: Optional[str] = None,
        **kwargs
    ) -> T:
        """Execute operation with adaptive timeout."""
        operation_name = operation_name or operation.__name__
        
        with tracer.start_as_current_span("timeout.execute") as span:
            span.set_attributes({
                "operation": operation_name,
                "strategy": self.config.strategy.value
            })
            
            # Get or create deadline context
            context = deadline_context.get()
            if context:
                context.push_operation(operation_name)
            
            # Calculate timeout
            if timeout:
                timeout_ms = timeout.total_seconds() * 1000
            else:
                timeout_ms = await self.timeout_calculator.calculate_timeout(
                    operation_name,
                    context
                )
            
            span.set_attribute("timeout_ms", timeout_ms)
            
            # Execute with timeout
            start_time = datetime.utcnow()
            
            try:
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=timeout_ms / 1000.0
                )
                
                # Record successful execution
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                await self.latency_tracker.record_latency(
                    operation_name,
                    duration_ms,
                    timed_out=False
                )
                
                # Update deadline context
                if context:
                    context.consume(duration_ms)
                    context.pop_operation()
                
                return result
                
            except asyncio.TimeoutError:
                # Record timeout
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                await self.latency_tracker.record_latency(
                    operation_name,
                    duration_ms,
                    timed_out=True
                )
                
                timeout_triggered.add(1, {
                    "operation": operation_name,
                    "timeout_ms": timeout_ms
                })
                
                timeout_duration.record(duration_ms, {
                    "operation": operation_name
                })
                
                # Check if deadline violated
                if context and context.remaining_ms <= 0:
                    deadline_violations.add(1, {
                        "operation": operation_name
                    })
                
                logger.warning(
                    f"Operation timed out",
                    operation=operation_name,
                    timeout_ms=timeout_ms,
                    duration_ms=duration_ms
                )
                
                raise TimeoutException(
                    f"Operation {operation_name} timed out after {timeout_ms}ms"
                )
            
            except Exception:
                # Update context on error
                if context:
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    context.consume(duration_ms)
                    context.pop_operation()
                
                raise
    
    async def with_deadline(
        self,
        deadline: datetime,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute operation with absolute deadline."""
        # Create deadline context
        context = DeadlineContext(absolute_deadline=deadline)
        
        # Set context
        token = deadline_context.set(context)
        
        try:
            return await self.execute(operation, *args, **kwargs)
        finally:
            # Reset context
            deadline_context.reset(token)
    
    async def with_budget(
        self,
        total_timeout: timedelta,
        operations: List[tuple[str, Callable, tuple, dict]],
        allocation: Optional[Dict[str, float]] = None
    ) -> List[Any]:
        """Execute multiple operations with timeout budget."""
        if not allocation:
            # Equal allocation by default
            allocation = {
                op[0]: 1.0 / len(operations)
                for op in operations
            }
        
        budget = TimeoutBudget(
            total_timeout.total_seconds() * 1000,
            allocation
        )
        
        results = []
        
        for phase, operation, args, kwargs in operations:
            if budget.is_exhausted():
                raise TimeoutException(f"Timeout budget exhausted at phase {phase}")
            
            # Get timeout for this phase
            phase_timeout_ms = budget.remaining_for_phase(phase)
            
            # Execute with phase timeout
            start_time = datetime.utcnow()
            
            result = await self.execute(
                operation,
                *args,
                timeout=timedelta(milliseconds=phase_timeout_ms),
                operation_name=phase,
                **kwargs
            )
            
            # Update budget
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            budget.consume(phase, duration_ms)
            
            results.append(result)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get timeout metrics."""
        return {
            "latency_stats": asyncio.run(self.latency_tracker.get_stats()),
            "timeout_effectiveness": self.latency_tracker.get_timeout_effectiveness(),
            "adaptive_timeouts": dict(self.timeout_calculator.adaptive_timeouts),
            "timeout_count": self.latency_tracker.timeout_count,
            "successful_count": self.latency_tracker.successful_count
        }


class TimeoutException(Exception):
    """Raised when operation times out."""
    pass


class DeadlineExceededException(Exception):
    """Raised when deadline is exceeded."""
    pass


# Convenience decorator
def with_timeout(
    timeout: Optional[timedelta] = None,
    strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE
):
    """
    Decorator for adding timeout to functions.
    
    Example:
        @with_timeout(timedelta(seconds=5))
        async def slow_operation():
            ...
    """
    def decorator(func):
        config = TimeoutConfig(
            strategy=strategy,
            default_timeout_ms=timeout.total_seconds() * 1000 if timeout else 5000
        )
        timeout_handler = AdaptiveTimeout(config)
        
        async def wrapper(*args, **kwargs):
            return await timeout_handler.execute(
                func,
                *args,
                timeout=timeout,
                **kwargs
            )
        
        return wrapper
    return decorator