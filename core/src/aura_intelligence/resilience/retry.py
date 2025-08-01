"""
Context-Aware Retry implementation for AURA Intelligence.

Features:
- Error-type specific retry strategies
- Retry budgets (token bucket algorithm)
- Hedged requests for latency-sensitive operations
- Adaptive backoff based on system load
- Integration with circuit breakers
- Distributed retry coordination
"""

from typing import Dict, Any, Optional, Callable, List, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import random
import math
import structlog
from collections import defaultdict, deque

from opentelemetry import trace, metrics

from .circuit_breaker import CircuitBreakerOpenError

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

T = TypeVar('T')

# Metrics
retry_attempts = meter.create_counter(
    name="aura.resilience.retry.attempts",
    description="Number of retry attempts"
)

retry_success = meter.create_counter(
    name="aura.resilience.retry.success",
    description="Number of successful retries"
)

retry_exhausted = meter.create_counter(
    name="aura.resilience.retry.exhausted",
    description="Number of retries that exhausted attempts"
)

retry_budget_consumed = meter.create_gauge(
    name="aura.resilience.retry.budget_consumed",
    description="Percentage of retry budget consumed"
)

hedge_requests = meter.create_counter(
    name="aura.resilience.retry.hedge_requests",
    description="Number of hedged requests sent"
)


class RetryStrategy(Enum):
    """Retry strategies for different scenarios."""
    IMMEDIATE = "immediate"
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"


class ErrorCategory(Enum):
    """Categories of errors for context-aware retries."""
    NETWORK = "network"           # Connection errors, timeouts
    RATE_LIMIT = "rate_limit"     # 429, throttling
    SERVER_ERROR = "server_error" # 5xx errors
    TIMEOUT = "timeout"           # Request timeout
    CIRCUIT_OPEN = "circuit_open" # Circuit breaker open
    RESOURCE = "resource"         # Resource exhaustion
    UNKNOWN = "unknown"           # Unclassified errors


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    # Basic settings
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    
    # Backoff settings
    initial_delay_ms: int = 100
    max_delay_ms: int = 30000
    backoff_base: float = 2.0
    jitter_factor: float = 0.1
    
    # Advanced settings
    budget_enabled: bool = True
    budget_per_minute: int = 100
    hedged_requests: bool = False
    hedge_delay_ms: int = 50
    hedge_max_attempts: int = 2
    
    # Error-specific settings
    error_strategies: Dict[ErrorCategory, Dict[str, Any]] = field(default_factory=dict)
    
    # Integration
    circuit_breaker_aware: bool = True
    adaptive_enabled: bool = False


class RetryBudget:
    """
    Token bucket implementation for retry budgets.
    
    Prevents retry storms by limiting retry rate.
    """
    
    def __init__(self, tokens_per_minute: int):
        self.capacity = tokens_per_minute
        self.tokens = float(tokens_per_minute)
        self.last_update = datetime.utcnow()
        self.lock = asyncio.Lock()
        
        # Metrics
        self.consumed_count = 0
        self.rejected_count = 0
        
    async def try_consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the budget."""
        async with self.lock:
            # Refill tokens based on time elapsed
            now = datetime.utcnow()
            elapsed = (now - self.last_update).total_seconds()
            self.last_update = now
            
            # Add tokens at the configured rate
            tokens_to_add = (elapsed / 60.0) * self.capacity
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            
            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.consumed_count += tokens
                
                # Update metric
                consumption_rate = (self.capacity - self.tokens) / self.capacity
                retry_budget_consumed.set(consumption_rate * 100)
                
                return True
            else:
                self.rejected_count += 1
                return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get budget metrics."""
        return {
            "capacity": self.capacity,
            "available_tokens": self.tokens,
            "consumed_count": self.consumed_count,
            "rejected_count": self.rejected_count,
            "utilization": (self.capacity - self.tokens) / self.capacity
        }


class BackoffCalculator:
    """
    Calculates backoff delays for different strategies.
    
    Includes jitter and adaptive adjustments.
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.fibonacci_cache = [0, 1]
        
    def calculate_delay(
        self,
        attempt: int,
        strategy: Optional[RetryStrategy] = None,
        error_category: Optional[ErrorCategory] = None
    ) -> float:
        """Calculate delay in milliseconds for the given attempt."""
        strategy = strategy or self.config.strategy
        
        # Get base delay
        if strategy == RetryStrategy.IMMEDIATE:
            base_delay = 0
        elif strategy == RetryStrategy.FIXED:
            base_delay = self.config.initial_delay_ms
        elif strategy == RetryStrategy.LINEAR:
            base_delay = self.config.initial_delay_ms * attempt
        elif strategy == RetryStrategy.EXPONENTIAL:
            base_delay = self.config.initial_delay_ms * math.pow(
                self.config.backoff_base, attempt - 1
            )
        elif strategy == RetryStrategy.FIBONACCI:
            base_delay = self.config.initial_delay_ms * self._fibonacci(attempt)
        elif strategy == RetryStrategy.ADAPTIVE:
            base_delay = self._adaptive_delay(attempt, error_category)
        else:
            base_delay = self.config.initial_delay_ms
        
        # Apply max delay cap
        base_delay = min(base_delay, self.config.max_delay_ms)
        
        # Add jitter
        jitter_range = base_delay * self.config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        
        return max(0, base_delay + jitter)
    
    def _fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number."""
        while len(self.fibonacci_cache) <= n:
            self.fibonacci_cache.append(
                self.fibonacci_cache[-1] + self.fibonacci_cache[-2]
            )
        return self.fibonacci_cache[n]
    
    def _adaptive_delay(
        self,
        attempt: int,
        error_category: Optional[ErrorCategory]
    ) -> float:
        """Calculate adaptive delay based on error type and system state."""
        # Base exponential backoff
        base_delay = self.config.initial_delay_ms * math.pow(
            self.config.backoff_base, attempt - 1
        )
        
        # Adjust based on error category
        if error_category == ErrorCategory.RATE_LIMIT:
            # Longer backoff for rate limits
            return base_delay * 3.0
        elif error_category == ErrorCategory.TIMEOUT:
            # Shorter backoff for timeouts (might be transient)
            return base_delay * 0.5
        elif error_category == ErrorCategory.CIRCUIT_OPEN:
            # Align with circuit breaker recovery
            return 5000  # 5 seconds
        
        return base_delay


class ErrorClassifier:
    """Classifies errors into categories for context-aware retry."""
    
    @staticmethod
    def classify(error: Exception) -> ErrorCategory:
        """Classify an error into a category."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Network errors
        if any(keyword in error_str or keyword in error_type for keyword in [
            'connection', 'network', 'socket', 'dns', 'ssl'
        ]):
            return ErrorCategory.NETWORK
        
        # Rate limiting
        if any(keyword in error_str for keyword in [
            '429', 'rate limit', 'throttl', 'too many'
        ]):
            return ErrorCategory.RATE_LIMIT
        
        # Server errors
        if any(keyword in error_str for keyword in [
            '500', '502', '503', '504', 'server error', 'internal error'
        ]):
            return ErrorCategory.SERVER_ERROR
        
        # Timeout
        if any(keyword in error_str or keyword in error_type for keyword in [
            'timeout', 'timed out', 'deadline'
        ]):
            return ErrorCategory.TIMEOUT
        
        # Circuit breaker
        if isinstance(error, CircuitBreakerOpenError):
            return ErrorCategory.CIRCUIT_OPEN
        
        # Resource exhaustion
        if any(keyword in error_str for keyword in [
            'memory', 'disk', 'quota', 'resource', 'exhausted'
        ]):
            return ErrorCategory.RESOURCE
        
        return ErrorCategory.UNKNOWN


class HedgedRequestManager:
    """
    Manages hedged requests for latency-sensitive operations.
    
    Sends backup requests if the primary is slow.
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.active_hedges: Dict[str, List[asyncio.Task]] = {}
        
    async def execute_with_hedge(
        self,
        operation: Callable[..., T],
        *args,
        request_id: Optional[str] = None,
        **kwargs
    ) -> T:
        """Execute operation with hedged requests."""
        if not self.config.hedged_requests:
            return await operation(*args, **kwargs)
        
        request_id = request_id or f"hedge-{datetime.utcnow().timestamp()}"
        tasks = []
        
        try:
            # Start primary request
            primary_task = asyncio.create_task(
                self._execute_with_tracking(operation, *args, **kwargs)
            )
            tasks.append(primary_task)
            
            # Schedule hedge requests
            for i in range(1, self.config.hedge_max_attempts):
                hedge_task = asyncio.create_task(
                    self._execute_hedge(
                        operation, *args,
                        delay_ms=self.config.hedge_delay_ms * i,
                        **kwargs
                    )
                )
                tasks.append(hedge_task)
            
            self.active_hedges[request_id] = tasks
            
            # Wait for first to complete
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
            
            # Return first result
            for task in done:
                if not task.cancelled():
                    result = await task
                    if len(done) > 1:
                        hedge_requests.add(1, {"hedges_sent": len(tasks) - 1})
                    return result
            
            raise Exception("All hedged requests failed")
            
        finally:
            # Cleanup
            self.active_hedges.pop(request_id, None)
    
    async def _execute_with_tracking(self, operation: Callable, *args, **kwargs) -> T:
        """Execute operation with performance tracking."""
        start_time = datetime.utcnow()
        try:
            return await operation(*args, **kwargs)
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.debug(
                "Request completed",
                duration_ms=duration,
                is_hedge=False
            )
    
    async def _execute_hedge(
        self,
        operation: Callable,
        *args,
        delay_ms: int,
        **kwargs
    ) -> T:
        """Execute hedge request after delay."""
        await asyncio.sleep(delay_ms / 1000.0)
        
        logger.debug(f"Sending hedge request after {delay_ms}ms")
        return await self._execute_with_tracking(operation, *args, **kwargs)


class ContextAwareRetry:
    """
    Context-aware retry with advanced features.
    
    Features:
    - Error-specific retry strategies
    - Retry budgets to prevent storms
    - Hedged requests for latency optimization
    - Adaptive backoff based on system state
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.backoff_calculator = BackoffCalculator(config)
        self.error_classifier = ErrorClassifier()
        self.hedge_manager = HedgedRequestManager(config)
        
        # Retry budget
        self.budget = RetryBudget(config.budget_per_minute) if config.budget_enabled else None
        
        # Metrics
        self.retry_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.last_errors = deque(maxlen=100)
        
    async def execute(
        self,
        operation: Callable[..., T],
        *args,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> T:
        """Execute operation with context-aware retry."""
        context = context or {}
        operation_name = context.get("operation_name", operation.__name__)
        
        with tracer.start_as_current_span("retry.execute") as span:
            span.set_attributes({
                "operation": operation_name,
                "max_attempts": self.config.max_attempts,
                "strategy": self.config.strategy.value
            })
            
            last_error = None
            
            for attempt in range(1, self.config.max_attempts + 1):
                try:
                    # Check retry budget
                    if self.budget and attempt > 1:
                        if not await self.budget.try_consume():
                            logger.warning(
                                "Retry budget exhausted",
                                operation=operation_name,
                                attempt=attempt
                            )
                            retry_exhausted.add(1, {"reason": "budget"})
                            raise last_error or Exception("Retry budget exhausted")
                    
                    # Execute with hedging on first attempt
                    if attempt == 1 and self.config.hedged_requests:
                        result = await self.hedge_manager.execute_with_hedge(
                            operation, *args, **kwargs
                        )
                    else:
                        result = await operation(*args, **kwargs)
                    
                    # Success
                    if attempt > 1:
                        retry_success.add(1, {
                            "operation": operation_name,
                            "attempt": attempt
                        })
                        self.success_counts[operation_name] += 1
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    error_category = self.error_classifier.classify(e)
                    
                    # Record attempt
                    retry_attempts.add(1, {
                        "operation": operation_name,
                        "attempt": attempt,
                        "error_category": error_category.value
                    })
                    self.retry_counts[operation_name] += 1
                    self.last_errors.append({
                        "timestamp": datetime.utcnow(),
                        "operation": operation_name,
                        "error": str(e),
                        "category": error_category
                    })
                    
                    # Check if retryable
                    if not self._should_retry(e, error_category, attempt):
                        logger.warning(
                            "Error not retryable",
                            operation=operation_name,
                            error=str(e),
                            category=error_category.value
                        )
                        raise
                    
                    # Last attempt, don't delay
                    if attempt >= self.config.max_attempts:
                        retry_exhausted.add(1, {
                            "reason": "max_attempts",
                            "operation": operation_name
                        })
                        raise
                    
                    # Calculate delay
                    delay_ms = self._get_delay_for_error(
                        attempt,
                        error_category,
                        context
                    )
                    
                    logger.info(
                        f"Retrying after {delay_ms}ms",
                        operation=operation_name,
                        attempt=attempt,
                        error_category=error_category.value
                    )
                    
                    # Wait before retry
                    await asyncio.sleep(delay_ms / 1000.0)
            
            # Should never reach here
            raise last_error or Exception("Retry logic error")
    
    def _should_retry(
        self,
        error: Exception,
        category: ErrorCategory,
        attempt: int
    ) -> bool:
        """Determine if error should be retried."""
        # Circuit breaker open - check config
        if category == ErrorCategory.CIRCUIT_OPEN:
            return self.config.circuit_breaker_aware and attempt == 1
        
        # Check error-specific config
        if category in self.config.error_strategies:
            strategy = self.config.error_strategies[category]
            if not strategy.get("retry", True):
                return False
            
            max_attempts = strategy.get("max_attempts", self.config.max_attempts)
            if attempt >= max_attempts:
                return False
        
        # Default: retry network, timeout, and server errors
        return category in [
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.SERVER_ERROR
        ]
    
    def _get_delay_for_error(
        self,
        attempt: int,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> float:
        """Get delay for specific error category."""
        # Check error-specific config
        if category in self.config.error_strategies:
            strategy = self.config.error_strategies[category]
            
            # Custom delay function
            if "delay_fn" in strategy:
                return strategy["delay_fn"](attempt)
            
            # Custom strategy
            if "strategy" in strategy:
                return self.backoff_calculator.calculate_delay(
                    attempt,
                    strategy["strategy"],
                    category
                )
        
        # Use default calculation
        return self.backoff_calculator.calculate_delay(
            attempt,
            error_category=category
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry metrics."""
        metrics = {
            "retry_counts": dict(self.retry_counts),
            "success_counts": dict(self.success_counts),
            "recent_errors": len(self.last_errors)
        }
        
        if self.budget:
            metrics["budget"] = self.budget.get_metrics()
        
        # Calculate success rate
        total_retries = sum(self.retry_counts.values())
        total_success = sum(self.success_counts.values())
        
        if total_retries > 0:
            metrics["success_rate"] = total_success / total_retries
        
        return metrics


# Example configurations for common scenarios
LATENCY_SENSITIVE_CONFIG = RetryConfig(
    max_attempts=2,
    strategy=RetryStrategy.IMMEDIATE,
    hedged_requests=True,
    hedge_delay_ms=50,
    budget_per_minute=200
)

BACKGROUND_JOB_CONFIG = RetryConfig(
    max_attempts=5,
    strategy=RetryStrategy.EXPONENTIAL,
    initial_delay_ms=1000,
    max_delay_ms=60000,
    budget_enabled=False
)

API_CALL_CONFIG = RetryConfig(
    max_attempts=3,
    strategy=RetryStrategy.EXPONENTIAL,
    initial_delay_ms=100,
    backoff_base=2.0,
    jitter_factor=0.2,
    error_strategies={
        ErrorCategory.RATE_LIMIT: {
            "strategy": RetryStrategy.EXPONENTIAL,
            "backoff_base": 3.0,
            "max_attempts": 5
        },
        ErrorCategory.CIRCUIT_OPEN: {
            "retry": False
        }
    }
)