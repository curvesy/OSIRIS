"""
AURA Intelligence Resilience Framework

Production-grade resilience patterns for distributed AI systems:
- Adaptive circuit breakers with ML-driven thresholds
- Dynamic bulkheads with auto-scaling
- Context-aware retry strategies
- AI-specific fallback chains
- Chaos engineering support

Based on 2025 best practices from Netflix, Google, AWS, and OpenAI.
"""

from typing import Dict, Any, Optional, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from opentelemetry import trace
from opentelemetry import metrics as otel_metrics

from .circuit_breaker import (
    AdaptiveCircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig
)

from .bulkhead import (
    DynamicBulkhead,
    BulkheadConfig,
    PriorityLevel
)

from .retry import (
    ContextAwareRetry,
    RetryStrategy,
    RetryBudget,
    RetryConfig
)

from .timeout import (
    AdaptiveTimeout,
    TimeoutConfig,
    DeadlineContext
)

from .metrics import (
    ResilienceMetrics,
    MetricsCollector
)

__version__ = "1.0.0"

# Type variable for generic resilience policies
T = TypeVar('T')

tracer = trace.get_tracer(__name__)
meter = otel_metrics.get_meter(__name__)


class ResilienceLevel(Enum):
    """Resilience levels for different criticality."""
    CRITICAL = "critical"      # Full resilience
    STANDARD = "standard"      # Balanced resilience
    BEST_EFFORT = "best_effort"  # Minimal resilience


@dataclass
class ResilienceConfig:
    """Global resilience configuration."""
    # Feature flags
    enable_circuit_breaker: bool = True
    enable_bulkhead: bool = True
    enable_retry: bool = True
    enable_timeout: bool = True
    enable_chaos: bool = False
    
    # Global settings
    default_timeout_ms: int = 5000
    max_retry_attempts: int = 3
    circuit_breaker_threshold: float = 0.5
    bulkhead_max_concurrent: int = 100
    
    # AI-specific
    enable_model_fallback: bool = True
    enable_consensus_resilience: bool = True
    
    # Observability
    metrics_enabled: bool = True
    detailed_tracing: bool = False


class ResiliencePolicy(Protocol):
    """Protocol for resilience policies."""
    
    async def execute(self, operation: Any) -> Any:
        """Execute operation with resilience."""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get policy metrics."""
        ...


@dataclass
class ResilienceContext:
    """Context for resilience decisions."""
    operation_name: str
    criticality: ResilienceLevel
    timeout: Optional[timedelta] = None
    retry_budget: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_critical(self) -> bool:
        return self.criticality == ResilienceLevel.CRITICAL
    
    def get(self, key: str, default=None):
        """Get attribute value like a dictionary for compatibility."""
        return getattr(self, key, default)


class ResilientOperation(Generic[T]):
    """
    Wraps an operation with multiple resilience patterns.
    
    Example:
        operation = ResilientOperation(
            my_async_function,
            context=ResilienceContext(
                operation_name="api_call",
                criticality=ResilienceLevel.CRITICAL
            )
        )
        result = await operation.execute()
    """
    
    def __init__(
        self,
        operation: Any,
        context: ResilienceContext,
        config: Optional[ResilienceConfig] = None
    ):
        self.operation = operation
        self.context = context
        self.config = config or ResilienceConfig()
        
        # Initialize resilience components
        self._init_components()
    
    def _init_components(self):
        """Initialize resilience components based on config."""
        if self.config.enable_circuit_breaker:
            self.circuit_breaker = AdaptiveCircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=self.config.circuit_breaker_threshold,
                    recovery_timeout=timedelta(seconds=30),
                    half_open_requests=5
                )
            )
        
        if self.config.enable_bulkhead:
            self.bulkhead = DynamicBulkhead(
                BulkheadConfig(
                    max_capacity=self.config.bulkhead_max_concurrent,
                    queue_size=1000,
                    priority_enabled=True
                )
            )
        
        if self.config.enable_retry:
            self.retry = ContextAwareRetry(
                RetryConfig(
                    max_attempts=self.config.max_retry_attempts,
                    backoff_base=1.0,
                    budget_enabled=True
                )
            )
        
        if self.config.enable_timeout:
            self.timeout = AdaptiveTimeout(
                TimeoutConfig(
                    default_timeout_ms=self.config.default_timeout_ms,
                    adaptive_enabled=True
                )
            )
    
    async def execute(self, *args, **kwargs) -> T:
        """Execute operation with full resilience stack."""
        with tracer.start_as_current_span(
            f"resilient.{self.context.operation_name}"
        ) as span:
            span.set_attributes({
                "resilience.level": self.context.criticality.value,
                "resilience.operation": self.context.operation_name
            })
            
            try:
                # Apply resilience layers in order
                result = await self._execute_with_resilience(*args, **kwargs)
                span.set_attribute("resilience.success", True)
                return result
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("resilience.success", False)
                raise
    
    async def _execute_with_resilience(self, *args, **kwargs) -> T:
        """Apply resilience patterns in order."""
        operation = self.operation
        
        # Layer 1: Bulkhead isolation
        if self.config.enable_bulkhead:
            operation = self._wrap_with_bulkhead(operation)
        
        # Layer 2: Circuit breaker
        if self.config.enable_circuit_breaker:
            operation = self._wrap_with_circuit_breaker(operation)
        
        # Layer 3: Timeout
        if self.config.enable_timeout:
            operation = self._wrap_with_timeout(operation)
        
        # Layer 4: Retry
        if self.config.enable_retry:
            operation = self._wrap_with_retry(operation)
        
        # Execute
        return await operation(*args, **kwargs)
    
    def _wrap_with_bulkhead(self, operation):
        """Wrap operation with bulkhead."""
        async def wrapped(*args, **kwargs):
            # Import ResourceRequest here to avoid circular imports
            from .bulkhead import ResourceRequest, ResourceType, PriorityLevel
            
            # Determine priority
            priority = (
                PriorityLevel.HIGH 
                if self.context.is_critical 
                else PriorityLevel.NORMAL
            )
            
            # Create a proper ResourceRequest
            request = ResourceRequest(
                id=f"req-{id(operation)}-{hash(str(args))}", 
                operation_name=operation.__name__,
                priority=priority,
                resources={ResourceType.AGENT_SLOT: 1.0}
            )
            
            # Remove any request from kwargs to avoid conflicts
            kwargs_clean = {k: v for k, v in kwargs.items() if k != 'request'}
            return await self.bulkhead.execute(
                operation, 
                request,
                *args, 
                **kwargs_clean
            )
        return wrapped
    
    def _wrap_with_circuit_breaker(self, operation):
        """Wrap operation with circuit breaker."""
        async def wrapped(*args, **kwargs):
            return await self.circuit_breaker.execute(
                operation,
                *args,
                **kwargs
            )
        return wrapped
    
    def _wrap_with_timeout(self, operation):
        """Wrap operation with timeout."""
        async def wrapped(*args, **kwargs):
            timeout = self.context.timeout or timedelta(
                milliseconds=self.config.default_timeout_ms
            )
            return await self.timeout.execute(
                operation,
                *args,
                timeout=timeout,
                **kwargs
            )
        return wrapped
    
    def _wrap_with_retry(self, operation):
        """Wrap operation with retry."""
        async def wrapped(*args, **kwargs):
            return await self.retry.execute(
                operation,
                *args,
                context=self.context,
                **kwargs
            )
        return wrapped


class ResilienceManager:
    """
    Central manager for resilience policies.
    
    Handles policy composition, metrics aggregation, and chaos testing.
    """
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.metrics = ResilienceMetrics()
        self.policies: Dict[str, ResiliencePolicy] = {}
        
    def create_resilient_operation(
        self,
        operation: Any,
        context: ResilienceContext
    ) -> ResilientOperation:
        """Create a resilient operation."""
        return ResilientOperation(operation, context, self.config)
    
    async def execute_with_resilience(
        self,
        operation: Any,
        context: ResilienceContext,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with resilience."""
        resilient_op = self.create_resilient_operation(operation, context)
        return await resilient_op.execute(*args, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated resilience metrics."""
        return self.metrics.get_all()
    
    def enable_chaos(self, experiment_name: str):
        """Enable chaos testing."""
        if not self.config.enable_chaos:
            raise RuntimeError("Chaos testing is disabled")
        # Chaos implementation in separate module
        from .chaos import ChaosExperiment
        return ChaosExperiment(experiment_name)


# Convenience decorators
def resilient(
    criticality: ResilienceLevel = ResilienceLevel.STANDARD,
    timeout_ms: Optional[int] = None,
    max_retries: Optional[int] = None
):
    """
    Decorator to make a function resilient.
    
    Example:
        @resilient(criticality=ResilienceLevel.CRITICAL)
        async def critical_operation():
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            context = ResilienceContext(
                operation_name=func.__name__,
                criticality=criticality,
                timeout=timedelta(milliseconds=timeout_ms) if timeout_ms else None
            )
            
            manager = ResilienceManager(ResilienceConfig())
            return await manager.execute_with_resilience(
                func, context, *args, **kwargs
            )
        
        return wrapper
    return decorator


__all__ = [
    # Core classes
    "ResilienceConfig",
    "ResilienceContext", 
    "ResilienceLevel",
    "ResiliencePolicy",
    "ResilientOperation",
    "ResilienceManager",
    
    # Circuit breaker
    "AdaptiveCircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    
    # Bulkhead
    "DynamicBulkhead",
    "BulkheadConfig",
    "PriorityLevel",
    
    # Retry
    "ContextAwareRetry",
    "RetryStrategy",
    "RetryBudget",
    "RetryConfig",
    
    # Timeout
    "AdaptiveTimeout",
    "TimeoutConfig",
    "DeadlineContext",
    
    # Metrics
    "ResilienceMetrics",
    "MetricsCollector",
    
    # Decorator
    "resilient"
]