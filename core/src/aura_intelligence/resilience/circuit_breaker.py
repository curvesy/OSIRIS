"""
Adaptive Circuit Breaker implementation for AURA Intelligence.

Features:
- ML-driven threshold adjustment based on system metrics
- Hierarchical breakers (service/method/resource)
- Gradual recovery with traffic shaping
- Integration with OpenTelemetry
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import asyncio
import numpy as np
from collections import deque
import structlog

from opentelemetry import trace, metrics

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
circuit_opens = meter.create_counter(
    name="aura.resilience.circuit_breaker.opens",
    description="Number of circuit breaker opens"
)

circuit_state = meter.create_gauge(
    name="aura.resilience.circuit_breaker.state",
    description="Current circuit breaker state (0=closed, 1=open, 2=half_open)"
)

failure_rate = meter.create_gauge(
    name="aura.resilience.circuit_breaker.failure_rate",
    description="Current failure rate"
)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Basic settings
    failure_threshold: float = 0.5
    recovery_timeout: timedelta = timedelta(seconds=30)
    half_open_requests: int = 5
    
    # Adaptive settings
    adaptive_enabled: bool = True
    min_threshold: float = 0.3
    max_threshold: float = 0.8
    
    # Window settings
    window_size: int = 100
    window_duration: timedelta = timedelta(seconds=60)
    
    # ML settings
    use_ml_prediction: bool = True
    prediction_window: int = 10


class CircuitBreakerMetrics:
    """Tracks circuit breaker metrics."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.requests = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
    def record_success(self, latency: float):
        """Record successful request."""
        self.requests.append(1)
        self.latencies.append(latency)
        self.timestamps.append(datetime.now(timezone.utc))
    
    def record_failure(self, latency: float):
        """Record failed request."""
        self.requests.append(0)
        self.latencies.append(latency)
        self.timestamps.append(datetime.now(timezone.utc))
    
    def get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if not self.requests:
            return 0.0
        return 1.0 - (sum(self.requests) / len(self.requests))
    
    def get_latency_percentile(self, percentile: float) -> float:
        """Get latency percentile."""
        if not self.latencies:
            return 0.0
        return float(np.percentile(list(self.latencies), percentile))
    
    def get_recent_pattern(self, n: int = 10) -> List[int]:
        """Get recent request pattern for ML prediction."""
        if len(self.requests) < n:
            return list(self.requests) + [1] * (n - len(self.requests))
        return list(self.requests)[-n:]


class ThresholdPredictor:
    """
    ML-based threshold predictor.
    
    Uses simple pattern recognition to predict optimal thresholds.
    In production, this would use a real ML model.
    """
    
    def __init__(self):
        self.history = deque(maxlen=1000)
        self.patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict[str, float]:
        """Initialize known failure patterns."""
        return {
            "cascading": 0.3,      # Lower threshold for cascading failures
            "intermittent": 0.6,   # Higher threshold for intermittent issues
            "degraded": 0.4,       # Medium threshold for degraded service
            "normal": 0.5          # Default threshold
        }
    
    def predict_threshold(
        self,
        metrics: CircuitBreakerMetrics,
        system_load: float
    ) -> float:
        """Predict optimal threshold based on patterns."""
        pattern = self._detect_pattern(metrics)
        base_threshold = self.patterns.get(pattern, 0.5)
        
        # Adjust based on system load
        load_factor = 1.0 - (system_load * 0.3)  # Lower threshold under high load
        
        return max(0.3, min(0.8, base_threshold * load_factor))
    
    def _detect_pattern(self, metrics: CircuitBreakerMetrics) -> str:
        """Detect failure pattern."""
        recent = metrics.get_recent_pattern()
        
        # Cascading: Increasing failures
        if recent[-5:].count(0) > recent[:5].count(0):
            return "cascading"
        
        # Intermittent: Mixed success/failure
        if 0.3 < sum(recent) / len(recent) < 0.7:
            return "intermittent"
        
        # Degraded: Consistent moderate failure
        if 0.2 < metrics.get_failure_rate() < 0.4:
            return "degraded"
        
        return "normal"


class AdaptiveCircuitBreaker:
    """
    Adaptive circuit breaker with ML-driven thresholds.
    
    Features:
    - Dynamic threshold adjustment
    - Predictive opening
    - Gradual recovery
    - Hierarchical support
    """
    
    def __init__(
        self,
        config: CircuitBreakerConfig,
        name: Optional[str] = None,
        parent: Optional['AdaptiveCircuitBreaker'] = None
    ):
        self.config = config
        self.name = name or "default"
        self.parent = parent
        
        # State
        self.state = CircuitBreakerState.CLOSED
        self.last_failure_time: Optional[datetime] = None
        self.half_open_requests = 0
        
        # Metrics
        self.metrics = CircuitBreakerMetrics(config.window_size)
        self.predictor = ThresholdPredictor() if config.use_ml_prediction else None
        
        # Children for hierarchical breakers
        self.children: Dict[str, AdaptiveCircuitBreaker] = {}
        
        # Update metrics
        self._update_state_metric()
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection."""
        # Check parent breaker first
        if self.parent and self.parent.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError(f"Parent breaker {self.parent.name} is open")
        
        # Check current state
        if self.state == CircuitBreakerState.OPEN:
            if await self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self._update_state_metric()
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        # Execute operation
        start_time = datetime.now(timezone.utc)
        
        try:
            result = await operation(*args, **kwargs)
            latency = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            await self._on_success(latency)
            return result
            
        except Exception as e:
            latency = (datetime.now(timezone.utc) - start_time).total_seconds()
            await self._on_failure(latency, e)
            raise
    
    async def _on_success(self, latency: float):
        """Handle successful execution."""
        self.metrics.record_success(latency)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_requests += 1
            
            if self.half_open_requests >= self.config.half_open_requests:
                # Enough successful requests, close breaker
                self.state = CircuitBreakerState.CLOSED
                self.half_open_requests = 0
                self._update_state_metric()
                logger.info(f"Circuit breaker {self.name} closed after recovery")
    
    async def _on_failure(self, latency: float, error: Exception):
        """Handle failed execution."""
        self.metrics.record_failure(latency)
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failure in half-open, reopen immediately
            self.state = CircuitBreakerState.OPEN
            self.half_open_requests = 0
            self._update_state_metric()
            circuit_opens.add(1, {"breaker": self.name, "reason": "half_open_failure"})
            logger.warning(f"Circuit breaker {self.name} reopened after half-open failure")
            
        elif self.state == CircuitBreakerState.CLOSED:
            # Check if should open
            if await self._should_open():
                self.state = CircuitBreakerState.OPEN
                self._update_state_metric()
                circuit_opens.add(1, {"breaker": self.name, "reason": "threshold_exceeded"})
                logger.warning(f"Circuit breaker {self.name} opened due to failures")
    
    async def _should_open(self) -> bool:
        """Determine if breaker should open."""
        current_failure_rate = self.metrics.get_failure_rate()
        
        # Update metric
        failure_rate.set(current_failure_rate, {"breaker": self.name})
        
        # Get threshold
        threshold = await self._get_threshold()
        
        # Check ML prediction if enabled
        if self.config.use_ml_prediction and self.predictor:
            predicted_failure = self._predict_future_failure()
            if predicted_failure > 0.8:  # High confidence of imminent failure
                logger.info(
                    f"Predictive opening for {self.name}",
                    predicted_failure=predicted_failure
                )
                return True
        
        return current_failure_rate > threshold
    
    async def _get_threshold(self) -> float:
        """Get current threshold (adaptive or static)."""
        if not self.config.adaptive_enabled:
            return self.config.failure_threshold
        
        # Get system load (simplified - in production, get from metrics)
        system_load = await self._get_system_load()
        
        # Use predictor if available
        if self.predictor:
            threshold = self.predictor.predict_threshold(self.metrics, system_load)
        else:
            # Simple adaptive: lower threshold under high load
            base = self.config.failure_threshold
            threshold = base * (1.0 - system_load * 0.3)
        
        # Clamp to configured bounds
        return max(
            self.config.min_threshold,
            min(self.config.max_threshold, threshold)
        )
    
    async def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset from open state."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure > self.config.recovery_timeout
    
    def _predict_future_failure(self) -> float:
        """Predict probability of future failure."""
        if not self.predictor:
            return 0.0
        
        # Simple prediction based on recent pattern
        recent = self.metrics.get_recent_pattern()
        if len(recent) < 5:
            return 0.0
        
        # If last 3 requests failed, high probability
        if sum(recent[-3:]) == 0:
            return 0.9
        
        # If failure rate increasing
        first_half = sum(recent[:5]) / 5
        second_half = sum(recent[5:]) / 5
        
        if second_half < first_half * 0.7:  # 30% worse
            return 0.7
        
        return 0.3
    
    async def _get_system_load(self) -> float:
        """Get current system load (0.0 to 1.0)."""
        # Simplified - in production, get from system metrics
        # Could check CPU, memory, queue depths, etc.
        return 0.5
    
    def _update_state_metric(self):
        """Update state metric."""
        state_value = {
            CircuitBreakerState.CLOSED: 0,
            CircuitBreakerState.OPEN: 1,
            CircuitBreakerState.HALF_OPEN: 2
        }[self.state]
        
        circuit_state.set(state_value, {"breaker": self.name})
    
    def get_child(self, name: str) -> 'AdaptiveCircuitBreaker':
        """Get or create child breaker for hierarchical protection."""
        if name not in self.children:
            child_config = dataclass.replace(self.config)
            self.children[name] = AdaptiveCircuitBreaker(
                child_config,
                name=f"{self.name}.{name}",
                parent=self
            )
        return self.children[name]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_rate": self.metrics.get_failure_rate(),
            "p99_latency": self.metrics.get_latency_percentile(99),
            "total_requests": len(self.metrics.requests),
            "half_open_requests": self.half_open_requests
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Example usage
async def example_usage():
    # Create adaptive circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=0.5,
        adaptive_enabled=True,
        use_ml_prediction=True
    )
    
    breaker = AdaptiveCircuitBreaker(config, name="api_service")
    
    # Create hierarchical breakers
    user_breaker = breaker.get_child("user_endpoint")
    admin_breaker = breaker.get_child("admin_endpoint")
    
    # Use breaker
    async def api_call():
        # Simulate API call
        await asyncio.sleep(0.1)
        if np.random.random() > 0.7:  # 30% failure rate
            raise Exception("API error")
        return {"status": "ok"}
    
    # Execute with circuit breaker
    try:
        result = await user_breaker.execute(api_call)
        print(f"Success: {result}")
    except CircuitBreakerOpenError:
        print("Circuit breaker is open, using fallback")
    except Exception as e:
        print(f"Operation failed: {e}")