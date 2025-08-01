"""
Resilience metrics collection and aggregation for AURA Intelligence.

Provides unified metrics for all resilience patterns with
OpenTelemetry integration and real-time dashboards.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import structlog

from opentelemetry import trace, metrics
from opentelemetry.metrics import Observation

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


@dataclass
class MetricWindow:
    """Time window for metric aggregation."""
    duration: timedelta
    buckets: int = 60
    
    @property
    def bucket_duration(self) -> timedelta:
        """Duration of each bucket."""
        return self.duration / self.buckets


class ResilienceMetrics:
    """
    Centralized metrics collection for resilience patterns.
    
    Aggregates metrics from circuit breakers, bulkheads, retries,
    and timeouts for unified observability.
    """
    
    def __init__(self):
        # Circuit breaker metrics
        self.circuit_breaker_states = meter.create_gauge(
            name="aura.resilience.circuit_breaker.state",
            description="Circuit breaker state (0=closed, 1=open, 2=half_open)"
        )
        
        self.circuit_breaker_failures = meter.create_counter(
            name="aura.resilience.circuit_breaker.failures",
            description="Number of circuit breaker failures"
        )
        
        # Bulkhead metrics
        self.bulkhead_utilization = meter.create_gauge(
            name="aura.resilience.bulkhead.utilization",
            description="Bulkhead utilization percentage"
        )
        
        self.bulkhead_queue_depth = meter.create_gauge(
            name="aura.resilience.bulkhead.queue_depth",
            description="Number of requests queued in bulkhead"
        )
        
        # Retry metrics
        self.retry_attempts = meter.create_histogram(
            name="aura.resilience.retry.attempts",
            description="Distribution of retry attempts",
            unit="attempts"
        )
        
        self.retry_delay = meter.create_histogram(
            name="aura.resilience.retry.delay",
            description="Retry delay distribution",
            unit="ms"
        )
        
        # Timeout metrics
        self.timeout_rate = meter.create_gauge(
            name="aura.resilience.timeout.rate",
            description="Timeout rate percentage"
        )
        
        self.operation_duration = meter.create_histogram(
            name="aura.resilience.operation.duration",
            description="Operation duration distribution",
            unit="ms"
        )
        
        # Composite metrics
        self.resilience_score = meter.create_gauge(
            name="aura.resilience.score",
            description="Overall resilience health score (0-100)"
        )
        
        self.failure_rate = meter.create_gauge(
            name="aura.resilience.failure_rate",
            description="Overall failure rate percentage"
        )
        
        # Time series storage
        self.time_series = defaultdict(lambda: deque(maxlen=1000))
        
    def record_circuit_breaker_state(
        self,
        breaker_name: str,
        state: str,
        failure_rate: float
    ):
        """Record circuit breaker state change."""
        state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
        
        self.circuit_breaker_states.set(
            state_value,
            {"breaker": breaker_name}
        )
        
        # Store time series
        self.time_series[f"cb_{breaker_name}_state"].append({
            "timestamp": datetime.utcnow(),
            "state": state,
            "failure_rate": failure_rate
        })
    
    def record_bulkhead_metrics(
        self,
        bulkhead_name: str,
        utilization: float,
        queue_depth: int,
        rejected: int
    ):
        """Record bulkhead metrics."""
        self.bulkhead_utilization.set(
            utilization * 100,
            {"bulkhead": bulkhead_name}
        )
        
        self.bulkhead_queue_depth.set(
            queue_depth,
            {"bulkhead": bulkhead_name}
        )
        
        # Store time series
        self.time_series[f"bh_{bulkhead_name}_metrics"].append({
            "timestamp": datetime.utcnow(),
            "utilization": utilization,
            "queue_depth": queue_depth,
            "rejected": rejected
        })
    
    def record_retry_attempt(
        self,
        operation: str,
        attempts: int,
        delay_ms: float,
        succeeded: bool
    ):
        """Record retry attempt."""
        self.retry_attempts.record(
            attempts,
            {"operation": operation, "succeeded": str(succeeded)}
        )
        
        if delay_ms > 0:
            self.retry_delay.record(
                delay_ms,
                {"operation": operation}
            )
    
    def record_timeout(
        self,
        operation: str,
        timeout_ms: float,
        actual_ms: float,
        timed_out: bool
    ):
        """Record timeout event."""
        self.operation_duration.record(
            actual_ms,
            {"operation": operation, "timed_out": str(timed_out)}
        )
        
        # Update timeout rate
        self._update_timeout_rate(operation, timed_out)
    
    def calculate_resilience_score(self) -> float:
        """
        Calculate overall resilience health score (0-100).
        
        Based on:
        - Circuit breaker health
        - Bulkhead utilization
        - Retry success rate
        - Timeout rate
        """
        scores = []
        
        # Circuit breaker score (higher is better)
        cb_score = self._calculate_circuit_breaker_score()
        if cb_score is not None:
            scores.append(cb_score)
        
        # Bulkhead score (moderate utilization is good)
        bh_score = self._calculate_bulkhead_score()
        if bh_score is not None:
            scores.append(bh_score)
        
        # Retry score (fewer retries is better)
        retry_score = self._calculate_retry_score()
        if retry_score is not None:
            scores.append(retry_score)
        
        # Timeout score (lower timeout rate is better)
        timeout_score = self._calculate_timeout_score()
        if timeout_score is not None:
            scores.append(timeout_score)
        
        if not scores:
            return 100.0  # No data, assume healthy
        
        # Weighted average
        score = sum(scores) / len(scores)
        
        # Update metric
        self.resilience_score.set(score)
        
        return score
    
    def _calculate_circuit_breaker_score(self) -> Optional[float]:
        """Calculate circuit breaker health score."""
        recent_states = []
        
        for key, values in self.time_series.items():
            if key.startswith("cb_") and key.endswith("_state"):
                recent = [v for v in values if 
                         datetime.utcnow() - v["timestamp"] < timedelta(minutes=5)]
                recent_states.extend(recent)
        
        if not recent_states:
            return None
        
        # Count open states
        open_count = sum(1 for s in recent_states if s["state"] == "open")
        open_ratio = open_count / len(recent_states)
        
        # Score: 100% if all closed, 0% if all open
        return (1 - open_ratio) * 100
    
    def _calculate_bulkhead_score(self) -> Optional[float]:
        """Calculate bulkhead health score."""
        recent_metrics = []
        
        for key, values in self.time_series.items():
            if key.startswith("bh_") and key.endswith("_metrics"):
                recent = [v for v in values if 
                         datetime.utcnow() - v["timestamp"] < timedelta(minutes=5)]
                recent_metrics.extend(recent)
        
        if not recent_metrics:
            return None
        
        # Average utilization
        avg_utilization = sum(m["utilization"] for m in recent_metrics) / len(recent_metrics)
        
        # Score: Best at 50-70% utilization
        if avg_utilization < 0.5:
            score = avg_utilization * 200  # Under-utilized
        elif avg_utilization < 0.7:
            score = 100  # Optimal
        else:
            score = 100 - (avg_utilization - 0.7) * 333  # Over-utilized
        
        return max(0, min(100, score))
    
    def _calculate_retry_score(self) -> Optional[float]:
        """Calculate retry health score."""
        # This would analyze retry patterns
        # For now, return a placeholder
        return 85.0
    
    def _calculate_timeout_score(self) -> Optional[float]:
        """Calculate timeout health score."""
        # This would analyze timeout rates
        # For now, return a placeholder
        return 90.0
    
    def _update_timeout_rate(self, operation: str, timed_out: bool):
        """Update timeout rate for operation."""
        key = f"timeout_{operation}"
        
        self.time_series[key].append({
            "timestamp": datetime.utcnow(),
            "timed_out": timed_out
        })
        
        # Calculate rate for last minute
        recent = [v for v in self.time_series[key] if 
                 datetime.utcnow() - v["timestamp"] < timedelta(minutes=1)]
        
        if recent:
            timeout_count = sum(1 for v in recent if v["timed_out"])
            rate = timeout_count / len(recent) * 100
            
            self.timeout_rate.set(rate, {"operation": operation})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "resilience_score": self.calculate_resilience_score(),
            "circuit_breakers": self._get_circuit_breaker_summary(),
            "bulkheads": self._get_bulkhead_summary(),
            "retries": self._get_retry_summary(),
            "timeouts": self._get_timeout_summary()
        }
    
    def _get_circuit_breaker_summary(self) -> Dict[str, Any]:
        """Get circuit breaker summary."""
        summary = {
            "total": 0,
            "open": 0,
            "half_open": 0,
            "closed": 0
        }
        
        # Count states from recent data
        for key, values in self.time_series.items():
            if key.startswith("cb_") and key.endswith("_state") and values:
                summary["total"] += 1
                latest_state = values[-1]["state"]
                summary[latest_state] = summary.get(latest_state, 0) + 1
        
        return summary
    
    def _get_bulkhead_summary(self) -> Dict[str, Any]:
        """Get bulkhead summary."""
        utilizations = []
        queue_depths = []
        
        for key, values in self.time_series.items():
            if key.startswith("bh_") and key.endswith("_metrics") and values:
                latest = values[-1]
                utilizations.append(latest["utilization"])
                queue_depths.append(latest["queue_depth"])
        
        if utilizations:
            return {
                "count": len(utilizations),
                "avg_utilization": sum(utilizations) / len(utilizations) * 100,
                "total_queued": sum(queue_depths)
            }
        
        return {"count": 0, "avg_utilization": 0, "total_queued": 0}
    
    def _get_retry_summary(self) -> Dict[str, Any]:
        """Get retry summary."""
        # Placeholder - would aggregate from actual retry data
        return {
            "total_retries": 0,
            "success_rate": 0.0,
            "avg_attempts": 0.0
        }
    
    def _get_timeout_summary(self) -> Dict[str, Any]:
        """Get timeout summary."""
        timeout_rates = []
        
        for key, values in self.time_series.items():
            if key.startswith("timeout_") and values:
                recent = [v for v in values if 
                         datetime.utcnow() - v["timestamp"] < timedelta(minutes=1)]
                if recent:
                    timeout_count = sum(1 for v in recent if v["timed_out"])
                    rate = timeout_count / len(recent) * 100
                    timeout_rates.append(rate)
        
        if timeout_rates:
            return {
                "operations": len(timeout_rates),
                "avg_timeout_rate": sum(timeout_rates) / len(timeout_rates),
                "max_timeout_rate": max(timeout_rates)
            }
        
        return {
            "operations": 0,
            "avg_timeout_rate": 0.0,
            "max_timeout_rate": 0.0
        }


class MetricsCollector:
    """
    Background metrics collector for resilience patterns.
    
    Periodically collects and aggregates metrics from all
    resilience components.
    """
    
    def __init__(
        self,
        metrics: ResilienceMetrics,
        collection_interval: timedelta = timedelta(seconds=10)
    ):
        self.metrics = metrics
        self.collection_interval = collection_interval
        self.components: Dict[str, Any] = {}
        self._collection_task: Optional[asyncio.Task] = None
        
    def register_component(self, name: str, component: Any):
        """Register a resilience component for metrics collection."""
        self.components[name] = component
        
    async def start(self):
        """Start metrics collection."""
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started resilience metrics collector")
        
    async def stop(self):
        """Stop metrics collection."""
        if self._collection_task:
            self._collection_task.cancel()
            
    async def _collection_loop(self):
        """Main collection loop."""
        while True:
            try:
                await asyncio.sleep(self.collection_interval.total_seconds())
                await self._collect_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _collect_metrics(self):
        """Collect metrics from all components."""
        for name, component in self.components.items():
            try:
                if hasattr(component, "get_metrics"):
                    metrics = component.get_metrics()
                    await self._process_component_metrics(name, metrics)
                    
            except Exception as e:
                logger.error(f"Error collecting metrics from {name}: {e}")
    
    async def _process_component_metrics(self, component_name: str, metrics: Dict[str, Any]):
        """Process metrics from a component."""
        # Circuit breaker metrics
        if "state" in metrics:
            self.metrics.record_circuit_breaker_state(
                component_name,
                metrics["state"],
                metrics.get("failure_rate", 0.0)
            )
        
        # Bulkhead metrics
        if "utilization" in metrics:
            self.metrics.record_bulkhead_metrics(
                component_name,
                metrics["utilization"],
                metrics.get("queue_size", 0),
                metrics.get("rejected", 0)
            )
        
        # Add more metric processing as needed


# Global metrics instance
resilience_metrics = ResilienceMetrics()
metrics_collector = MetricsCollector(resilience_metrics)