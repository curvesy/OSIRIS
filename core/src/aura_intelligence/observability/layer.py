"""
Observability Layer for AURA Intelligence
"""

from typing import Dict, Any, Optional, Callable
import logging
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class ObservabilityLayer:
    """
    Unified observability layer for metrics, tracing, and logging.
    """
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.traces: list = []
        self.enabled = True
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if not self.enabled:
            return
            
        metric_key = f"{name}:{tags}" if tags else name
        if metric_key not in self.metrics:
            self.metrics[metric_key] = []
        self.metrics[metric_key].append(value)
        
    def start_trace(self, operation: str) -> Dict[str, Any]:
        """Start a new trace."""
        if not self.enabled:
            return {}
            
        trace = {
            "operation": operation,
            "start_time": time.time(),
            "status": "in_progress"
        }
        self.traces.append(trace)
        return trace
        
    def end_trace(self, trace: Dict[str, Any], status: str = "success"):
        """End a trace."""
        if not self.enabled or not trace:
            return
            
        trace["end_time"] = time.time()
        trace["status"] = status
        trace["duration"] = trace["end_time"] - trace["start_time"]
        
    @contextmanager
    def trace(self, operation: str):
        """Context manager for tracing."""
        trace = self.start_trace(operation)
        try:
            yield trace
            self.end_trace(trace, "success")
        except Exception as e:
            self.end_trace(trace, "error")
            raise
            
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event."""
        if not self.enabled:
            return
            
        logger.info(f"Event: {event_type}", extra=data)
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recorded metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                }
        return summary
        
    def clear(self):
        """Clear all recorded data."""
        self.metrics.clear()
        self.traces.clear()