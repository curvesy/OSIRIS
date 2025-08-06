"""
Production Observability for Shape Memory V2
===========================================

Implements 2025-grade observability using OpenTelemetry for distributed
tracing and structured logging, and Prometheus for metrics. This module
is designed to be a high-performance, low-overhead foundation for
monitoring the entire memory system.
"""

import time
import functools
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode, Span
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# --- Prometheus Metrics Setup ---
# Using an explicit registry allows for better isolation and testing.
# In a real microservice, this would be exposed via a /metrics endpoint.
PROMETHEUS_REGISTRY = CollectorRegistry()

# The "Golden Signals" of SRE: Latency, Traffic, Errors, Saturation
QUERY_LATENCY = Histogram(
    'shape_memory_query_duration_seconds',
    'Query latency in seconds.',
    ['operation', 'backend'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=PROMETHEUS_REGISTRY
)

QUERY_TRAFFIC = Counter(
    'shape_memory_queries_total',
    'Total number of queries.',
    ['operation', 'status'],
    registry=PROMETHEUS_REGISTRY
)

RECALL_GAUGE = Gauge(
    'shape_memory_recall_at_5',
    'Estimated recall@5 from the nightly watchdog job.',
    registry=PROMETHEUS_REGISTRY
)

FALSE_POSITIVE_GAUGE = Gauge(
    'shape_memory_false_positive_rate',
    'False positive rate from shadow deployment.',
    registry=PROMETHEUS_REGISTRY
)

MEMORY_VECTORS = Gauge(
    'shape_memory_vectors_total',
    'Total number of vectors in the memory store.',
    ['backend'],
    registry=PROMETHEUS_REGISTRY
)

EMBEDDING_AGE = Histogram(
    'shape_memory_embedding_age_hours',
    'Age of embeddings in hours for staleness tracking.',
    buckets=[1, 6, 12, 24, 48, 72, 168],  # 1h to 1 week
    registry=PROMETHEUS_REGISTRY
)

# --- OpenTelemetry Tracing Setup ---
# This setup assumes an OTel collector is running and configured.
tracer = trace.get_tracer("aura_intelligence.shape_memory_v2", "1.0.0")

# --- Structured Logging ---
# A custom formatter to inject trace context directly into logs.
class TraceContextFormatter(logging.Formatter):
    def format(self, record):
        span = trace.get_current_span()
        if span.is_recording():
            ctx = span.get_span_context()
            record.otelTraceID = format(ctx.trace_id, '032x')
            record.otelSpanID = format(ctx.span_id, '016x')
        else:
            record.otelTraceID = "0" * 32
            record.otelSpanID = "0" * 16
        return super().format(record)

def setup_logging():
    """Configure structured logging with trace correlation."""
    handler = logging.StreamHandler()
    formatter = TraceContextFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - %(message)s'
    )
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

# --- Decorator for Tracing and Metrics ---

def instrument(operation: str, backend: str = "redis"):
    """
    A decorator that provides production-grade tracing and metrics for a function.
    This is the core of our observability strategy.

    Args:
        operation: The logical name of the operation (e.g., 'store', 'retrieve').
        backend: The system being interacted with (e.g., 'redis', 'neo4j').
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Start a new span in the trace
            with tracer.start_as_current_span(f"ShapeMemory.{operation}") as span:
                # Add useful attributes to the span for debugging
                span.set_attribute("memory.operation", operation)
                span.set_attribute("memory.backend", backend)
                if "k" in kwargs:
                    span.set_attribute("memory.retrieval.k", kwargs["k"])

                start_time = time.perf_counter()
                try:
                    # 2. Execute the actual function
                    result = func(*args, **kwargs)

                    # 3. Record success metrics
                    QUERY_TRAFFIC.labels(operation=operation, status="success").inc()
                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    # 4. Record failure metrics and log the exception
                    QUERY_TRAFFIC.labels(operation=operation, status="error").inc()
                    span.set_status(Status(StatusCode.ERROR, description=str(e)))
                    span.record_exception(e)
                    logging.getLogger(func.__module__).error(
                        f"Operation '{operation}' failed: {e}", exc_info=True
                    )
                    raise

                finally:
                    # 5. Always record latency
                    duration = time.perf_counter() - start_time
                    QUERY_LATENCY.labels(operation=operation, backend=backend).observe(duration)
        return wrapper
    return decorator

# --- Centralized Metric Updates ---

def update_recall(value: float):
    """Updates the recall@5 metric from the watchdog job."""
    RECALL_GAUGE.set(value)

def update_false_positive_rate(value: float):
    """Updates the false positive rate from shadow deployment."""
    FALSE_POSITIVE_GAUGE.set(value)

def update_vector_count(backend: str, count: int):
    """Updates the total vector count for a given backend."""
    MEMORY_VECTORS.labels(backend=backend).set(count)

def record_embedding_age(age_hours: float):
    """Records the age of an embedding for staleness tracking."""
    EMBEDDING_AGE.observe(age_hours)

# --- Context Manager for Manual Tracing ---

@contextmanager
def trace_operation(operation: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for manual tracing of operations.
    
    Usage:
        with trace_operation("custom_operation", {"key": "value"}):
            # do something
    """
    with tracer.start_as_current_span(operation) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(f"shape_memory.{key}", value)
        
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

# Initialize logging on module import
setup_logging()

# Export the old interface for backward compatibility
# (will be removed in next refactoring)
class ObservabilityManager:
    """Deprecated - use module-level functions instead."""
    
    def __init__(self, config=None):
        import warnings
        warnings.warn(
            "ObservabilityManager is deprecated. Use module-level functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def update_memory_count(self, backend: str, count: int):
        update_vector_count(backend, count)
    
    def record_embedding_age(self, age_hours: float):
        record_embedding_age(age_hours)

# Keep singleton for backward compatibility
observability = ObservabilityManager()

# Backward compatibility for traced decorator
def traced(operation: str):
    """Deprecated - use @instrument instead."""
    import warnings
    warnings.warn(
        "traced() is deprecated. Use @instrument(operation, backend) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return instrument(operation, "redis")