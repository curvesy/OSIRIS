"""
OpenTelemetry integration for distributed tracing
"""

import asyncio
from functools import wraps
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider, sampling
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Callable
import time
import logging

from .config import ObservabilityConfig

logger = logging.getLogger(__name__)


class AdaptiveSampler(sampling.Sampler):
    """
    Adaptive sampler that adjusts sampling rate based on error rate
    and performance characteristics.
    """
    
    def __init__(self, base_rate: float = 0.1):
        self.base_rate = base_rate
        self.error_rate = 0.0
        self.sample_count = 0
        self.error_count = 0
        
    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None
    ) -> sampling.SamplingResult:
        # Always sample if parent was sampled
        if parent_context and parent_context.is_valid:
            parent_span_context = trace.get_current_span(parent_context).get_span_context()
            if parent_span_context.is_valid and parent_span_context.trace_flags.sampled:
                return sampling.SamplingResult(
                    sampling.Decision.RECORD_AND_SAMPLE,
                    attributes={"sampling.reason": "parent_sampled"}
                )
        
        # Increase sampling for errors
        if attributes and attributes.get("error", False):
            return sampling.SamplingResult(
                sampling.Decision.RECORD_AND_SAMPLE,
                attributes={"sampling.reason": "error_detected"}
            )
        
        # Adaptive sampling based on error rate
        if self.error_rate > 0.05:  # If error rate > 5%, sample more
            adjusted_rate = min(1.0, self.base_rate * (1 + self.error_rate * 10))
        else:
            adjusted_rate = self.base_rate
            
        # Sample based on trace ID
        if (trace_id & 0xffffffff) / 0xffffffff < adjusted_rate:
            return sampling.SamplingResult(
                sampling.Decision.RECORD_AND_SAMPLE,
                attributes={"sampling.rate": adjusted_rate}
            )
        else:
            return sampling.SamplingResult(
                sampling.Decision.DROP
            )
    
    def update_error_rate(self, is_error: bool):
        """Update error rate for adaptive sampling"""
        self.sample_count += 1
        if is_error:
            self.error_count += 1
        
        # Update error rate with exponential moving average
        if self.sample_count > 100:
            self.error_rate = self.error_count / self.sample_count
            # Reset counters periodically
            if self.sample_count > 10000:
                self.sample_count = 100
                self.error_count = int(self.error_rate * 100)
    
    def get_description(self) -> str:
        return f"AdaptiveSampler(base_rate={self.base_rate}, current_error_rate={self.error_rate:.2%})"


class OpenTelemetryManager:
    """
    Manages OpenTelemetry configuration and instrumentation
    """
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.tracer_provider: Optional[TracerProvider] = None
        self.sampler: Optional[AdaptiveSampler] = None
        
    def initialize(self):
        """Initialize OpenTelemetry with OTLP exporter"""
        if not self.config.enable_tracing:
            logger.info("Tracing disabled in configuration")
            return
            
        try:
            # Create resource with service information
            resource = Resource.create({
                SERVICE_NAME: self.config.service_name,
                "service.version": "1.0.0",
                "deployment.environment": "production"
            })
            
            # Setup sampler
            if self.config.adaptive_sampling:
                sampler = AdaptiveSampler(self.config.sample_rate)
            else:
                sampler = sampling.TraceIdRatioBased(self.config.sample_rate)
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(
                resource=resource,
                sampler=sampler
            )
            
            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=self.config.otlp_insecure
            )
            
            # Add batch processor
            span_processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                max_export_interval_millis=5000
            )
            
            self.tracer_provider.add_span_processor(span_processor)
            
            # Set as global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            # Set up propagator
            set_global_textmap(TraceContextTextMapPropagator())
            
            # Auto-instrument libraries
            self._setup_auto_instrumentation()
            
            logger.info(
                "OpenTelemetry initialized successfully",
                extra={
                    "endpoint": self.config.otlp_endpoint,
                    "service_name": self.config.service_name,
                    "sample_rate": self.config.sample_rate
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            raise
    
    def _setup_auto_instrumentation(self):
        """Setup automatic instrumentation for common libraries"""
        try:
            # HTTP clients
            RequestsInstrumentor().instrument()
            AioHttpClientInstrumentor().instrument()
            
            # Databases
            SQLAlchemyInstrumentor().instrument()
            RedisInstrumentor().instrument()
            Psycopg2Instrumentor().instrument()
            
            logger.info("Auto-instrumentation configured")
            
        except Exception as e:
            logger.warning(f"Some auto-instrumentation failed: {e}")
    
    def get_tracer(self, name: str = None) -> trace.Tracer:
        """Get a tracer instance"""
        if name is None:
            name = self.config.service_name
        return trace.get_tracer(name)
    
    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL
    ):
        """
        Context manager for tracing operations
        
        Usage:
            with tracer.trace_operation("process_data", {"data.size": 1024}):
                # Your code here
                pass
        """
        tracer = self.get_tracer()
        
        with tracer.start_as_current_span(
            operation_name,
            kind=kind,
            attributes=attributes or {}
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if self.sampler and isinstance(self.sampler, AdaptiveSampler):
                    self.sampler.update_error_rate(True)
                raise
            else:
                if self.sampler and isinstance(self.sampler, AdaptiveSampler):
                    self.sampler.update_error_rate(False)
    
    def record_metric(
        self,
        span: trace.Span,
        metric_name: str,
        value: float,
        unit: str = None
    ):
        """Record a metric as a span event"""
        event_attributes = {
            "metric.name": metric_name,
            "metric.value": value
        }
        if unit:
            event_attributes["metric.unit"] = unit
            
        span.add_event("metric.recorded", attributes=event_attributes)
    
    def add_baggage(self, key: str, value: str):
        """Add baggage item to context"""
        # Baggage propagation for cross-service context
        from opentelemetry import baggage
        return baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage item from context"""
        from opentelemetry import baggage
        return baggage.get_baggage(key)
    
    def inject_context(self, carrier: Dict[str, str]):
        """Inject trace context into carrier for propagation"""
        from opentelemetry.propagate import inject
        inject(carrier)
        
    def extract_context(self, carrier: Dict[str, str]):
        """Extract trace context from carrier"""
        from opentelemetry.propagate import extract
        return extract(carrier)
    
    def shutdown(self):
        """Shutdown tracing and flush remaining spans"""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            logger.info("OpenTelemetry shutdown complete")


# Convenience functions
_manager: Optional[OpenTelemetryManager] = None


def initialize_tracing(config: ObservabilityConfig):
    """Initialize global tracing"""
    global _manager
    _manager = OpenTelemetryManager(config)
    _manager.initialize()
    return _manager


def get_tracer(name: str = None) -> trace.Tracer:
    """Get a tracer instance"""
    if _manager:
        return _manager.get_tracer(name)
    # Return default tracer if not initialized
    return trace.get_tracer(name or "aura_intelligence")


def trace_operation(operation_name: str, **kwargs):
    """Decorator for tracing operations"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(operation_name) as span:
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(operation_name) as span:
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


def trace_span(name: str, **kwargs):
    """Decorator for tracing function execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            async with tracer.start_as_current_span(name) as span:
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


class TracingContext:
    """Simple context manager for tracing."""
    
    def __init__(self, service: str, operation: str):
        self.service = service
        self.operation = operation
        self.span = None
        
    async def __aenter__(self):
        tracer = get_tracer()
        self.span = tracer.start_span(f"{self.service}.{self.operation}")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.end()
            
    def __enter__(self):
        tracer = get_tracer()
        self.span = tracer.start_span(f"{self.service}.{self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.end()