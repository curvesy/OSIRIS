"""
🔍 Distributed Tracing for Streaming TDA
OpenTelemetry integration for cross-service tracing with minimal overhead
"""

import asyncio
from typing import Optional, Dict, Any, Callable, AsyncContextManager, TypeVar
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import time

from opentelemetry import trace, context, baggage
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace import TracerProvider, sampling
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.utils import unwrap
import structlog

from prometheus_client import Counter, Histogram

logger = structlog.get_logger(__name__)

# Metrics
SPANS_CREATED = Counter('tracing_spans_created_total', 'Total spans created', ['service', 'operation'])
SPAN_DURATION = Histogram('tracing_span_duration_seconds', 'Span duration', ['service', 'operation'])
CONTEXT_PROPAGATIONS = Counter('tracing_context_propagations_total', 'Context propagations', ['direction'])

T = TypeVar('T')


class TracingConfig:
    """Configuration for distributed tracing"""
    def __init__(
        self,
        service_name: str = "streaming-tda",
        endpoint: str = "localhost:4317",
        insecure: bool = True,
        sample_rate: float = 1.0,
        max_export_batch_size: int = 512,
        export_timeout_millis: int = 30000,
        adaptive_sampling: bool = True
    ):
        self.service_name = service_name
        self.endpoint = endpoint
        self.insecure = insecure
        self.sample_rate = sample_rate
        self.max_export_batch_size = max_export_batch_size
        self.export_timeout_millis = export_timeout_millis
        self.adaptive_sampling = adaptive_sampling


class AdaptiveSampler(sampling.Sampler):
    """Adaptive sampler that adjusts based on load and errors"""
    
    def __init__(self, base_rate: float = 0.1):
        self.base_rate = base_rate
        self.error_rate = 1.0  # Always sample errors
        self.high_latency_rate = 0.5
        self.latency_threshold_ms = 100
        
    def should_sample(
        self,
        parent_context: Optional[context.Context],
        trace_id: int,
        name: str,
        kind: trace.SpanKind,
        attributes: Dict[str, Any] = None,
        links: Any = None
    ) -> sampling.SamplingResult:
        # Always sample if parent was sampled
        parent_span_context = trace.get_current_span(parent_context).get_span_context()
        if parent_span_context and parent_span_context.is_valid and parent_span_context.trace_flags.sampled:
            return sampling.SamplingResult(
                decision=sampling.Decision.RECORD_AND_SAMPLE,
                attributes=attributes
            )
        
        # Sample errors at higher rate
        if attributes and attributes.get("error", False):
            if trace_id % 100 < self.error_rate * 100:
                return sampling.SamplingResult(
                    decision=sampling.Decision.RECORD_AND_SAMPLE,
                    attributes=attributes
                )
        
        # Sample high latency operations
        if attributes and attributes.get("latency_ms", 0) > self.latency_threshold_ms:
            if trace_id % 100 < self.high_latency_rate * 100:
                return sampling.SamplingResult(
                    decision=sampling.Decision.RECORD_AND_SAMPLE,
                    attributes=attributes
                )
        
        # Default sampling
        if trace_id % 100 < self.base_rate * 100:
            return sampling.SamplingResult(
                decision=sampling.Decision.RECORD_AND_SAMPLE,
                attributes=attributes
            )
            
        return sampling.SamplingResult(
            decision=sampling.Decision.DROP,
            attributes=attributes
        )
    
    def get_description(self) -> str:
        return f"AdaptiveSampler(base_rate={self.base_rate})"


class DistributedTracer:
    """Main distributed tracing implementation"""
    
    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self.tracer: Optional[trace.Tracer] = None
        self.propagator = TraceContextTextMapPropagator()
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the tracer with OTLP exporter"""
        if self._initialized:
            return
            
        try:
            # Create resource
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
            provider = TracerProvider(resource=resource, sampler=sampler)
            
            # Create OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.endpoint,
                insecure=self.config.insecure
            )
            
            # Add batch processor
            span_processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                max_export_batch_size=self.config.max_export_batch_size,
                export_timeout_millis=self.config.export_timeout_millis
            )
            provider.add_span_processor(span_processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            
            self._initialized = True
            logger.info("Distributed tracing initialized", 
                       endpoint=self.config.endpoint,
                       service=self.config.service_name)
            
        except Exception as e:
            logger.error("Failed to initialize tracing", error=str(e))
            raise
    
    @contextmanager
    def trace_operation(self, name: str, **attributes) -> Span:
        """Context manager for tracing operations"""
        if not self.tracer:
            # No-op if not initialized
            yield None
            return
            
        with self.tracer.start_as_current_span(name) as span:
            SPANS_CREATED.labels(
                service=self.config.service_name,
                operation=name
            ).inc()
            
            # Add attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
                
            start_time = time.time()
            try:
                yield span
            finally:
                duration = time.time() - start_time
                SPAN_DURATION.labels(
                    service=self.config.service_name,
                    operation=name
                ).observe(duration)
    
    @asynccontextmanager
    async def trace_async_operation(self, name: str, **attributes) -> AsyncContextManager[Span]:
        """Async context manager for tracing operations"""
        if not self.tracer:
            # No-op if not initialized
            yield None
            return
            
        with self.tracer.start_as_current_span(name) as span:
            SPANS_CREATED.labels(
                service=self.config.service_name,
                operation=name
            ).inc()
            
            # Add attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
                
            start_time = time.time()
            try:
                yield span
            finally:
                duration = time.time() - start_time
                SPAN_DURATION.labels(
                    service=self.config.service_name,
                    operation=name
                ).observe(duration)
    
    def add_span_attributes(self, **attributes) -> None:
        """Add attributes to current span"""
        span = trace.get_current_span()
        if span and span.is_recording():
            for key, value in attributes.items():
                span.set_attribute(key, value)
    
    def inject_context(self, carrier: Dict[str, Any]) -> None:
        """Inject trace context into carrier for propagation"""
        self.propagator.inject(carrier)
        CONTEXT_PROPAGATIONS.labels(direction="inject").inc()
    
    def extract_context(self, carrier: Dict[str, Any]) -> context.Context:
        """Extract trace context from carrier"""
        ctx = self.propagator.extract(carrier)
        CONTEXT_PROPAGATIONS.labels(direction="extract").inc()
        return ctx
    
    def trace_method(self, operation_name: Optional[str] = None):
        """Decorator for tracing methods"""
        def decorator(func: Callable) -> Callable:
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.trace_async_operation(name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.trace_operation(name):
                        return func(*args, **kwargs)
                return sync_wrapper
                
        return decorator
    
    def record_exception(self, exception: Exception) -> None:
        """Record exception in current span"""
        span = trace.get_current_span()
        if span and span.is_recording():
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to current span"""
        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(name, attributes=attributes or {})
    
    def create_baggage(self, key: str, value: str) -> context.Context:
        """Create baggage for cross-service context propagation"""
        return baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage value from context"""
        return baggage.get_baggage(key)
    
    async def shutdown(self) -> None:
        """Graceful shutdown of tracing"""
        if self._initialized and self.tracer:
            provider = trace.get_tracer_provider()
            if hasattr(provider, 'shutdown'):
                provider.shutdown()
            self._initialized = False
            logger.info("Distributed tracing shutdown complete")


# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None


def get_tracer() -> DistributedTracer:
    """Get global tracer instance"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DistributedTracer()
    return _global_tracer


async def initialize_tracing(config: Optional[TracingConfig] = None) -> DistributedTracer:
    """Initialize global tracing"""
    tracer = get_tracer()
    if config:
        tracer.config = config
    await tracer.initialize()
    return tracer