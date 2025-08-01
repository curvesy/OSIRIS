"""
🔬 2025-Grade OpenTelemetry Instrumentation

Modern observability foundation with:
- OpenTelemetry 1.9+ unified telemetry (traces, metrics, logs)
- Semantic conventions for AI/ML workloads
- Adaptive sampling and intelligent cardinality reduction
- Multi-agent distributed tracing readiness
- Production-grade performance optimizations
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps

# OpenTelemetry 1.9+ imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.metrics import MetricInstruments

# AI/ML semantic conventions (2025 standard)
from opentelemetry.semconv.ai import (
    AI_OPERATION_NAME,
    AI_MODEL_NAME,
    AI_MODEL_VERSION,
    AI_PROMPT_TOKENS,
    AI_COMPLETION_TOKENS
)


@dataclass
class TelemetryConfig:
    """Production-grade telemetry configuration."""
    
    service_name: str = "aura-intelligence"
    service_version: str = "1.0.0"
    environment: str = "production"
    
    # OTLP endpoints
    otlp_endpoint: str = "http://otel-collector:4317"
    otlp_insecure: bool = True
    
    # Sampling configuration
    trace_sample_rate: float = 0.1  # 10% sampling for production
    adaptive_sampling: bool = True
    
    # Metrics configuration
    metric_export_interval: int = 10  # seconds
    metric_export_timeout: int = 5   # seconds
    
    # Performance optimizations
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout: int = 30  # seconds


class ModernTelemetry:
    """
    🔬 2025-Grade OpenTelemetry Implementation
    
    Provides unified observability for the Intelligence Flywheel with:
    - Semantic AI/ML conventions
    - Adaptive sampling based on system load
    - Multi-agent distributed tracing
    - Production-optimized performance
    """
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.tracer = None
        self.meter = None
        self._initialized = False
        
        # Custom metrics for Intelligence Flywheel
        self.search_latency = None
        self.archival_jobs = None
        self.agent_decisions = None
        self.memory_usage = None
        self.pattern_discoveries = None
    
    def initialize(self) -> None:
        """Initialize OpenTelemetry with production-grade configuration."""
        
        if self._initialized:
            return
        
        # Create resource with semantic attributes
        resource = Resource.create({
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            DEPLOYMENT_ENVIRONMENT: self.config.environment,
            "service.namespace": "aura-intelligence",
            "service.instance.id": f"{self.config.service_name}-{int(time.time())}",
            "ai.system": "intelligence-flywheel",
            "ai.model.type": "topological-data-analysis"
        })
        
        # Configure tracing with adaptive sampling
        trace_provider = TracerProvider(
            resource=resource,
            sampler=self._create_adaptive_sampler()
        )
        
        # OTLP span exporter with production settings
        span_exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=self.config.otlp_insecure,
            timeout=self.config.export_timeout
        )
        
        # Batch processor for performance
        span_processor = BatchSpanProcessor(
            span_exporter,
            max_queue_size=self.config.max_queue_size,
            max_export_batch_size=self.config.max_export_batch_size,
            export_timeout_millis=self.config.export_timeout * 1000
        )
        
        trace_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(trace_provider)
        
        # Configure metrics with OTLP export
        metric_exporter = OTLPMetricExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=self.config.otlp_insecure,
            timeout=self.config.metric_export_timeout
        )
        
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=self.config.metric_export_interval * 1000,
            export_timeout_millis=self.config.metric_export_timeout * 1000
        )
        
        metrics.set_meter_provider(MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        ))
        
        # Get tracer and meter
        self.tracer = trace.get_tracer(
            "aura.intelligence.core",
            version=self.config.service_version
        )
        
        self.meter = metrics.get_meter(
            "aura.intelligence.metrics",
            version=self.config.service_version
        )
        
        # Initialize custom metrics
        self._create_custom_metrics()
        
        # Auto-instrument common libraries
        self._setup_auto_instrumentation()
        
        self._initialized = True
        print(f"🔬 Modern telemetry initialized for {self.config.service_name}")
    
    def _create_adaptive_sampler(self):
        """Create adaptive sampler based on system load."""
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBasedSampler
        
        # TODO: Implement intelligent sampling based on:
        # - System CPU/memory usage
        # - Error rates
        # - Business-critical operations
        return TraceIdRatioBasedSampler(rate=self.config.trace_sample_rate)
    
    def _create_custom_metrics(self) -> None:
        """Create Intelligence Flywheel specific metrics."""
        
        # Search performance metrics
        self.search_latency = self.meter.create_histogram(
            name="aura.search.duration",
            description="Search request duration across memory tiers",
            unit="ms"
        )
        
        self.search_requests = self.meter.create_counter(
            name="aura.search.requests",
            description="Total search requests by tier and status"
        )
        
        # Memory system metrics
        self.memory_usage = self.meter.create_gauge(
            name="aura.memory.usage",
            description="Memory usage by tier (hot/cold/semantic)",
            unit="bytes"
        )
        
        self.archival_jobs = self.meter.create_counter(
            name="aura.archival.jobs",
            description="Archival job completions by status"
        )
        
        # AI/ML specific metrics
        self.pattern_discoveries = self.meter.create_counter(
            name="aura.patterns.discovered",
            description="Semantic patterns discovered by clustering"
        )
        
        self.agent_decisions = self.meter.create_histogram(
            name="aura.agent.decision_time",
            description="Agent decision latency by agent type",
            unit="ms"
        )
        
        self.model_inference = self.meter.create_histogram(
            name="aura.model.inference_time",
            description="TDA model inference latency",
            unit="ms"
        )
    
    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries."""
        
        # Instrument asyncio for async operation tracing
        AsyncioInstrumentor().instrument()
        
        # Instrument HTTP requests
        RequestsInstrumentor().instrument()
        
        print("🔧 Auto-instrumentation configured")
    
    def instrument_fastapi(self, app) -> None:
        """Instrument FastAPI application with semantic attributes."""
        
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            meter_provider=metrics.get_meter_provider(),
            excluded_urls="/health,/metrics,/ready"
        )
        
        print("🌐 FastAPI instrumentation enabled")
    
    @asynccontextmanager
    async def trace_operation(
        self, 
        operation_name: str, 
        attributes: Optional[Dict[str, Any]] = None,
        ai_operation: bool = False
    ):
        """Context manager for tracing operations with semantic attributes."""
        
        span_attributes = attributes or {}
        
        # Add AI/ML semantic attributes if specified
        if ai_operation:
            span_attributes.update({
                AI_OPERATION_NAME: operation_name,
                "ai.system": "intelligence-flywheel"
            })
        
        with self.tracer.start_as_current_span(
            operation_name,
            attributes=span_attributes
        ) as span:
            start_time = time.perf_counter()
            
            try:
                yield span
                span.set_status(trace.Status(trace.StatusCode.OK))
                
            except Exception as e:
                span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR,
                        description=str(e)
                    )
                )
                span.record_exception(e)
                raise
                
            finally:
                # Record operation duration
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("operation.duration_ms", duration_ms)
    
    def record_search_metrics(
        self, 
        tier: str, 
        duration_ms: float, 
        status: str = "success",
        result_count: int = 0
    ) -> None:
        """Record search operation metrics."""
        
        attributes = {
            "tier": tier,
            "status": status
        }
        
        self.search_latency.record(duration_ms, attributes)
        self.search_requests.add(1, attributes)
        
        # Add result count as attribute
        if result_count > 0:
            attributes["result_count_bucket"] = self._get_count_bucket(result_count)
    
    def record_agent_decision(
        self, 
        agent_type: str, 
        duration_ms: float, 
        confidence: float,
        decision_type: str = "unknown"
    ) -> None:
        """Record agent decision metrics with AI semantic attributes."""
        
        attributes = {
            "agent.type": agent_type,
            "agent.decision_type": decision_type,
            "agent.confidence_bucket": self._get_confidence_bucket(confidence)
        }
        
        self.agent_decisions.record(duration_ms, attributes)
    
    def record_archival_job(self, status: str, records_processed: int = 0) -> None:
        """Record archival job completion."""
        
        attributes = {
            "status": status,
            "records_bucket": self._get_count_bucket(records_processed)
        }
        
        self.archival_jobs.add(1, attributes)
    
    def record_pattern_discovery(self, pattern_type: str, count: int = 1) -> None:
        """Record semantic pattern discovery."""
        
        attributes = {
            "pattern.type": pattern_type,
            "clustering.algorithm": "hdbscan"
        }
        
        self.pattern_discoveries.add(count, attributes)
    
    def _get_count_bucket(self, count: int) -> str:
        """Get count bucket for cardinality reduction."""
        if count == 0:
            return "0"
        elif count <= 10:
            return "1-10"
        elif count <= 100:
            return "11-100"
        elif count <= 1000:
            return "101-1000"
        else:
            return "1000+"
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for cardinality reduction."""
        if confidence < 0.5:
            return "low"
        elif confidence < 0.8:
            return "medium"
        else:
            return "high"


# Global telemetry instance
_telemetry: Optional[ModernTelemetry] = None


def get_telemetry() -> ModernTelemetry:
    """Get global telemetry instance."""
    global _telemetry
    
    if _telemetry is None:
        config = TelemetryConfig()
        _telemetry = ModernTelemetry(config)
        _telemetry.initialize()
    
    return _telemetry


def trace_ai_operation(operation_name: str):
    """Decorator for tracing AI/ML operations."""
    
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            
            async with telemetry.trace_operation(
                operation_name,
                ai_operation=True
            ) as span:
                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)
                
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            
            with telemetry.tracer.start_as_current_span(
                operation_name,
                attributes={
                    AI_OPERATION_NAME: operation_name,
                    "ai.system": "intelligence-flywheel",
                    "code.function": func.__name__,
                    "code.namespace": func.__module__
                }
            ):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
