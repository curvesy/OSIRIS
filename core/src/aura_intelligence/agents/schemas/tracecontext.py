"""
🔍 OpenTelemetry Trace Context - W3C Standard Integration

W3C trace context fields and utilities for distributed tracing:
- Traceparent and tracestate handling
- Span ID management
- Context propagation utilities
- OpenTelemetry integration
- Correlation ID support

Ensures end-to-end observability across all agent interactions.
"""

import uuid
import re
from typing import Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field, validator
from opentelemetry import trace
from opentelemetry.propagate import inject, extract
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

try:
    from .base import ImmutableBaseModel
except ImportError:
    # Fallback for direct import (testing/isolation)
    from base import ImmutableBaseModel


# ============================================================================
# W3C TRACE CONTEXT CONSTANTS
# ============================================================================

# W3C traceparent format: version-trace_id-parent_id-trace_flags
TRACEPARENT_REGEX = re.compile(r'^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$')

# Valid trace flags
TRACE_FLAGS_SAMPLED = '01'
TRACE_FLAGS_NOT_SAMPLED = '00'

# Maximum tracestate length (W3C spec)
MAX_TRACESTATE_LENGTH = 512


# ============================================================================
# TRACE CONTEXT MODELS
# ============================================================================

class TraceContext(ImmutableBaseModel):
    """
    W3C trace context information for distributed tracing.
    
    Implements the W3C Trace Context specification for
    propagating trace information across service boundaries.
    """
    
    traceparent: str = Field(
        ...,
        description="W3C traceparent header (version-trace_id-parent_id-trace_flags)"
    )
    
    tracestate: Optional[str] = Field(
        None,
        description="W3C tracestate header for vendor-specific trace data"
    )
    
    span_id: Optional[str] = Field(
        None,
        description="OpenTelemetry span ID for this operation"
    )
    
    correlation_id: Optional[str] = Field(
        None,
        description="Application-level correlation ID"
    )
    
    @validator('traceparent')
    def validate_traceparent(cls, v):
        """Validate traceparent format according to W3C spec."""
        if not v:
            raise ValueError("Traceparent cannot be empty")
        
        if not TRACEPARENT_REGEX.match(v):
            raise ValueError(
                "Traceparent must be in format: version-trace_id-parent_id-trace_flags "
                "(e.g., '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01')"
            )
        
        # Extract components for additional validation
        version, trace_id, parent_id, trace_flags = v.split('-')
        
        # Validate version (currently only 00 is supported)
        if version != '00':
            raise ValueError("Only traceparent version '00' is currently supported")
        
        # Validate trace_id is not all zeros
        if trace_id == '00000000000000000000000000000000':
            raise ValueError("Trace ID cannot be all zeros")
        
        # Validate parent_id is not all zeros
        if parent_id == '0000000000000000':
            raise ValueError("Parent ID cannot be all zeros")
        
        # Validate trace_flags
        if trace_flags not in [TRACE_FLAGS_SAMPLED, TRACE_FLAGS_NOT_SAMPLED]:
            raise ValueError(f"Trace flags must be '{TRACE_FLAGS_SAMPLED}' or '{TRACE_FLAGS_NOT_SAMPLED}'")
        
        return v
    
    @validator('tracestate')
    def validate_tracestate(cls, v):
        """Validate tracestate format according to W3C spec."""
        if v is None:
            return v
        
        if len(v) > MAX_TRACESTATE_LENGTH:
            raise ValueError(f"Tracestate cannot exceed {MAX_TRACESTATE_LENGTH} characters")
        
        # Basic format validation (key=value pairs separated by commas)
        if v and not re.match(r'^[a-z0-9_\-*/]+(=[^,]*)?(?:,[a-z0-9_\-*/]+(=[^,]*)?)*$', v):
            raise ValueError("Tracestate format is invalid")
        
        return v
    
    @validator('span_id')
    def validate_span_id(cls, v):
        """Validate span ID format."""
        if v is None:
            return v
        
        if not re.match(r'^[0-9a-f]{16}$', v):
            raise ValueError("Span ID must be 16 hex characters")
        
        if v == '0000000000000000':
            raise ValueError("Span ID cannot be all zeros")
        
        return v
    
    def get_trace_id(self) -> str:
        """Extract trace ID from traceparent."""
        return self.traceparent.split('-')[1]
    
    def get_parent_id(self) -> str:
        """Extract parent ID from traceparent."""
        return self.traceparent.split('-')[2]
    
    def get_trace_flags(self) -> str:
        """Extract trace flags from traceparent."""
        return self.traceparent.split('-')[3]
    
    def is_sampled(self) -> bool:
        """Check if trace is sampled."""
        return self.get_trace_flags() == TRACE_FLAGS_SAMPLED
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {'traceparent': self.traceparent}
        if self.tracestate:
            headers['tracestate'] = self.tracestate
        return headers
    
    def create_child_context(self, new_span_id: Optional[str] = None) -> 'TraceContext':
        """Create a child trace context with new span ID."""
        if new_span_id is None:
            new_span_id = generate_span_id()
        
        # Create new traceparent with current span as parent
        version, trace_id, _, trace_flags = self.traceparent.split('-')
        new_traceparent = f"{version}-{trace_id}-{new_span_id}-{trace_flags}"
        
        return TraceContext(
            traceparent=new_traceparent,
            tracestate=self.tracestate,
            span_id=new_span_id,
            correlation_id=self.correlation_id
        )


# ============================================================================
# TRACE CONTEXT UTILITIES
# ============================================================================

def generate_trace_id() -> str:
    """Generate a new 32-character hex trace ID."""
    return uuid.uuid4().hex + uuid.uuid4().hex[:16]


def generate_span_id() -> str:
    """Generate a new 16-character hex span ID."""
    return uuid.uuid4().hex[:16]


def generate_traceparent(sampled: bool = True) -> str:
    """
    Generate a new W3C traceparent header.
    
    Args:
        sampled: Whether the trace should be sampled
        
    Returns:
        W3C traceparent string
    """
    version = '00'
    trace_id = generate_trace_id()
    span_id = generate_span_id()
    trace_flags = TRACE_FLAGS_SAMPLED if sampled else TRACE_FLAGS_NOT_SAMPLED
    
    return f"{version}-{trace_id}-{span_id}-{trace_flags}"


def create_trace_context(
    traceparent: Optional[str] = None,
    tracestate: Optional[str] = None,
    correlation_id: Optional[str] = None,
    sampled: bool = True
) -> TraceContext:
    """
    Create a new trace context.
    
    Args:
        traceparent: Existing traceparent or None to generate new
        tracestate: Tracestate header
        correlation_id: Application correlation ID
        sampled: Whether to sample the trace (if generating new)
        
    Returns:
        TraceContext instance
    """
    if traceparent is None:
        traceparent = generate_traceparent(sampled)
    
    span_id = traceparent.split('-')[2] if traceparent else None
    
    return TraceContext(
        traceparent=traceparent,
        tracestate=tracestate,
        span_id=span_id,
        correlation_id=correlation_id
    )


def extract_trace_context_from_headers(headers: Dict[str, str]) -> Optional[TraceContext]:
    """
    Extract trace context from HTTP headers.
    
    Args:
        headers: HTTP headers dictionary
        
    Returns:
        TraceContext if valid headers found, None otherwise
    """
    traceparent = headers.get('traceparent')
    if not traceparent:
        return None
    
    try:
        return TraceContext(
            traceparent=traceparent,
            tracestate=headers.get('tracestate'),
            span_id=traceparent.split('-')[2],
            correlation_id=headers.get('x-correlation-id')
        )
    except ValueError:
        return None


def inject_trace_context_to_headers(context: TraceContext, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Inject trace context into HTTP headers.
    
    Args:
        context: Trace context to inject
        headers: Existing headers dictionary or None
        
    Returns:
        Headers dictionary with trace context
    """
    if headers is None:
        headers = {}
    
    headers.update(context.to_headers())
    
    if context.correlation_id:
        headers['x-correlation-id'] = context.correlation_id
    
    return headers


# ============================================================================
# OPENTELEMETRY INTEGRATION
# ============================================================================

def get_current_trace_context() -> Optional[TraceContext]:
    """
    Get trace context from current OpenTelemetry span.
    
    Returns:
        TraceContext if active span exists, None otherwise
    """
    span = trace.get_current_span()
    if not span or not span.is_recording():
        return None
    
    span_context = span.get_span_context()
    if not span_context.is_valid:
        return None
    
    # Format trace and span IDs
    trace_id = format(span_context.trace_id, '032x')
    span_id = format(span_context.span_id, '016x')
    
    # Determine trace flags
    trace_flags = TRACE_FLAGS_SAMPLED if span_context.trace_flags.sampled else TRACE_FLAGS_NOT_SAMPLED
    
    # Create traceparent
    traceparent = f"00-{trace_id}-{span_id}-{trace_flags}"
    
    return TraceContext(
        traceparent=traceparent,
        span_id=span_id
    )


def create_child_span_with_context(
    name: str,
    parent_context: Optional[TraceContext] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> Tuple[trace.Span, TraceContext]:
    """
    Create a child span with trace context.
    
    Args:
        name: Span name
        parent_context: Parent trace context
        attributes: Span attributes
        
    Returns:
        Tuple of (span, trace_context)
    """
    # If no parent context, use current span or create new
    if parent_context is None:
        parent_context = get_current_trace_context()
        if parent_context is None:
            parent_context = create_trace_context()
    
    # Extract trace context for OpenTelemetry
    headers = parent_context.to_headers()
    otel_context = extract(headers)
    
    # Create child span
    tracer = trace.get_tracer(__name__)
    span = tracer.start_span(name, context=otel_context)
    
    # Set attributes
    if attributes:
        span.set_attributes(attributes)
    
    # Create child trace context
    child_context = parent_context.create_child_context()
    
    return span, child_context


def propagate_trace_context(context: TraceContext) -> Dict[str, str]:
    """
    Prepare trace context for propagation to other services.
    
    Args:
        context: Trace context to propagate
        
    Returns:
        Headers dictionary for HTTP requests
    """
    headers = {}
    
    # Use OpenTelemetry propagator
    propagator = TraceContextTextMapPropagator()
    
    # Create a temporary context with the trace information
    temp_headers = context.to_headers()
    otel_context = extract(temp_headers)
    
    # Inject into headers
    inject(headers, context=otel_context)
    
    # Add correlation ID if present
    if context.correlation_id:
        headers['x-correlation-id'] = context.correlation_id
    
    return headers


# ============================================================================
# TRACE CONTEXT MIXINS
# ============================================================================

class TraceContextMixin:
    """
    Mixin for models that need trace context.
    
    Provides trace context fields and utilities
    for distributed tracing integration.
    
    Note: This is a proper mixin that doesn't inherit from BaseModel
    to avoid MRO conflicts when used with other base classes.
    """
    
    traceparent: str = Field(
        ...,
        description="W3C trace context for OpenTelemetry correlation"
    )
    
    tracestate: Optional[str] = Field(
        None,
        description="W3C trace state for vendor-specific data"
    )
    
    span_id: Optional[str] = Field(
        None,
        description="OpenTelemetry span ID"
    )
    
    def get_trace_context(self) -> TraceContext:
        """Get trace context from this model."""
        return TraceContext(
            traceparent=self.traceparent,
            tracestate=self.tracestate,
            span_id=self.span_id,
            correlation_id=getattr(self, 'correlation_id', None)
        )
    
    def create_child_trace_context(self) -> TraceContext:
        """Create a child trace context."""
        return self.get_trace_context().create_child_context()


# ============================================================================
# CORRELATION ID UTILITIES
# ============================================================================

def generate_correlation_id(prefix: str = "corr") -> str:
    """Generate a correlation ID for request tracing."""
    return f"{prefix}_{uuid.uuid4().hex}"


def extract_correlation_id_from_context(context: TraceContext) -> Optional[str]:
    """Extract correlation ID from trace context."""
    return context.correlation_id


def create_correlation_context(
    correlation_id: Optional[str] = None,
    traceparent: Optional[str] = None
) -> TraceContext:
    """
    Create trace context with correlation ID.
    
    Args:
        correlation_id: Correlation ID or None to generate
        traceparent: Existing traceparent or None to generate
        
    Returns:
        TraceContext with correlation ID
    """
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    return create_trace_context(
        traceparent=traceparent,
        correlation_id=correlation_id
    )


# Export public interface
__all__ = [
    # Models
    'TraceContext', 'TraceContextMixin',
    
    # Constants
    'TRACE_FLAGS_SAMPLED', 'TRACE_FLAGS_NOT_SAMPLED', 'MAX_TRACESTATE_LENGTH',
    
    # Generation utilities
    'generate_trace_id', 'generate_span_id', 'generate_traceparent', 'generate_correlation_id',
    
    # Context creation
    'create_trace_context', 'create_correlation_context',
    
    # Header utilities
    'extract_trace_context_from_headers', 'inject_trace_context_to_headers', 'propagate_trace_context',
    
    # OpenTelemetry integration
    'get_current_trace_context', 'create_child_span_with_context',
    
    # Correlation utilities
    'extract_correlation_id_from_context'
]
