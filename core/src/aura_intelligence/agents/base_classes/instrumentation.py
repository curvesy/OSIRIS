"""
📊 Agent Instrumentation - OpenTelemetry Integration

Comprehensive observability instrumentation for agents with:
- Automatic tracing of agent methods
- Performance metrics collection
- Error tracking and alerting
- Custom span attributes and events
- Integration with existing observability cockpit

Based on the 2025-grade observability from the existing system.
"""

import functools
import time
import inspect
from typing import Any, Callable, Dict, Optional, List
from datetime import datetime, timezone

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.metrics import Counter, Histogram, Gauge
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
except ImportError:
    # OpenTelemetry is optional - provide fallbacks
    trace = None
    metrics = None
    Status = None
    StatusCode = None
    Span = None
    Counter = None
    Histogram = None
    Gauge = None
    inject = None
    extract = None
    TraceContextTextMapPropagator = None

from ..schemas.acp import ACPEnvelope
from ..schemas.state import AgentState
from ..schemas.log import AgentActionEvent, ActionType, ActionResult

# Initialize tracer with fallback
if trace:
    tracer = trace.get_tracer(__name__)
else:
    # Fallback tracer
    class NoOpTracer:
        def start_as_current_span(self, name, **kwargs):
            def decorator(func):
                return func
            return decorator
    tracer = NoOpTracer()

# Initialize meter with fallback
if metrics:
    meter = metrics.get_meter(__name__)
else:
    # Fallback meter
    class NoOpMeter:
        def create_counter(self, **kwargs):
            class NoOpCounter:
                def add(self, value, attributes=None):
                    pass
            return NoOpCounter()
        def create_histogram(self, **kwargs):
            class NoOpHistogram:
                def record(self, value, attributes=None):
                    pass
            return NoOpHistogram()
        def create_gauge(self, **kwargs):
            class NoOpGauge:
                def set(self, value, attributes=None):
                    pass
            return NoOpGauge()
    meter = NoOpMeter()


class AgentMetrics:
    """Centralized metrics collection for agents."""
    
    def __init__(self):
        # Counters
        self.method_calls = meter.create_counter(
            name="agent_method_calls_total",
            description="Total number of agent method calls",
            unit="1"
        )
        
        self.method_errors = meter.create_counter(
            name="agent_method_errors_total",
            description="Total number of agent method errors",
            unit="1"
        )
        
        self.memory_operations = meter.create_counter(
            name="agent_memory_operations_total",
            description="Total number of memory operations",
            unit="1"
        )
        
        self.message_operations = meter.create_counter(
            name="agent_message_operations_total",
            description="Total number of message operations",
            unit="1"
        )
        
        # Histograms
        self.method_duration = meter.create_histogram(
            name="agent_method_duration_ms",
            description="Agent method execution duration",
            unit="ms"
        )
        
        self.memory_query_duration = meter.create_histogram(
            name="agent_memory_query_duration_ms",
            description="Memory query duration",
            unit="ms"
        )
        
        self.message_processing_duration = meter.create_histogram(
            name="agent_message_processing_duration_ms",
            description="Message processing duration",
            unit="ms"
        )
        
        # Gauges
        self.active_tasks = meter.create_gauge(
            name="agent_active_tasks",
            description="Number of active tasks per agent",
            unit="1"
        )
        
        self.confidence_score = meter.create_histogram(
            name="agent_confidence_score",
            description="Agent confidence scores",
            unit="1"
        )


# Global metrics instance
agent_metrics = AgentMetrics()


def instrument_agent(
    operation_type: str = "general",
    record_args: bool = False,
    record_result: bool = False,
    extract_correlation_id: bool = True
):
    """
    Decorator to instrument agent methods with OpenTelemetry.
    
    Args:
        operation_type: Type of operation for categorization
        record_args: Whether to record method arguments
        record_result: Whether to record method results
        extract_correlation_id: Whether to extract correlation ID from args
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract agent info from self
            agent_self = args[0] if args else None
            agent_id = getattr(agent_self, 'agent_id', 'unknown')
            agent_role = getattr(agent_self, 'role', 'unknown')
            if hasattr(agent_role, 'value'):
                agent_role = agent_role.value
            
            # Create span name
            span_name = f"agent_{operation_type}_{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                # Set basic attributes
                span.set_attributes({
                    "agent.id": agent_id,
                    "agent.role": agent_role,
                    "operation.type": operation_type,
                    "method.name": func.__name__,
                    "method.module": func.__module__
                })
                
                # Extract correlation ID if requested
                correlation_id = None
                if extract_correlation_id:
                    correlation_id = _extract_correlation_id(args, kwargs)
                    if correlation_id:
                        span.set_attribute("correlation.id", correlation_id)
                
                # Record arguments if requested
                if record_args:
                    _record_method_args(span, args, kwargs)
                
                start_time = time.time()
                
                try:
                    # Execute the method
                    if inspect.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Record metrics
                    agent_metrics.method_calls.add(1, {
                        "agent_id": agent_id,
                        "agent_role": agent_role,
                        "operation_type": operation_type,
                        "method": func.__name__,
                        "status": "success"
                    })
                    
                    agent_metrics.method_duration.record(duration_ms, {
                        "agent_id": agent_id,
                        "agent_role": agent_role,
                        "operation_type": operation_type,
                        "method": func.__name__
                    })
                    
                    # Record result if requested
                    if record_result and result is not None:
                        _record_method_result(span, result)
                    
                    # Set span attributes
                    span.set_attributes({
                        "method.duration_ms": duration_ms,
                        "method.status": "success"
                    })
                    
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Record error metrics
                    agent_metrics.method_errors.add(1, {
                        "agent_id": agent_id,
                        "agent_role": agent_role,
                        "operation_type": operation_type,
                        "method": func.__name__,
                        "error_type": type(e).__name__
                    })
                    
                    agent_metrics.method_duration.record(duration_ms, {
                        "agent_id": agent_id,
                        "agent_role": agent_role,
                        "operation_type": operation_type,
                        "method": func.__name__
                    })
                    
                    # Record exception in span
                    span.record_exception(e)
                    span.set_attributes({
                        "method.duration_ms": duration_ms,
                        "method.status": "error",
                        "error.type": type(e).__name__,
                        "error.message": str(e)
                    })
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar implementation for sync functions
            agent_self = args[0] if args else None
            agent_id = getattr(agent_self, 'agent_id', 'unknown')
            agent_role = getattr(agent_self, 'role', 'unknown')
            if hasattr(agent_role, 'value'):
                agent_role = agent_role.value
            
            span_name = f"agent_{operation_type}_{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                span.set_attributes({
                    "agent.id": agent_id,
                    "agent.role": agent_role,
                    "operation.type": operation_type,
                    "method.name": func.__name__,
                    "method.module": func.__module__
                })
                
                if extract_correlation_id:
                    correlation_id = _extract_correlation_id(args, kwargs)
                    if correlation_id:
                        span.set_attribute("correlation.id", correlation_id)
                
                if record_args:
                    _record_method_args(span, args, kwargs)
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    agent_metrics.method_calls.add(1, {
                        "agent_id": agent_id,
                        "agent_role": agent_role,
                        "operation_type": operation_type,
                        "method": func.__name__,
                        "status": "success"
                    })
                    
                    agent_metrics.method_duration.record(duration_ms, {
                        "agent_id": agent_id,
                        "agent_role": agent_role,
                        "operation_type": operation_type,
                        "method": func.__name__
                    })
                    
                    if record_result and result is not None:
                        _record_method_result(span, result)
                    
                    span.set_attributes({
                        "method.duration_ms": duration_ms,
                        "method.status": "success"
                    })
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    agent_metrics.method_errors.add(1, {
                        "agent_id": agent_id,
                        "agent_role": agent_role,
                        "operation_type": operation_type,
                        "method": func.__name__,
                        "error_type": type(e).__name__
                    })
                    
                    span.record_exception(e)
                    span.set_attributes({
                        "method.duration_ms": duration_ms,
                        "method.status": "error",
                        "error.type": type(e).__name__,
                        "error.message": str(e)
                    })
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    raise
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def instrument_memory_operation(operation: str = "query"):
    """Decorator specifically for memory operations."""
    return instrument_agent(
        operation_type=f"memory_{operation}",
        record_args=True,
        record_result=False,  # Don't record full results (could be large)
        extract_correlation_id=True
    )


def instrument_message_operation(operation: str = "send"):
    """Decorator specifically for message operations."""
    return instrument_agent(
        operation_type=f"message_{operation}",
        record_args=False,  # Don't record message content for security
        record_result=False,
        extract_correlation_id=True
    )


def instrument_task_processing():
    """Decorator specifically for task processing."""
    return instrument_agent(
        operation_type="task_processing",
        record_args=False,  # AgentState could be large
        record_result=False,
        extract_correlation_id=True
    )


def _extract_correlation_id(args: tuple, kwargs: dict) -> Optional[str]:
    """Extract correlation ID from method arguments."""
    # Check kwargs first
    if 'correlation_id' in kwargs:
        return kwargs['correlation_id']
    
    # Check for AgentState in args
    for arg in args:
        if isinstance(arg, AgentState):
            return arg.correlation_id
        elif isinstance(arg, ACPEnvelope):
            return arg.correlation_id
        elif isinstance(arg, dict) and 'correlation_id' in arg:
            return arg['correlation_id']
    
    return None


def _record_method_args(span: Span, args: tuple, kwargs: dict) -> None:
    """Record method arguments in span (safely)."""
    try:
        # Record number of args
        span.set_attribute("method.args_count", len(args))
        span.set_attribute("method.kwargs_count", len(kwargs))
        
        # Record specific argument types and values (safely)
        for i, arg in enumerate(args[1:], 1):  # Skip self
            arg_type = type(arg).__name__
            span.set_attribute(f"method.arg_{i}_type", arg_type)
            
            # Record safe values
            if isinstance(arg, (str, int, float, bool)):
                if isinstance(arg, str) and len(arg) < 100:
                    span.set_attribute(f"method.arg_{i}_value", arg)
                elif not isinstance(arg, str):
                    span.set_attribute(f"method.arg_{i}_value", str(arg))
        
        # Record safe kwargs
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) < 100:
                    span.set_attribute(f"method.kwarg_{key}", value)
                elif not isinstance(value, str):
                    span.set_attribute(f"method.kwarg_{key}", str(value))
    
    except Exception:
        # Don't fail the method if argument recording fails
        pass


def _record_method_result(span: Span, result: Any) -> None:
    """Record method result in span (safely)."""
    try:
        result_type = type(result).__name__
        span.set_attribute("method.result_type", result_type)
        
        # Record safe result information
        if isinstance(result, (str, int, float, bool)):
            if isinstance(result, str) and len(result) < 100:
                span.set_attribute("method.result_value", result)
            elif not isinstance(result, str):
                span.set_attribute("method.result_value", str(result))
        elif isinstance(result, (list, tuple)):
            span.set_attribute("method.result_length", len(result))
        elif isinstance(result, dict):
            span.set_attribute("method.result_keys_count", len(result))
        elif hasattr(result, '__len__'):
            try:
                span.set_attribute("method.result_length", len(result))
            except:
                pass
    
    except Exception:
        # Don't fail the method if result recording fails
        pass


def create_child_span(
    name: str,
    parent_context: Optional[Dict[str, str]] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> Span:
    """
    Create a child span with optional parent context.
    
    Args:
        name: Span name
        parent_context: Parent trace context
        attributes: Span attributes
        
    Returns:
        New span
    """
    # Extract parent context if provided
    if parent_context:
        context = extract(parent_context)
        token = trace.set_span_in_context(trace.get_current_span(context))
        span = tracer.start_span(name, context=context)
    else:
        span = tracer.start_span(name)
    
    # Set attributes
    if attributes:
        span.set_attributes(attributes)
    
    return span


def inject_trace_context() -> Dict[str, str]:
    """
    Inject current trace context for propagation.
    
    Returns:
        Trace context headers
    """
    headers = {}
    inject(headers)
    return headers


def record_agent_event(
    event_name: str,
    agent_id: str,
    attributes: Optional[Dict[str, Any]] = None
) -> None:
    """
    Record a custom agent event.
    
    Args:
        event_name: Name of the event
        agent_id: Agent identifier
        attributes: Event attributes
    """
    span = trace.get_current_span()
    
    event_attributes = {
        "agent.id": agent_id,
        "event.timestamp": datetime.now(timezone.utc).isoformat(),
        **(attributes or {})
    }
    
    span.add_event(event_name, event_attributes)


def record_confidence_score(
    agent_id: str,
    agent_role: str,
    confidence: float,
    operation: str = "general"
) -> None:
    """
    Record agent confidence score.
    
    Args:
        agent_id: Agent identifier
        agent_role: Agent role
        confidence: Confidence score (0.0-1.0)
        operation: Operation type
    """
    agent_metrics.confidence_score.record(confidence, {
        "agent_id": agent_id,
        "agent_role": agent_role,
        "operation": operation
    })


def update_active_tasks_gauge(agent_id: str, agent_role: str, count: int) -> None:
    """
    Update active tasks gauge.
    
    Args:
        agent_id: Agent identifier
        agent_role: Agent role
        count: Number of active tasks
    """
    agent_metrics.active_tasks.set(count, {
        "agent_id": agent_id,
        "agent_role": agent_role
    })
