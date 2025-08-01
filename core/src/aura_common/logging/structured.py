"""
📝 Structured Logging for AURA
Modern structured logging with OpenTelemetry integration.
"""

from typing import Any, Optional, MutableMapping, ClassVar
from collections.abc import Mapping
import structlog
from structlog.types import FilteringBoundLogger, Processor
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
import json
import sys
from datetime import datetime
from contextvars import ContextVar

# Context variable for storing logger context
_logger_context: ContextVar[dict[str, Any]] = ContextVar('logger_context', default={})


class AuraLogger:
    """
    Modern structured logger with OpenTelemetry integration.
    
    Features:
    - Structured JSON logging
    - Automatic trace context injection
    - Correlation ID tracking
    - Performance metrics
    - Error fingerprinting
    """
    
    _instances: ClassVar[dict[str, 'AuraLogger']] = {}
    _configured: ClassVar[bool] = False
    
    def __init__(self, name: str, **context: Any):
        """Initialize logger with service name and context."""
        self.name = name
        self.context = context
        self._logger = self._get_structured_logger()
        self._tracer = trace.get_tracer(name)
    
    @classmethod
    def _configure_structlog(cls) -> None:
        """Configure structlog with our processors."""
        if cls._configured:
            return
            
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                cls._add_trace_context,
                cls._add_correlation_id,
                cls._add_service_info,
                cls._add_error_fingerprint,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        cls._configured = True
    
    @staticmethod
    def _add_trace_context(
        logger: FilteringBoundLogger, 
        method_name: str, 
        event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Add OpenTelemetry trace context to logs."""
        span = trace.get_current_span()
        if span.is_recording():
            span_context = span.get_span_context()
            event_dict['trace_id'] = format(span_context.trace_id, '032x')
            event_dict['span_id'] = format(span_context.span_id, '016x')
            event_dict['trace_flags'] = format(span_context.trace_flags, '02x')
        return event_dict
    
    @staticmethod
    def _add_correlation_id(
        logger: FilteringBoundLogger,
        method_name: str,
        event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Add correlation ID from context."""
        from .correlation import get_correlation_id
        correlation_id = get_correlation_id()
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
        return event_dict
    
    @staticmethod
    def _add_service_info(
        logger: FilteringBoundLogger,
        method_name: str,
        event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Add service metadata."""
        event_dict['service'] = {
            'name': logger.name,
            'version': '2.0.0',
            'environment': 'production'  # Should come from config
        }
        return event_dict
    
    @staticmethod
    def _add_error_fingerprint(
        logger: FilteringBoundLogger,
        method_name: str,
        event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Add error fingerprint for deduplication."""
        if 'exception' in event_dict:
            import hashlib
            exc_info = event_dict['exception']
            fingerprint = hashlib.sha256(
                f"{exc_info['type']}{exc_info['value']}".encode()
            ).hexdigest()[:12]
            event_dict['error_fingerprint'] = fingerprint
        return event_dict
    
    def _get_structured_logger(self) -> FilteringBoundLogger:
        """Get or create structured logger instance."""
        self._configure_structlog()
        logger = structlog.get_logger(self.name)
        return logger.bind(**self.context)
    
    def bind(self, **context: Any) -> 'AuraLogger':
        """Create new logger with additional context."""
        new_context = {**self.context, **context}
        return AuraLogger(self.name, **new_context)
    
    def with_span(self, name: str, **attributes: Any) -> Span:
        """Create a new span with automatic logging."""
        span = self._tracer.start_span(name, attributes=attributes)
        self.info(f"Started span: {name}", span_name=name, **attributes)
        return span
    
    # Logging methods with modern type hints
    def debug(self, msg: str, /, **kwargs: Any) -> None:
        """Log debug message with structured data."""
        self._logger.debug(msg, **kwargs)
    
    def info(self, msg: str, /, **kwargs: Any) -> None:
        """Log info message with structured data."""
        self._logger.info(msg, **kwargs)
    
    def warning(self, msg: str, /, **kwargs: Any) -> None:
        """Log warning message with structured data."""
        self._logger.warning(msg, **kwargs)
    
    def error(
        self, 
        msg: str, 
        /, 
        exc_info: Optional[Exception] = None,
        **kwargs: Any
    ) -> None:
        """Log error message with exception info."""
        if exc_info:
            kwargs['exc_info'] = exc_info
        self._logger.error(msg, **kwargs)
    
    def critical(
        self, 
        msg: str, 
        /, 
        exc_info: Optional[Exception] = None,
        **kwargs: Any
    ) -> None:
        """Log critical message with exception info."""
        if exc_info:
            kwargs['exc_info'] = exc_info
        self._logger.critical(msg, **kwargs)
    
    def measure_performance(
        self, 
        operation: str,
        duration_ms: float,
        /, 
        **metadata: Any
    ) -> None:
        """Log performance metrics."""
        self.info(
            f"Performance: {operation}",
            performance={
                'operation': operation,
                'duration_ms': duration_ms,
                'metadata': metadata
            }
        )
    
    def log_event(
        self,
        event_type: str,
        /, 
        severity: str = "info",
        **event_data: Any
    ) -> None:
        """Log a structured event."""
        log_method = getattr(self, severity, self.info)
        log_method(
            f"Event: {event_type}",
            event={
                'type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data': event_data
            }
        )


# Global logger cache
_logger_cache: dict[str, AuraLogger] = {}


def get_logger(
    name: str, 
    /, 
    **context: Any
) -> AuraLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        **context: Additional context to bind
        
    Returns:
        AuraLogger instance
        
    Example:
        ```python
        logger = get_logger(__name__, component="tda-engine")
        logger.info("Engine started", engine_type="mojo")
        ```
    """
    cache_key = f"{name}:{json.dumps(context, sort_keys=True)}"
    
    if cache_key not in _logger_cache:
        _logger_cache[cache_key] = AuraLogger(name, **context)
    
    return _logger_cache[cache_key]