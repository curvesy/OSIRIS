"""
Logging utilities for AURA Intelligence
"""

import logging
from typing import Optional, Any, Dict
from functools import wraps
import uuid

# Use standard logging if structlog is not available
try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    structlog = None


def get_logger(name: str):
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structured logger or standard logger
    """
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    else:
        # Fallback to standard logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


def with_correlation_id(correlation_id: Optional[str] = None):
    """
    Decorator to add correlation ID to function execution.
    
    Args:
        correlation_id: Optional correlation ID, generates one if not provided
        
    Returns:
        Decorated function with correlation ID in context
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cid = correlation_id or str(uuid.uuid4())
            logger = get_logger(func.__module__)
            
            # Bind correlation ID to logger context
            bound_logger = logger.bind(correlation_id=cid)
            
            # Store in kwargs for function use
            kwargs['_correlation_id'] = cid
            kwargs['_logger'] = bound_logger
            
            try:
                bound_logger.info(f"Starting {func.__name__}", 
                                 correlation_id=cid)
                result = func(*args, **kwargs)
                bound_logger.info(f"Completed {func.__name__}", 
                                 correlation_id=cid)
                return result
            except Exception as e:
                bound_logger.error(f"Error in {func.__name__}", 
                                  error=str(e), 
                                  correlation_id=cid)
                raise
                
        return wrapper
    return decorator


def get_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


# Configure structlog
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)