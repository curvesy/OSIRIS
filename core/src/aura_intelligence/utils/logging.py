"""
Logging utilities for AURA Intelligence.

Provides structured logging with JSON support, log rotation, and performance tracking.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import structlog
from pydantic import BaseModel


class LogConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"  # "json" or "text"
    file_path: Optional[Path] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True


def setup_logging(
    config: Optional[LogConfig] = None,
    service_name: str = "aura-intelligence"
) -> None:
    """
    Set up structured logging for the application.
    
    Args:
        config: Logging configuration
        service_name: Name of the service for log context
    """
    if config is None:
        config = LogConfig()
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]
    
    # Add service context
    processors.append(
        structlog.processors.add_log_level_number
    )
    processors.append(
        lambda _, __, event_dict: {**event_dict, "service": service_name}
    )
    
    # Configure output format
    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))
        
        if config.format == "json":
            console_handler.setFormatter(JsonFormatter(service_name=service_name))
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.enable_file and config.file_path:
        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        
        if config.format == "json":
            file_handler.setFormatter(JsonFormatter(service_name=service_name))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        
        root_logger.addHandler(file_handler)


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def __init__(self, service_name: str = "aura-intelligence"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName", "exc_info", "exc_text"
            }:
                log_data[key] = value
        
        return json.dumps(log_data)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary log context."""
    
    def __init__(self, logger: structlog.BoundLogger, **kwargs: Any):
        self.logger = logger
        self.context = kwargs
        self.token = None
    
    def __enter__(self) -> structlog.BoundLogger:
        """Enter context and bind values."""
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self.logger
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and unbind values."""
        if self.token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_with_context(logger: structlog.BoundLogger, **context: Any) -> LogContext:
    """
    Create a context manager for temporary log context.
    
    Example:
        with log_with_context(logger, request_id="123", user_id="456"):
            logger.info("Processing request")  # Will include request_id and user_id
    """
    return LogContext(logger, **context)