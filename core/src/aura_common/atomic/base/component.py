"""
Base atomic component implementation.

This module provides the foundation for all atomic components in the AURA system.
Each component follows strict single-responsibility principles and provides
built-in instrumentation for observability.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import time
import structlog
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from .exceptions import ComponentError, ConfigurationError, ProcessingError

# Type variables for generic component interface
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
ConfigT = TypeVar('ConfigT')


@dataclass
class ComponentMetrics:
    """Metrics collected during component execution."""
    
    execution_time_ms: float
    input_size: int
    output_size: int
    success: bool
    error: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/monitoring."""
        return {
            "execution_time_ms": self.execution_time_ms,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "success": self.success,
            "error": self.error,
            **self.custom_metrics
        }


class AtomicComponent(ABC, Generic[InputT, OutputT, ConfigT]):
    """
    Base class for all atomic components.
    
    This class enforces single responsibility principle and provides:
    - Automatic instrumentation (logging, tracing, metrics)
    - Configuration validation
    - Error handling
    - Size constraints (implementation must be <150 lines)
    """
    
    def __init__(
        self,
        name: str,
        config: ConfigT,
        logger: Optional[structlog.BoundLogger] = None,
        tracer: Optional[trace.Tracer] = None
    ):
        """
        Initialize atomic component.
        
        Args:
            name: Component name for identification
            config: Component-specific configuration
            logger: Optional structured logger
            tracer: Optional OpenTelemetry tracer
        """
        self.name = name
        self.config = config
        self.logger = logger or structlog.get_logger()
        self.tracer = tracer or trace.get_tracer(__name__)
        
        # Bind component name to logger
        self.logger = self.logger.bind(component=name)
        
        # Validate configuration
        try:
            self._validate_config()
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration for {name}: {str(e)}")
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate component configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def _process(self, input_data: InputT) -> OutputT:
        """
        Core processing logic.
        
        This method must:
        - Implement the component's single responsibility
        - Be under 150 lines of code
        - Handle errors appropriately
        
        Args:
            input_data: Input to process
            
        Returns:
            Processed output
            
        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    async def process(self, input_data: InputT) -> Tuple[OutputT, ComponentMetrics]:
        """
        Process input with automatic instrumentation.
        
        This method wraps the core processing logic with:
        - OpenTelemetry tracing
        - Structured logging
        - Metrics collection
        - Error handling
        
        Args:
            input_data: Input to process
            
        Returns:
            Tuple of (output, metrics)
            
        Raises:
            ComponentError: If processing fails
        """
        with self.tracer.start_as_current_span(f"{self.name}.process") as span:
            start_time = time.time()
            
            # Initialize metrics
            metrics = ComponentMetrics(
                execution_time_ms=0,
                input_size=self._calculate_size(input_data),
                output_size=0,
                success=False
            )
            
            # Set span attributes
            span.set_attribute("component.name", self.name)
            span.set_attribute("input.size", metrics.input_size)
            
            try:
                # Log start
                self.logger.info(
                    "component.processing.started",
                    input_size=metrics.input_size
                )
                
                # Execute core processing
                output = await self._process(input_data)
                
                # Calculate final metrics
                metrics.execution_time_ms = (time.time() - start_time) * 1000
                metrics.output_size = self._calculate_size(output)
                metrics.success = True
                
                # Log success
                self.logger.info(
                    "component.processing.completed",
                    **metrics.to_dict()
                )
                
                # Update span
                span.set_attribute("output.size", metrics.output_size)
                span.set_attribute("execution.time_ms", metrics.execution_time_ms)
                span.set_status(StatusCode.OK)
                
                return output, metrics
                
            except Exception as e:
                # Calculate error metrics
                metrics.execution_time_ms = (time.time() - start_time) * 1000
                metrics.error = str(e)
                
                # Log error
                self.logger.error(
                    "component.processing.failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    **metrics.to_dict()
                )
                
                # Record exception in span
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, str(e))
                
                # Wrap and re-raise
                if isinstance(e, ComponentError):
                    raise
                else:
                    raise ProcessingError(
                        f"Processing failed in {self.name}: {str(e)}"
                    ) from e
    
    def _calculate_size(self, data: Any) -> int:
        """
        Calculate approximate size of data in bytes.
        
        Args:
            data: Data to measure
            
        Returns:
            Approximate size in bytes
        """
        if data is None:
            return 0
        elif isinstance(data, (str, bytes)):
            return len(data)
        elif isinstance(data, (list, tuple)):
            return sum(self._calculate_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(
                self._calculate_size(k) + self._calculate_size(v)
                for k, v in data.items()
            )
        else:
            # Fallback for other types
            return len(str(data))
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform component health check.
        
        Returns:
            Health status dictionary
        """
        return {
            "component": self.name,
            "status": "healthy",
            "config_valid": True
        }