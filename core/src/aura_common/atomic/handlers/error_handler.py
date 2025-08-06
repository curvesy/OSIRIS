"""
Error handler atomic component.

Provides centralized error handling with configurable strategies,
error classification, and recovery mechanisms.
"""

from typing import Any, Callable, Optional, Dict, List, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import traceback

from ..base import AtomicComponent
from ..base.exceptions import ComponentError, RetryableError


class ErrorHandlingStrategy(Enum):
    """Error handling strategies."""
    
    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    FALLBACK = "fallback"
    LOG_AND_CONTINUE = "log_and_continue"
    CUSTOM = "custom"


@dataclass
class ErrorClassification:
    """Classification of an error."""
    
    error_type: Type[Exception]
    is_retryable: bool
    is_critical: bool
    category: str
    suggested_action: str


@dataclass
class ErrorHandlerConfig:
    """Configuration for error handling."""
    
    default_strategy: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST
    capture_stack_trace: bool = True
    max_error_history: int = 100
    error_classifications: List[ErrorClassification] = field(default_factory=list)
    fallback_value: Any = None
    custom_handler: Optional[Callable[[Exception], Any]] = None
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.max_error_history <= 0:
            raise ValueError("max_error_history must be positive")
        
        if self.default_strategy == ErrorHandlingStrategy.CUSTOM and not self.custom_handler:
            raise ValueError("custom_handler required for CUSTOM strategy")


@dataclass
class HandledError:
    """Result of error handling."""
    
    original_error: Exception
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    timestamp: datetime
    strategy_used: ErrorHandlingStrategy
    classification: Optional[ErrorClassification]
    recovery_action: Optional[str]
    handled_successfully: bool
    result: Any = None


class ErrorHandler(AtomicComponent[Exception, HandledError, ErrorHandlerConfig]):
    """
    Atomic component for centralized error handling.
    
    Features:
    - Error classification and categorization
    - Configurable handling strategies
    - Error history tracking
    - Stack trace capture
    - Recovery suggestions
    """
    
    def __init__(self, name: str, config: ErrorHandlerConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self._error_history: List[HandledError] = []
        self._setup_default_classifications()
    
    def _validate_config(self) -> None:
        """Validate component configuration."""
        self.config.validate()
    
    def _setup_default_classifications(self) -> None:
        """Set up default error classifications."""
        defaults = [
            ErrorClassification(
                error_type=RetryableError,
                is_retryable=True,
                is_critical=False,
                category="transient",
                suggested_action="Retry with backoff"
            ),
            ErrorClassification(
                error_type=ConnectionError,
                is_retryable=True,
                is_critical=False,
                category="network",
                suggested_action="Check network connectivity"
            ),
            ErrorClassification(
                error_type=TimeoutError,
                is_retryable=True,
                is_critical=False,
                category="performance",
                suggested_action="Increase timeout or optimize operation"
            ),
            ErrorClassification(
                error_type=ValueError,
                is_retryable=False,
                is_critical=True,
                category="validation",
                suggested_action="Fix input data"
            )
        ]
        
        # Add defaults if not already present
        existing_types = {c.error_type for c in self.config.error_classifications}
        for default in defaults:
            if default.error_type not in existing_types:
                self.config.error_classifications.append(default)
    
    async def _process(self, error: Exception) -> HandledError:
        """
        Handle an error according to configured strategy.
        
        Args:
            error: Exception to handle
            
        Returns:
            HandledError with handling details
        """
        # Classify error
        classification = self._classify_error(error)
        
        # Capture details
        error_details = HandledError(
            original_error=error,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc() if self.config.capture_stack_trace else None,
            timestamp=datetime.now(timezone.utc),
            strategy_used=self._determine_strategy(error, classification),
            classification=classification,
            recovery_action=classification.suggested_action if classification else None,
            handled_successfully=False
        )
        
        # Apply handling strategy
        try:
            if error_details.strategy_used == ErrorHandlingStrategy.FAIL_FAST:
                error_details.handled_successfully = False
                
            elif error_details.strategy_used == ErrorHandlingStrategy.RETRY:
                error_details.recovery_action = "Retry operation"
                error_details.handled_successfully = True
                
            elif error_details.strategy_used == ErrorHandlingStrategy.FALLBACK:
                error_details.result = self.config.fallback_value
                error_details.handled_successfully = True
                
            elif error_details.strategy_used == ErrorHandlingStrategy.LOG_AND_CONTINUE:
                self.logger.error(
                    "Error handled with log_and_continue",
                    error_type=error_details.error_type,
                    error_message=error_details.error_message
                )
                error_details.handled_successfully = True
                
            elif error_details.strategy_used == ErrorHandlingStrategy.CUSTOM:
                error_details.result = self.config.custom_handler(error)
                error_details.handled_successfully = True
                
        except Exception as handler_error:
            self.logger.error(f"Error handler failed: {handler_error}")
            error_details.handled_successfully = False
        
        # Store in history
        self._add_to_history(error_details)
        
        return error_details
    
    def _classify_error(self, error: Exception) -> Optional[ErrorClassification]:
        """Classify an error based on configured classifications."""
        for classification in self.config.error_classifications:
            if isinstance(error, classification.error_type):
                return classification
        return None
    
    def _determine_strategy(
        self,
        error: Exception,
        classification: Optional[ErrorClassification]
    ) -> ErrorHandlingStrategy:
        """Determine handling strategy for error."""
        # Use classification hints if available
        if classification:
            if classification.is_critical:
                return ErrorHandlingStrategy.FAIL_FAST
            elif classification.is_retryable:
                return ErrorHandlingStrategy.RETRY
        
        # Fall back to default
        return self.config.default_strategy
    
    def _add_to_history(self, handled_error: HandledError) -> None:
        """Add error to history with size limit."""
        self._error_history.append(handled_error)
        
        # Trim history if needed
        if len(self._error_history) > self.config.max_error_history:
            self._error_history = self._error_history[-self.config.max_error_history:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self._error_history:
            return {
                "total_errors": 0,
                "error_types": {},
                "strategies_used": {},
                "success_rate": 1.0
            }
        
        error_types = {}
        strategies = {}
        successful = 0
        
        for error in self._error_history:
            # Count error types
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
            # Count strategies
            strategy = error.strategy_used.value
            strategies[strategy] = strategies.get(strategy, 0) + 1
            
            # Count successes
            if error.handled_successfully:
                successful += 1
        
        return {
            "total_errors": len(self._error_history),
            "error_types": error_types,
            "strategies_used": strategies,
            "success_rate": successful / len(self._error_history)
        }