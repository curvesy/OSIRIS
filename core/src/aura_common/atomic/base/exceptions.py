"""
Exception hierarchy for atomic components.

These exceptions provide clear error categorization and help with
debugging and error handling in production systems.
"""

from typing import Optional, Dict, Any


class ComponentError(Exception):
    """Base exception for all component-related errors."""
    
    def __init__(
        self,
        message: str,
        component_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize component error.
        
        Args:
            message: Error message
            component_name: Name of the component that raised the error
            details: Additional error details
        """
        super().__init__(message)
        self.component_name = component_name
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "component_name": self.component_name,
            "details": self.details
        }


class ConfigurationError(ComponentError):
    """Raised when component configuration is invalid."""
    pass


class ProcessingError(ComponentError):
    """Raised when component processing fails."""
    pass


class ValidationError(ComponentError):
    """Raised when input validation fails."""
    pass


class ConnectionError(ComponentError):
    """Raised when connection to external system fails."""
    pass


class TimeoutError(ComponentError):
    """Raised when operation times out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        component_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize timeout error.
        
        Args:
            message: Error message
            timeout_seconds: Timeout duration in seconds
            component_name: Name of the component
            details: Additional error details
        """
        super().__init__(message, component_name, details)
        self.timeout_seconds = timeout_seconds
        if self.details is not None:
            self.details["timeout_seconds"] = timeout_seconds


class RetryableError(ComponentError):
    """Base class for errors that can be retried."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        max_retries: Optional[int] = None,
        component_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize retryable error.
        
        Args:
            message: Error message
            retry_after: Suggested retry delay in seconds
            max_retries: Maximum number of retries
            component_name: Name of the component
            details: Additional error details
        """
        super().__init__(message, component_name, details)
        self.retry_after = retry_after
        self.max_retries = max_retries
        if self.details is not None:
            if retry_after is not None:
                self.details["retry_after"] = retry_after
            if max_retries is not None:
                self.details["max_retries"] = max_retries


class CircuitBreakerError(ComponentError):
    """Raised when circuit breaker is open."""
    
    def __init__(
        self,
        message: str,
        reset_after: float,
        failure_count: int,
        component_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize circuit breaker error.
        
        Args:
            message: Error message
            reset_after: Time until circuit breaker resets (seconds)
            failure_count: Number of failures that triggered the breaker
            component_name: Name of the component
            details: Additional error details
        """
        super().__init__(message, component_name, details)
        self.reset_after = reset_after
        self.failure_count = failure_count
        if self.details is not None:
            self.details.update({
                "reset_after": reset_after,
                "failure_count": failure_count
            })