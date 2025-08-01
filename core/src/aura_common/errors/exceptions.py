"""
⚡ AURA Exception Hierarchy
Well-structured exceptions with rich context.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback
import json


class AuraError(Exception):
    """
    Base exception for all AURA errors.
    
    Features:
    - Rich error context
    - Correlation ID tracking
    - Error categorization
    - JSON serialization
    """
    
    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        correlation_id: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ):
        """
        Initialize with rich context.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            cause: Original exception that caused this error
            correlation_id: Request correlation ID
            suggestions: List of suggestions to fix the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.correlation_id = correlation_id
        self.suggestions = suggestions or []
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc()
        
        # Auto-populate correlation ID if available
        if not self.correlation_id:
            try:
                from ..logging.correlation import get_correlation_id
                self.correlation_id = get_correlation_id()
            except ImportError:
                pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'correlation_id': self.correlation_id,
            'suggestions': self.suggestions,
            'timestamp': self.timestamp.isoformat(),
            'cause': str(self.cause) if self.cause else None,
            'traceback': self.traceback
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def with_details(self, **kwargs: Any) -> 'AuraError':
        """Add additional details to the error."""
        self.details.update(kwargs)
        return self
    
    def with_suggestion(self, suggestion: str) -> 'AuraError':
        """Add a suggestion for fixing the error."""
        self.suggestions.append(suggestion)
        return self


class ConfigurationError(AuraError):
    """Error in configuration or settings."""
    
    def __init__(
        self,
        message: str,
        *,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize with configuration context."""
        details = kwargs.pop('details', {})
        if config_key:
            details['config_key'] = config_key
        if config_file:
            details['config_file'] = config_file
        
        super().__init__(
            message,
            error_code='CONFIG_ERROR',
            details=details,
            **kwargs
        )


class ValidationError(AuraError):
    """Data validation error."""
    
    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraint: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize with validation context."""
        details = kwargs.pop('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        if constraint:
            details['constraint'] = constraint
        
        super().__init__(
            message,
            error_code='VALIDATION_ERROR',
            details=details,
            **kwargs
        )


class IntegrationError(AuraError):
    """External service integration error."""
    
    def __init__(
        self,
        message: str,
        *,
        service: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any
    ):
        """Initialize with integration context."""
        details = kwargs.pop('details', {})
        if service:
            details['service'] = service
        if endpoint:
            details['endpoint'] = endpoint
        if status_code:
            details['status_code'] = status_code
        
        super().__init__(
            message,
            error_code='INTEGRATION_ERROR',
            details=details,
            **kwargs
        )


class ResourceError(AuraError):
    """Resource access or availability error."""
    
    def __init__(
        self,
        message: str,
        *,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize with resource context."""
        details = kwargs.pop('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message,
            error_code='RESOURCE_ERROR',
            details=details,
            **kwargs
        )


class SecurityError(AuraError):
    """Security-related error."""
    
    def __init__(
        self,
        message: str,
        *,
        action: Optional[str] = None,
        principal: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize with security context."""
        details = kwargs.pop('details', {})
        if action:
            details['action'] = action
        if principal:
            details['principal'] = principal
        if resource:
            details['resource'] = resource
        
        # Security errors should have limited details in production
        if kwargs.get('production', True):
            message = "Security error occurred"
            details = {'error_id': details.get('error_id', 'SEC_ERR')}
        
        super().__init__(
            message,
            error_code='SECURITY_ERROR',
            details=details,
            **kwargs
        )


class StateError(AuraError):
    """Invalid state transition or state error."""
    
    def __init__(
        self,
        message: str,
        *,
        current_state: Optional[str] = None,
        attempted_transition: Optional[str] = None,
        valid_transitions: Optional[List[str]] = None,
        **kwargs: Any
    ):
        """Initialize with state context."""
        details = kwargs.pop('details', {})
        if current_state:
            details['current_state'] = current_state
        if attempted_transition:
            details['attempted_transition'] = attempted_transition
        if valid_transitions:
            details['valid_transitions'] = valid_transitions
        
        super().__init__(
            message,
            error_code='STATE_ERROR',
            details=details,
            **kwargs
        )