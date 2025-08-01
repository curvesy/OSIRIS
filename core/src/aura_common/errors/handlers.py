"""
⚡ Error Handlers
Global and specific error handling strategies.
"""

from typing import Optional, Callable, Any, Dict, Type
from functools import wraps
import asyncio
import sys

from .exceptions import AuraError
from ..logging import get_logger

logger = get_logger(__name__)


class ErrorHandler:
    """Base error handler with logging and metrics."""
    
    def __init__(self, name: str = "ErrorHandler"):
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
    
    async def handle(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Handle an error with context."""
        # Log the error
        self.logger.error(
            f"Error handled by {self.name}",
            exc_info=error,
            error_type=type(error).__name__,
            context=context
        )
        
        # Specific handling based on error type
        if isinstance(error, AuraError):
            return await self._handle_aura_error(error, context)
        else:
            return await self._handle_generic_error(error, context)
    
    async def _handle_aura_error(
        self,
        error: AuraError,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Handle AURA-specific errors."""
        # Log with full context
        self.logger.error(
            "AURA error occurred",
            error_dict=error.to_dict(),
            context=context
        )
        
        # Could implement specific recovery strategies here
        return None
    
    async def _handle_generic_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Handle generic errors."""
        self.logger.error(
            "Generic error occurred",
            exc_info=error,
            context=context
        )
        return None


class GlobalErrorHandler:
    """Global error handler for uncaught exceptions."""
    
    _instance: Optional['GlobalErrorHandler'] = None
    _installed: bool = False
    
    def __new__(cls) -> 'GlobalErrorHandler':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.handlers: Dict[Type[Exception], ErrorHandler] = {}
    
    def register_handler(
        self,
        exception_type: Type[Exception],
        handler: ErrorHandler
    ) -> None:
        """Register a handler for specific exception type."""
        self.handlers[exception_type] = handler
    
    def install(self) -> None:
        """Install global exception handlers."""
        if self._installed:
            return
        
        # Handle uncaught exceptions
        sys.excepthook = self._handle_exception
        
        # Handle unhandled async exceptions
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(self._handle_async_exception)
        
        self._installed = True
        self.logger.info("Global error handler installed")
    
    def _handle_exception(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_traceback: Any
    ) -> None:
        """Handle uncaught exceptions."""
        self.logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Find appropriate handler
        for error_type, handler in self.handlers.items():
            if issubclass(exc_type, error_type):
                asyncio.create_task(
                    handler.handle(exc_value, {"traceback": exc_traceback})
                )
                return
    
    def _handle_async_exception(
        self,
        loop: asyncio.AbstractEventLoop,
        context: Dict[str, Any]
    ) -> None:
        """Handle unhandled async exceptions."""
        exception = context.get('exception')
        if exception:
            self.logger.critical(
                "Unhandled async exception",
                exc_info=exception,
                context=context
            )
            
            # Find appropriate handler
            for error_type, handler in self.handlers.items():
                if isinstance(exception, error_type):
                    asyncio.create_task(
                        handler.handle(exception, context)
                    )
                    return