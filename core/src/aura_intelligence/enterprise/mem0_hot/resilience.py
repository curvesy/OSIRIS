"""
🛡️ Resilience & Error Handling

Exponential backoff, dead letter queues, circuit breakers, and robust failure handling
for DuckDB and Redis operations in the Phase 2C Intelligence Flywheel.
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json

from aura_intelligence.utils.logger import get_logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    # Retry conditions
    retry_on_exceptions: List[type] = None
    retry_on_status_codes: List[int] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout_seconds: float = 60.0  # Time before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    
    # Monitoring
    window_size_seconds: float = 300.0  # 5-minute sliding window
    min_requests: int = 10  # Minimum requests before considering failure rate


class ExponentialBackoff:
    """
    🔄 Exponential Backoff with Jitter
    
    Implements exponential backoff with optional jitter for retry logic.
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def retry(self, 
                   func: Callable,
                   *args,
                   operation_name: str = "operation",
                   **kwargs) -> Any:
        """
        Execute function with exponential backoff retry.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            operation_name: Name for logging
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries exhausted
        """
        
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"✅ {operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry(e):
                    self.logger.error(f"❌ {operation_name} failed with non-retryable error: {e}")
                    raise e
                
                # Calculate delay for next attempt
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    
                    self.logger.warning(
                        f"⚠️ {operation_name} failed (attempt {attempt + 1}/{self.config.max_attempts}): {e}"
                    )
                    self.logger.info(f"🔄 Retrying in {delay:.2f}s...")
                    
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self.logger.error(f"❌ {operation_name} failed after {self.config.max_attempts} attempts")
        raise last_exception
    
    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        
        if self.config.retry_on_exceptions:
            return any(isinstance(exception, exc_type) for exc_type in self.config.retry_on_exceptions)
        
        # Default: retry on common transient errors
        transient_errors = [
            ConnectionError,
            TimeoutError,
            OSError,
            # Add database-specific errors
        ]
        
        return any(isinstance(exception, exc_type) for exc_type in transient_errors)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        
        # Exponential backoff
        delay = self.config.base_delay_seconds * (self.config.exponential_base ** attempt)
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay_seconds)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            delay += jitter
        
        return delay


class CircuitBreaker:
    """
    ⚡ Circuit Breaker Pattern
    
    Prevents cascading failures by temporarily blocking requests to failing services.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
        # Sliding window for failure rate calculation
        self.request_history: List[Dict[str, Any]] = []
        
        self.logger = get_logger(__name__)
        self.logger.info(f"⚡ Circuit breaker '{name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"⚡ Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            # Execute function
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(str(e))
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from OPEN to HALF_OPEN."""
        
        if not self.last_failure_time:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout_seconds
    
    def _record_success(self, execution_time: float):
        """Record successful execution."""
        
        self.request_history.append({
            "timestamp": time.time(),
            "success": True,
            "execution_time": execution_time
        })
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"⚡ Circuit breaker '{self.name}' CLOSED - service recovered")
        
        self._cleanup_old_history()
    
    def _record_failure(self, error_message: str):
        """Record failed execution."""
        
        self.request_history.append({
            "timestamp": time.time(),
            "success": False,
            "error": error_message
        })
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during half-open, go back to open
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.logger.warning(f"⚡ Circuit breaker '{self.name}' back to OPEN - service still failing")
        
        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self._should_open_circuit():
                self.state = CircuitState.OPEN
                self.logger.error(f"⚡ Circuit breaker '{self.name}' OPENED - too many failures")
        
        self._cleanup_old_history()
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened due to failures."""
        
        # Simple threshold check
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Failure rate check within sliding window
        recent_requests = [
            req for req in self.request_history
            if time.time() - req["timestamp"] <= self.config.window_size_seconds
        ]
        
        if len(recent_requests) < self.config.min_requests:
            return False
        
        failure_rate = sum(1 for req in recent_requests if not req["success"]) / len(recent_requests)
        
        return failure_rate >= 0.5  # 50% failure rate threshold
    
    def _cleanup_old_history(self):
        """Remove old entries from request history."""
        
        cutoff_time = time.time() - self.config.window_size_seconds
        self.request_history = [
            req for req in self.request_history
            if req["timestamp"] > cutoff_time
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        
        recent_requests = [
            req for req in self.request_history
            if time.time() - req["timestamp"] <= self.config.window_size_seconds
        ]
        
        failure_rate = 0.0
        if recent_requests:
            failure_rate = sum(1 for req in recent_requests if not req["success"]) / len(recent_requests)
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "recent_requests": len(recent_requests),
            "failure_rate": failure_rate,
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class DeadLetterQueue:
    """
    💀 Dead Letter Queue
    
    Stores failed operations for later retry or manual intervention.
    """
    
    def __init__(self, name: str, max_size: int = 1000):
        self.name = name
        self.max_size = max_size
        self.queue: List[Dict[str, Any]] = []
        
        self.logger = get_logger(__name__)
        self.logger.info(f"💀 Dead letter queue '{name}' initialized (max_size: {max_size})")
    
    def add_failed_operation(self, 
                           operation_name: str,
                           operation_data: Dict[str, Any],
                           error_message: str,
                           retry_count: int = 0):
        """Add failed operation to dead letter queue."""
        
        if len(self.queue) >= self.max_size:
            # Remove oldest entry
            removed = self.queue.pop(0)
            self.logger.warning(f"💀 DLQ '{self.name}' full - removed oldest entry: {removed['operation_name']}")
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation_name": operation_name,
            "operation_data": operation_data,
            "error_message": error_message,
            "retry_count": retry_count,
            "id": f"{operation_name}_{int(time.time())}_{random.randint(1000, 9999)}"
        }
        
        self.queue.append(entry)
        
        self.logger.error(f"💀 Added to DLQ '{self.name}': {operation_name} - {error_message}")
    
    def get_failed_operations(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get failed operations from queue."""
        
        if limit:
            return self.queue[-limit:]
        return self.queue.copy()
    
    def remove_operation(self, operation_id: str) -> bool:
        """Remove operation from queue by ID."""
        
        for i, entry in enumerate(self.queue):
            if entry["id"] == operation_id:
                removed = self.queue.pop(i)
                self.logger.info(f"💀 Removed from DLQ '{self.name}': {removed['operation_name']}")
                return True
        
        return False
    
    def clear_queue(self) -> int:
        """Clear all entries from queue."""
        
        count = len(self.queue)
        self.queue.clear()
        
        self.logger.info(f"💀 Cleared DLQ '{self.name}' - removed {count} entries")
        
        return count
    
    def get_status(self) -> Dict[str, Any]:
        """Get dead letter queue status."""
        
        return {
            "name": self.name,
            "queue_size": len(self.queue),
            "max_size": self.max_size,
            "oldest_entry": self.queue[0]["timestamp"] if self.queue else None,
            "newest_entry": self.queue[-1]["timestamp"] if self.queue else None
        }
