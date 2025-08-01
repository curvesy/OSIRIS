"""
Bulkhead Pattern for Resource Isolation

Implements the bulkhead pattern to isolate resources and prevent
resource exhaustion from affecting the entire system.
"""

import asyncio
from typing import TypeVar, Callable, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from opentelemetry import trace, metrics
import structlog

# Type variable for generic return types
T = TypeVar('T')

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Bulkhead metrics
bulkhead_active_gauge = meter.create_up_down_counter(
    name="bulkhead.active_executions",
    description="Number of active executions in bulkhead",
    unit="1"
)

bulkhead_rejected_counter = meter.create_counter(
    name="bulkhead.rejected",
    description="Number of rejected executions due to bulkhead full",
    unit="1"
)

bulkhead_queue_size_gauge = meter.create_up_down_counter(
    name="bulkhead.queue_size",
    description="Number of executions waiting in queue",
    unit="1"
)


class BulkheadFullError(Exception):
    """Raised when bulkhead capacity is exceeded."""
    
    def __init__(self, message: str, bulkhead_name: str, active: int, max_concurrent: int):
        super().__init__(message)
        self.bulkhead_name = bulkhead_name
        self.active = active
        self.max_concurrent = max_concurrent


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead."""
    
    name: str
    max_concurrent: int = 10           # Max concurrent executions
    max_queue_size: int = 100          # Max waiting in queue
    timeout: timedelta = timedelta(seconds=30)  # Max wait time in queue
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if self.max_queue_size < 0:
            raise ValueError("max_queue_size must be non-negative")
        if self.timeout.total_seconds() <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class BulkheadStats:
    """Statistics for bulkhead."""
    
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    rejected_executions: int = 0
    timeout_executions: int = 0
    current_active: int = 0
    current_queued: int = 0
    peak_active: int = 0
    peak_queued: int = 0
    
    def record_execution_start(self) -> None:
        """Record start of execution."""
        self.total_executions += 1
        self.current_active += 1
        self.peak_active = max(self.peak_active, self.current_active)
    
    def record_execution_end(self, success: bool) -> None:
        """Record end of execution."""
        self.current_active -= 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
    
    def record_rejection(self) -> None:
        """Record rejected execution."""
        self.rejected_executions += 1
    
    def record_timeout(self) -> None:
        """Record timeout in queue."""
        self.timeout_executions += 1
    
    def record_queue_add(self) -> None:
        """Record addition to queue."""
        self.current_queued += 1
        self.peak_queued = max(self.peak_queued, self.current_queued)
    
    def record_queue_remove(self) -> None:
        """Record removal from queue."""
        self.current_queued -= 1


class Bulkhead:
    """
    Bulkhead implementation for resource isolation.
    
    Features:
    - Limits concurrent executions
    - Optional queueing with timeout
    - Full observability
    - Thread-safe async implementation
    """
    
    def __init__(self, config: BulkheadConfig):
        """Initialize bulkhead."""
        config.validate()
        self.config = config
        self.stats = BulkheadStats()
        self.logger = structlog.get_logger().bind(bulkhead=config.name)
        
        # Semaphore for limiting concurrent executions
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        
        # Queue for waiting executions
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._lock = asyncio.Lock()
    
    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function with bulkhead protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            BulkheadFullError: If bulkhead and queue are full
            asyncio.TimeoutError: If queue timeout exceeded
        """
        # Check if we can execute immediately
        if self._semaphore.locked() and self._queue.full():
            self.stats.record_rejection()
            bulkhead_rejected_counter.add(
                1,
                {"bulkhead.name": self.config.name}
            )
            
            raise BulkheadFullError(
                f"Bulkhead '{self.config.name}' is full",
                self.config.name,
                self.stats.current_active,
                self.config.max_concurrent
            )
        
        # Create execution task
        execution_task = asyncio.create_task(
            self._execute_with_bulkhead(func, *args, **kwargs)
        )
        
        # If semaphore is available, execute immediately
        if not self._semaphore.locked():
            return await execution_task
        
        # Otherwise, add to queue
        async with self._lock:
            self.stats.record_queue_add()
            bulkhead_queue_size_gauge.add(
                1,
                {"bulkhead.name": self.config.name}
            )
        
        try:
            # Wait in queue with timeout
            return await asyncio.wait_for(
                execution_task,
                timeout=self.config.timeout.total_seconds()
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self.stats.record_timeout()
                self.stats.record_queue_remove()
                bulkhead_queue_size_gauge.add(
                    -1,
                    {"bulkhead.name": self.config.name}
                )
            
            # Cancel the execution task
            execution_task.cancel()
            
            self.logger.warning(
                "Bulkhead queue timeout",
                timeout_seconds=self.config.timeout.total_seconds(),
                queue_size=self.stats.current_queued
            )
            
            raise
        finally:
            # Ensure queue counter is updated
            if not execution_task.done():
                async with self._lock:
                    self.stats.record_queue_remove()
                    bulkhead_queue_size_gauge.add(
                        -1,
                        {"bulkhead.name": self.config.name}
                    )
    
    async def _execute_with_bulkhead(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute function with bulkhead semaphore."""
        async with self._semaphore:
            # Update stats
            async with self._lock:
                self.stats.record_execution_start()
                bulkhead_active_gauge.add(
                    1,
                    {"bulkhead.name": self.config.name}
                )
            
            # Execute with tracing
            with tracer.start_as_current_span(
                f"bulkhead.{self.config.name}",
                attributes={
                    "bulkhead.name": self.config.name,
                    "bulkhead.active": self.stats.current_active,
                    "bulkhead.queued": self.stats.current_queued
                }
            ) as span:
                success = False
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
                    
                finally:
                    # Update stats
                    async with self._lock:
                        self.stats.record_execution_end(success)
                        bulkhead_active_gauge.add(
                            -1,
                            {"bulkhead.name": self.config.name}
                        )
    
    @asynccontextmanager
    async def acquire(self):
        """
        Context manager for bulkhead protection.
        
        Usage:
            async with bulkhead.acquire():
                # Protected code here
                pass
        """
        if self._semaphore.locked() and self._queue.full():
            self.stats.record_rejection()
            raise BulkheadFullError(
                f"Bulkhead '{self.config.name}' is full",
                self.config.name,
                self.stats.current_active,
                self.config.max_concurrent
            )
        
        async with self._semaphore:
            # Update stats
            async with self._lock:
                self.stats.record_execution_start()
                bulkhead_active_gauge.add(
                    1,
                    {"bulkhead.name": self.config.name}
                )
            
            success = False
            try:
                yield
                success = True
            finally:
                # Update stats
                async with self._lock:
                    self.stats.record_execution_end(success)
                    bulkhead_active_gauge.add(
                        -1,
                        {"bulkhead.name": self.config.name}
                    )
    
    def get_stats(self) -> BulkheadStats:
        """Get bulkhead statistics."""
        return self.stats
    
    def is_full(self) -> bool:
        """Check if bulkhead is at capacity."""
        return self._semaphore.locked()
    
    def available_slots(self) -> int:
        """Get number of available execution slots."""
        return self.config.max_concurrent - self.stats.current_active
    
    async def health_check(self) -> Dict[str, Any]:
        """Check bulkhead health."""
        utilization = self.stats.current_active / self.config.max_concurrent
        
        health_status = "healthy"
        if utilization > 0.9:
            health_status = "degraded"
        if self.is_full():
            health_status = "critical"
        
        return {
            "status": health_status,
            "utilization": utilization,
            "active": self.stats.current_active,
            "queued": self.stats.current_queued,
            "available": self.available_slots(),
            "stats": {
                "total": self.stats.total_executions,
                "successful": self.stats.successful_executions,
                "failed": self.stats.failed_executions,
                "rejected": self.stats.rejected_executions,
                "timeout": self.stats.timeout_executions,
                "peak_active": self.stats.peak_active,
                "peak_queued": self.stats.peak_queued
            }
        }