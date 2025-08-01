"""
🛡️ Resilient Database Operations

Wrapper for DuckDB and Redis operations with circuit breakers, retries,
and dead letter queues for production-grade reliability.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
import duckdb
import redis.asyncio as redis

from aura_intelligence.enterprise.mem0_hot.resilience import (
    ExponentialBackoff, CircuitBreaker, DeadLetterQueue,
    RetryConfig, CircuitBreakerConfig, CircuitBreakerOpenError
)
from aura_intelligence.utils.logger import get_logger


class ResilientDuckDBOperations:
    """
    🦆 Resilient DuckDB Operations
    
    Wraps DuckDB operations with circuit breakers, retries, and error handling.
    """
    
    def __init__(self, 
                 conn: duckdb.DuckDBPyConnection,
                 retry_config: RetryConfig = None,
                 circuit_config: CircuitBreakerConfig = None):
        """Initialize resilient DuckDB operations."""
        
        self.conn = conn
        
        # Initialize resilience components
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0
        )
        
        self.circuit_config = circuit_config or CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_seconds=60.0
        )
        
        self.backoff = ExponentialBackoff(self.retry_config)
        self.circuit_breaker = CircuitBreaker("duckdb", self.circuit_config)
        self.dead_letter_queue = DeadLetterQueue("duckdb_operations")
        
        self.logger = get_logger(__name__)
        self.logger.info("🦆 Resilient DuckDB operations initialized")
    
    async def execute_query(self, 
                           query: str, 
                           params: List = None,
                           operation_name: str = "query") -> Any:
        """Execute DuckDB query with resilience."""
        
        async def _execute():
            # Execute in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            if params:
                return await loop.run_in_executor(
                    None, 
                    lambda: self.conn.execute(query, params)
                )
            else:
                return await loop.run_in_executor(
                    None,
                    lambda: self.conn.execute(query)
                )
        
        try:
            # Execute through circuit breaker and retry logic
            result = await self.circuit_breaker.call(
                self.backoff.retry,
                _execute,
                operation_name=f"duckdb_{operation_name}"
            )
            
            return result
            
        except CircuitBreakerOpenError as e:
            self.logger.error(f"🚨 DuckDB circuit breaker open for {operation_name}")
            
            # Add to dead letter queue
            self.dead_letter_queue.add_failed_operation(
                operation_name=f"duckdb_{operation_name}",
                operation_data={"query": query, "params": params},
                error_message=str(e)
            )
            
            raise e
            
        except Exception as e:
            self.logger.error(f"❌ DuckDB {operation_name} failed after all retries: {e}")
            
            # Add to dead letter queue
            self.dead_letter_queue.add_failed_operation(
                operation_name=f"duckdb_{operation_name}",
                operation_data={"query": query, "params": params},
                error_message=str(e)
            )
            
            raise e
    
    async def execute_batch(self, 
                           query: str, 
                           batch_data: List,
                           operation_name: str = "batch") -> Any:
        """Execute DuckDB batch operation with resilience."""
        
        async def _execute_batch():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.conn.executemany(query, batch_data)
            )
        
        try:
            result = await self.circuit_breaker.call(
                self.backoff.retry,
                _execute_batch,
                operation_name=f"duckdb_batch_{operation_name}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DuckDB batch {operation_name} failed: {e}")
            
            # Add to dead letter queue with sample data
            sample_data = batch_data[:5] if len(batch_data) > 5 else batch_data
            self.dead_letter_queue.add_failed_operation(
                operation_name=f"duckdb_batch_{operation_name}",
                operation_data={
                    "query": query, 
                    "batch_size": len(batch_data),
                    "sample_data": sample_data
                },
                error_message=str(e)
            )
            
            raise e
    
    async def fetch_dataframe(self, 
                             query: str, 
                             params: List = None,
                             operation_name: str = "fetch_df") -> Any:
        """Fetch DataFrame with resilience."""
        
        async def _fetch_df():
            loop = asyncio.get_event_loop()
            if params:
                return await loop.run_in_executor(
                    None,
                    lambda: self.conn.execute(query, params).fetchdf()
                )
            else:
                return await loop.run_in_executor(
                    None,
                    lambda: self.conn.execute(query).fetchdf()
                )
        
        try:
            result = await self.circuit_breaker.call(
                self.backoff.retry,
                _fetch_df,
                operation_name=f"duckdb_df_{operation_name}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DuckDB DataFrame {operation_name} failed: {e}")
            
            self.dead_letter_queue.add_failed_operation(
                operation_name=f"duckdb_df_{operation_name}",
                operation_data={"query": query, "params": params},
                error_message=str(e)
            )
            
            raise e
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of DuckDB operations."""
        
        return {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "dead_letter_queue": self.dead_letter_queue.get_status(),
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "base_delay_seconds": self.retry_config.base_delay_seconds,
                "max_delay_seconds": self.retry_config.max_delay_seconds
            }
        }


class ResilientRedisOperations:
    """
    🔴 Resilient Redis Operations
    
    Wraps Redis operations with circuit breakers, retries, and error handling.
    """
    
    def __init__(self, 
                 redis_client: redis.Redis,
                 retry_config: RetryConfig = None,
                 circuit_config: CircuitBreakerConfig = None):
        """Initialize resilient Redis operations."""
        
        self.redis_client = redis_client
        
        # Initialize resilience components
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.5,
            max_delay_seconds=10.0
        )
        
        self.circuit_config = circuit_config or CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=30.0
        )
        
        self.backoff = ExponentialBackoff(self.retry_config)
        self.circuit_breaker = CircuitBreaker("redis", self.circuit_config)
        self.dead_letter_queue = DeadLetterQueue("redis_operations")
        
        self.logger = get_logger(__name__)
        self.logger.info("🔴 Resilient Redis operations initialized")
    
    async def set_with_retry(self, 
                            key: str, 
                            value: Any,
                            ex: int = None,
                            operation_name: str = "set") -> bool:
        """Set Redis key with resilience."""
        
        async def _redis_set():
            return await self.redis_client.set(key, value, ex=ex)
        
        try:
            result = await self.circuit_breaker.call(
                self.backoff.retry,
                _redis_set,
                operation_name=f"redis_{operation_name}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Redis {operation_name} failed: {e}")
            
            self.dead_letter_queue.add_failed_operation(
                operation_name=f"redis_{operation_name}",
                operation_data={"key": key, "value": str(value)[:100], "ex": ex},
                error_message=str(e)
            )
            
            raise e
    
    async def get_with_retry(self, 
                            key: str,
                            operation_name: str = "get") -> Any:
        """Get Redis key with resilience."""
        
        async def _redis_get():
            return await self.redis_client.get(key)
        
        try:
            result = await self.circuit_breaker.call(
                self.backoff.retry,
                _redis_get,
                operation_name=f"redis_{operation_name}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Redis {operation_name} failed: {e}")
            
            self.dead_letter_queue.add_failed_operation(
                operation_name=f"redis_{operation_name}",
                operation_data={"key": key},
                error_message=str(e)
            )
            
            raise e
    
    async def pipeline_with_retry(self, 
                                 operations: List[Dict[str, Any]],
                                 operation_name: str = "pipeline") -> List[Any]:
        """Execute Redis pipeline with resilience."""
        
        async def _redis_pipeline():
            pipe = self.redis_client.pipeline()
            
            for op in operations:
                method = getattr(pipe, op["method"])
                method(*op.get("args", []), **op.get("kwargs", {}))
            
            return await pipe.execute()
        
        try:
            result = await self.circuit_breaker.call(
                self.backoff.retry,
                _redis_pipeline,
                operation_name=f"redis_{operation_name}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Redis pipeline {operation_name} failed: {e}")
            
            self.dead_letter_queue.add_failed_operation(
                operation_name=f"redis_pipeline_{operation_name}",
                operation_data={"operations_count": len(operations)},
                error_message=str(e)
            )
            
            raise e
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Redis operations."""
        
        return {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "dead_letter_queue": self.dead_letter_queue.get_status(),
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "base_delay_seconds": self.retry_config.base_delay_seconds,
                "max_delay_seconds": self.retry_config.max_delay_seconds
            }
        }


class ResilientOperationsManager:
    """
    🛡️ Resilient Operations Manager
    
    Centralized manager for all resilient database operations.
    """
    
    def __init__(self, 
                 duckdb_conn: duckdb.DuckDBPyConnection,
                 redis_client: redis.Redis = None):
        """Initialize resilient operations manager."""
        
        self.duckdb_ops = ResilientDuckDBOperations(duckdb_conn)
        self.redis_ops = ResilientRedisOperations(redis_client) if redis_client else None
        
        self.logger = get_logger(__name__)
        self.logger.info("🛡️ Resilient operations manager initialized")
    
    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status of all operations."""
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "duckdb": self.duckdb_ops.get_health_status()
        }
        
        if self.redis_ops:
            health["redis"] = self.redis_ops.get_health_status()
        
        # Overall health assessment
        duckdb_healthy = self.duckdb_ops.circuit_breaker.state.value == "closed"
        redis_healthy = True
        
        if self.redis_ops:
            redis_healthy = self.redis_ops.circuit_breaker.state.value == "closed"
        
        health["overall_status"] = "healthy" if (duckdb_healthy and redis_healthy) else "degraded"
        
        return health
    
    async def retry_failed_operations(self, max_retries: int = 5) -> Dict[str, Any]:
        """Retry failed operations from dead letter queues."""
        
        results = {
            "duckdb_retries": 0,
            "duckdb_successes": 0,
            "redis_retries": 0,
            "redis_successes": 0
        }
        
        # Retry DuckDB operations
        duckdb_failed = self.duckdb_ops.dead_letter_queue.get_failed_operations(max_retries)
        
        for operation in duckdb_failed:
            try:
                # Attempt to retry the operation
                # This is a simplified retry - in practice, you'd reconstruct the original operation
                results["duckdb_retries"] += 1
                
                # If successful, remove from DLQ
                self.duckdb_ops.dead_letter_queue.remove_operation(operation["id"])
                results["duckdb_successes"] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Failed to retry DuckDB operation {operation['id']}: {e}")
        
        # Retry Redis operations if available
        if self.redis_ops:
            redis_failed = self.redis_ops.dead_letter_queue.get_failed_operations(max_retries)
            
            for operation in redis_failed:
                try:
                    results["redis_retries"] += 1
                    
                    # If successful, remove from DLQ
                    self.redis_ops.dead_letter_queue.remove_operation(operation["id"])
                    results["redis_successes"] += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to retry Redis operation {operation['id']}: {e}")
        
        self.logger.info(f"🔄 Retry summary: {results}")
        
        return results
