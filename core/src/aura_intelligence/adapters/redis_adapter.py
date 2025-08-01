"""
Redis Adapter for AURA Intelligence.

Provides async interface to Redis for caching with:
- Context window caching
- TTL management
- Serialization support
- Pipeline operations
- Full observability
"""

from typing import Dict, Any, List, Optional, Union, TypeVar, Type
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import json
import pickle
from enum import Enum

import structlog
from opentelemetry import trace
import redis.asyncio as redis
from redis.asyncio.client import Pipeline

from ..resilience import resilient, ResilienceLevel
from ..observability import create_tracer

logger = structlog.get_logger()
tracer = create_tracer("redis_adapter")

T = TypeVar('T')


class SerializationType(str, Enum):
    """Serialization types for Redis values."""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"


@dataclass
class RedisConfig:
    """Configuration for Redis connection."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    # Connection pool settings
    max_connections: int = 50
    connection_timeout: float = 20.0
    socket_timeout: float = 20.0
    socket_keepalive: bool = True
    
    # Retry settings
    retry_on_timeout: bool = True
    retry_on_error: List[type] = None
    max_retries: int = 3
    
    # Performance settings
    decode_responses: bool = False  # Keep False for binary data
    health_check_interval: int = 30
    
    # Default TTL settings
    default_ttl_seconds: int = 3600  # 1 hour
    context_window_ttl: int = 7200  # 2 hours
    decision_cache_ttl: int = 86400  # 24 hours


class RedisAdapter:
    """Async adapter for Redis operations."""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._client: Optional[redis.Redis] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the Redis client."""
        if self._initialized:
            return
            
        with tracer.start_as_current_span("redis_initialize") as span:
            span.set_attribute("redis.host", self.config.host)
            span.set_attribute("redis.port", self.config.port)
            span.set_attribute("redis.db", self.config.db)
            
            try:
                # Create connection pool
                pool = redis.ConnectionPool(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    max_connections=self.config.max_connections,
                    decode_responses=self.config.decode_responses,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.connection_timeout,
                    socket_keepalive=self.config.socket_keepalive,
                    socket_keepalive_options={},
                    retry_on_timeout=self.config.retry_on_timeout,
                    retry_on_error=self.config.retry_on_error or [],
                    health_check_interval=self.config.health_check_interval
                )
                
                # Create client
                self._client = redis.Redis(connection_pool=pool)
                
                # Verify connectivity
                await self._client.ping()
                
                self._initialized = True
                logger.info("Redis adapter initialized", 
                           host=self.config.host,
                           port=self.config.port)
                
            except Exception as e:
                logger.error("Failed to initialize Redis", error=str(e))
                raise
                
    async def close(self):
        """Close the Redis client."""
        if self._client:
            await self._client.close()
            await self._client.connection_pool.disconnect()
            self._initialized = False
            logger.info("Redis adapter closed")
            
    def _serialize(self, value: Any, serialization: SerializationType) -> bytes:
        """Serialize value based on type."""
        if serialization == SerializationType.JSON:
            return json.dumps(value, default=str).encode('utf-8')
        elif serialization == SerializationType.PICKLE:
            return pickle.dumps(value)
        elif serialization == SerializationType.STRING:
            return str(value).encode('utf-8')
        else:
            raise ValueError(f"Unknown serialization type: {serialization}")
            
    def _deserialize(self, data: bytes, serialization: SerializationType) -> Any:
        """Deserialize value based on type."""
        if data is None:
            return None
            
        if serialization == SerializationType.JSON:
            return json.loads(data.decode('utf-8'))
        elif serialization == SerializationType.PICKLE:
            return pickle.loads(data)
        elif serialization == SerializationType.STRING:
            return data.decode('utf-8')
        else:
            raise ValueError(f"Unknown serialization type: {serialization}")
            
    @resilient(level=ResilienceLevel.NORMAL)
    async def get(
        self,
        key: str,
        serialization: SerializationType = SerializationType.JSON
    ) -> Optional[Any]:
        """Get a value from cache."""
        with tracer.start_as_current_span("redis_get") as span:
            span.set_attribute("redis.key", key)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                data = await self._client.get(key)
                value = self._deserialize(data, serialization)
                
                span.set_attribute("redis.hit", data is not None)
                return value
                
            except Exception as e:
                logger.error("Failed to get from Redis", 
                           key=key,
                           error=str(e))
                raise
                
    @resilient(level=ResilienceLevel.NORMAL)
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialization: SerializationType = SerializationType.JSON
    ) -> bool:
        """Set a value in cache."""
        with tracer.start_as_current_span("redis_set") as span:
            span.set_attribute("redis.key", key)
            span.set_attribute("redis.ttl", ttl or self.config.default_ttl_seconds)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                data = self._serialize(value, serialization)
                ttl = ttl or self.config.default_ttl_seconds
                
                result = await self._client.set(key, data, ex=ttl)
                return bool(result)
                
            except Exception as e:
                logger.error("Failed to set in Redis", 
                           key=key,
                           error=str(e))
                raise
                
    async def delete(self, key: Union[str, List[str]]) -> int:
        """Delete key(s) from cache."""
        with tracer.start_as_current_span("redis_delete") as span:
            if isinstance(key, str):
                keys = [key]
            else:
                keys = key
                
            span.set_attribute("redis.keys_count", len(keys))
            
            if not self._initialized:
                await self.initialize()
                
            try:
                return await self._client.delete(*keys)
            except Exception as e:
                logger.error("Failed to delete from Redis", 
                           keys=keys,
                           error=str(e))
                raise
                
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._initialized:
            await self.initialize()
            
        return bool(await self._client.exists(key))
        
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key."""
        if not self._initialized:
            await self.initialize()
            
        return bool(await self._client.expire(key, ttl))
        
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        if not self._initialized:
            await self.initialize()
            
        return await self._client.ttl(key)
        
    # Batch operations
    
    async def mget(
        self,
        keys: List[str],
        serialization: SerializationType = SerializationType.JSON
    ) -> Dict[str, Any]:
        """Get multiple values."""
        with tracer.start_as_current_span("redis_mget") as span:
            span.set_attribute("redis.keys_count", len(keys))
            
            if not self._initialized:
                await self.initialize()
                
            try:
                values = await self._client.mget(keys)
                result = {}
                
                for key, data in zip(keys, values):
                    if data is not None:
                        result[key] = self._deserialize(data, serialization)
                        
                span.set_attribute("redis.hits", len(result))
                return result
                
            except Exception as e:
                logger.error("Failed to mget from Redis", 
                           keys_count=len(keys),
                           error=str(e))
                raise
                
    async def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        serialization: SerializationType = SerializationType.JSON
    ) -> bool:
        """Set multiple values."""
        with tracer.start_as_current_span("redis_mset") as span:
            span.set_attribute("redis.keys_count", len(mapping))
            
            if not self._initialized:
                await self.initialize()
                
            try:
                # Serialize all values
                serialized = {}
                for key, value in mapping.items():
                    serialized[key] = self._serialize(value, serialization)
                    
                # Use pipeline for atomic operation with TTL
                async with self._client.pipeline() as pipe:
                    pipe.mset(serialized)
                    
                    if ttl:
                        for key in mapping.keys():
                            pipe.expire(key, ttl)
                            
                    results = await pipe.execute()
                    
                return all(results)
                
            except Exception as e:
                logger.error("Failed to mset in Redis", 
                           keys_count=len(mapping),
                           error=str(e))
                raise
                
    # Context-specific methods
    
    async def cache_context_window(
        self,
        agent_id: str,
        context_id: str,
        context_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache a context window for an agent."""
        key = f"context:{agent_id}:{context_id}"
        ttl = ttl or self.config.context_window_ttl
        
        return await self.set(key, context_data, ttl=ttl)
        
    async def get_context_window(
        self,
        agent_id: str,
        context_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached context window."""
        key = f"context:{agent_id}:{context_id}"
        return await self.get(key)
        
    async def cache_decision(
        self,
        decision_id: str,
        decision_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache a decision for fast retrieval."""
        key = f"decision:{decision_id}"
        ttl = ttl or self.config.decision_cache_ttl
        
        return await self.set(key, decision_data, ttl=ttl)
        
    async def get_cached_decision(
        self,
        decision_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached decision."""
        key = f"decision:{decision_id}"
        return await self.get(key)
        
    async def cache_embeddings(
        self,
        embeddings: Dict[str, List[float]],
        ttl: int = 3600
    ) -> bool:
        """Cache embeddings with their keys."""
        mapping = {f"embedding:{k}": v for k, v in embeddings.items()}
        return await self.mset(mapping, ttl=ttl, serialization=SerializationType.PICKLE)
        
    async def get_embeddings(
        self,
        keys: List[str]
    ) -> Dict[str, List[float]]:
        """Get cached embeddings."""
        redis_keys = [f"embedding:{k}" for k in keys]
        cached = await self.mget(redis_keys, serialization=SerializationType.PICKLE)
        
        # Map back to original keys
        result = {}
        for key, redis_key in zip(keys, redis_keys):
            if redis_key in cached:
                result[key] = cached[redis_key]
                
        return result
        
    # Utility methods
    
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            if not self._initialized:
                await self.initialize()
            return await self._client.ping()
        except Exception:
            return False
            
    async def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self._initialized:
            await self.initialize()
            
        cursor = 0
        deleted = 0
        
        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            if keys:
                deleted += await self._client.delete(*keys)
                
            if cursor == 0:
                break
                
        return deleted
        
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        if not self._initialized:
            await self.initialize()
            
        info = await self._client.info()
        
        return {
            "version": info.get("redis_version"),
            "connected_clients": info.get("connected_clients"),
            "used_memory_human": info.get("used_memory_human"),
            "total_connections_received": info.get("total_connections_received"),
            "total_commands_processed": info.get("total_commands_processed"),
            "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec"),
            "keyspace": info.get("db0", {})
        }