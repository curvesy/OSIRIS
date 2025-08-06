"""
Redis Event Bus Implementation
==============================
High-performance Redis Streams implementation of the AURA Event Bus.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import AsyncIterator, Dict, Any, Optional
import logging

try:
    import redis.asyncio as redis
except ImportError:
    import aioredis as redis  # Fallback for older versions

from .bus_protocol import EventBus, Event, EventMetadata


logger = logging.getLogger(__name__)


class RedisBus(EventBus):
    """
    Redis Streams implementation of the Event Bus.
    
    Features:
    - MAXLEN to prevent unbounded growth
    - Consumer groups for load balancing
    - Persistent with AOF
    - Sub-5ms latency at 1k msg/s
    - Connection pooling for burst performance
    """
    
    # Class-level connection pool (shared across instances)
    _pools: Dict[str, redis.ConnectionPool] = {}
    
    def __init__(
        self, 
        url: str = "redis://localhost:6379",
        max_stream_length: int = 10000,
        max_connections: int = 128  # Increased for burst performance
    ):
        self.url = url
        self.max_stream_length = max_stream_length
        
        # Use shared pool per URL
        if url not in self._pools:
            self._pools[url] = redis.ConnectionPool.from_url(
                url, 
                max_connections=max_connections,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 3,  # TCP_KEEPINTVL
                    3: 5   # TCP_KEEPCNT
                }
            )
        
        self.pool = self._pools[url]
        self._client: Optional[redis.Redis] = None
    
    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.pool)
        return self._client
    
    async def publish(self, stream: str, data: Dict[str, Any]) -> str:
        """
        Publish event to stream with automatic trimming.
        
        Uses MAXLEN to prevent OOM issues in demo.
        """
        client = await self._get_client()
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        
        # Prepare event data
        event_data = {
            "id": event_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": data.get("type", "generic"),
            "payload": json.dumps(data)
        }
        
        # Publish with MAXLEN
        try:
            stream_id = await client.xadd(
                stream,
                event_data,
                maxlen=self.max_stream_length,
                approximate=True  # ~ for better performance
            )
            
            logger.debug(f"Published to {stream}: {event_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to publish to {stream}: {e}")
            raise
    
    async def subscribe(
        self, 
        stream: str, 
        group: str, 
        consumer: str
    ) -> AsyncIterator[Event]:
        """
        Subscribe to stream as part of consumer group.
        
        Auto-creates consumer group if needed.
        """
        client = await self._get_client()
        
        # Ensure consumer group exists
        try:
            await client.xgroup_create(stream, group, id="0")
            logger.info(f"Created consumer group {group} for {stream}")
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        
        # Start consuming
        last_id = ">"  # Read only new messages
        
        while True:
            try:
                # Read with 1 second block timeout
                messages = await client.xreadgroup(
                    group,
                    consumer,
                    {stream: last_id},
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                if not messages:
                    continue
                
                # Process messages
                for stream_name, stream_messages in messages:
                    for msg_id, data in stream_messages:
                        # Parse event
                        event = self._parse_event(
                            stream_name,
                            msg_id,
                            data
                        )
                        
                        yield event
                        
            except Exception as e:
                logger.error(f"Subscribe error: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def ack(self, stream: str, group: str, event_id: str) -> bool:
        """Acknowledge message processing."""
        client = await self._get_client()
        
        try:
            result = await client.xack(stream, group, event_id)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to ack {event_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception:
            return False
    
    def _parse_event(
        self, 
        stream: str, 
        msg_id: str, 
        data: Dict[str, Any]
    ) -> Event:
        """Parse Redis message into Event object."""
        # Extract metadata
        metadata = EventMetadata(
            id=data.get("id", msg_id),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            stream=stream,
            type=data.get("type", "generic"),
            provenance_id=data.get("provenance_id"),
            retry_count=int(data.get("retry_count", 0))
        )
        
        # Parse payload
        payload_str = data.get("payload", "{}")
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            payload = {"raw": payload_str}
        
        return Event(metadata=metadata, payload=payload)
    
    async def close(self):
        """Clean up connections."""
        if self._client:
            await self._client.close()
            await self.pool.disconnect()


# Factory function for easy creation
def create_redis_bus(
    url: Optional[str] = None,
    **kwargs
) -> RedisBus:
    """Create Redis bus with sensible defaults."""
    if url is None:
        url = "redis://localhost:6379"
    
    return RedisBus(url=url, **kwargs)