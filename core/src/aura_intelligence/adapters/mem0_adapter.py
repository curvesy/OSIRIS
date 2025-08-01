"""
Mem0 Adapter for AURA Intelligence.

Provides async interface to Mem0 memory management system with:
- Memory search and retrieval
- Batch operations for efficiency
- Embedding support
- Automatic memory pruning
- Full observability
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
from enum import Enum

import structlog
from opentelemetry import trace
import httpx

from ..resilience import resilient, ResilienceLevel
from ..observability import create_tracer

logger = structlog.get_logger()
tracer = create_tracer("mem0_adapter")


class MemoryType(str, Enum):
    """Types of memories in Mem0."""
    DECISION = "decision"
    OBSERVATION = "observation"
    LEARNING = "learning"
    CONTEXT = "context"
    PATTERN = "pattern"
    ADAPTATION = "adaptation"


@dataclass
class Mem0Config:
    """Configuration for Mem0 connection."""
    base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    
    # Connection settings
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Memory settings
    default_retention_days: int = 30
    max_memory_size_mb: int = 100
    embedding_dimension: int = 768
    
    # Batch settings
    batch_size: int = 100
    batch_timeout: float = 5.0
    
    # Search settings
    default_limit: int = 10
    similarity_threshold: float = 0.7
    
    # Performance settings
    connection_pool_size: int = 10
    keepalive_expiry: float = 30.0


@dataclass
class Memory:
    """Represents a memory in Mem0."""
    id: str
    agent_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: Optional[int] = None
    relevance_score: float = 1.0


@dataclass
class SearchQuery:
    """Query for searching memories."""
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    agent_ids: Optional[List[str]] = None
    memory_types: Optional[List[MemoryType]] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    offset: int = 0


class Mem0Adapter:
    """Async adapter for Mem0 operations."""
    
    def __init__(self, config: Mem0Config):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the Mem0 client."""
        if self._initialized:
            return
            
        with tracer.start_as_current_span("mem0_initialize") as span:
            span.set_attribute("mem0.base_url", self.config.base_url)
            
            try:
                # Create async HTTP client with connection pooling
                self._client = httpx.AsyncClient(
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    limits=httpx.Limits(
                        max_connections=self.config.connection_pool_size,
                        keepalive_expiry=self.config.keepalive_expiry
                    ),
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                        "Content-Type": "application/json"
                    }
                )
                
                # Verify connectivity
                response = await self._client.get("/health")
                response.raise_for_status()
                
                self._initialized = True
                logger.info("Mem0 adapter initialized", base_url=self.config.base_url)
                
            except Exception as e:
                logger.error("Failed to initialize Mem0", error=str(e))
                raise
                
    async def close(self):
        """Close the Mem0 client."""
        if self._client:
            await self._client.aclose()
            self._initialized = False
            logger.info("Mem0 adapter closed")
            
    @resilient(level=ResilienceLevel.CRITICAL)
    async def add_memory(
        self,
        memory: Memory
    ) -> str:
        """Add a new memory."""
        with tracer.start_as_current_span("mem0_add_memory") as span:
            span.set_attribute("mem0.agent_id", memory.agent_id)
            span.set_attribute("mem0.memory_type", memory.memory_type.value)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                payload = {
                    "agent_id": memory.agent_id,
                    "type": memory.memory_type.value,
                    "content": memory.content,
                    "embedding": memory.embedding,
                    "metadata": memory.metadata,
                    "timestamp": memory.timestamp.isoformat(),
                    "ttl_seconds": memory.ttl_seconds
                }
                
                response = await self._client.post(
                    "/memories",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                memory_id = result["id"]
                
                span.set_attribute("mem0.memory_id", memory_id)
                return memory_id
                
            except httpx.HTTPError as e:
                logger.error("Failed to add memory", 
                           agent_id=memory.agent_id,
                           error=str(e))
                raise
                
    @resilient(level=ResilienceLevel.CRITICAL)
    async def add_memories_batch(
        self,
        memories: List[Memory]
    ) -> List[str]:
        """Add multiple memories in batch."""
        with tracer.start_as_current_span("mem0_add_memories_batch") as span:
            span.set_attribute("mem0.batch_size", len(memories))
            
            if not self._initialized:
                await self.initialize()
                
            try:
                payloads = []
                for memory in memories:
                    payloads.append({
                        "agent_id": memory.agent_id,
                        "type": memory.memory_type.value,
                        "content": memory.content,
                        "embedding": memory.embedding,
                        "metadata": memory.metadata,
                        "timestamp": memory.timestamp.isoformat(),
                        "ttl_seconds": memory.ttl_seconds
                    })
                
                response = await self._client.post(
                    "/memories/batch",
                    json={"memories": payloads}
                )
                response.raise_for_status()
                
                result = response.json()
                memory_ids = result["ids"]
                
                span.set_attribute("mem0.memories_added", len(memory_ids))
                return memory_ids
                
            except httpx.HTTPError as e:
                logger.error("Failed to add memories batch", 
                           batch_size=len(memories),
                           error=str(e))
                raise
                
    @resilient(level=ResilienceLevel.CRITICAL)
    async def search_memories(
        self,
        query: SearchQuery
    ) -> List[Memory]:
        """Search for memories."""
        with tracer.start_as_current_span("mem0_search_memories") as span:
            span.set_attribute("mem0.query_text", query.query_text or "")
            span.set_attribute("mem0.limit", query.limit)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                payload = {
                    "query_text": query.query_text,
                    "query_embedding": query.query_embedding,
                    "filters": {
                        "agent_ids": query.agent_ids,
                        "memory_types": [t.value for t in query.memory_types] if query.memory_types else None,
                        "metadata": query.metadata_filters
                    },
                    "limit": query.limit,
                    "offset": query.offset
                }
                
                # Add time range if specified
                if query.time_range:
                    payload["filters"]["time_range"] = {
                        "start": query.time_range[0].isoformat(),
                        "end": query.time_range[1].isoformat()
                    }
                
                response = await self._client.post(
                    "/memories/search",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                memories = []
                
                for item in result["memories"]:
                    memory = Memory(
                        id=item["id"],
                        agent_id=item["agent_id"],
                        memory_type=MemoryType(item["type"]),
                        content=item["content"],
                        embedding=item.get("embedding"),
                        metadata=item.get("metadata", {}),
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        relevance_score=item.get("score", 1.0)
                    )
                    memories.append(memory)
                
                span.set_attribute("mem0.results_count", len(memories))
                return memories
                
            except httpx.HTTPError as e:
                logger.error("Failed to search memories", 
                           query=query.query_text,
                           error=str(e))
                raise
                
    async def get_memory(
        self,
        memory_id: str
    ) -> Optional[Memory]:
        """Get a specific memory by ID."""
        with tracer.start_as_current_span("mem0_get_memory") as span:
            span.set_attribute("mem0.memory_id", memory_id)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                response = await self._client.get(f"/memories/{memory_id}")
                
                if response.status_code == 404:
                    return None
                    
                response.raise_for_status()
                item = response.json()
                
                return Memory(
                    id=item["id"],
                    agent_id=item["agent_id"],
                    memory_type=MemoryType(item["type"]),
                    content=item["content"],
                    embedding=item.get("embedding"),
                    metadata=item.get("metadata", {}),
                    timestamp=datetime.fromisoformat(item["timestamp"])
                )
                
            except httpx.HTTPError as e:
                logger.error("Failed to get memory", 
                           memory_id=memory_id,
                           error=str(e))
                raise
                
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an existing memory."""
        with tracer.start_as_current_span("mem0_update_memory") as span:
            span.set_attribute("mem0.memory_id", memory_id)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                response = await self._client.patch(
                    f"/memories/{memory_id}",
                    json=updates
                )
                response.raise_for_status()
                return True
                
            except httpx.HTTPError as e:
                logger.error("Failed to update memory", 
                           memory_id=memory_id,
                           error=str(e))
                return False
                
    async def delete_memory(
        self,
        memory_id: str
    ) -> bool:
        """Delete a memory."""
        with tracer.start_as_current_span("mem0_delete_memory") as span:
            span.set_attribute("mem0.memory_id", memory_id)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                response = await self._client.delete(f"/memories/{memory_id}")
                response.raise_for_status()
                return True
                
            except httpx.HTTPError as e:
                logger.error("Failed to delete memory", 
                           memory_id=memory_id,
                           error=str(e))
                return False
                
    async def prune_memories(
        self,
        agent_id: Optional[str] = None,
        older_than: Optional[datetime] = None,
        memory_type: Optional[MemoryType] = None
    ) -> int:
        """Prune old memories based on criteria."""
        with tracer.start_as_current_span("mem0_prune_memories") as span:
            span.set_attribute("mem0.agent_id", agent_id or "all")
            
            if not self._initialized:
                await self.initialize()
                
            try:
                payload = {
                    "agent_id": agent_id,
                    "older_than": older_than.isoformat() if older_than else None,
                    "memory_type": memory_type.value if memory_type else None
                }
                
                response = await self._client.post(
                    "/memories/prune",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                pruned_count = result["pruned_count"]
                
                span.set_attribute("mem0.pruned_count", pruned_count)
                return pruned_count
                
            except httpx.HTTPError as e:
                logger.error("Failed to prune memories", 
                           agent_id=agent_id,
                           error=str(e))
                raise
                
    # Context-specific methods for LNN integration
    
    async def get_context_window(
        self,
        agent_id: str,
        window_size: int = 100,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[Memory]:
        """Get a context window of recent memories for an agent."""
        query = SearchQuery(
            agent_ids=[agent_id],
            memory_types=memory_types or [
                MemoryType.DECISION,
                MemoryType.OBSERVATION,
                MemoryType.PATTERN
            ],
            limit=window_size
        )
        
        memories = await self.search_memories(query)
        
        # Sort by timestamp (most recent first)
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        
        return memories[:window_size]
        
    async def find_similar_decisions(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = None
    ) -> List[Memory]:
        """Find similar past decisions based on embedding."""
        query = SearchQuery(
            query_embedding=embedding,
            memory_types=[MemoryType.DECISION],
            limit=limit
        )
        
        memories = await self.search_memories(query)
        
        # Filter by similarity threshold if specified
        if threshold:
            memories = [m for m in memories if m.relevance_score >= threshold]
            
        return memories
        
    async def get_agent_history(
        self,
        agent_id: str,
        hours: int = 24,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[Memory]:
        """Get agent's recent history."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        query = SearchQuery(
            agent_ids=[agent_id],
            memory_types=memory_types,
            time_range=(start_time, end_time),
            limit=1000  # Get all within time range
        )
        
        return await self.search_memories(query)