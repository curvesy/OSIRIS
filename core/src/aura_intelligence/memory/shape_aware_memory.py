"""
Shape-Aware Memory System for AURA Intelligence
==============================================

This module implements memory retrieval based on topological similarity,
enabling the system to find contextually relevant information based on
the "shape" of data rather than just keywords or embeddings.

This is the core innovation that makes AURA unique.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
from datetime import datetime, timezone
import json
from neo4j import AsyncGraphDatabase
import redis.asyncio as redis
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    # Fallback for older scipy versions
    from scipy.spatial.distance import euclidean as wasserstein_distance

from ..tda.models import TDAResult, BettiNumbers
from ..observability.metrics import metrics_collector


@dataclass
class TopologicalSignature:
    """Represents the topological signature of a memory."""
    
    betti_numbers: BettiNumbers
    persistence_diagram: np.ndarray
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def distance_to(self, other: 'TopologicalSignature') -> float:
        """Calculate topological distance to another signature."""
        # Betti number distance
        betti_dist = np.sqrt(
            (self.betti_numbers.b0 - other.betti_numbers.b0) ** 2 +
            (self.betti_numbers.b1 - other.betti_numbers.b1) ** 2 +
            (self.betti_numbers.b2 - other.betti_numbers.b2) ** 2
        )
        
        # Wasserstein distance for persistence diagrams
        if self.persistence_diagram.size > 0 and other.persistence_diagram.size > 0:
            # Ensure diagrams have same shape
            min_points = min(len(self.persistence_diagram), len(other.persistence_diagram))
            pd1 = self.persistence_diagram[:min_points]
            pd2 = other.persistence_diagram[:min_points]
            
            # Calculate Wasserstein distance
            if pd1.ndim == 2 and pd2.ndim == 2:
                birth1, death1 = pd1[:, 0], pd1[:, 1]
                birth2, death2 = pd2[:, 0], pd2[:, 1]
                
                birth_dist = wasserstein_distance(birth1, birth2)
                death_dist = wasserstein_distance(death1, death2)
                persistence_dist = (birth_dist + death_dist) / 2
            else:
                persistence_dist = 0.0
        else:
            persistence_dist = 0.0
        
        # Weighted combination
        return 0.3 * betti_dist + 0.7 * persistence_dist
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "betti_0": float(self.betti_numbers.b0),
            "betti_1": float(self.betti_numbers.b1),
            "betti_2": float(self.betti_numbers.b2),
            "persistence_diagram": self.persistence_diagram.tolist(),
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TopologicalSignature':
        """Create from dictionary."""
        return cls(
            betti_numbers=BettiNumbers(
                b0=data["betti_0"],
                b1=data["betti_1"],
                b2=data["betti_2"]
            ),
            persistence_diagram=np.array(data["persistence_diagram"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class ShapeMemory:
    """A memory entry with topological context."""
    
    memory_id: str
    content: Dict[str, Any]
    signature: TopologicalSignature
    context_type: str
    relevance_score: float = 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        # Boost relevance based on access patterns
        self.relevance_score = min(self.relevance_score * 1.1, 10.0)


class ShapeAwareMemorySystem:
    """
    Revolutionary memory system that retrieves context based on topological similarity.
    This is AURA's key differentiator - understanding the "shape" of problems.
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        redis_url: str,
        similarity_threshold: float = 0.3
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.redis_url = redis_url
        self.similarity_threshold = similarity_threshold
        
        self._driver: Optional[AsyncGraphDatabase.driver] = None
        self._redis: Optional[redis.Redis] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        if self._initialized:
            return
        
        # Neo4j for persistent storage
        self._driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Redis for fast cache
        self._redis = await redis.from_url(self.redis_url)
        
        # Create Neo4j indices
        async with self._driver.session() as session:
            await session.run("""
                CREATE INDEX IF NOT EXISTS FOR (m:ShapeMemory) ON (m.memory_id)
            """)
            await session.run("""
                CREATE INDEX IF NOT EXISTS FOR (m:ShapeMemory) ON (m.betti_0, m.betti_1, m.betti_2)
            """)
        
        self._initialized = True
        metrics_collector.shape_memory_initialized.inc()
    
    async def store_memory(
        self,
        content: Dict[str, Any],
        tda_result: TDAResult,
        context_type: str = "general"
    ) -> ShapeMemory:
        """Store a memory with its topological signature."""
        # Create topological signature
        signature = TopologicalSignature(
            betti_numbers=tda_result.betti_numbers,
            persistence_diagram=tda_result.persistence_diagram
        )
        
        # Create memory entry
        memory = ShapeMemory(
            memory_id=f"mem_{datetime.now(timezone.utc).timestamp()}_{np.random.randint(1000)}",
            content=content,
            signature=signature,
            context_type=context_type
        )
        
        # Store in Neo4j
        async with self._driver.session() as session:
            await session.run("""
                CREATE (m:ShapeMemory {
                    memory_id: $memory_id,
                    content: $content,
                    betti_0: $betti_0,
                    betti_1: $betti_1,
                    betti_2: $betti_2,
                    persistence_diagram: $persistence_diagram,
                    context_type: $context_type,
                    relevance_score: $relevance_score,
                    created_at: $created_at
                })
            """, {
                "memory_id": memory.memory_id,
                "content": json.dumps(content),
                "betti_0": float(signature.betti_numbers.b0),
                "betti_1": float(signature.betti_numbers.b1),
                "betti_2": float(signature.betti_numbers.b2),
                "persistence_diagram": json.dumps(signature.persistence_diagram.tolist()),
                "context_type": context_type,
                "relevance_score": memory.relevance_score,
                "created_at": memory.created_at.isoformat()
            })
        
        # Cache in Redis for fast access
        await self._cache_memory(memory)
        
        metrics_collector.memories_stored.labels(context_type=context_type).inc()
        return memory
    
    async def retrieve_by_shape(
        self,
        query_signature: TopologicalSignature,
        limit: int = 10,
        context_filter: Optional[str] = None
    ) -> List[ShapeMemory]:
        """
        Retrieve memories with similar topological shape.
        This is the core innovation - finding contextually similar patterns.
        """
        start_time = asyncio.get_event_loop().time()
        
        # First, try Redis cache for recent memories
        cached_memories = await self._search_cache(query_signature, limit)
        
        if len(cached_memories) >= limit:
            metrics_collector.shape_retrievals.labels(source="cache").inc()
            return cached_memories[:limit]
        
        # Search Neo4j for comprehensive results
        async with self._driver.session() as session:
            # Use Betti numbers for initial filtering
            query = """
                MATCH (m:ShapeMemory)
                WHERE abs(m.betti_0 - $b0) < $threshold
                  AND abs(m.betti_1 - $b1) < $threshold
                  AND abs(m.betti_2 - $b2) < $threshold
            """
            
            if context_filter:
                query += " AND m.context_type = $context_type"
            
            query += """
                RETURN m.memory_id as memory_id,
                       m.content as content,
                       m.betti_0 as b0,
                       m.betti_1 as b1,
                       m.betti_2 as b2,
                       m.persistence_diagram as pd,
                       m.context_type as context_type,
                       m.relevance_score as relevance_score,
                       m.created_at as created_at
                ORDER BY m.relevance_score DESC
                LIMIT $limit
            """
            
            params = {
                "b0": float(query_signature.betti_numbers.b0),
                "b1": float(query_signature.betti_numbers.b1),
                "b2": float(query_signature.betti_numbers.b2),
                "threshold": self.similarity_threshold * 2,  # Wider initial filter
                "limit": limit * 3  # Get more candidates for refined filtering
            }
            
            if context_filter:
                params["context_type"] = context_filter
            
            result = await session.run(query, params)
            records = await result.data()
        
        # Convert to ShapeMemory objects and calculate exact distances
        memories_with_distances = []
        
        for record in records:
            # Reconstruct signature
            signature = TopologicalSignature(
                betti_numbers=BettiNumbers(
                    b0=record["b0"],
                    b1=record["b1"],
                    b2=record["b2"]
                ),
                persistence_diagram=np.array(json.loads(record["pd"]))
            )
            
            # Calculate exact topological distance
            distance = query_signature.distance_to(signature)
            
            if distance <= self.similarity_threshold:
                memory = ShapeMemory(
                    memory_id=record["memory_id"],
                    content=json.loads(record["content"]),
                    signature=signature,
                    context_type=record["context_type"],
                    relevance_score=record["relevance_score"],
                    created_at=datetime.fromisoformat(record["created_at"])
                )
                memories_with_distances.append((distance, memory))
        
        # Sort by distance and relevance
        memories_with_distances.sort(key=lambda x: (x[0], -x[1].relevance_score))
        
        # Extract memories
        retrieved_memories = [m for _, m in memories_with_distances[:limit]]
        
        # Update access counts
        for memory in retrieved_memories:
            await self._update_access(memory)
        
        # Record metrics
        retrieval_time = asyncio.get_event_loop().time() - start_time
        metrics_collector.shape_retrieval_latency.observe(retrieval_time * 1000)
        metrics_collector.shape_retrievals.labels(source="neo4j").inc()
        
        return retrieved_memories
    
    async def find_anomaly_patterns(
        self,
        anomaly_signature: TopologicalSignature,
        historical_window: int = 30  # days
    ) -> List[Tuple[ShapeMemory, float]]:
        """
        Find historical patterns similar to current anomaly.
        Returns memories with similarity scores.
        """
        # Search for similar anomalies in recent history
        cutoff_date = datetime.now(timezone.utc).timestamp() - (historical_window * 86400)
        
        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (m:ShapeMemory)
                WHERE m.context_type = 'anomaly'
                  AND m.created_at > $cutoff_date
                RETURN m.memory_id as memory_id,
                       m.content as content,
                       m.betti_0 as b0,
                       m.betti_1 as b1,
                       m.betti_2 as b2,
                       m.persistence_diagram as pd,
                       m.relevance_score as relevance_score,
                       m.created_at as created_at
                ORDER BY m.created_at DESC
                LIMIT 100
            """, {"cutoff_date": cutoff_date})
            
            records = await result.data()
        
        # Calculate similarity scores
        similar_patterns = []
        
        for record in records:
            signature = TopologicalSignature(
                betti_numbers=BettiNumbers(
                    b0=record["b0"],
                    b1=record["b1"],
                    b2=record["b2"]
                ),
                persistence_diagram=np.array(json.loads(record["pd"]))
            )
            
            similarity = 1.0 - min(anomaly_signature.distance_to(signature), 1.0)
            
            if similarity > 0.7:  # High similarity threshold for anomalies
                memory = ShapeMemory(
                    memory_id=record["memory_id"],
                    content=json.loads(record["content"]),
                    signature=signature,
                    context_type="anomaly",
                    relevance_score=record["relevance_score"],
                    created_at=datetime.fromisoformat(record["created_at"])
                )
                similar_patterns.append((memory, similarity))
        
        # Sort by similarity
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return similar_patterns
    
    async def _cache_memory(self, memory: ShapeMemory) -> None:
        """Cache memory in Redis for fast access."""
        key = f"shape_memory:{memory.memory_id}"
        value = {
            "content": json.dumps(memory.content),
            "signature": memory.signature.to_dict(),
            "context_type": memory.context_type,
            "relevance_score": memory.relevance_score
        }
        
        # Store with TTL based on relevance
        ttl = int(3600 * memory.relevance_score)  # 1-10 hours based on relevance
        await self._redis.hset(key, mapping=value)
        await self._redis.expire(key, ttl)
    
    async def _search_cache(
        self,
        query_signature: TopologicalSignature,
        limit: int
    ) -> List[ShapeMemory]:
        """Search Redis cache for similar memories."""
        # Get all cached memories (simplified for demo)
        pattern = "shape_memory:*"
        memories = []
        
        async for key in self._redis.scan_iter(match=pattern):
            data = await self._redis.hgetall(key)
            if data:
                signature_data = json.loads(data[b"signature"])
                signature = TopologicalSignature.from_dict(signature_data)
                
                distance = query_signature.distance_to(signature)
                if distance <= self.similarity_threshold:
                    memory = ShapeMemory(
                        memory_id=key.decode().split(":")[-1],
                        content=json.loads(data[b"content"]),
                        signature=signature,
                        context_type=data[b"context_type"].decode(),
                        relevance_score=float(data[b"relevance_score"])
                    )
                    memories.append((distance, memory))
        
        # Sort by distance
        memories.sort(key=lambda x: x[0])
        return [m for _, m in memories[:limit]]
    
    async def _update_access(self, memory: ShapeMemory) -> None:
        """Update access statistics for a memory."""
        memory.update_access()
        
        async with self._driver.session() as session:
            await session.run("""
                MATCH (m:ShapeMemory {memory_id: $memory_id})
                SET m.access_count = m.access_count + 1,
                    m.last_accessed = $now,
                    m.relevance_score = $relevance_score
            """, {
                "memory_id": memory.memory_id,
                "now": memory.last_accessed.isoformat(),
                "relevance_score": memory.relevance_score
            })
    
    async def cleanup(self) -> None:
        """Clean up connections."""
        if self._driver:
            await self._driver.close()
        if self._redis:
            await self._redis.close()


# Example usage
async def demo_shape_aware_memory():
    """Demonstrate shape-aware memory retrieval."""
    
    # Initialize system
    memory_system = ShapeAwareMemorySystem(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        redis_url="redis://localhost:6379"
    )
    
    await memory_system.initialize()
    
    # Store a memory with topological signature
    tda_result = TDAResult(
        betti_numbers=BettiNumbers(b0=1, b1=2, b2=0),
        persistence_diagram=np.array([[0.1, 0.5], [0.2, 0.8], [0.3, 0.7]]),
        topological_features={"holes": 2, "components": 1}
    )
    
    memory = await memory_system.store_memory(
        content={
            "event": "Network anomaly detected",
            "severity": "high",
            "pattern": "Unusual topology in traffic flow"
        },
        tda_result=tda_result,
        context_type="anomaly"
    )
    
    print(f"Stored memory: {memory.memory_id}")
    
    # Retrieve similar memories by shape
    query_signature = TopologicalSignature(
        betti_numbers=BettiNumbers(b0=1, b1=2, b2=0),
        persistence_diagram=np.array([[0.1, 0.6], [0.2, 0.7], [0.3, 0.8]])
    )
    
    similar_memories = await memory_system.retrieve_by_shape(
        query_signature=query_signature,
        limit=5,
        context_filter="anomaly"
    )
    
    print(f"\nFound {len(similar_memories)} similar memories by shape:")
    for mem in similar_memories:
        print(f"  - {mem.memory_id}: {mem.content}")
    
    await memory_system.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_shape_aware_memory())