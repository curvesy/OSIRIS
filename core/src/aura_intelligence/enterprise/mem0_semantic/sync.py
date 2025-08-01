"""
🔄 Semantic Memory Synchronization

Batch consolidation from hot tier to AWS MemoryDB Redis.
Implements semantic clustering and long-term memory persistence.

Based on partab.md: "batch consolidation, Redis writes" specification.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis.asyncio as redis
import numpy as np
from starlette.concurrency import run_in_threadpool

# Redis search imports for vector indexing
try:
    from redis.commands.search.field import VectorField, TagField, TextField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_SEARCH_AVAILABLE = True
except ImportError:
    REDIS_SEARCH_AVAILABLE = False

from aura_intelligence.enterprise.data_structures import TopologicalSignature
from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer
from aura_intelligence.utils.logger import get_logger


@dataclass
class SemanticCluster:
    """Semantic cluster of related signatures."""
    cluster_id: str
    centroid_vector: np.ndarray
    signature_hashes: Set[str]
    creation_time: datetime
    last_updated: datetime
    cluster_score: float


@dataclass
class MemoryConsolidationBatch:
    """Batch of signatures for semantic consolidation."""
    signatures: List[TopologicalSignature]
    vectors: List[np.ndarray]
    timestamp: datetime
    batch_id: str


class SemanticMemorySync:
    """
    🔄 Semantic Memory Synchronization Service
    
    Features:
    - Batch consolidation from DuckDB hot tier to Redis
    - Semantic clustering using vector similarity
    - Long-term memory persistence with TTL
    - Incremental updates and conflict resolution
    - Performance monitoring and alerting
    """
    
    def __init__(self,
                 redis_url: str,
                 vectorizer: SignatureVectorizer,
                 cluster_threshold: float = 0.8):
        """Initialize semantic memory sync service."""

        self.redis_url = redis_url
        self.vectorizer = vectorizer
        self.cluster_threshold = cluster_threshold

        # Redis connection
        self.redis_client = None

        # Vector search indexes
        self.index_name = "semantic_memory_index"
        self.cluster_index_name = "semantic_cluster_index"
        self.vector_dimension = 128  # Must match vectorizer output

        # Clustering state
        self.active_clusters: Dict[str, SemanticCluster] = {}
        self.cluster_counter = 0
        
        # Performance tracking
        self.sync_count = 0
        self.total_sync_time = 0.0
        self.total_signatures_synced = 0
        self.sync_errors = 0
        
        # Background sync
        self.sync_task = None
        self.is_running = False
        
        self.logger = get_logger(__name__)

        if not REDIS_SEARCH_AVAILABLE:
            self.logger.warning("⚠️ Redis search module not available - vector search will be limited")

        self.logger.info("🔄 Semantic Memory Sync initialized")

    async def initialize(self):
        """Initialize Redis connection, create vector indexes, and load existing clusters."""

        try:
            # Connect to Redis
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Keep binary for vector data
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )

            # Test connection
            await self.redis_client.ping()
            self.logger.info("✅ Redis connection established")

            # Create vector search indexes
            if REDIS_SEARCH_AVAILABLE:
                await self._create_vector_indexes()

            # Load existing clusters
            await self._load_existing_clusters()

            return True

        except Exception as e:
            self.logger.error(f"❌ Redis initialization failed: {e}")
            return False

    async def _create_vector_indexes(self):
        """Create Redis vector search indexes for semantic memories and clusters."""

        try:
            # Create semantic memory index
            await self._create_semantic_memory_index()

            # Create cluster index
            await self._create_cluster_index()

            self.logger.info("✅ Vector search indexes created successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to create vector indexes: {e}")

    async def _create_semantic_memory_index(self):
        """Create the main semantic memory vector search index."""

        try:
            # Check if index exists
            await run_in_threadpool(
                lambda: self.redis_client.ft(self.index_name).info()
            )
            self.logger.debug(f"Index '{self.index_name}' already exists")
            return

        except:
            # Index doesn't exist, create it
            pass

        try:
            # Define the schema for searchable semantic memories
            schema = (
                # Core vector field for similarity search
                VectorField(
                    "embedding",
                    "HNSW",  # High-performance HNSW algorithm
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dimension,  # 128-dimensional vectors
                        "DISTANCE_METRIC": "COSINE",  # Cosine similarity
                    },
                ),
                # Searchable metadata fields
                TagField("agent_id"),
                TextField("event_type"),
                TextField("signature_hash"),
                TagField("memory_tier"),
                TextField("betti_numbers"),
                TextField("timestamp"),
            )

            # Create index definition
            definition = IndexDefinition(
                prefix=["semantic:signature:"],
                index_type=IndexType.HASH
            )

            # Create the index
            await run_in_threadpool(
                lambda: self.redis_client.ft(self.index_name).create_index(
                    fields=schema,
                    definition=definition
                )
            )

            self.logger.info(f"✅ Created semantic memory index '{self.index_name}'")

        except Exception as e:
            self.logger.error(f"❌ Failed to create semantic memory index: {e}")
            raise

    async def _create_cluster_index(self):
        """Create the cluster centroid vector search index."""

        try:
            # Check if cluster index exists
            await run_in_threadpool(
                lambda: self.redis_client.ft(self.cluster_index_name).info()
            )
            self.logger.debug(f"Cluster index '{self.cluster_index_name}' already exists")
            return

        except:
            # Index doesn't exist, create it
            pass

        try:
            # Define schema for cluster centroids
            schema = (
                # Cluster centroid vector
                VectorField(
                    "centroid",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dimension,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                # Cluster metadata
                TextField("cluster_id"),
                TextField("cluster_label"),
                TextField("signature_count"),
                TextField("created_at"),
                TextField("signature_hashes"),
            )

            # Create cluster index definition
            definition = IndexDefinition(
                prefix=["semantic:cluster:"],
                index_type=IndexType.HASH
            )

            # Create the cluster index
            await run_in_threadpool(
                lambda: self.redis_client.ft(self.cluster_index_name).create_index(
                    fields=schema,
                    definition=definition
                )
            )

            self.logger.info(f"✅ Created cluster index '{self.cluster_index_name}'")

        except Exception as e:
            self.logger.error(f"❌ Failed to create cluster index: {e}")
            raise

    async def start_background_sync(self, interval_minutes: int = 15):
        """Start background synchronization process."""
        
        if self.is_running:
            self.logger.warning("⚠️ Background sync already running")
            return
        
        if not self.redis_client:
            await self.initialize()
        
        self.is_running = True
        self.sync_task = asyncio.create_task(
            self._background_sync_loop(interval_minutes)
        )
        
        self.logger.info(f"🔄 Background sync started (interval: {interval_minutes}min)")
    
    async def stop_background_sync(self):
        """Stop background synchronization process."""
        
        if not self.is_running:
            return
        
        self.is_running = False
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("⏹️ Background sync stopped")
    
    async def _background_sync_loop(self, interval_minutes: int):
        """Background loop for periodic synchronization."""
        
        while self.is_running:
            try:
                # This would integrate with the hot tier to get new signatures
                # For now, we'll implement the sync logic structure
                await self._perform_sync_cycle()
                await asyncio.sleep(interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Background sync error: {e}")
                self.sync_errors += 1
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def sync_batch(self, batch: MemoryConsolidationBatch) -> Dict[str, Any]:
        """
        Synchronize a batch of signatures to semantic long-term memory.
        
        Args:
            batch: Batch of signatures with vectors
            
        Returns:
            Sync result with cluster assignments and performance metrics
        """
        
        try:
            start_time = time.time()
            
            # Process signatures into semantic clusters
            cluster_assignments = await self._cluster_signatures(batch)
            
            # Update Redis with consolidated clusters
            redis_updates = await self._update_redis_clusters(cluster_assignments)
            
            # Update local cluster state
            await self._update_local_clusters(cluster_assignments)
            
            # Performance tracking
            sync_time = time.time() - start_time
            self.sync_count += 1
            self.total_sync_time += sync_time
            self.total_signatures_synced += len(batch.signatures)
            
            result = {
                "status": "success",
                "batch_id": batch.batch_id,
                "signatures_processed": len(batch.signatures),
                "clusters_updated": len(cluster_assignments),
                "redis_operations": redis_updates,
                "sync_time_seconds": sync_time
            }
            
            self.logger.info(f"🔄 Synced batch {batch.batch_id}: {len(batch.signatures)} signatures in {sync_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Batch sync failed for {batch.batch_id}: {e}")
            self.sync_errors += 1
            return {"status": "error", "batch_id": batch.batch_id, "error": str(e)}
    
    async def _cluster_signatures(self, batch: MemoryConsolidationBatch) -> Dict[str, List[int]]:
        """Cluster signatures based on vector similarity."""
        
        cluster_assignments = {}
        
        for i, (signature, vector) in enumerate(zip(batch.signatures, batch.vectors)):
            # Find best matching cluster
            best_cluster_id = None
            best_similarity = 0.0
            
            for cluster_id, cluster in self.active_clusters.items():
                similarity = self.vectorizer.compute_similarity(
                    vector, cluster.centroid_vector, metric="cosine"
                )
                
                if similarity > best_similarity and similarity >= self.cluster_threshold:
                    best_similarity = similarity
                    best_cluster_id = cluster_id
            
            # Create new cluster if no match found
            if best_cluster_id is None:
                best_cluster_id = await self._create_new_cluster(signature, vector)
            
            # Add to cluster assignment
            if best_cluster_id not in cluster_assignments:
                cluster_assignments[best_cluster_id] = []
            cluster_assignments[best_cluster_id].append(i)
        
        return cluster_assignments
    
    async def _create_new_cluster(self, signature: TopologicalSignature, vector: np.ndarray) -> str:
        """Create a new semantic cluster."""
        
        self.cluster_counter += 1
        cluster_id = f"cluster_{self.cluster_counter}_{int(time.time())}"
        
        cluster = SemanticCluster(
            cluster_id=cluster_id,
            centroid_vector=vector.copy(),
            signature_hashes={signature.hash},
            creation_time=datetime.now(),
            last_updated=datetime.now(),
            cluster_score=1.0
        )
        
        self.active_clusters[cluster_id] = cluster
        
        self.logger.debug(f"🆕 Created new cluster: {cluster_id}")
        
        return cluster_id
    
    async def _update_redis_clusters(self, cluster_assignments: Dict[str, List[int]]) -> int:
        """Update Redis with cluster information using vector search index."""

        if not self.redis_client:
            return 0

        operations = 0

        try:
            pipe = self.redis_client.pipeline()

            for cluster_id, signature_indices in cluster_assignments.items():
                cluster = self.active_clusters[cluster_id]

                # Store cluster metadata with vector index support
                cluster_key = f"semantic:cluster:{cluster_id}"

                # Prepare cluster data for vector index
                cluster_data = {
                    "cluster_id": cluster_id,
                    "cluster_label": f"cluster_{cluster_id}",
                    "signature_count": str(len(cluster.signature_hashes)),
                    "created_at": cluster.creation_time.isoformat(),
                    "signature_hashes": ",".join(cluster.signature_hashes)
                }

                # Add centroid vector for vector search (binary format)
                if REDIS_SEARCH_AVAILABLE:
                    cluster_data["centroid"] = cluster.centroid_vector.astype(np.float32).tobytes()
                else:
                    # Fallback to JSON for non-vector search
                    cluster_data["centroid_vector"] = json.dumps(cluster.centroid_vector.tolist())

                pipe.hset(cluster_key, mapping=cluster_data)
                pipe.expire(cluster_key, 86400 * 30)  # 30 day TTL
                operations += 2

                # Store signature hashes in cluster (for backward compatibility)
                signatures_key = f"semantic:signatures:{cluster_id}"
                pipe.sadd(signatures_key, *cluster.signature_hashes)
                pipe.expire(signatures_key, 86400 * 30)
                operations += 2

            # Execute pipeline
            await pipe.execute()

            return operations

        except Exception as e:
            self.logger.error(f"❌ Redis cluster update failed: {e}")
            return 0

    async def sync_consolidated_memories(self, memories: List[Dict[str, Any]]) -> int:
        """
        Write consolidated memories to Redis with vector search index.

        This is the PRODUCTION-GRADE method that writes memories to the
        vector search index for high-performance similarity search.
        """

        if not self.redis_client or not memories:
            return 0

        operations = 0

        try:
            pipe = self.redis_client.pipeline()

            for memory in memories:
                signature_key = f"semantic:signature:{memory['hash']}"

                # Prepare memory data for vector index
                memory_data = {
                    "signature_hash": memory['hash'],
                    "agent_id": memory.get('agent_id', 'unknown'),
                    "event_type": memory.get('event_type', 'unknown'),
                    "memory_tier": "SEMANTIC",
                    "betti_numbers": ",".join(map(str, memory.get('betti_numbers', [0, 0, 0]))),
                    "timestamp": memory.get('timestamp', datetime.now().isoformat()),
                }

                # Add embedding vector for vector search (binary format)
                if REDIS_SEARCH_AVAILABLE and 'embedding' in memory:
                    embedding_array = np.array(memory['embedding'], dtype=np.float32)
                    memory_data["embedding"] = embedding_array.tobytes()
                else:
                    # Fallback for non-vector search
                    if 'embedding' in memory:
                        memory_data["embedding_json"] = json.dumps(memory['embedding'])

                pipe.hset(signature_key, mapping=memory_data)
                pipe.expire(signature_key, 86400 * 90)  # 90 day TTL for semantic memories
                operations += 2

            # Execute pipeline
            await pipe.execute()

            self.logger.info(f"✅ Synced {len(memories)} memories to semantic store with vector index")

            return operations

        except Exception as e:
            self.logger.error(f"❌ Failed to sync consolidated memories: {e}")
            return 0
    
    async def _update_local_clusters(self, cluster_assignments: Dict[str, List[int]]):
        """Update local cluster state."""
        
        for cluster_id in cluster_assignments:
            if cluster_id in self.active_clusters:
                self.active_clusters[cluster_id].last_updated = datetime.now()
    
    async def _load_existing_clusters(self):
        """Load existing clusters from Redis."""
        
        if not self.redis_client:
            return
        
        try:
            # Get all cluster keys
            cluster_keys = await self.redis_client.keys("semantic:cluster:*")
            
            for cluster_key in cluster_keys:
                cluster_id = cluster_key.split(":")[-1]
                
                # Load cluster data
                cluster_data = await self.redis_client.hgetall(cluster_key)
                
                if cluster_data:
                    # Load signature hashes
                    signatures_key = f"semantic:signatures:{cluster_id}"
                    signature_hashes = await self.redis_client.smembers(signatures_key)
                    
                    # Reconstruct cluster
                    cluster = SemanticCluster(
                        cluster_id=cluster_id,
                        centroid_vector=np.array(json.loads(cluster_data["centroid_vector"])),
                        signature_hashes=set(signature_hashes),
                        creation_time=datetime.fromisoformat(cluster_data["creation_time"]),
                        last_updated=datetime.fromisoformat(cluster_data["last_updated"]),
                        cluster_score=float(cluster_data["cluster_score"])
                    )
                    
                    self.active_clusters[cluster_id] = cluster
            
            self.logger.info(f"📥 Loaded {len(self.active_clusters)} existing clusters")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load existing clusters: {e}")
    
    async def _perform_sync_cycle(self):
        """Perform a complete sync cycle."""
        
        # This would integrate with the hot tier to get new data
        # For now, we'll implement the structure
        
        self.logger.debug("🔄 Performing sync cycle")
        
        # In a real implementation, this would:
        # 1. Query hot tier for new signatures since last sync
        # 2. Create MemoryConsolidationBatch
        # 3. Call sync_batch()
        # 4. Update sync timestamps
    
    def get_sync_metrics(self) -> Dict[str, Any]:
        """Get synchronization performance metrics."""
        
        avg_sync_time = self.total_sync_time / max(self.sync_count, 1)
        
        return {
            "sync_count": self.sync_count,
            "total_signatures_synced": self.total_signatures_synced,
            "total_sync_time_seconds": self.total_sync_time,
            "avg_sync_time_seconds": avg_sync_time,
            "sync_errors": self.sync_errors,
            "active_clusters": len(self.active_clusters),
            "cluster_threshold": self.cluster_threshold,
            "is_running": self.is_running
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on semantic sync service."""
        
        try:
            # Check Redis connectivity
            redis_healthy = False
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    redis_healthy = True
                except Exception:
                    pass
            
            # Check cluster state
            cluster_health = {
                "active_clusters": len(self.active_clusters),
                "avg_cluster_size": sum(len(c.signature_hashes) for c in self.active_clusters.values()) / max(len(self.active_clusters), 1)
            }
            
            return {
                "status": "healthy" if redis_healthy else "unhealthy",
                "redis_healthy": redis_healthy,
                "background_running": self.is_running,
                "cluster_health": cluster_health,
                "metrics": self.get_sync_metrics()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "redis_healthy": False
            }


# Alias for backward compatibility
SemanticSync = SemanticMemorySync
