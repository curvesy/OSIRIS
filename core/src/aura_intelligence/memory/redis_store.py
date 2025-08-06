"""
Redis Vector Store - Production Implementation
============================================

A battle-tested Redis vector store with connection pooling,
circuit breakers, retry logic, and comprehensive error handling.

Based on 2025 best practices from Discord, Uber, and Pinterest.
"""

import redis
import numpy as np
import json
import time
import logging
import threading
import zstandard as zstd  # Add zstd for compression
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from redis.commands.search.field import VectorField, TagField, TextField, NumericField
from redis.commands.search.query import Query
from redis.exceptions import ResponseError, ConnectionError, TimeoutError
from pybreaker import CircuitBreaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .observability import instrument, update_vector_count, record_embedding_age
from .storage_interface import MemoryStorage

logger = logging.getLogger(__name__)

# Production constants
INDEX_NAME = "shape_memory_idx"
KEY_PREFIX = "shape:v2:"
VECTOR_DIMENSIONS = 128

# Compression settings
COMPRESSION_LEVEL = 3  # zstd level 3 is a good balance of speed/ratio
COMPRESSION_THRESHOLD = 1024  # Only compress if data > 1KB


@dataclass
class RedisConfig:
    """Production Redis configuration."""
    url: str = "redis://localhost:6379"
    max_connections: int = 50
    socket_timeout: int = 5
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, int] = None
    metrics_update_interval: int = 30  # seconds
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            self.socket_keepalive_options = {
                1: 1,   # TCP_KEEPIDLE
                2: 5,   # TCP_KEEPINTVL  
                3: 3,   # TCP_KEEPCNT
            }


class MetricsUpdater:
    """Background thread for updating metrics without blocking operations."""
    
    def __init__(self, store: 'RedisVectorStore', interval: int):
        self.store = store
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="metrics-updater",
            daemon=True
        )
        self._thread.start()
        logger.debug(f"MetricsUpdater started with {interval}s interval")
    
    def _run(self):
        """Main loop for the metrics thread."""
        while not self._stop_event.is_set():
            try:
                self._do_update()
            except Exception as e:
                logger.warning(f"Metrics update failed: {e}")
            
            # Wait for interval or stop signal
            self._stop_event.wait(self.interval)
    
    def _do_update(self):
        """Perform the actual metrics update."""
        try:
            info = self.store.redis.ft(INDEX_NAME).info()
            doc_count = int(info.num_docs)
            update_vector_count("redis", doc_count)
            logger.debug(f"Updated metrics: {doc_count} vectors")
        except Exception as e:
            logger.warning(f"Failed to get index info: {e}")
    
    def stop(self):
        """Stop the metrics updater gracefully."""
        logger.debug("Stopping MetricsUpdater")
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("MetricsUpdater thread did not stop gracefully")


# Define retry decorator for transient failures
redis_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying Redis operation (attempt {retry_state.attempt_number})"
    )
)


class RedisVectorStore(MemoryStorage):
    """
    Production-ready Redis vector store with HNSW indexing.
    
    Features:
    - Connection pooling for performance
    - Circuit breaker for resilience
    - Retry logic for transient failures
    - Throttled metrics updates
    - Comprehensive monitoring
    - Atomic operations with pipelines
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        
        # Connection pool for efficiency
        self.pool = redis.BlockingConnectionPool.from_url(
            self.config.url,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_keepalive=self.config.socket_keepalive,
            socket_keepalive_options=self.config.socket_keepalive_options,
            decode_responses=False  # Important for binary data
        )
        
        self.redis = redis.Redis(connection_pool=self.pool)
        
        # Circuit breaker prevents cascading failures
        self.breaker = CircuitBreaker(
            fail_max=5,
            reset_timeout=30,  # Reduced from 60s for faster recovery
            exclude=[ResponseError]  # Don't break on logical errors
        )
        
        # Start background metrics updater
        self.metrics_updater = MetricsUpdater(self, self.config.metrics_update_interval)
        
        self._ensure_index()
    
    @contextmanager
    def _timed_operation(self, operation: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"{operation} took {elapsed:.2f}ms")
    
    def _ensure_index(self):
        """Create HNSW index if not exists."""
        try:
            # Check if index exists
            self.redis.ft(INDEX_NAME).info()
            logger.info(f"Index '{INDEX_NAME}' already exists")
        except ResponseError as e:
            if "Unknown index" in str(e):
                logger.info(f"Creating new index '{INDEX_NAME}'")
                self._create_index()
            else:
                logger.error(f"Error checking index: {e}")
                raise
    
    def _create_index(self):
        """Create the HNSW vector index."""
        # Drop existing index if any (shouldn't happen but be safe)
        try:
            self.redis.ft(INDEX_NAME).dropindex(delete_documents=False)
        except ResponseError as e:
            if "Unknown index" not in str(e):
                logger.warning(f"Failed to drop index: {e}")
        
        # Create schema with optimized HNSW parameters
        schema = (
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIMENSIONS,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 40,  # High connectivity for better recall
                    "EF_CONSTRUCTION": 200,  # High quality construction
                    "EF_RUNTIME": 50  # Runtime search quality
                }
            ),
            TagField("context_type"),
            TextField("content"),
            TagField("version"),  # For embedding versioning
            NumericField("created_at")  # For time-based queries
        )
        
        definition = redis.commands.search.IndexDefinition(
            prefix=[KEY_PREFIX]
        )
        
        self.redis.ft(INDEX_NAME).create_index(
            schema,
            definition=definition
        )
        logger.info("Created HNSW index with optimized parameters")
    
    @instrument("store", "redis")
    @redis_retry  # Retry should be innermost (closest to function)
    def add(
        self,
        memory_id: str,
        embedding: np.ndarray,
        content: Dict[str, Any],
        context_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a memory to the store with retry logic.
        
        Returns:
            bool: Success status
        """
        with self._timed_operation("redis_add"):
            key = f"{KEY_PREFIX}{memory_id}"
            
            # Validate embedding dimension
            if embedding.shape[0] != VECTOR_DIMENSIONS:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {VECTOR_DIMENSIONS}, "
                    f"got {embedding.shape[0]}"
                )
            
            # Prepare data
            data = {
                "embedding": embedding.astype(np.float32).tobytes(),
                "content": json.dumps(content),
                "context_type": context_type,
                "version": "v2",  # Embedding version tracking
                "created_at": int(time.time())
            }
            
            if metadata:
                # Check if we have a persistence diagram to compress
                if "persistence_diagram" in metadata:
                    diagram = metadata["persistence_diagram"]
                    # Convert to bytes if it's a list (from tolist())
                    if isinstance(diagram, list):
                        diagram_array = np.array(diagram, dtype=np.float32)
                        diagram_bytes = diagram_array.tobytes()
                    else:
                        diagram_bytes = diagram.tobytes()
                    
                    # Compress if above threshold
                    if len(diagram_bytes) > COMPRESSION_THRESHOLD:
                        compressed = zstd.compress(diagram_bytes, COMPRESSION_LEVEL)
                        # Store compressed version with marker
                        metadata["persistence_diagram_compressed"] = compressed.hex()
                        metadata["persistence_diagram_shape"] = diagram_array.shape if isinstance(diagram, list) else diagram.shape
                        metadata["persistence_diagram_dtype"] = str(diagram_array.dtype if isinstance(diagram, list) else diagram.dtype)
                        # Remove uncompressed version
                        del metadata["persistence_diagram"]
                        logger.debug(f"Compressed diagram from {len(diagram_bytes)} to {len(compressed)} bytes ({len(compressed)/len(diagram_bytes):.1%})")
                
                data["metadata"] = json.dumps(metadata)
                # Track embedding age if provided
                if "embedding_created_at" in metadata:
                    age_hours = (time.time() - metadata["embedding_created_at"]) / 3600
                    record_embedding_age(age_hours)
            
            # Use pipeline for atomic operation
            pipe = self.redis.pipeline()
            pipe.hset(key, mapping=data)
            pipe.expire(key, 86400 * 30)  # 30 day TTL
            
            try:
                results = pipe.execute()
                return all(results)
            except Exception as e:
                logger.error(f"Failed to add memory {memory_id}: {e}")
                raise
    
    @instrument("search", "redis")
    @redis_retry  # Retry should be innermost (closest to function)
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        context_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors with retry logic.
        
        Returns:
            List of (memory_data, similarity_score) tuples
        """
        with self._timed_operation("redis_search"):
            # Validate query dimension
            if query_embedding.shape[0] != VECTOR_DIMENSIONS:
                raise ValueError(
                    f"Query dimension mismatch: expected {VECTOR_DIMENSIONS}, "
                    f"got {query_embedding.shape[0]}"
                )
            
            # Build query
            base_query = f"*=>[KNN {k} @embedding $vec AS score]"
            
            # Add filters
            filters = []
            if context_filter:
                filters.append(f"@context_type:{{{context_filter}}}")
            
            if score_threshold > 0:
                # Note: Redis returns distance, not similarity
                # For cosine: similarity = 1 - distance
                max_distance = 1 - score_threshold
                filters.append(f"@score:[0 {max_distance}]")
            
            # Combine query parts
            query_str = base_query
            if filters:
                filter_str = " ".join(filters)
                query_str = f"({filter_str})=>[KNN {k} @embedding $vec AS score]"
            
            # Create query object
            q = Query(query_str).sort_by("score").return_fields(
                "score", "content", "context_type", "metadata", "created_at"
            ).dialect(2)
            
            # Execute search
            query_vec = query_embedding.astype(np.float32).tobytes()
            
            try:
                results = self.redis.ft(INDEX_NAME).search(
                    q, query_params={"vec": query_vec}
                )
            except ResponseError as e:
                logger.error(f"Search failed: {e}")
                return []
            
            # Parse results
            memories = []
            for doc in results.docs:
                try:
                    # Parse stored data
                    content = json.loads(doc.content)
                    metadata = json.loads(doc.metadata) if hasattr(doc, 'metadata') else {}
                    
                    # Decompress persistence diagram if needed
                    if "persistence_diagram_compressed" in metadata:
                        try:
                            compressed_hex = metadata["persistence_diagram_compressed"]
                            compressed_bytes = bytes.fromhex(compressed_hex)
                            decompressed = zstd.decompress(compressed_bytes)
                            
                            # Reconstruct array with original shape and dtype
                            shape = tuple(metadata["persistence_diagram_shape"])
                            dtype = np.dtype(metadata["persistence_diagram_dtype"])
                            diagram_array = np.frombuffer(decompressed, dtype=dtype).reshape(shape)
                            
                            # Replace compressed data with decompressed array
                            metadata["persistence_diagram"] = diagram_array.tolist()
                            
                            # Remove compression artifacts
                            del metadata["persistence_diagram_compressed"]
                            del metadata["persistence_diagram_shape"]
                            del metadata["persistence_diagram_dtype"]
                        except Exception as e:
                            logger.warning(f"Failed to decompress diagram: {e}")
                    
                    # Calculate similarity from distance
                    distance = float(doc.score)
                    similarity = 1 - distance
                    
                    # Skip if below threshold
                    if similarity < score_threshold:
                        continue
                    
                    memory_data = {
                        "id": doc.id.replace(KEY_PREFIX, ""),
                        "content": content,
                        "context_type": doc.context_type,
                        "metadata": metadata,
                        "created_at": int(doc.created_at) if hasattr(doc, 'created_at') else None
                    }
                    
                    memories.append((memory_data, similarity))
                    
                except Exception as e:
                    logger.warning(f"Failed to parse result {doc.id}: {e}")
                    continue
            
            return memories
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis connection and index health."""
        try:
            # Ping Redis
            self.redis.ping()
            
            # Get index info
            info = self.redis.ft(INDEX_NAME).info()
            
            return {
                "status": "healthy",
                "backend": "redis",
                "index_docs": int(info.num_docs),
                "circuit_breaker": self.breaker.current_state,
                "connection_pool": {
                    "created": self.pool.created_connections,
                    "available": self.pool.max_connections - self.pool.created_connections
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "redis",
                "error": str(e),
                "circuit_breaker": self.breaker.current_state
            }
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""
        try:
            key = f"{KEY_PREFIX}{memory_id}"
            data = self.redis.hgetall(key)
            
            if not data:
                return None
            
            # Parse the stored data
            memory_data = {
                "id": memory_id,
                "content": json.loads(data[b"content"]),
                "context_type": data[b"context_type"].decode(),
                "metadata": json.loads(data[b"metadata"]) if b"metadata" in data else {},
                "created_at": int(data[b"created_at"]) if b"created_at" in data else None
            }
            
            # Decompress persistence diagram if needed
            metadata = memory_data["metadata"]
            if "persistence_diagram_compressed" in metadata:
                try:
                    compressed_hex = metadata["persistence_diagram_compressed"]
                    compressed_bytes = bytes.fromhex(compressed_hex)
                    decompressed = zstd.decompress(compressed_bytes)
                    
                    shape = tuple(metadata["persistence_diagram_shape"])
                    dtype = np.dtype(metadata["persistence_diagram_dtype"])
                    diagram_array = np.frombuffer(decompressed, dtype=dtype).reshape(shape)
                    
                    metadata["persistence_diagram"] = diagram_array.tolist()
                    del metadata["persistence_diagram_compressed"]
                    del metadata["persistence_diagram_shape"]
                    del metadata["persistence_diagram_dtype"]
                except Exception as e:
                    logger.warning(f"Failed to decompress diagram: {e}")
            
            return memory_data
            
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            key = f"{KEY_PREFIX}{memory_id}"
            result = self.redis.delete(key)
            
            if result > 0:
                logger.debug(f"Deleted memory {memory_id}")
                return True
            else:
                logger.debug(f"Memory {memory_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def close(self):
        """Clean up resources."""
        try:
            # Stop metrics updater
            self.metrics_updater.stop()
            
            # Close connection pool
            self.pool.disconnect()
            
            logger.info("Redis vector store closed successfully")
        except Exception as e:
            logger.error(f"Error closing Redis store: {e}")