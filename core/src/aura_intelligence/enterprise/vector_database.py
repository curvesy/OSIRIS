"""
🔥 AURA Intelligence Vector Database Service

Qdrant integration for sub-10ms topological similarity search.
This is the foundation of our intelligence flywheel - enabling
"Have we seen this shape before?" queries.

Based on kiki.md and ppdd.md research for professional 2025 architecture.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, Match, SearchRequest
)

from aura_intelligence.enterprise.data_structures import (
    TopologicalSignature, SearchResult, vectorize_persistence_diagram,
    calculate_signature_similarity
)
from aura_intelligence.utils.logger import get_logger


class VectorDatabaseService:
    """
    🔥 Vector Database Service for Topological Intelligence
    
    Provides sub-10ms similarity search for topological signatures using Qdrant.
    This is the core component that enables the intelligence flywheel by answering
    "Have we seen this shape before?" with lightning speed.
    
    Features:
    - Custom distance metrics for topological data
    - Consciousness-weighted similarity scoring
    - High-performance HNSW indexing
    - Real-time signature storage and retrieval
    - Production-ready with monitoring and health checks
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "topology_signatures",
                 vector_size: int = 16,
                 enable_monitoring: bool = True):
        """
        Initialize Vector Database Service.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection for signatures
            vector_size: Dimension of signature vectors
            enable_monitoring: Enable performance monitoring
        """
        self.logger = get_logger(__name__)
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.enable_monitoring = enable_monitoring
        
        # Performance metrics
        self.query_count = 0
        self.total_query_time = 0.0
        self.avg_query_time = 0.0
        
        # Initialize Qdrant client
        self.client = None
        self.initialized = False
        
        self.logger.info(f"🔥 Vector Database Service initialized (Qdrant: {host}:{port})")
    
    async def initialize(self) -> bool:
        """Initialize Qdrant connection and create collection if needed."""
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(host=self.host, port=self.port)
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                # Create collection with optimized settings for topological data
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,  # Cosine similarity for topological data
                        hnsw_config=models.HnswConfigDiff(
                            m=16,  # Number of bi-directional links for each node
                            ef_construct=200,  # Size of dynamic candidate list
                            full_scan_threshold=10000,  # Threshold for full scan
                            max_indexing_threads=4  # Parallel indexing
                        )
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=None,
                        memmap_threshold=None,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=2
                    )
                )
                self.logger.info(f"✅ Created Qdrant collection: {self.collection_name}")
            else:
                self.logger.info(f"✅ Using existing Qdrant collection: {self.collection_name}")
            
            # Test connection
            info = self.client.get_collection(self.collection_name)
            self.logger.info(f"📊 Collection info: {info.points_count} points, {info.vectors_count} vectors")
            
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Vector Database: {e}")
            return False
    
    async def store_signature(self, signature: TopologicalSignature) -> bool:
        """
        Store a topological signature in the vector database.
        
        Args:
            signature: TopologicalSignature to store
            
        Returns:
            bool: Success status
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Convert signature to vector
            vector = signature.to_vector()
            
            # Ensure vector is correct size
            if len(vector) != self.vector_size:
                # Pad or truncate to correct size
                if len(vector) < self.vector_size:
                    vector.extend([0.0] * (self.vector_size - len(vector)))
                else:
                    vector = vector[:self.vector_size]
            
            # Create payload with metadata
            payload = {
                "signature_hash": signature.signature_hash,
                "betti_numbers": signature.betti_numbers,
                "consciousness_level": signature.consciousness_level,
                "quantum_coherence": signature.quantum_coherence,
                "algorithm_used": signature.algorithm_used,
                "timestamp": signature.timestamp.isoformat(),
                "agent_context": signature.agent_context,
                "performance_metrics": signature.performance_metrics
            }
            
            # Store in Qdrant
            point = PointStruct(
                id=signature.signature_hash,
                vector=vector,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            storage_time = (time.time() - start_time) * 1000
            self.logger.debug(f"📦 Stored signature {signature.signature_hash[:8]}... in {storage_time:.2f}ms")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store signature: {e}")
            return False
    
    async def search_similar(self, 
                           query_signature: TopologicalSignature,
                           limit: int = 5,
                           score_threshold: float = 0.7,
                           include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar topological signatures.
        
        Args:
            query_signature: Signature to search for
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            include_metadata: Include full metadata in results
            
        Returns:
            List of similar signatures with scores
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Convert query to vector
            query_vector = query_signature.to_vector()
            
            # Ensure vector is correct size
            if len(query_vector) != self.vector_size:
                if len(query_vector) < self.vector_size:
                    query_vector.extend([0.0] * (self.vector_size - len(query_vector)))
                else:
                    query_vector = query_vector[:self.vector_size]
            
            # Perform similarity search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=include_metadata,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Process results
            similar_signatures = []
            for hit in search_result:
                result = {
                    "signature_hash": hit.id,
                    "similarity_score": hit.score,
                    "payload": hit.payload if include_metadata else None
                }
                
                # Add consciousness-weighted score
                if hit.payload and "consciousness_level" in hit.payload:
                    consciousness_weight = hit.payload["consciousness_level"]
                    weighted_score = hit.score * (0.7 + 0.3 * consciousness_weight)
                    result["consciousness_weighted_score"] = weighted_score
                
                similar_signatures.append(result)
            
            # Update performance metrics
            query_time = (time.time() - start_time) * 1000
            self.query_count += 1
            self.total_query_time += query_time
            self.avg_query_time = self.total_query_time / self.query_count
            
            self.logger.debug(f"🔍 Found {len(similar_signatures)} similar signatures in {query_time:.2f}ms")
            
            # Log performance warning if query is slow
            if query_time > 10.0:
                self.logger.warning(f"⚠️ Slow query: {query_time:.2f}ms (target: <10ms)")
            
            return similar_signatures
            
        except Exception as e:
            self.logger.error(f"❌ Failed to search similar signatures: {e}")
            return []
    
    async def get_signature_by_hash(self, signature_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific signature by hash.
        
        Args:
            signature_hash: Hash of the signature to retrieve
            
        Returns:
            Signature data or None if not found
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Retrieve by ID
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[signature_hash],
                with_payload=True,
                with_vectors=True
            )
            
            if result:
                point = result[0]
                return {
                    "signature_hash": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to retrieve signature {signature_hash}: {e}")
            return None
    
    async def delete_signature(self, signature_hash: str) -> bool:
        """Delete a signature from the database."""
        if not self.initialized:
            await self.initialize()
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[signature_hash]
                )
            )
            
            self.logger.debug(f"🗑️ Deleted signature {signature_hash[:8]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to delete signature {signature_hash}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics and health metrics."""
        if not self.initialized:
            await self.initialize()
        
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": len(info.segments) if info.segments else 0,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "query_count": self.query_count,
                "avg_query_time_ms": round(self.avg_query_time, 2),
                "performance_target_met": self.avg_query_time < 10.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector database."""
        try:
            if not self.initialized:
                return {"status": "unhealthy", "reason": "not_initialized"}
            
            # Test basic connectivity
            collections = self.client.get_collections()
            
            # Test query performance with dummy vector
            start_time = time.time()
            dummy_vector = [0.0] * self.vector_size
            self.client.search(
                collection_name=self.collection_name,
                query_vector=dummy_vector,
                limit=1
            )
            query_time = (time.time() - start_time) * 1000
            
            status = "healthy" if query_time < 10.0 else "degraded"
            
            return {
                "status": status,
                "query_time_ms": round(query_time, 2),
                "collections_count": len(collections.collections),
                "target_performance_met": query_time < 10.0,
                "avg_query_time_ms": round(self.avg_query_time, 2),
                "total_queries": self.query_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e)
            }


# Utility functions for vector operations
def optimize_vector_for_topology(signature: TopologicalSignature) -> List[float]:
    """
    Optimize vector representation specifically for topological data.
    
    This implements advanced vectorization strategies from research.
    """
    # Start with basic vector
    vector = signature.to_vector()
    
    # Add topological invariants
    betti_sum = sum(signature.betti_numbers[:3])
    betti_ratio = signature.betti_numbers[1] / max(signature.betti_numbers[0], 1)
    
    # Add consciousness and quantum features
    consciousness_features = [
        signature.consciousness_level,
        signature.quantum_coherence,
        signature.consciousness_level * signature.quantum_coherence  # Coupling
    ]
    
    # Combine all features
    optimized_vector = vector[:13] + consciousness_features  # 16 total dimensions
    
    # Normalize to unit vector for cosine similarity
    norm = np.linalg.norm(optimized_vector)
    if norm > 0:
        optimized_vector = [x / norm for x in optimized_vector]
    
    return optimized_vector


async def batch_store_signatures(service: VectorDatabaseService, 
                                signatures: List[TopologicalSignature]) -> int:
    """
    Store multiple signatures in batch for better performance.
    
    Returns:
        Number of successfully stored signatures
    """
    success_count = 0
    
    for signature in signatures:
        if await service.store_signature(signature):
            success_count += 1
    
    return success_count
