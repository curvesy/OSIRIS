"""
Shape-Aware Memory V2: Clean Architecture
========================================

A modular implementation of shape-aware memory that combines:
- Topological feature extraction
- FastRP embeddings
- k-NN index for fast retrieval
- Simple storage abstraction
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

import numpy as np

from ..tda.models import TDAResult, BettiNumbers
from .topo_features import TopologicalFeatureExtractor
from .fastrp import FastRP, FastRPConfig
from .knn_index import KNNIndex, KNNConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory with its metadata."""
    id: str
    content: Dict[str, Any]
    embedding: np.ndarray
    betti_numbers: BettiNumbers
    created_at: datetime = field(default_factory=datetime.now)
    context_type: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "betti_numbers": {
                "b0": self.betti_numbers.b0,
                "b1": self.betti_numbers.b1,
                "b2": self.betti_numbers.b2
            },
            "created_at": self.created_at.isoformat(),
            "context_type": self.context_type
        }


@dataclass
class ShapeMemoryConfig:
    """Configuration for Shape Memory V2."""
    embedding_dim: int = 128
    fastrp_iterations: int = 3
    knn_metric: str = "cosine"
    knn_backend: str = "sklearn"


class ShapeMemoryV2:
    """
    Shape-aware memory system with fast topological retrieval.
    
    This implementation focuses on clarity and modularity.
    Each component has a single responsibility and is easily
    testable in isolation.
    """
    
    def __init__(self, config: Optional[ShapeMemoryConfig] = None):
        self.config = config or ShapeMemoryConfig()
        
        # Initialize components
        self.feature_extractor = TopologicalFeatureExtractor()
        
        # FastRP for embeddings
        feature_dim = self.feature_extractor.feature_dimension
        fastrp_config = FastRPConfig(
            embedding_dim=self.config.embedding_dim,
            iterations=self.config.fastrp_iterations
        )
        self.embedder = FastRP(feature_dim, fastrp_config)
        
        # k-NN index
        knn_config = KNNConfig(
            metric=self.config.knn_metric,
            backend=self.config.knn_backend
        )
        self.index = KNNIndex(self.config.embedding_dim, knn_config)
        
        # Simple in-memory storage
        self.storage: Dict[str, MemoryEntry] = {}
        
        logger.info("ShapeMemoryV2 initialized")
    
    def store(
        self,
        content: Dict[str, Any],
        tda_result: TDAResult,
        context_type: str = "general"
    ) -> str:
        """
        Store a memory with its topological signature.
        
        Args:
            content: The memory content
            tda_result: Topological analysis result
            context_type: Type of memory (general, anomaly, etc.)
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        start_time = time.time()
        
        # Generate unique ID
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        
        # Extract features
        features = self.feature_extractor.extract(
            tda_result.betti_numbers,
            tda_result.persistence_diagram
        )
        
        # Generate embedding
        embedding = self.embedder.transform(features.combined.reshape(1, -1))[0]
        
        # Create memory entry
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding,
            betti_numbers=tda_result.betti_numbers,
            context_type=context_type
        )
        
        # Store in memory
        self.storage[memory_id] = entry
        
        # Add to index
        self.index.add(embedding.reshape(1, -1), [memory_id])
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Stored memory {memory_id} in {elapsed:.2f}ms")
        
        return memory_id
    
    def retrieve(
        self,
        query_tda: TDAResult,
        k: int = 10,
        context_filter: Optional[str] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve memories similar to the query.
        
        Args:
            query_tda: Topological signature to search for
            k: Number of results to return
            context_filter: Optional filter by context type
            
        Returns:
            List of (memory, similarity_score) tuples
        """
        start_time = time.time()
        
        # Extract features and generate embedding
        features = self.feature_extractor.extract(
            query_tda.betti_numbers,
            query_tda.persistence_diagram
        )
        query_embedding = self.embedder.transform(features.combined.reshape(1, -1))[0]
        
        # Search index
        results = self.index.search(query_embedding, k=k * 2)  # Get extra for filtering
        
        # Retrieve memories and filter
        memories = []
        for memory_id, distance in results:
            entry = self.storage.get(memory_id)
            if entry:
                # Apply context filter if specified
                if context_filter and entry.context_type != context_filter:
                    continue
                
                # Convert distance to similarity (assuming cosine)
                similarity = 1.0 - distance
                memories.append((entry, similarity))
                
                if len(memories) >= k:
                    break
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Retrieved {len(memories)} memories in {elapsed:.2f}ms")
        
        return memories
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        context_counts = {}
        for entry in self.storage.values():
            context_counts[entry.context_type] = context_counts.get(entry.context_type, 0) + 1
        
        return {
            "total_memories": len(self.storage),
            "index_size": len(self.index),
            "context_distribution": context_counts,
            "embedding_dim": self.config.embedding_dim
        }