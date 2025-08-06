"""
Shape Memory V2 - Production Ready
=================================

The main orchestrator that combines all components with
proper abstraction, monitoring, and error handling.
"""

import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .topo_features import TopologicalFeatureExtractor
from .fastrp import FastRP, FastRPConfig
from .storage_interface import MemoryStorage, InMemoryStorage
from .redis_store import RedisVectorStore, RedisConfig
from .fusion_scorer import AdaptiveFusionScorer, FusionConfig
from .observability import instrument, update_recall, update_false_positive_rate
from ..tda.models import TDAResult, BettiNumbers

logger = logging.getLogger(__name__)


@dataclass
class ShapeMemoryConfig:
    """Production configuration for Shape Memory V2."""
    storage_backend: str = "redis"  # "redis" or "memory"
    redis_url: str = "redis://localhost:6379"
    embedding_dim: int = 128
    fastrp_iterations: int = 3
    feature_flag_enabled: bool = True
    enable_fusion_scoring: bool = True
    fusion_alpha: float = 0.7
    fusion_beta: float = 0.3


class ShapeMemoryV2:
    """
    Production-ready shape-aware memory system.
    
    Features:
    - Pluggable storage backends
    - Comprehensive monitoring
    - Feature flag support
    - Shadow mode capability
    """
    
    def __init__(self, config: Optional[ShapeMemoryConfig] = None):
        self.config = config or ShapeMemoryConfig()
        
        # Initialize components
        self.feature_extractor = TopologicalFeatureExtractor()
        
        # FastRP embedder
        feature_dim = self.feature_extractor.feature_dimension
        fastrp_config = FastRPConfig(
            embedding_dim=self.config.embedding_dim,
            iterations=self.config.fastrp_iterations
        )
        self.embedder = FastRP(feature_dim, fastrp_config)
        
        # Storage backend
        self.storage = self._create_storage()
        
        # Initialize fusion scorer if enabled
        if self.config.enable_fusion_scoring:
            self.fusion_scorer = AdaptiveFusionScorer(
                FusionConfig(
                    base_alpha=self.config.fusion_alpha,
                    base_beta=self.config.fusion_beta
                )
            )
        else:
            self.fusion_scorer = None
        
        # Metrics
        self._store_counter = 0
        self._retrieve_counter = 0
        
        logger.info(f"ShapeMemoryV2 initialized with {self.config.storage_backend} backend")
    
    def _create_storage(self) -> MemoryStorage:
        """Create appropriate storage backend."""
        if self.config.storage_backend == "redis":
            redis_config = RedisConfig(url=self.config.redis_url)
            return RedisVectorStore(redis_config)
        else:
            return InMemoryStorage()
    
    @instrument("store", "memory")
    def store(
        self,
        content: Dict[str, Any],
        tda_result: TDAResult,
        context_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a new memory with its topological signature.
        
        Args:
            content: The actual content to store
            tda_result: Topological analysis result
            context_type: Type of context (e.g., "general", "danger")
            metadata: Additional metadata
            
        Returns:
            str: Unique memory ID
        """
        try:
            # Check feature flag
            if not self.config.feature_flag_enabled:
                logger.warning("Shape memory is disabled by feature flag")
                return ""
            
            # Generate memory ID
            memory_id = str(uuid.uuid4())
            
            # Extract features
            features = self.feature_extractor.extract(
                tda_result.betti_numbers,
                tda_result.persistence_diagram
            )
            
            # Generate embedding
            embedding = self.embedder.transform(features.combined.reshape(1, -1))[0]
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata["embedding_created_at"] = time.time()
            metadata["betti_numbers"] = {
                "b0": tda_result.betti_numbers.b0,
                "b1": tda_result.betti_numbers.b1,
                "b2": tda_result.betti_numbers.b2
            }
            
            # CRITICAL: Store the actual persistence diagram for deterministic reconstruction
            if tda_result.persistence_diagram is not None:
                # Convert to list for JSON serialization
                metadata["persistence_diagram"] = tda_result.persistence_diagram.tolist()
            
            # Store in backend
            success = self.storage.add(
                memory_id=memory_id,
                embedding=embedding,
                content=content,
                context_type=context_type,
                metadata=metadata
            )
            
            if success:
                self._store_counter += 1
                logger.debug(f"Stored memory {memory_id} with context {context_type}")
                return memory_id
            else:
                logger.error(f"Failed to store memory {memory_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    @instrument("retrieve", "memory")
    def retrieve(
        self,
        query_tda: TDAResult,
        k: int = 10,
        context_filter: Optional[str] = None,
        score_threshold: float = 0.0,
        enable_fusion: Optional[bool] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve memories similar to the query topology.
        
        Args:
            query_tda: Query topological signature
            k: Number of results to return
            context_filter: Filter by context type
            score_threshold: Minimum similarity score
            enable_fusion: Override fusion scoring setting
            
        Returns:
            List of (memory, score) tuples
        """
        try:
            # Check feature flag
            if not self.config.feature_flag_enabled:
                logger.warning("Shape memory is disabled by feature flag")
                return []
            
            # Extract features and generate embedding
            features = self.feature_extractor.extract(
                query_tda.betti_numbers,
                query_tda.persistence_diagram
            )
            query_embedding = self.embedder.transform(features.combined.reshape(1, -1))[0]
            
            # Determine if fusion scoring should be used
            use_fusion = enable_fusion if enable_fusion is not None else self.config.enable_fusion_scoring
            
            # Search with larger k if fusion scoring is enabled
            search_k = k * 2 if use_fusion and self.fusion_scorer else k
            
            # Search in storage
            results = self.storage.search(
                query_embedding=query_embedding,
                k=search_k,
                context_filter=context_filter,
                score_threshold=0.0  # Apply threshold after fusion
            )
            
            # Apply fusion scoring if enabled
            if use_fusion and self.fusion_scorer and results:
                scored_results = []
                
                for memory_data, cosine_similarity in results:
                    # Get stored persistence diagram
                    stored_diagram = self._reconstruct_diagram(memory_data.get("metadata", {}))
                    
                    # Calculate embedding age
                    age_hours = 0
                    if "metadata" in memory_data and "embedding_created_at" in memory_data["metadata"]:
                        age_hours = (time.time() - memory_data["metadata"]["embedding_created_at"]) / 3600
                    
                    # Compute fusion score
                    score_result = self.fusion_scorer.score(
                        fastrp_similarity=cosine_similarity,
                        persistence_diagram1=query_tda.persistence_diagram,
                        persistence_diagram2=stored_diagram,
                        embedding_age_hours=age_hours,
                        enable_exploration=False  # No exploration in production
                    )
                    
                    final_score = score_result["final_score"]
                    
                    # Apply threshold
                    if final_score >= score_threshold:
                        scored_results.append((memory_data, final_score))
                
                # Sort by score and return top k
                scored_results.sort(key=lambda x: x[1], reverse=True)
                results = scored_results[:k]
            else:
                # Filter by threshold if no fusion
                results = [(m, s) for m, s in results if s >= score_threshold][:k]
            
            self._retrieve_counter += 1
            logger.debug(f"Retrieved {len(results)} memories")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            # CRITICAL FIX: Don't hide failures - let them propagate
            # This allows proper error handling and monitoring
            raise RuntimeError(f"Memory retrieval failed: {e}") from e
    
    def _reconstruct_diagram(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Reconstruct persistence diagram from metadata."""
        # CRITICAL FIX: Make reconstruction deterministic
        if "betti_numbers" in metadata:
            betti = metadata["betti_numbers"]
            
            # Check if we have the actual stored diagram
            if "persistence_diagram" in metadata:
                # Reconstruct from stored data
                diagram_data = metadata["persistence_diagram"]
                if isinstance(diagram_data, list):
                    return np.array(diagram_data)
                elif isinstance(diagram_data, dict) and "data" in diagram_data:
                    return np.array(diagram_data["data"])
            
            # FALLBACK: Create deterministic synthetic diagram
            # Use Betti numbers as seed for reproducibility
            total_features = betti["b0"] + betti["b1"] + betti["b2"]
            if total_features == 0:
                return np.array([[0, 1]])
            
            # Generate deterministic birth-death pairs based on topology
            births = np.linspace(0, 0.5, total_features)
            # Higher Betti numbers indicate more complex topology -> longer lifetimes
            complexity_factor = (betti["b1"] + 2 * betti["b2"]) / (total_features + 1)
            lifetimes = np.linspace(0.1, 0.5 + complexity_factor, total_features)
            deaths = births + lifetimes
            
            return np.column_stack([births, deaths])
        
        return np.array([[0, 1]])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "stores": self._store_counter,
            "retrieves": self._retrieve_counter,
            "backend": self.config.storage_backend,
            "fusion_enabled": self.config.enable_fusion_scoring
        }
        
        # Add storage-specific stats
        if hasattr(self.storage, 'health_check'):
            health = self.storage.health_check()
            stats.update(health)
        
        # Add fusion scorer stats
        if self.fusion_scorer:
            fusion_stats = self.fusion_scorer.get_stats()
            stats["fusion"] = fusion_stats
        
        return stats
    
    def close(self):
        """Clean up resources."""
        if hasattr(self.storage, 'close'):
            self.storage.close()
        logger.info("ShapeMemoryV2 closed")


# Utility functions for metrics updates
def update_shape_memory_recall(value: float):
    """Update the recall@5 metric."""
    update_recall(value)

def update_shape_memory_false_positive_rate(value: float):
    """Update the false positive rate metric."""
    update_false_positive_rate(value)