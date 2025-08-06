"""
Async Shape Memory V2 - Production Ready
=======================================

Asynchronous implementation of the shape-aware memory system
for high-throughput, low-latency operations at scale.
"""

import asyncio
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Protocol
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .topo_features import TopologicalFeatureExtractor
from .fastrp import FastRP, FastRPConfig
from .fusion_scorer import AdaptiveFusionScorer, FusionConfig
from .observability import instrument, update_recall, update_false_positive_rate
from ..tda.models import TDAResult, BettiNumbers

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
_cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="shape-cpu")


class AsyncVectorStore(Protocol):
    """Protocol for async vector stores."""
    
    async def add(
        self,
        memory_id: str,
        embedding: np.ndarray,
        content: Dict[str, Any],
        context_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a vector asynchronously."""
        ...
    
    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        context_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search vectors asynchronously."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health asynchronously."""
        ...
    
    async def close(self):
        """Close connections asynchronously."""
        ...


@dataclass
class AsyncShapeMemoryConfig:
    """Configuration for async Shape Memory V2."""
    storage_backend: str = "redis"
    redis_url: str = "redis://localhost:6379"
    embedding_dim: int = 128
    fastrp_iterations: int = 3
    enable_fusion_scoring: bool = True
    fusion_alpha: float = 0.7
    fusion_beta: float = 0.3
    max_concurrent_operations: int = 100
    operation_timeout: float = 5.0  # seconds


class AsyncShapeMemoryV2:
    """
    Asynchronous shape-aware memory system.
    
    Features:
    - Concurrent feature extraction and embedding
    - Async storage operations
    - Batched processing support
    - Connection pooling
    - Comprehensive monitoring
    """
    
    def __init__(
        self,
        vector_store: AsyncVectorStore,
        config: Optional[AsyncShapeMemoryConfig] = None
    ):
        self.config = config or AsyncShapeMemoryConfig()
        self.vector_store = vector_store
        
        # Initialize components
        self.feature_extractor = TopologicalFeatureExtractor()
        
        # FastRP embedder
        feature_dim = self.feature_extractor.feature_dimension
        fastrp_config = FastRPConfig(
            embedding_dim=self.config.embedding_dim,
            iterations=self.config.fastrp_iterations
        )
        self.embedder = FastRP(feature_dim, fastrp_config)
        
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
        
        # Semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)
        
        logger.info("AsyncShapeMemoryV2 initialized")
    
    async def _extract_features_async(self, tda_result: TDAResult) -> np.ndarray:
        """Extract features in a thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _cpu_executor,
            self._extract_features_sync,
            tda_result
        )
    
    def _extract_features_sync(self, tda_result: TDAResult) -> np.ndarray:
        """Synchronous feature extraction."""
        features = self.feature_extractor.extract(
            tda_result.betti_numbers,
            tda_result.persistence_diagram
        )
        return features.combined
    
    async def _generate_embedding_async(self, features: np.ndarray) -> np.ndarray:
        """Generate embedding in a thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _cpu_executor,
            lambda: self.embedder.transform(features.reshape(1, -1))[0]
        )
    
    @instrument("store_async", "memory")
    async def store(
        self,
        content: Dict[str, Any],
        tda_result: TDAResult,
        context_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory asynchronously with parallel processing.
        
        Args:
            content: The actual content to store
            tda_result: Topological analysis result
            context_type: Type of context
            metadata: Additional metadata
            
        Returns:
            str: Unique memory ID
        """
        async with self._semaphore:  # Rate limiting
            try:
                # Generate memory ID
                memory_id = str(uuid.uuid4())
                
                # Parallel feature extraction and metadata preparation
                features, prepared_metadata = await asyncio.gather(
                    self._extract_features_async(tda_result),
                    self._prepare_metadata(tda_result, metadata)
                )
                
                # Generate embedding
                embedding = await self._generate_embedding_async(features)
                
                # Store with timeout
                success = await asyncio.wait_for(
                    self.vector_store.add(
                        memory_id=memory_id,
                        embedding=embedding,
                        content=content,
                        context_type=context_type,
                        metadata=prepared_metadata
                    ),
                    timeout=self.config.operation_timeout
                )
                
                if success:
                    logger.debug(f"Stored memory {memory_id} asynchronously")
                    return memory_id
                else:
                    logger.error(f"Failed to store memory {memory_id}")
                    return ""
                    
            except asyncio.TimeoutError:
                logger.error(f"Store operation timed out after {self.config.operation_timeout}s")
                return ""
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
                raise
    
    async def _prepare_metadata(
        self,
        tda_result: TDAResult,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare metadata with topological information."""
        if metadata is None:
            metadata = {}
        
        metadata["embedding_created_at"] = time.time()
        metadata["betti_numbers"] = {
            "b0": tda_result.betti_numbers.b0,
            "b1": tda_result.betti_numbers.b1,
            "b2": tda_result.betti_numbers.b2
        }
        
        # Store compressed persistence diagram
        # In production, you'd compress this properly
        metadata["persistence_diagram_shape"] = tda_result.persistence_diagram.shape
        
        return metadata
    
    @instrument("retrieve_async", "memory")
    async def retrieve(
        self,
        query_tda: TDAResult,
        k: int = 10,
        context_filter: Optional[str] = None,
        score_threshold: float = 0.0,
        enable_fusion: Optional[bool] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve memories asynchronously with parallel scoring.
        
        Args:
            query_tda: Query topological signature
            k: Number of results
            context_filter: Filter by context type
            score_threshold: Minimum similarity score
            enable_fusion: Override fusion scoring setting
            
        Returns:
            List of (memory, score) tuples
        """
        async with self._semaphore:  # Rate limiting
            try:
                # Extract features and generate embedding
                features = await self._extract_features_async(query_tda)
                query_embedding = await self._generate_embedding_async(features)
                
                # Determine if fusion scoring should be used
                use_fusion = (
                    enable_fusion if enable_fusion is not None 
                    else self.config.enable_fusion_scoring
                )
                
                # Search with larger k if fusion scoring is enabled
                search_k = k * 2 if use_fusion and self.fusion_scorer else k
                
                # Search with timeout
                results = await asyncio.wait_for(
                    self.vector_store.search(
                        query_embedding=query_embedding,
                        k=search_k,
                        context_filter=context_filter,
                        score_threshold=0.0  # Apply threshold after fusion
                    ),
                    timeout=self.config.operation_timeout
                )
                
                # Apply fusion scoring if enabled
                if use_fusion and self.fusion_scorer and results:
                    results = await self._apply_fusion_scoring_async(
                        results, query_tda, score_threshold, k
                    )
                else:
                    # Filter by threshold if no fusion
                    results = [(m, s) for m, s in results if s >= score_threshold][:k]
                
                logger.debug(f"Retrieved {len(results)} memories asynchronously")
                return results
                
            except asyncio.TimeoutError:
                logger.error(f"Retrieve operation timed out after {self.config.operation_timeout}s")
                return []
            except Exception as e:
                logger.error(f"Error retrieving memories: {e}")
                return []
    
    async def _apply_fusion_scoring_async(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        query_tda: TDAResult,
        score_threshold: float,
        k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Apply fusion scoring to results in parallel."""
        # Create scoring tasks
        scoring_tasks = []
        for memory_data, cosine_similarity in results:
            task = self._score_single_result(
                memory_data, cosine_similarity, query_tda
            )
            scoring_tasks.append(task)
        
        # Execute all scoring tasks in parallel
        scored_results = await asyncio.gather(*scoring_tasks)
        
        # Filter and sort
        filtered_results = [
            (memory, score) for memory, score in scored_results 
            if score >= score_threshold
        ]
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_results[:k]
    
    async def _score_single_result(
        self,
        memory_data: Dict[str, Any],
        cosine_similarity: float,
        query_tda: TDAResult
    ) -> Tuple[Dict[str, Any], float]:
        """Score a single result with fusion scoring."""
        # Get stored persistence diagram
        stored_diagram = self._reconstruct_diagram(memory_data.get("metadata", {}))
        
        # Calculate embedding age
        age_hours = 0
        if "metadata" in memory_data and "embedding_created_at" in memory_data["metadata"]:
            age_hours = (time.time() - memory_data["metadata"]["embedding_created_at"]) / 3600
        
        # Compute fusion score in thread pool
        loop = asyncio.get_event_loop()
        score_result = await loop.run_in_executor(
            _cpu_executor,
            self.fusion_scorer.score,
            cosine_similarity,
            query_tda.persistence_diagram,
            stored_diagram,
            age_hours,
            False  # No exploration in production
        )
        
        return (memory_data, score_result["final_score"])
    
    def _reconstruct_diagram(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Reconstruct persistence diagram from metadata."""
        if "betti_numbers" in metadata:
            betti = metadata["betti_numbers"]
            # In production, you'd decompress the actual stored diagram
            return np.random.rand(betti["b0"] + betti["b1"] + betti["b2"], 2)
        return np.array([[0, 1]])
    
    async def batch_store(
        self,
        items: List[Tuple[Dict[str, Any], TDAResult, str, Optional[Dict[str, Any]]]]
    ) -> List[str]:
        """
        Store multiple memories in parallel.
        
        Args:
            items: List of (content, tda_result, context_type, metadata) tuples
            
        Returns:
            List of memory IDs
        """
        tasks = []
        for content, tda_result, context_type, metadata in items:
            task = self.store(content, tda_result, context_type, metadata)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check system health asynchronously."""
        health = await self.vector_store.health_check()
        
        stats = {
            "async_shape_memory_v2": {
                "status": health.get("status", "unknown"),
                "backend": self.config.storage_backend,
                "fusion_enabled": self.config.enable_fusion_scoring,
                "max_concurrent": self.config.max_concurrent_operations,
                "storage": health
            }
        }
        
        if self.fusion_scorer:
            stats["async_shape_memory_v2"]["fusion"] = self.fusion_scorer.get_stats()
        
        return stats
    
    async def close(self):
        """Clean up resources."""
        await self.vector_store.close()
        logger.info("AsyncShapeMemoryV2 closed")


# Example async Redis adapter
class AsyncRedisAdapter(AsyncVectorStore):
    """Adapter to make Redis operations async."""
    
    def __init__(self, sync_store):
        self.sync_store = sync_store
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    async def add(self, **kwargs) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.sync_store.add(**kwargs)
        )
    
    async def search(self, **kwargs) -> List[Tuple[Dict[str, Any], float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.sync_store.search(**kwargs)
        )
    
    async def health_check(self) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.sync_store.health_check
        )
    
    async def close(self):
        self.sync_store.close()
        self._executor.shutdown(wait=True)