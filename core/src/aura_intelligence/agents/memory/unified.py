"""
🧠 Unified Memory Interface - LlamaIndex Fusion over Hot→Cold→Wise

Advanced memory interface that wraps the existing Intelligence Flywheel with
LlamaIndex fusion retrieval for sophisticated multi-tier querying.

Key Features:
- LlamaIndex FusionRetriever over existing memory tiers
- Intelligent query routing and result fusion
- Context-aware retrieval with confidence scoring
- Integration with existing DuckDB, S3, and Redis systems
- OpenTelemetry instrumentation for observability

Based on the advanced patterns from phas02d.md and kakakagan.md research.
"""

import asyncio
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Import your existing memory components with graceful fallbacks
try:
    from ...enterprise.mem0_hot.ingest import HotMemoryIngest
except ImportError:
    HotMemoryIngest = None

try:
    from ...enterprise.mem0_semantic.sync import SemanticSync
except ImportError:
    SemanticSync = None

try:
    from ...enterprise.cold_storage.archive import ArchivalJob
except ImportError:
    ArchivalJob = None

try:
    from ...api.search import SearchRouter
except ImportError:
    SearchRouter = None

# LlamaIndex imports (will be installed as dependency)
try:
    from llama_index.core import VectorStoreIndex, SummaryIndex
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from llama_index.core.retrievers.fusion import FusionRetriever
    from llama_index.core.postprocessor import SimilarityPostprocessor
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    # Fallback classes for when LlamaIndex is not available
    class BaseRetriever:
        pass
    class NodeWithScore:
        pass
    class QueryBundle:
        pass


tracer = trace.get_tracer(__name__)


class MemoryTier(str, Enum):
    """Memory tiers in the Intelligence Flywheel."""
    HOT = "hot"           # DuckDB recent memory (<48 hours)
    COLD = "cold"         # S3 Parquet historical data
    SEMANTIC = "semantic" # Redis vector patterns
    AUTO = "auto"         # Intelligent tier selection


class QueryType(str, Enum):
    """Types of memory queries."""
    SIMILARITY = "similarity"     # Vector similarity search
    KEYWORD = "keyword"          # Text-based search
    TEMPORAL = "temporal"        # Time-based queries
    PATTERN = "pattern"          # Pattern matching
    HYBRID = "hybrid"            # Multi-modal search


@dataclass
class QueryResult:
    """Result from a memory query."""
    content: Any
    source: MemoryTier
    confidence: float
    metadata: Dict[str, Any]
    retrieval_time_ms: float
    
    def __post_init__(self):
        # Ensure confidence is between 0 and 1
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class FusedQueryResult:
    """Result from fusion retrieval across multiple tiers."""
    results: List[QueryResult]
    fusion_confidence: float
    total_retrieval_time_ms: float
    tiers_queried: List[MemoryTier]
    fusion_method: str
    
    def get_best_result(self) -> Optional[QueryResult]:
        """Get the highest confidence result."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.confidence)
    
    def get_results_by_tier(self, tier: MemoryTier) -> List[QueryResult]:
        """Get results from a specific tier."""
        return [r for r in self.results if r.source == tier]


class UnifiedMemory:
    """
    Unified memory interface with LlamaIndex fusion over Hot→Cold→Wise tiers.
    
    Provides intelligent query routing, result fusion, and context-aware retrieval
    over the existing operational Intelligence Flywheel.
    """
    
    def __init__(
        self,
        search_router: SearchRouter,
        hot_memory: Optional[HotMemoryIngest] = None,
        semantic_sync: Optional[SemanticSync] = None,
        archival_job: Optional[ArchivalJob] = None,
        enable_fusion: bool = True,
        fusion_top_k: int = 10,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize the unified memory interface.
        
        Args:
            search_router: Existing search router from Phase 2C
            hot_memory: Hot memory ingest service
            semantic_sync: Semantic memory sync service
            archival_job: Cold storage archival service
            enable_fusion: Whether to use LlamaIndex fusion
            fusion_top_k: Number of results to retrieve per tier
            confidence_threshold: Minimum confidence for results
        """
        self.search_router = search_router
        self.hot_memory = hot_memory
        self.semantic_sync = semantic_sync
        self.archival_job = archival_job
        
        self.enable_fusion = enable_fusion and LLAMA_INDEX_AVAILABLE
        self.fusion_top_k = fusion_top_k
        self.confidence_threshold = confidence_threshold
        
        # Initialize LlamaIndex components if available
        self.fusion_retriever = None
        if self.enable_fusion:
            self._initialize_fusion_retriever()
    
    def _initialize_fusion_retriever(self) -> None:
        """Initialize LlamaIndex fusion retriever."""
        if not LLAMA_INDEX_AVAILABLE:
            return
        
        try:
            # Create retrievers for each tier
            retrievers = []
            
            # Hot memory retriever (DuckDB)
            hot_retriever = self._create_hot_retriever()
            if hot_retriever:
                retrievers.append(hot_retriever)
            
            # Semantic memory retriever (Redis)
            semantic_retriever = self._create_semantic_retriever()
            if semantic_retriever:
                retrievers.append(semantic_retriever)
            
            # Cold storage retriever (S3)
            cold_retriever = self._create_cold_retriever()
            if cold_retriever:
                retrievers.append(cold_retriever)
            
            if retrievers:
                self.fusion_retriever = FusionRetriever(
                    retrievers=retrievers,
                    similarity_top_k=self.fusion_top_k,
                    num_queries=3,  # Generate multiple query variations
                    mode="reciprocal_rerank",  # Use reciprocal rank fusion
                    use_async=True
                )
        
        except Exception as e:
            print(f"Warning: Could not initialize fusion retriever: {e}")
            self.enable_fusion = False
    
    def _create_hot_retriever(self) -> Optional[BaseRetriever]:
        """Create retriever for hot memory tier."""
        # This would wrap your existing DuckDB search
        # For now, return None - will implement custom retriever
        return None
    
    def _create_semantic_retriever(self) -> Optional[BaseRetriever]:
        """Create retriever for semantic memory tier."""
        # This would wrap your existing Redis vector search
        # For now, return None - will implement custom retriever
        return None
    
    def _create_cold_retriever(self) -> Optional[BaseRetriever]:
        """Create retriever for cold storage tier."""
        # This would wrap your existing S3 Parquet search
        # For now, return None - will implement custom retriever
        return None
    
    @tracer.start_as_current_span("unified_memory_query")
    async def query(
        self,
        query: str,
        tier: MemoryTier = MemoryTier.AUTO,
        query_type: QueryType = QueryType.HYBRID,
        limit: int = 10,
        threshold: float = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FusedQueryResult:
        """
        Query the unified memory system.
        
        Args:
            query: Query string or vector
            tier: Which memory tier(s) to query
            query_type: Type of query to perform
            limit: Maximum number of results
            threshold: Confidence threshold (uses default if None)
            context: Additional context for the query
            
        Returns:
            Fused query results from multiple tiers
        """
        span = trace.get_current_span()
        span.set_attributes({
            "query": query[:100],  # Truncate for logging
            "tier": tier.value,
            "query_type": query_type.value,
            "limit": limit
        })
        
        start_time = time.time()
        threshold = threshold or self.confidence_threshold
        
        try:
            if self.enable_fusion and tier == MemoryTier.AUTO:
                # Use LlamaIndex fusion retrieval
                results = await self._fusion_query(query, limit, threshold, context)
            else:
                # Use direct tier querying
                results = await self._direct_query(query, tier, query_type, limit, threshold, context)
            
            total_time = (time.time() - start_time) * 1000
            
            span.set_attributes({
                "results_count": len(results.results),
                "fusion_confidence": results.fusion_confidence,
                "retrieval_time_ms": total_time
            })
            span.set_status(Status(StatusCode.OK))
            
            return results
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
    
    async def _fusion_query(
        self,
        query: str,
        limit: int,
        threshold: float,
        context: Optional[Dict[str, Any]]
    ) -> FusedQueryResult:
        """Perform fusion query across multiple tiers."""
        
        if not self.fusion_retriever:
            # Fallback to direct querying
            return await self._direct_query(query, MemoryTier.AUTO, QueryType.HYBRID, limit, threshold, context)
        
        start_time = time.time()
        
        try:
            # Use LlamaIndex fusion retrieval
            query_bundle = QueryBundle(query_str=query)
            nodes = await self.fusion_retriever.aretrieve(query_bundle)
            
            # Convert LlamaIndex results to our format
            results = []
            for node in nodes[:limit]:
                if hasattr(node, 'score') and node.score >= threshold:
                    result = QueryResult(
                        content=node.node.text if hasattr(node.node, 'text') else str(node.node),
                        source=self._determine_source_tier(node),
                        confidence=node.score,
                        metadata=node.node.metadata if hasattr(node.node, 'metadata') else {},
                        retrieval_time_ms=0  # Individual timing not available
                    )
                    results.append(result)
            
            total_time = (time.time() - start_time) * 1000
            
            # Calculate fusion confidence
            fusion_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
            
            return FusedQueryResult(
                results=results,
                fusion_confidence=fusion_confidence,
                total_retrieval_time_ms=total_time,
                tiers_queried=[MemoryTier.HOT, MemoryTier.SEMANTIC, MemoryTier.COLD],
                fusion_method="llama_index_reciprocal_rerank"
            )
            
        except Exception as e:
            # Fallback to direct querying
            return await self._direct_query(query, MemoryTier.AUTO, QueryType.HYBRID, limit, threshold, context)
    
    async def _direct_query(
        self,
        query: str,
        tier: MemoryTier,
        query_type: QueryType,
        limit: int,
        threshold: float,
        context: Optional[Dict[str, Any]]
    ) -> FusedQueryResult:
        """Perform direct query using existing search router."""
        
        start_time = time.time()
        results = []
        tiers_queried = []
        
        if tier == MemoryTier.AUTO:
            # Query all tiers and fuse results
            tier_queries = [
                (MemoryTier.HOT, "hot"),
                (MemoryTier.SEMANTIC, "semantic"),
                (MemoryTier.COLD, "cold")
            ]
        else:
            # Query specific tier
            tier_mapping = {
                MemoryTier.HOT: "hot",
                MemoryTier.SEMANTIC: "semantic", 
                MemoryTier.COLD: "cold"
            }
            tier_queries = [(tier, tier_mapping[tier])]
        
        # Execute queries
        for memory_tier, search_tier in tier_queries:
            try:
                tier_start = time.time()
                
                # Use existing search router
                search_result = await self.search_router.search(
                    query=query,
                    tier=search_tier,
                    limit=limit,
                    threshold=threshold
                )
                
                tier_time = (time.time() - tier_start) * 1000
                tiers_queried.append(memory_tier)
                
                # Convert search results to QueryResult format
                if hasattr(search_result, 'results') and search_result.results:
                    for item in search_result.results:
                        confidence = getattr(item, 'confidence', 0.5)
                        if confidence >= threshold:
                            result = QueryResult(
                                content=item,
                                source=memory_tier,
                                confidence=confidence,
                                metadata=getattr(item, 'metadata', {}),
                                retrieval_time_ms=tier_time
                            )
                            results.append(result)
                            
            except Exception as e:
                # Log error but continue with other tiers
                print(f"Error querying {memory_tier.value} tier: {e}")
        
        # Sort results by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)
        results = results[:limit]
        
        total_time = (time.time() - start_time) * 1000
        
        # Calculate fusion confidence
        fusion_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
        
        return FusedQueryResult(
            results=results,
            fusion_confidence=fusion_confidence,
            total_retrieval_time_ms=total_time,
            tiers_queried=tiers_queried,
            fusion_method="direct_confidence_ranking"
        )
    
    def _determine_source_tier(self, node) -> MemoryTier:
        """Determine which tier a LlamaIndex node came from."""
        # This would analyze the node metadata to determine source
        # For now, default to semantic
        return MemoryTier.SEMANTIC
    
    @tracer.start_as_current_span("unified_memory_store")
    async def store(
        self,
        content: Any,
        tier: MemoryTier = MemoryTier.HOT,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store content in the specified memory tier.
        
        Args:
            content: Content to store
            tier: Which tier to store in
            metadata: Additional metadata
            
        Returns:
            Storage ID or reference
        """
        span = trace.get_current_span()
        span.set_attributes({
            "tier": tier.value,
            "content_type": type(content).__name__
        })
        
        try:
            if tier == MemoryTier.HOT and self.hot_memory:
                # Store in hot memory (DuckDB)
                result = await self.hot_memory.add_signature(
                    timestamp=datetime.now(),
                    signature=content,
                    metadata=metadata or {}
                )
                span.set_status(Status(StatusCode.OK))
                return result
                
            elif tier == MemoryTier.SEMANTIC and self.semantic_sync:
                # Store in semantic memory (Redis)
                result = await self.semantic_sync.store_pattern(content, metadata)
                span.set_status(Status(StatusCode.OK))
                return result
                
            elif tier == MemoryTier.COLD and self.archival_job:
                # Store in cold storage (S3)
                result = await self.archival_job.store_data(content, metadata)
                span.set_status(Status(StatusCode.OK))
                return result
                
            else:
                raise ValueError(f"Unsupported storage tier: {tier}")
                
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage across tiers."""
        stats = {
            "hot_tier": {"available": False},
            "semantic_tier": {"available": False},
            "cold_tier": {"available": False},
            "fusion_enabled": self.enable_fusion
        }
        
        # Get stats from each tier if available
        try:
            if self.search_router:
                # Use existing health check or stats endpoints
                health = await self.search_router.get_health()
                stats.update(health)
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup any resources if needed
        pass
