"""
🚀 FastAPI Endpoints for mem0 Search API

/analyze /search /memory endpoints with hot/semantic memory integration.
Implements unified intelligence interface for Phase 2C.

Based on partab.md: "/analyze /search /memory" endpoint specification.
"""

import asyncio
import time
import uuid
import duckdb
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from aura_intelligence.enterprise.mem0_search.schemas import (
    AnalyzeRequest, AnalyzeResponse, SearchRequest, SearchResponse,
    MemoryRequest, MemoryResponse, SignatureMatch, MemoryCluster,
    SystemHealthResponse, ComponentHealth, HealthStatus, ErrorResponse,
    TopologicalSignatureModel, EventType, MemoryTier
)
from aura_intelligence.enterprise.mem0_search.deps import (
    get_hot_memory, get_semantic_memory, get_ranking_service,
    get_current_agent, check_duckdb_health, check_redis_health, check_services_health
)
from aura_intelligence.enterprise.mem0_hot.ingest import HotEpisodicIngestor
from aura_intelligence.enterprise.mem0_semantic.sync import SemanticMemorySync
from aura_intelligence.enterprise.mem0_semantic.rank import MemoryRankingService
from aura_intelligence.enterprise.data_structures import TopologicalSignature
from aura_intelligence.utils.logger import get_logger

logger = get_logger(__name__)


def create_search_router() -> APIRouter:
    """Create the search API router with all endpoints."""
    
    router = APIRouter(prefix="/api/v1", tags=["mem0-search"])
    
    @router.post("/analyze", response_model=AnalyzeResponse)
    async def analyze_signature(
        request: AnalyzeRequest,
        background_tasks: BackgroundTasks,
        agent_id: str = Depends(get_current_agent),
        hot_memory: HotEpisodicIngestor = Depends(get_hot_memory),
        semantic_memory: SemanticMemorySync = Depends(get_semantic_memory),
        ranking_service: MemoryRankingService = Depends(get_ranking_service)
    ) -> AnalyzeResponse:
        """
        🔍 Analyze topological signature for anomalies and patterns.
        
        Performs comprehensive analysis including:
        - Anomaly detection based on signature properties
        - Similarity search against hot and semantic memory
        - Memory storage with intelligent ranking
        - Background synchronization to long-term memory
        """
        
        try:
            start_time = time.time()
            analysis_id = str(uuid.uuid4())
            
            logger.info(f"🔍 Starting analysis {analysis_id} for agent {agent_id}")
            
            # Convert Pydantic model to internal structure
            signature = TopologicalSignature(
                hash=request.signature.hash,
                betti_0=request.signature.betti_0,
                betti_1=request.signature.betti_1,
                betti_2=request.signature.betti_2,
                anomaly_score=request.signature.anomaly_score
            )
            
            # Anomaly detection
            anomaly_detected = signature.anomaly_score > 0.7
            confidence_score = min(signature.anomaly_score * 1.2, 1.0)
            
            # Find similar signatures if requested
            similar_signatures = []
            if request.include_similar:
                similar_signatures = await _find_similar_signatures(
                    signature, request.similarity_threshold, request.max_results,
                    hot_memory, semantic_memory
                )
            
            # Store in hot memory
            memory_stored = await hot_memory.ingest_signature(
                signature=signature,
                agent_id=agent_id,
                event_type="analysis",
                agent_meta={"analysis_id": analysis_id},
                full_event=request.context
            )
            
            # Background tasks
            if memory_stored:
                # Score memory for ranking
                background_tasks.add_task(
                    _score_memory_background,
                    ranking_service, signature.hash, request.context
                )
                
                # Sync to semantic memory
                background_tasks.add_task(
                    _sync_to_semantic_background,
                    semantic_memory, signature, agent_id
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            response = AnalyzeResponse(
                analysis_id=analysis_id,
                input_signature=request.signature,
                anomaly_detected=anomaly_detected,
                confidence_score=confidence_score,
                similar_signatures=similar_signatures,
                memory_stored=memory_stored,
                processing_time_ms=processing_time
            )
            
            logger.info(f"✅ Analysis {analysis_id} completed in {processing_time:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {str(e)}"
            )
    
    @router.post("/search", response_model=SearchResponse)
    async def search_memory(
        request: SearchRequest,
        agent_id: str = Depends(get_current_agent),
        hot_memory: HotEpisodicIngestor = Depends(get_hot_memory),
        semantic_memory: SemanticMemorySync = Depends(get_semantic_memory)
    ) -> SearchResponse:
        """
        🔎 Search memory for similar signatures and patterns.
        
        Supports multiple search modes:
        - Signature-based similarity search
        - Direct vector search
        - Filtered search by agent, event type, time range
        - Cross-tier search (hot + semantic memory)
        """
        
        try:
            start_time = time.time()
            search_id = str(uuid.uuid4())
            
            logger.info(f"🔎 Starting search {search_id} for agent {agent_id}")
            
            matches = []
            clusters = []
            memory_tiers_searched = []
            
            # Determine search targets
            search_hot = request.memory_tier in [MemoryTier.HOT, MemoryTier.BOTH]
            search_semantic = request.memory_tier in [MemoryTier.SEMANTIC, MemoryTier.BOTH]
            
            # Search hot memory
            if search_hot:
                hot_matches = await _search_hot_memory(request, hot_memory)
                matches.extend(hot_matches)
                memory_tiers_searched.append(MemoryTier.HOT)
            
            # Search semantic memory
            if search_semantic:
                semantic_matches, semantic_clusters = await _search_semantic_memory(
                    request, semantic_memory
                )
                matches.extend(semantic_matches)
                clusters.extend(semantic_clusters)
                memory_tiers_searched.append(MemoryTier.SEMANTIC)
            
            # Sort by similarity score
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit results
            matches = matches[:request.max_results]
            
            search_time = (time.time() - start_time) * 1000
            
            response = SearchResponse(
                search_id=search_id,
                query_processed=True,
                matches=matches,
                clusters=clusters,
                total_matches=len(matches),
                search_time_ms=search_time,
                memory_tiers_searched=memory_tiers_searched
            )
            
            logger.info(f"✅ Search {search_id} completed: {len(matches)} matches in {search_time:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}"
            )
    
    @router.post("/memory", response_model=MemoryResponse)
    async def memory_operations(
        request: MemoryRequest,
        agent_id: str = Depends(get_current_agent),
        hot_memory: HotEpisodicIngestor = Depends(get_hot_memory),
        semantic_memory: SemanticMemorySync = Depends(get_semantic_memory)
    ) -> MemoryResponse:
        """
        💾 Perform memory operations (store, retrieve, update, delete).
        
        Supports operations on both hot and semantic memory tiers
        with intelligent routing based on operation type and data.
        """
        
        try:
            start_time = time.time()
            operation_id = str(uuid.uuid4())
            
            logger.info(f"💾 Starting memory operation {request.operation} ({operation_id}) for agent {agent_id}")
            
            # Route operation based on type
            if request.operation == "store":
                result = await _store_memory(request, agent_id, hot_memory)
            elif request.operation == "retrieve":
                result = await _retrieve_memory(request, hot_memory, semantic_memory)
            elif request.operation == "update":
                result = await _update_memory(request, agent_id, hot_memory)
            elif request.operation == "delete":
                result = await _delete_memory(request, hot_memory, semantic_memory)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported operation: {request.operation}"
                )
            
            operation_time = (time.time() - start_time) * 1000
            
            response = MemoryResponse(
                operation_id=operation_id,
                operation=request.operation,
                success=result["success"],
                signature_hash=result.get("signature_hash"),
                memory_tier=request.memory_tier,
                operation_time_ms=operation_time,
                message=result["message"],
                data=result.get("data")
            )
            
            logger.info(f"✅ Memory operation {request.operation} completed in {operation_time:.2f}ms")
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Memory operation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Memory operation failed: {str(e)}"
            )
    
    @router.get("/health", response_model=SystemHealthResponse)
    async def health_check() -> SystemHealthResponse:
        """
        🏥 System health check endpoint.
        
        Checks health of all components:
        - DuckDB hot memory
        - Redis semantic memory
        - All services and background processes
        """
        
        try:
            check_start = time.time()
            
            # Check individual components
            duckdb_health = await check_duckdb_health()
            redis_health = await check_redis_health()
            services_health = await check_services_health()
            
            # Compile component health
            components = [
                ComponentHealth(
                    component="duckdb",
                    status=HealthStatus(duckdb_health["status"]),
                    last_check=datetime.now(),
                    response_time_ms=duckdb_health.get("response_time_ms"),
                    error_message=duckdb_health.get("error"),
                    metrics={"connection": duckdb_health.get("connection", "unknown")}
                ),
                ComponentHealth(
                    component="redis",
                    status=HealthStatus(redis_health["status"]),
                    last_check=datetime.now(),
                    response_time_ms=redis_health.get("response_time_ms"),
                    error_message=redis_health.get("error"),
                    metrics={"connection": redis_health.get("connection", "unknown")}
                )
            ]
            
            # Add service health
            for service_name, service_health in services_health.items():
                components.append(
                    ComponentHealth(
                        component=service_name,
                        status=HealthStatus(service_health["status"]),
                        last_check=datetime.now(),
                        error_message=service_health.get("error"),
                        metrics=service_health.get("metrics", {})
                    )
                )
            
            # Determine overall status
            unhealthy_components = [c for c in components if c.status != HealthStatus.HEALTHY]
            if not unhealthy_components:
                overall_status = HealthStatus.HEALTHY
            elif len(unhealthy_components) < len(components) / 2:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.UNHEALTHY
            
            check_time = (time.time() - check_start) * 1000
            
            response = SystemHealthResponse(
                overall_status=overall_status,
                check_timestamp=datetime.now(),
                components=components,
                system_metrics={
                    "health_check_time_ms": check_time,
                    "total_components": len(components),
                    "healthy_components": len([c for c in components if c.status == HealthStatus.HEALTHY])
                },
                uptime_seconds=time.time()  # Simplified uptime
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )
    
    return router


# Helper functions for endpoint implementations
async def _find_similar_signatures(
    signature: TopologicalSignature,
    threshold: float,
    max_results: int,
    hot_memory: HotEpisodicIngestor,
    semantic_memory: SemanticMemorySync
) -> List[SignatureMatch]:
    """Find similar signatures across memory tiers using vector similarity."""

    matches = []

    try:
        from starlette.concurrency import run_in_threadpool
        from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer

        logger.debug(f"🔍 Similarity search for {signature.hash[:8]}... (threshold: {threshold})")

        # Convert signature to vector for similarity search
        vectorizer = SignatureVectorizer()
        query_vector = vectorizer.vectorize_signature(signature)

        # Search hot memory tier first (most recent data)
        hot_matches = await run_in_threadpool(
            _search_hot_memory_sync,
            hot_memory.conn, query_vector, threshold, max_results // 2
        )
        matches.extend(hot_matches)

        # Search semantic memory tier for historical patterns
        if len(matches) < max_results:
            remaining_slots = max_results - len(matches)
            semantic_matches = await _search_semantic_memory_tier(
                semantic_memory, query_vector, threshold, remaining_slots
            )
            matches.extend(semantic_matches)

        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.debug(f"✅ Found {len(matches)} similar signatures")
        return matches[:max_results]

    except Exception as e:
        logger.error(f"❌ Similarity search failed: {e}")
        return []


async def _search_hot_memory(
    request: SearchRequest,
    hot_memory: HotEpisodicIngestor
) -> List[SignatureMatch]:
    """Search hot memory tier using DuckDB vector similarity."""

    try:
        from starlette.concurrency import run_in_threadpool
        from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer

        logger.debug("🔥 Searching hot memory tier")

        # Convert query to vector if needed
        if hasattr(request, 'signature') and request.signature:
            vectorizer = SignatureVectorizer()
            query_vector = vectorizer.vectorize_signature(request.signature)
        elif hasattr(request, 'query_vector') and request.query_vector:
            query_vector = request.query_vector
        else:
            logger.warning("No query vector or signature provided for search")
            return []

        # Execute vector similarity search in thread pool
        matches = await run_in_threadpool(
            _search_hot_memory_sync,
            hot_memory.conn,
            query_vector,
            getattr(request, 'similarity_threshold', 0.7),
            getattr(request, 'max_results', 10)
        )

        logger.debug(f"✅ Hot memory search found {len(matches)} matches")
        return matches

    except Exception as e:
        logger.error(f"❌ Hot memory search failed: {e}")
        return []


async def _search_semantic_memory(
    request: SearchRequest,
    semantic_memory: SemanticMemorySync
) -> tuple[List[SignatureMatch], List[MemoryCluster]]:
    """Search semantic memory tier using Redis vector similarity."""

    matches = []
    clusters = []

    try:
        from starlette.concurrency import run_in_threadpool
        from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer

        logger.debug("🧠 Searching semantic memory tier")

        # Convert query to vector if needed
        if hasattr(request, 'signature') and request.signature:
            vectorizer = SignatureVectorizer()
            query_vector = vectorizer.vectorize_signature(request.signature)
        elif hasattr(request, 'query_vector') and request.query_vector:
            query_vector = request.query_vector
        else:
            logger.warning("No query vector or signature provided for semantic search")
            return [], []

        # Search semantic memory (Redis-based)
        matches = await _search_semantic_memory_tier(
            semantic_memory,
            query_vector,
            getattr(request, 'similarity_threshold', 0.7),
            getattr(request, 'max_results', 10)
        )

        # Get memory clusters for context
        clusters = await _get_memory_clusters(
            semantic_memory,
            query_vector,
            getattr(request, 'cluster_threshold', 0.8)
        )

        logger.debug(f"✅ Semantic memory search found {len(matches)} matches, {len(clusters)} clusters")
        return matches, clusters

    except Exception as e:
        logger.error(f"❌ Semantic memory search failed: {e}")
        return [], []


async def _store_memory(
    request: MemoryRequest,
    agent_id: str,
    hot_memory: HotEpisodicIngestor
) -> Dict[str, Any]:
    """Store signature in memory."""

    try:
        if not request.signature:
            return {
                "success": False,
                "message": "No signature provided for store operation"
            }

        # Convert to internal structure
        signature = TopologicalSignature(
            hash=request.signature.hash,
            betti_0=request.signature.betti_0,
            betti_1=request.signature.betti_1,
            betti_2=request.signature.betti_2,
            anomaly_score=request.signature.anomaly_score
        )

        # Store in hot memory
        success = await hot_memory.ingest_signature(
            signature=signature,
            agent_id=agent_id,
            event_type=request.event_type.value,
            agent_meta={"operation": "store"},
            full_event=request.metadata
        )

        return {
            "success": success,
            "signature_hash": signature.hash,
            "message": "Signature stored successfully" if success else "Failed to store signature"
        }

    except Exception as e:
        logger.error(f"❌ Store operation failed: {e}")
        return {
            "success": False,
            "message": f"Store operation failed: {str(e)}"
        }


async def _retrieve_memory(
    request: MemoryRequest,
    hot_memory: HotEpisodicIngestor,
    semantic_memory: SemanticMemorySync
) -> Dict[str, Any]:
    """Retrieve signature from memory."""

    try:
        if not request.signature_hash:
            return {
                "success": False,
                "message": "No signature hash provided for retrieve operation"
            }

        # This would implement actual retrieval from DuckDB/Redis
        # For now, return placeholder

        return {
            "success": False,
            "signature_hash": request.signature_hash,
            "message": "Retrieve operation not yet implemented",
            "data": None
        }

    except Exception as e:
        logger.error(f"❌ Retrieve operation failed: {e}")
        return {
            "success": False,
            "message": f"Retrieve operation failed: {str(e)}"
        }


async def _update_memory(
    request: MemoryRequest,
    agent_id: str,
    hot_memory: HotEpisodicIngestor
) -> Dict[str, Any]:
    """Update signature in memory."""

    try:
        # This would implement actual update logic
        # For now, return placeholder

        return {
            "success": False,
            "message": "Update operation not yet implemented"
        }

    except Exception as e:
        logger.error(f"❌ Update operation failed: {e}")
        return {
            "success": False,
            "message": f"Update operation failed: {str(e)}"
        }


async def _delete_memory(
    request: MemoryRequest,
    hot_memory: HotEpisodicIngestor,
    semantic_memory: SemanticMemorySync
) -> Dict[str, Any]:
    """Delete signature from memory."""

    try:
        # This would implement actual deletion logic
        # For now, return placeholder

        return {
            "success": False,
            "message": "Delete operation not yet implemented"
        }

    except Exception as e:
        logger.error(f"❌ Delete operation failed: {e}")
        return {
            "success": False,
            "message": f"Delete operation failed: {str(e)}"
        }


async def _score_memory_background(
    ranking_service: MemoryRankingService,
    signature_hash: str,
    context: Dict[str, Any]
):
    """Background task to score memory for ranking."""

    try:
        await ranking_service.score_memory(signature_hash, context)
        logger.debug(f"✅ Memory scored: {signature_hash[:8]}...")
    except Exception as e:
        logger.error(f"❌ Background memory scoring failed: {e}")


async def _sync_to_semantic_background(
    semantic_memory: SemanticMemorySync,
    signature: TopologicalSignature,
    agent_id: str
):
    """Background task to sync to semantic memory."""

    try:
        # This would implement actual sync logic
        logger.debug(f"🔄 Syncing to semantic memory: {signature.hash[:8]}...")
    except Exception as e:
        logger.error(f"❌ Background semantic sync failed: {e}")


# Synchronous database query functions (for thread pool execution)

def _search_hot_memory_sync(
    conn: duckdb.DuckDBPyConnection,
    query_vector: List[float],
    threshold: float,
    max_results: int
) -> List[SignatureMatch]:
    """Synchronous DuckDB vector similarity search."""

    try:
        # Install VSS extension if not already installed
        try:
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
        except:
            pass  # Extension might already be loaded

        # Vector similarity search using DuckDB VSS
        query_sql = """
        SELECT
            signature_hash,
            betti_0, betti_1, betti_2,
            agent_id, event_type,
            timestamp,
            array_cosine_similarity(signature_vector, CAST(? AS FLOAT[128])) as similarity_score,
            agent_meta,
            full_event
        FROM recent_activity
        WHERE array_cosine_similarity(signature_vector, CAST(? AS FLOAT[128])) >= ?
        ORDER BY similarity_score DESC
        LIMIT ?
        """

        # Convert query vector to float32 for DuckDB compatibility
        query_vector_float = [float(x) for x in query_vector]
        result = conn.execute(query_sql, [query_vector_float, query_vector_float, threshold, max_results])
        rows = result.fetchall()

        matches = []
        for row in rows:
            # Create TopologicalSignatureModel from row data
            # Row format: [signature_hash, betti_0, betti_1, betti_2, agent_id, event_type, timestamp, similarity_score, agent_meta, full_event]
            signature_model = TopologicalSignatureModel(
                hash=row[0],
                betti_0=row[1],
                betti_1=row[2],
                betti_2=row[3],
                anomaly_score=0.5,  # Default value, would need to be stored separately
                timestamp=row[6] if isinstance(row[6], datetime) else datetime.now()
            )

            match = SignatureMatch(
                signature=signature_model,
                similarity_score=row[7],  # similarity_score is at index 7
                memory_tier=MemoryTier.HOT,
                last_accessed=datetime.now(),
                access_count=1
            )
            matches.append(match)

        return matches

    except Exception as e:
        logger.error(f"❌ DuckDB vector search failed: {e}")
        return []


async def _search_semantic_memory_tier(
    semantic_memory: SemanticMemorySync,
    query_vector: List[float],
    threshold: float,
    max_results: int
) -> List[SignatureMatch]:
    """Search Redis-based semantic memory tier using vector similarity."""

    try:
        from starlette.concurrency import run_in_threadpool
        import numpy as np

        logger.debug(f"🔍 Searching Redis semantic memory (threshold: {threshold}, max: {max_results})")

        # Execute Redis vector search in thread pool to avoid blocking
        matches = await run_in_threadpool(
            _search_redis_semantic_sync,
            semantic_memory,
            query_vector,
            threshold,
            max_results
        )

        logger.debug(f"✅ Redis semantic search found {len(matches)} matches")
        return matches

    except Exception as e:
        logger.error(f"❌ Redis semantic search failed: {e}")
        return []


def _search_redis_semantic_sync(
    semantic_memory: SemanticMemorySync,
    query_vector: List[float],
    threshold: float,
    max_results: int
) -> List[SignatureMatch]:
    """
    PRODUCTION-GRADE Redis vector similarity search using server-side vector index.

    This is the CORRECT, scalable implementation that performs vector search
    ON THE REDIS SERVER using the vector search index, not client-side brute force.
    """

    try:
        import numpy as np
        from datetime import datetime

        # Get Redis client from semantic memory
        redis_client = semantic_memory.redis_client
        if not redis_client:
            logger.warning("Redis client not available for semantic search")
            return []

        # Check if Redis search is available
        try:
            from redis.commands.search.query import Query
            REDIS_SEARCH_AVAILABLE = True
        except ImportError:
            REDIS_SEARCH_AVAILABLE = False

        if not REDIS_SEARCH_AVAILABLE:
            logger.warning("Redis search not available - using fallback client-side search")
            return _search_redis_semantic_fallback(redis_client, query_vector, threshold, max_results)

        try:
            index_name = "semantic_memory_index"

            # Build the K-Nearest Neighbor (KNN) query
            # This query executes entirely ON THE REDIS SERVER
            redis_query = (
                Query(f"*=>[KNN {max_results} @embedding $vector as similarity_score]")
                .sort_by("similarity_score", asc=False)  # Sort by similarity, highest first
                .return_fields(
                    "signature_hash",
                    "betti_numbers",
                    "agent_id",
                    "event_type",
                    "timestamp",
                    "similarity_score"
                )
                .dialect(2)
            )

            # Prepare query parameters
            query_params = {
                "vector": np.array(query_vector, dtype=np.float32).tobytes()
            }

            # Execute the search query on the Redis server
            search_results = redis_client.ft(index_name).search(redis_query, query_params)

            # Process the results into our Pydantic models
            matches = []
            for doc in search_results.docs:
                try:
                    # Redis returns cosine distance (0=identical), convert to similarity (1=identical)
                    similarity = 1 - float(doc.similarity_score)

                    # Filter by threshold
                    if similarity >= threshold:
                        # Parse betti numbers
                        betti_parts = doc.betti_numbers.split(',') if hasattr(doc, 'betti_numbers') else ['0', '0', '0']
                        betti_0, betti_1, betti_2 = [int(b) for b in betti_parts[:3]]

                        # Create signature model
                        signature_model = TopologicalSignatureModel(
                            hash=doc.signature_hash,
                            betti_0=betti_0,
                            betti_1=betti_1,
                            betti_2=betti_2,
                            anomaly_score=0.0,  # Not stored in semantic tier
                            timestamp=datetime.fromisoformat(doc.timestamp) if hasattr(doc, 'timestamp') else datetime.now()
                        )

                        # Create match
                        match = SignatureMatch(
                            signature=signature_model,
                            similarity_score=float(similarity),
                            memory_tier=MemoryTier.SEMANTIC,
                            last_accessed=datetime.now(),
                            access_count=1  # Default for semantic tier
                        )

                        matches.append(match)

                except (ValueError, AttributeError) as e:
                    # Skip malformed results
                    logger.warning(f"Failed to parse search result: {e}")
                    continue

            logger.debug(f"✅ Redis vector search found {len(matches)} matches")
            return matches

        except Exception as e:
            logger.warning(f"Redis vector search failed, using fallback: {e}")
            # If vector search fails, fallback to client-side search
            return _search_redis_semantic_fallback(redis_client, query_vector, threshold, max_results)

    except Exception as e:
        logger.error(f"❌ Redis semantic search sync failed: {e}")
        return []


def _search_redis_semantic_fallback(
    redis_client,
    query_vector: List[float],
    threshold: float,
    max_results: int
) -> List[SignatureMatch]:
    """
    Fallback client-side search for when Redis vector search is not available.

    WARNING: This is NOT production-grade and will not scale.
    Only use for development/testing when Redis search modules are unavailable.
    """

    try:
        import numpy as np
        import json
        from datetime import datetime

        # Search for semantic signatures using Redis hash operations
        pattern = "semantic:signature:*"
        keys = redis_client.keys(pattern)

        if not keys:
            logger.debug("No semantic signatures found in Redis")
            return []

        matches = []
        query_np = np.array(query_vector, dtype=np.float32)

        # Batch retrieve signature data
        pipe = redis_client.pipeline()
        for key in keys[:max_results * 2]:  # Get more than needed for filtering
            pipe.hgetall(key)
        results = pipe.execute()

        for key, data in zip(keys, results):
            if not data:
                continue

            try:
                # Extract signature hash from key
                if isinstance(key, bytes):
                    signature_hash = key.decode('utf-8').split(':')[-1]
                else:
                    signature_hash = key.split(':')[-1]

                # Try to get vector from binary format first (new format)
                stored_vector = None
                if b'embedding' in data:
                    vector_bytes = data[b'embedding']
                    stored_vector = np.frombuffer(vector_bytes, dtype=np.float32)
                elif b'embedding_json' in data:
                    # Fallback to JSON format
                    vector_str = data[b'embedding_json'].decode('utf-8')
                    stored_vector = np.array(json.loads(vector_str), dtype=np.float32)
                elif b'vector' in data:
                    # Legacy format
                    vector_str = data[b'vector'].decode('utf-8')
                    stored_vector = np.array([float(x) for x in vector_str.split(',')], dtype=np.float32)

                if stored_vector is None:
                    continue

                # Calculate cosine similarity
                similarity = np.dot(query_np, stored_vector) / (
                    np.linalg.norm(query_np) * np.linalg.norm(stored_vector)
                )

                if similarity >= threshold:
                    # Parse betti numbers
                    betti_str = data.get(b'betti_numbers', b'0,0,0').decode('utf-8')
                    betti_parts = betti_str.split(',')
                    betti_0, betti_1, betti_2 = [int(b) for b in betti_parts[:3]]

                    # Create signature model
                    signature_model = TopologicalSignatureModel(
                        hash=signature_hash,
                        betti_0=betti_0,
                        betti_1=betti_1,
                        betti_2=betti_2,
                        anomaly_score=0.0,
                        timestamp=datetime.now()
                    )

                    match = SignatureMatch(
                        signature=signature_model,
                        similarity_score=float(similarity),
                        memory_tier=MemoryTier.SEMANTIC,
                        last_accessed=datetime.now(),
                        access_count=1
                    )
                    matches.append(match)

            except Exception as e:
                logger.warning(f"Failed to parse semantic signature {key}: {e}")
                continue

        # Sort by similarity score (highest first) and limit results
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:max_results]

    except Exception as e:
        logger.error(f"❌ Redis semantic fallback search failed: {e}")
        return []


async def _get_memory_clusters(
    semantic_memory: SemanticMemorySync,
    query_vector: List[float],
    threshold: float
) -> List[MemoryCluster]:
    """Get memory clusters from semantic memory using Redis cluster data."""

    try:
        from starlette.concurrency import run_in_threadpool

        logger.debug(f"🧠 Getting memory clusters (threshold: {threshold})")

        # Execute cluster retrieval in thread pool
        clusters = await run_in_threadpool(
            _get_redis_clusters_sync,
            semantic_memory,
            query_vector,
            threshold
        )

        logger.debug(f"✅ Found {len(clusters)} memory clusters")
        return clusters

    except Exception as e:
        logger.error(f"❌ Memory cluster retrieval failed: {e}")
        return []


def _get_redis_clusters_sync(
    semantic_memory: SemanticMemorySync,
    query_vector: List[float],
    threshold: float
) -> List[MemoryCluster]:
    """Synchronous Redis cluster retrieval."""

    try:
        import numpy as np
        from datetime import datetime

        # Get Redis client
        redis_client = semantic_memory.redis_client
        if not redis_client:
            logger.warning("Redis client not available for cluster retrieval")
            return []

        # Search for memory clusters
        # Pattern: semantic:cluster:{cluster_id} -> {centroid, signatures, metadata}
        pattern = "semantic:cluster:*"
        keys = redis_client.keys(pattern)

        if not keys:
            logger.debug("No memory clusters found in Redis")
            return []

        clusters = []
        query_np = np.array(query_vector, dtype=np.float32)

        # Retrieve cluster data
        pipe = redis_client.pipeline()
        for key in keys:
            pipe.hgetall(key)
        results = pipe.execute()

        for key, data in zip(keys, results):
            if not data:
                continue

            try:
                # Extract cluster ID from key
                cluster_id = key.decode('utf-8').split(':')[-1]

                # Parse centroid vector
                centroid_str = data.get(b'centroid', b'').decode('utf-8')
                if not centroid_str:
                    continue

                centroid_vector = np.array([float(x) for x in centroid_str.split(',')], dtype=np.float32)

                # Calculate similarity to cluster centroid
                similarity = np.dot(query_np, centroid_vector) / (
                    np.linalg.norm(query_np) * np.linalg.norm(centroid_vector)
                )

                if similarity >= threshold:
                    # Parse cluster metadata
                    signature_count = int(data.get(b'signature_count', b'0'))
                    cluster_label = data.get(b'label', b'').decode('utf-8')
                    created_at_str = data.get(b'created_at', b'').decode('utf-8')

                    # Parse timestamp
                    try:
                        created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
                    except:
                        created_at = datetime.now()

                    # Parse signature hashes in cluster
                    signatures_str = data.get(b'signatures', b'').decode('utf-8')
                    signature_hashes = signatures_str.split(',') if signatures_str else []

                    cluster = MemoryCluster(
                        cluster_id=cluster_id,
                        centroid_vector=centroid_vector.tolist(),
                        signature_count=signature_count,
                        similarity_score=float(similarity),
                        cluster_label=cluster_label or f"cluster_{cluster_id}",
                        created_at=created_at,
                        signature_hashes=signature_hashes
                    )
                    clusters.append(cluster)

            except Exception as e:
                logger.warning(f"Failed to parse memory cluster {key}: {e}")
                continue

        # Sort by similarity score (highest first)
        clusters.sort(key=lambda x: x.similarity_score, reverse=True)
        return clusters

    except Exception as e:
        logger.error(f"❌ Redis cluster retrieval sync failed: {e}")
        return []
