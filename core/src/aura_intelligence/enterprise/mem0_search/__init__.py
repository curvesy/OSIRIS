"""
🔍 mem0_search: FastAPI service layer

Enhanced search endpoints with hot/semantic memory integration.
Unified /analyze /search /memory endpoints for Phase 2C.

Based on partab.md blueprint for production-grade Phase 2C implementation.
"""

from .endpoints import create_search_router
from .deps import get_hot_memory, get_semantic_memory, get_ranking_service
from .schemas import (
    AnalyzeRequest, AnalyzeResponse,
    SearchRequest, SearchResponse,
    MemoryRequest, MemoryResponse,
    SignatureMatch, MemoryCluster
)

__all__ = [
    "create_search_router",
    "get_hot_memory",
    "get_semantic_memory", 
    "get_ranking_service",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "SearchRequest",
    "SearchResponse",
    "MemoryRequest",
    "MemoryResponse",
    "SignatureMatch",
    "MemoryCluster"
]
