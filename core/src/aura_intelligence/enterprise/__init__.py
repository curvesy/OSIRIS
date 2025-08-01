"""
🧠 AURA Intelligence Enterprise Components

This module contains enterprise-grade components for the AURA Intelligence system:
- Vector Database Service (Qdrant) for similarity search
- Knowledge Graph Service (Neo4j) for causal reasoning
- Search API Service for unified intelligence interface
- Enhanced Knowledge Graph with GDS 2.19 for advanced graph ML
- Phase 2C: Hot Episodic Memory (DuckDB) with ultra-low-latency access
- Phase 2C: Semantic Long-term Memory (Redis) with intelligent ranking
- Phase 2C: Unified Search API with /analyze /search /memory endpoints

These components form the Intelligence Flywheel that transforms raw computational
power into true intelligence through the Topological Search & Memory Layer.
"""

from .vector_database import VectorDatabaseService
from .knowledge_graph import KnowledgeGraphService
from .enhanced_knowledge_graph import EnhancedKnowledgeGraphService
from .search_api import SearchAPIService

# Phase 2C: Hot Episodic Memory Components
from .mem0_hot import (
    HotEpisodicIngestor, SignatureVectorizer, ArchivalManager,
    DuckDBSettings, create_schema, RECENT_ACTIVITY_TABLE
)

# Phase 2C: Semantic Long-term Memory Components
from .mem0_semantic import (
    SemanticMemorySync, MemoryRankingService
)

# Phase 2C: Search API Components
from .mem0_search import (
    create_search_router, AnalyzeRequest, AnalyzeResponse,
    SearchRequest, SearchResponse, MemoryRequest, MemoryResponse
)
from .data_structures import (
    TopologicalSignature,
    SystemEvent,
    AgentAction,
    Outcome
)

# Enterprise feature stubs
class EnterpriseSecurityManager:
    def __init__(self): pass

class ComplianceManager:
    def __init__(self): pass

class EnterpriseMonitoring:
    def __init__(self): pass

class DeploymentManager:
    def __init__(self): pass

__all__ = [
    # Phase 2A & 2B Components
    "VectorDatabaseService",
    "KnowledgeGraphService",
    "EnhancedKnowledgeGraphService",
    "SearchAPIService",

    # Phase 2C: Hot Episodic Memory
    "HotEpisodicIngestor",
    "SignatureVectorizer",
    "ArchivalManager",
    "DuckDBSettings",
    "create_schema",
    "RECENT_ACTIVITY_TABLE",

    # Phase 2C: Semantic Long-term Memory
    "SemanticMemorySync",
    "MemoryRankingService",

    # Phase 2C: Search API
    "create_search_router",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "SearchRequest",
    "SearchResponse",
    "MemoryRequest",
    "MemoryResponse",

    # Data Structures
    "TopologicalSignature",
    "SystemEvent",
    "AgentAction",
    "Outcome",

    # Enterprise Stubs
    "EnterpriseSecurityManager",
    "ComplianceManager",
    "EnterpriseMonitoring",
    "DeploymentManager",
]
