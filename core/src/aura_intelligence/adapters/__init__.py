"""
Adapters Package for AURA Intelligence.

Provides adapters for external systems integration:
- Neo4j knowledge graph
- Mem0 memory management
- Redis caching
- Other external services
"""

# Neo4j adapter - graceful import
try:
    from .neo4j_adapter import Neo4jAdapter, Neo4jConfig
    _neo4j_available = True
except ImportError:
    _neo4j_available = False
    Neo4jAdapter = None
    Neo4jConfig = None

# Mem0 adapter - graceful import
try:
    from .mem0_adapter import Mem0Adapter, Mem0Config, Memory, MemoryType, SearchQuery
    _mem0_available = True
except ImportError:
    _mem0_available = False
    Mem0Adapter = None
    Mem0Config = None
    Memory = None
    MemoryType = None
    SearchQuery = None

# Redis adapter - graceful import
try:
    from .redis_adapter import RedisAdapter, RedisConfig, SerializationType
    _redis_available = True
except ImportError:
    _redis_available = False
    RedisAdapter = None
    RedisConfig = None
    SerializationType = None

__all__ = []

# Add available adapters
if _neo4j_available:
    __all__.extend(["Neo4jAdapter", "Neo4jConfig"])

if _mem0_available:
    __all__.extend(["Mem0Adapter", "Mem0Config", "Memory", "MemoryType", "SearchQuery"])
    
if _redis_available:
    __all__.extend(["RedisAdapter", "RedisConfig", "SerializationType"])