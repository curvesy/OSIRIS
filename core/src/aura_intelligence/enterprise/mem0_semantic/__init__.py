"""
🧠 mem0_semantic: MemoryDB long-term layer

AWS MemoryDB (Redis 6.4) with RedisVector for semantic long-term memory.
Batch consolidation, TTL policies, and decay scoring.

Based on partab.md blueprint for production-grade Phase 2C implementation.
"""

from .sync import SemanticMemorySync
from .rank import MemoryRankingService

__all__ = [
    "SemanticMemorySync",
    "MemoryRankingService"
]
