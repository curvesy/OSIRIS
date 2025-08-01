"""
🔥 mem0_hot: DuckDB-based episodic layer

High-throughput async writer, schema management, and vectorization
for ultra-low-latency access to recent topological signatures.

Based on partab.md blueprint for production-grade Phase 2C implementation.
"""

from .ingest import HotEpisodicIngestor
from .schema import create_schema, RECENT_ACTIVITY_TABLE
from .archive import ArchivalManager
from .vectorize import SignatureVectorizer
from .settings import DuckDBSettings, DEV_SETTINGS, PRODUCTION_SETTINGS, DEFAULT_SETTINGS
from .scheduler import ArchivalScheduler, SchedulerConfig

__all__ = [
    "HotEpisodicIngestor",
    "create_schema",
    "RECENT_ACTIVITY_TABLE",
    "ArchivalManager",
    "SignatureVectorizer",
    "DuckDBSettings",
    "DEV_SETTINGS",
    "PRODUCTION_SETTINGS",
    "DEFAULT_SETTINGS",
    "ArchivalScheduler",
    "SchedulerConfig"
]
