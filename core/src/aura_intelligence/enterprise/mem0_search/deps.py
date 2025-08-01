"""
🔌 Dependency Injection for mem0 Search API

Connection pools, authentication, and service dependencies.
Manages hot memory, semantic memory, and ranking service instances.

Based on partab.md: "connection pools, auth" specification.
"""

import asyncio
from typing import Optional, Dict, Any
from functools import lru_cache
import duckdb
import redis.asyncio as redis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from aura_intelligence.enterprise.mem0_hot.ingest import HotEpisodicIngestor
from aura_intelligence.enterprise.mem0_hot.vectorize import SignatureVectorizer
from aura_intelligence.enterprise.mem0_hot.archive import ArchivalManager
from aura_intelligence.enterprise.mem0_hot.settings import DuckDBSettings, DEFAULT_SETTINGS
from aura_intelligence.enterprise.mem0_semantic.sync import SemanticMemorySync
from aura_intelligence.enterprise.mem0_semantic.rank import MemoryRankingService
from aura_intelligence.utils.logger import get_logger

logger = get_logger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Global service instances
_hot_memory_service: Optional[HotEpisodicIngestor] = None
_semantic_memory_service: Optional[SemanticMemorySync] = None
_ranking_service: Optional[MemoryRankingService] = None
_duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None
_redis_connection: Optional[redis.Redis] = None

# Configuration
_duckdb_settings: DuckDBSettings = DEFAULT_SETTINGS
_redis_url: str = "redis://localhost:6379/0"
_auth_enabled: bool = False
_valid_api_keys: Dict[str, str] = {}


def configure_dependencies(
    duckdb_settings: DuckDBSettings = None,
    redis_url: str = "redis://localhost:6379/0",
    auth_enabled: bool = False,
    api_keys: Dict[str, str] = None
):
    """Configure global dependency settings."""
    
    global _duckdb_settings, _redis_url, _auth_enabled, _valid_api_keys
    
    if duckdb_settings:
        _duckdb_settings = duckdb_settings
    
    _redis_url = redis_url
    _auth_enabled = auth_enabled
    _valid_api_keys = api_keys or {}
    
    logger.info(f"🔌 Dependencies configured (auth: {auth_enabled}, redis: {redis_url})")


async def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Get or create DuckDB connection."""
    
    global _duckdb_connection
    
    if _duckdb_connection is None:
        try:
            # Create connection with settings
            _duckdb_connection = duckdb.connect(
                database=_duckdb_settings.db_path,
                config=_duckdb_settings.get_duckdb_config()
            )
            
            # Install VSS extension if enabled
            if _duckdb_settings.enable_vss:
                try:
                    _duckdb_connection.execute("INSTALL vss")
                    _duckdb_connection.execute("LOAD vss")
                    logger.info("✅ DuckDB VSS extension loaded")
                except Exception as e:
                    logger.warning(f"⚠️ VSS extension failed to load: {e}")
            
            # Create schema
            from aura_intelligence.enterprise.mem0_hot.schema import create_schema
            create_schema(_duckdb_connection, _duckdb_settings.vector_dimension)
            
            logger.info("✅ DuckDB connection established")
            
        except Exception as e:
            logger.error(f"❌ DuckDB connection failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed"
            )
    
    return _duckdb_connection


async def get_redis_connection() -> redis.Redis:
    """Get or create Redis connection."""
    
    global _redis_connection
    
    if _redis_connection is None:
        try:
            _redis_connection = redis.from_url(
                _redis_url,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Test connection
            await _redis_connection.ping()
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis connection failed"
            )
    
    return _redis_connection


async def get_hot_memory() -> HotEpisodicIngestor:
    """Get hot episodic memory service."""
    
    global _hot_memory_service
    
    if _hot_memory_service is None:
        try:
            # Get dependencies
            duckdb_conn = await get_duckdb_connection()
            
            # Create vectorizer
            vectorizer = SignatureVectorizer(_duckdb_settings.vector_dimension)
            
            # Create hot memory service
            _hot_memory_service = HotEpisodicIngestor(
                conn=duckdb_conn,
                settings=_duckdb_settings,
                vectorizer=vectorizer
            )
            
            logger.info("✅ Hot memory service initialized")
            
        except Exception as e:
            logger.error(f"❌ Hot memory service initialization failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Hot memory service unavailable"
            )
    
    return _hot_memory_service


async def get_semantic_memory() -> SemanticMemorySync:
    """Get semantic memory synchronization service."""
    
    global _semantic_memory_service
    
    if _semantic_memory_service is None:
        try:
            # Create vectorizer
            vectorizer = SignatureVectorizer(_duckdb_settings.vector_dimension)
            
            # Create semantic memory service
            _semantic_memory_service = SemanticMemorySync(
                redis_url=_redis_url,
                vectorizer=vectorizer,
                cluster_threshold=0.8
            )
            
            # Initialize Redis connection
            await _semantic_memory_service.initialize()
            
            logger.info("✅ Semantic memory service initialized")
            
        except Exception as e:
            logger.error(f"❌ Semantic memory service initialization failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Semantic memory service unavailable"
            )
    
    return _semantic_memory_service


async def get_ranking_service() -> MemoryRankingService:
    """Get memory ranking service."""
    
    global _ranking_service
    
    if _ranking_service is None:
        try:
            # Create ranking service
            _ranking_service = MemoryRankingService(redis_url=_redis_url)
            
            # Initialize Redis connection
            await _ranking_service.initialize()
            
            logger.info("✅ Memory ranking service initialized")
            
        except Exception as e:
            logger.error(f"❌ Memory ranking service initialization failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory ranking service unavailable"
            )
    
    return _ranking_service


async def get_archival_manager() -> ArchivalManager:
    """Get archival manager service."""
    
    try:
        duckdb_conn = await get_duckdb_connection()
        
        archival_manager = ArchivalManager(
            conn=duckdb_conn,
            settings=_duckdb_settings
        )
        
        return archival_manager
        
    except Exception as e:
        logger.error(f"❌ Archival manager initialization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Archival manager unavailable"
        )


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key authentication."""
    
    if not _auth_enabled:
        return "anonymous"  # No auth required
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    api_key = credentials.credentials
    
    if api_key not in _valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return _valid_api_keys[api_key]  # Return user/agent ID


async def get_current_agent(agent_id: str = Depends(verify_api_key)) -> str:
    """Get current authenticated agent ID."""
    return agent_id


# Health check dependencies
async def check_duckdb_health() -> Dict[str, Any]:
    """Check DuckDB health."""
    
    try:
        conn = await get_duckdb_connection()
        result = conn.execute("SELECT 1").fetchone()
        
        return {
            "status": "healthy" if result[0] == 1 else "unhealthy",
            "connection": "active",
            "response_time_ms": 1.0  # Placeholder
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "failed",
            "error": str(e)
        }


async def check_redis_health() -> Dict[str, Any]:
    """Check Redis health."""
    
    try:
        redis_conn = await get_redis_connection()
        await redis_conn.ping()
        
        return {
            "status": "healthy",
            "connection": "active",
            "response_time_ms": 1.0  # Placeholder
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "failed",
            "error": str(e)
        }


async def check_services_health() -> Dict[str, Any]:
    """Check all services health."""
    
    health_checks = {}
    
    # Check hot memory service
    try:
        hot_memory = await get_hot_memory()
        health_checks["hot_memory"] = await hot_memory.health_check()
    except Exception as e:
        health_checks["hot_memory"] = {"status": "unhealthy", "error": str(e)}
    
    # Check semantic memory service
    try:
        semantic_memory = await get_semantic_memory()
        health_checks["semantic_memory"] = await semantic_memory.health_check()
    except Exception as e:
        health_checks["semantic_memory"] = {"status": "unhealthy", "error": str(e)}
    
    # Check ranking service
    try:
        ranking_service = await get_ranking_service()
        health_checks["ranking_service"] = await ranking_service.health_check()
    except Exception as e:
        health_checks["ranking_service"] = {"status": "unhealthy", "error": str(e)}
    
    return health_checks


# Cleanup functions
async def cleanup_connections():
    """Clean up all connections and services."""
    
    global _hot_memory_service, _semantic_memory_service, _ranking_service
    global _duckdb_connection, _redis_connection
    
    try:
        # Stop background services
        if _semantic_memory_service:
            await _semantic_memory_service.stop_background_sync()
        
        if _ranking_service:
            await _ranking_service.stop_background_cleanup()
        
        # Close connections
        if _redis_connection:
            await _redis_connection.close()
            _redis_connection = None
        
        if _duckdb_connection:
            _duckdb_connection.close()
            _duckdb_connection = None
        
        # Reset services
        _hot_memory_service = None
        _semantic_memory_service = None
        _ranking_service = None
        
        logger.info("🧹 All connections and services cleaned up")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")


# Startup and shutdown handlers
async def startup_dependencies():
    """Initialize all dependencies on startup."""
    
    try:
        # Initialize connections
        await get_duckdb_connection()
        await get_redis_connection()
        
        # Initialize services
        await get_hot_memory()
        await get_semantic_memory()
        await get_ranking_service()
        
        # Start background services
        semantic_memory = await get_semantic_memory()
        await semantic_memory.start_background_sync(interval_minutes=15)
        
        ranking_service = await get_ranking_service()
        await ranking_service.start_background_cleanup(interval_hours=6)
        
        logger.info("🚀 All dependencies initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Dependency initialization failed: {e}")
        raise


async def shutdown_dependencies():
    """Clean up all dependencies on shutdown."""
    
    await cleanup_connections()
    logger.info("🛑 All dependencies shut down")
