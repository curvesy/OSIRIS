"""
Configuration Management for Shape Memory V2
===========================================

Production-grade configuration using Pydantic with:
- Environment variable support
- Type validation
- Default values
- Secret management
- Feature flags
"""

from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, validator, SecretStr
from enum import Enum


class StorageBackend(str, Enum):
    """Supported storage backends."""
    REDIS = "redis"
    MEMORY = "memory"
    FAISS = "faiss"
    WEAVIATE = "weaviate"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ShapeMemorySettings(BaseSettings):
    """
    Main configuration for Shape Memory V2.
    
    All settings can be overridden via environment variables
    with the prefix SHAPE_MEMORY_.
    
    Example:
        SHAPE_MEMORY_REDIS_URL=redis://prod-redis:6379
        SHAPE_MEMORY_EMBEDDING_DIM=256
    """
    
    # Storage Configuration
    storage_backend: StorageBackend = Field(
        StorageBackend.REDIS,
        description="Storage backend to use"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        "redis://localhost:6379",
        env="SHAPE_MEMORY_REDIS_URL",
        description="Redis connection URL"
    )
    redis_password: Optional[SecretStr] = Field(
        None,
        env="SHAPE_MEMORY_REDIS_PASSWORD",
        description="Redis password (if required)"
    )
    redis_max_connections: int = Field(
        50,
        ge=10,
        le=1000,
        description="Maximum Redis connections"
    )
    redis_socket_timeout: int = Field(
        5,
        ge=1,
        le=30,
        description="Redis socket timeout in seconds"
    )
    
    # Neo4j Configuration
    neo4j_url: str = Field(
        "bolt://localhost:7687",
        env="SHAPE_MEMORY_NEO4J_URL",
        description="Neo4j connection URL"
    )
    neo4j_user: str = Field(
        "neo4j",
        env="SHAPE_MEMORY_NEO4J_USER",
        description="Neo4j username"
    )
    neo4j_password: SecretStr = Field(
        ...,
        env="SHAPE_MEMORY_NEO4J_PASSWORD",
        description="Neo4j password"
    )
    
    # Embedding Configuration
    embedding_dim: int = Field(
        128,
        ge=32,
        le=1024,
        description="Embedding dimension"
    )
    fastrp_iterations: int = Field(
        3,
        ge=1,
        le=10,
        description="FastRP iteration count"
    )
    
    # Fusion Scoring Configuration
    enable_fusion_scoring: bool = Field(
        True,
        description="Enable adaptive fusion scoring"
    )
    fusion_alpha: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Weight for FastRP similarity"
    )
    fusion_beta: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Weight for Wasserstein distance"
    )
    fusion_tau_hours: float = Field(
        168.0,
        ge=1.0,
        le=720.0,
        description="Embedding decay time constant (hours)"
    )
    
    # Performance Configuration
    max_concurrent_operations: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum concurrent operations"
    )
    operation_timeout: float = Field(
        5.0,
        ge=0.1,
        le=60.0,
        description="Operation timeout in seconds"
    )
    metrics_update_interval: int = Field(
        30,
        ge=5,
        le=300,
        description="Metrics update interval in seconds"
    )
    
    # Circuit Breaker Configuration
    circuit_breaker_fail_max: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum failures before circuit opens"
    )
    circuit_breaker_reset_timeout: int = Field(
        30,
        ge=5,
        le=300,
        description="Circuit breaker reset timeout in seconds"
    )
    
    # Retry Configuration
    retry_max_attempts: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )
    retry_wait_min: float = Field(
        0.1,
        ge=0.01,
        le=1.0,
        description="Minimum retry wait time"
    )
    retry_wait_max: float = Field(
        2.0,
        ge=0.1,
        le=10.0,
        description="Maximum retry wait time"
    )
    
    # ETL Configuration
    etl_batch_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="ETL batch size"
    )
    etl_similarity_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for graph edges"
    )
    etl_max_similar_edges: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum similar edges per node"
    )
    
    # Feature Flags
    feature_flag_enabled: bool = Field(
        True,
        env="SHAPE_MEMORY_FEATURE_ENABLED",
        description="Master feature flag"
    )
    enable_shadow_mode: bool = Field(
        False,
        env="SHAPE_MEMORY_SHADOW_MODE",
        description="Enable shadow deployment mode"
    )
    enable_graph_algorithms: bool = Field(
        True,
        description="Enable Neo4j graph algorithms"
    )
    enable_danger_detection: bool = Field(
        True,
        description="Enable danger ring detection"
    )
    
    # Observability Configuration
    log_level: LogLevel = Field(
        LogLevel.INFO,
        description="Logging level"
    )
    enable_tracing: bool = Field(
        True,
        description="Enable OpenTelemetry tracing"
    )
    enable_metrics: bool = Field(
        True,
        description="Enable Prometheus metrics"
    )
    otlp_endpoint: Optional[str] = Field(
        None,
        env="SHAPE_MEMORY_OTLP_ENDPOINT",
        description="OpenTelemetry collector endpoint"
    )
    
    # Validation
    @validator("fusion_alpha", "fusion_beta")
    def validate_fusion_weights(cls, v, values):
        """Ensure fusion weights sum to 1.0."""
        if "fusion_alpha" in values and "fusion_beta" in values:
            total = values["fusion_alpha"] + values["fusion_beta"]
            if not (0.99 <= total <= 1.01):  # Allow small floating point error
                raise ValueError(f"Fusion weights must sum to 1.0, got {total}")
        return v
    
    @validator("redis_url")
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("Redis URL must start with redis:// or rediss://")
        return v
    
    @validator("neo4j_url")
    def validate_neo4j_url(cls, v):
        """Validate Neo4j URL format."""
        if not v.startswith(("bolt://", "neo4j://")):
            raise ValueError("Neo4j URL must start with bolt:// or neo4j://")
        return v
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "SHAPE_MEMORY_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Allow extra fields for forward compatibility
        extra = "allow"
        
        # Use enum values
        use_enum_values = True


class BenchmarkSettings(BaseSettings):
    """Configuration for benchmarking."""
    
    num_vectors: int = Field(
        1_000_000,
        ge=1000,
        le=10_000_000,
        description="Number of vectors to benchmark"
    )
    vector_dim: int = Field(
        128,
        ge=32,
        le=1024,
        description="Vector dimension"
    )
    num_queries: int = Field(
        10_000,
        ge=100,
        le=100_000,
        description="Number of queries to run"
    )
    num_threads: int = Field(
        8,
        ge=1,
        le=64,
        description="Number of benchmark threads"
    )
    k_values: list[int] = Field(
        [5, 10, 50],
        description="k values to test"
    )
    
    class Config:
        env_prefix = "BENCHMARK_"


# Singleton instances
_settings: Optional[ShapeMemorySettings] = None
_benchmark_settings: Optional[BenchmarkSettings] = None


def get_settings() -> ShapeMemorySettings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = ShapeMemorySettings()
    return _settings


def get_benchmark_settings() -> BenchmarkSettings:
    """Get or create benchmark settings singleton."""
    global _benchmark_settings
    if _benchmark_settings is None:
        _benchmark_settings = BenchmarkSettings()
    return _benchmark_settings


def reload_settings():
    """Reload settings from environment."""
    global _settings, _benchmark_settings
    _settings = None
    _benchmark_settings = None


# Export commonly used settings
settings = get_settings()
benchmark_settings = get_benchmark_settings()