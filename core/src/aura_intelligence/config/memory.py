"""
Memory configuration for AURA Intelligence.

Defines settings for memory management, vector stores, and caching.
"""

from typing import Optional

from pydantic import Field, field_validator

from .base import BaseSettings


class MemorySettings(BaseSettings):
    """
    Memory system configuration.
    
    Controls memory management, vector stores, and caching behavior.
    Environment variables: AURA_MEMORY__*
    """
    
    model_config = BaseSettings.model_config.copy()
    model_config.update(env_prefix="AURA_MEMORY__")

    
    # Vector store settings
    vector_store_type: str = Field(
        default="chroma",
        description="Type of vector store (chroma, pinecone, weaviate)"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for vector store"
    )
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        description="Dimension of embeddings"
    )
    
    # Memory limits
    max_memories: int = Field(
        default=10000,
        ge=100,
        description="Maximum number of memories to store"
    )
    memory_ttl_hours: int = Field(
        default=720,  # 30 days
        ge=1,
        description="Time-to-live for memories in hours"
    )
    
    # Cache settings
    enable_cache: bool = Field(
        default=True,
        description="Enable memory caching"
    )
    cache_size_mb: int = Field(
        default=512,
        ge=64,
        description="Cache size in MB"
    )
    cache_ttl_minutes: int = Field(
        default=60,
        ge=1,
        description="Cache TTL in minutes"
    )
    
    # Persistence settings
    enable_persistence: bool = Field(
        default=True,
        description="Enable memory persistence"
    )
    persistence_path: Optional[str] = Field(
        default=None,
        description="Path for memory persistence"
    )
    checkpoint_interval_minutes: int = Field(
        default=30,
        ge=5,
        description="Checkpoint interval in minutes"
    )
    
    # Advanced features
    enable_semantic_search: bool = Field(
        default=True,
        description="Enable semantic search capabilities"
    )
    enable_memory_consolidation: bool = Field(
        default=True,
        description="Enable memory consolidation"
    )
    consolidation_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for memory consolidation"
    )
    
    # Performance tuning
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for vector operations"
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of workers for parallel processing"
    )
    
    @field_validator("vector_store_type")
    @classmethod
    def validate_vector_store(cls, v: str) -> str:
        """Validate vector store type."""
        allowed = {"chroma", "pinecone", "weaviate", "faiss", "qdrant"}
        if v not in allowed:
            raise ValueError(f"Vector store must be one of {allowed}")
        return v
    
    @property
    def requires_api_key(self) -> bool:
        """Check if vector store requires API key."""
        return self.vector_store_type in {"pinecone", "weaviate", "qdrant"}