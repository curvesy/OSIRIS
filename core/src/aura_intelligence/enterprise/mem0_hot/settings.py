"""
🔧 DuckDB Configuration Settings

Memory limits, threading, and performance tuning for embedded DuckDB 0.11
following partab.md best practices.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class DuckDBSettings:
    """
    DuckDB configuration for hot episodic memory layer.
    
    Based on partab.md recommendations for production deployment.
    """
    
    # Memory configuration
    memory_limit: str = "4GB"
    temp_directory: str = "/tmp/duckdb"
    
    # Threading configuration  
    threads: int = 4
    
    # Database configuration
    db_path: str = ":memory:"  # In-memory for hot tier
    enable_vss: bool = True    # Vector similarity search
    
    # Retention configuration
    retention_hours: int = 24
    
    # Performance tuning
    enable_optimizer: bool = True
    enable_profiling: bool = False
    
    # Archival configuration
    s3_bucket: Optional[str] = None
    s3_prefix: str = "aura-intelligence/hot-archive"
    parquet_compression: str = "snappy"
    
    # Vector configuration
    vector_dimension: int = 128
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        """Validate settings after initialization."""
        
        # Ensure temp directory exists
        temp_path = Path(self.temp_directory)
        temp_path.mkdir(parents=True, exist_ok=True)
        
        # Validate memory limit format
        if not self.memory_limit.endswith(('GB', 'MB', 'KB')):
            raise ValueError("memory_limit must end with GB, MB, or KB")
        
        # Validate thread count
        if self.threads < 1 or self.threads > 32:
            raise ValueError("threads must be between 1 and 32")
        
        # Validate retention hours
        if self.retention_hours < 1:
            raise ValueError("retention_hours must be at least 1")
        
        # Validate vector dimension
        if self.vector_dimension < 1 or self.vector_dimension > 2048:
            raise ValueError("vector_dimension must be between 1 and 2048")
    
    def get_duckdb_config(self) -> dict:
        """Get DuckDB configuration dictionary."""
        
        config = {
            "memory_limit": self.memory_limit,
            "temp_directory": self.temp_directory,
            "threads": self.threads,
            "enable_optimizer": self.enable_optimizer,
            "enable_profiling": self.enable_profiling
        }
        
        return config
    
    def get_connection_string(self) -> str:
        """Get DuckDB connection string with configuration."""
        
        if self.db_path == ":memory:":
            return ":memory:"
        
        return f"{self.db_path}?memory_limit={self.memory_limit}&threads={self.threads}"


# Default settings instance
DEFAULT_SETTINGS = DuckDBSettings()


# Production settings with higher limits
PRODUCTION_SETTINGS = DuckDBSettings(
    memory_limit="8GB",
    threads=8,
    retention_hours=24,
    enable_profiling=True,
    s3_bucket="aura-intelligence-prod",
    vector_dimension=256
)


# Development settings with lower resource usage
DEV_SETTINGS = DuckDBSettings(
    memory_limit="1GB", 
    threads=2,
    retention_hours=6,
    enable_profiling=True,
    temp_directory="/tmp/duckdb-dev"
)
