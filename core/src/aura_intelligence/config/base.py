"""
Base configuration classes and enums for AURA Intelligence.

Uses Pydantic v2 for type-safe, environment-aware configuration.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict


class EnvironmentType(str, Enum):
    """Environment types for deployment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


class EnhancementLevel(str, Enum):
    """Agent enhancement levels."""
    BASIC = "basic"
    ADVANCED = "advanced"
    ULTIMATE = "ultimate"
    CONSCIOUSNESS = "consciousness"


class BaseSettings(PydanticBaseSettings):
    """
    Base settings class with common configuration.
    
    All settings can be overridden via environment variables with the prefix AURA_.
    For example: AURA_ENVIRONMENT=production
    """
    
    model_config = SettingsConfigDict(
        env_prefix="AURA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Core settings
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent,
        description="Project root directory"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".aura_intelligence",
        description="Data storage directory"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path.home() / ".aura_intelligence" / "logs",
        description="Logs directory"
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to ensure directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment in (EnvironmentType.PRODUCTION, EnvironmentType.ENTERPRISE)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == EnvironmentType.DEVELOPMENT