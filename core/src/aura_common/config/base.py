"""
⚙️ Base Configuration Classes
Type-safe configuration with Pydantic v2.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import timedelta
from pathlib import Path
import os


class ConfigSection(BaseModel):
    """Base class for configuration sections."""
    
    model_config = ConfigDict(
        extra='forbid',  # Strict validation
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "description": "Base configuration section"
        }
    )


class LoggingConfig(ConfigSection):
    """Logging configuration."""
    
    level: str = Field(
        default="INFO",
        description="Logging level",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    format: str = Field(
        default="json",
        description="Log format",
        pattern="^(json|text)$"
    )
    correlation_header: str = Field(
        default="X-Correlation-ID",
        description="HTTP header for correlation ID"
    )
    shadow_mode_enabled: bool = Field(
        default=True,
        description="Enable shadow mode logging"
    )
    shadow_mode_dir: Path = Field(
        default=Path("shadow_logs"),
        description="Directory for shadow mode logs"
    )


class TDAConfig(ConfigSection):
    """TDA engine configuration."""
    
    preferred_engine: str = Field(
        default="auto",
        description="Preferred TDA engine",
        pattern="^(auto|mojo|cuda|python)$"
    )
    cuda_device: Optional[int] = Field(
        default=None,
        description="CUDA device ID"
    )
    max_points: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum points for TDA"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable TDA result caching"
    )
    cache_ttl: timedelta = Field(
        default=timedelta(hours=1),
        description="Cache time-to-live"
    )


class AgentConfig(ConfigSection):
    """Agent system configuration."""
    
    consensus_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Consensus threshold for decisions"
    )
    max_agents: int = Field(
        default=10,
        ge=3,
        le=100,
        description="Maximum number of agents"
    )
    timeout: timedelta = Field(
        default=timedelta(seconds=30),
        description="Agent operation timeout"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry attempts for failed operations"
    )


class MemoryConfig(ConfigSection):
    """Memory system configuration."""
    
    hot_cache_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Hot cache size in MB"
    )
    archive_threshold: int = Field(
        default=10000,
        ge=1000,
        description="Archive threshold in records"
    )
    vector_dimensions: int = Field(
        default=768,
        ge=128,
        le=4096,
        description="Vector embedding dimensions"
    )
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for search"
    )


class SecurityConfig(ConfigSection):
    """Security configuration."""
    
    encryption_enabled: bool = Field(
        default=True,
        description="Enable encryption"
    )
    signature_algorithm: str = Field(
        default="Ed25519",
        description="Signature algorithm",
        pattern="^(HMAC|RSA|ECDSA|Ed25519|Dilithium2|Falcon512)$"
    )
    key_rotation_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Key rotation period in days"
    )
    audit_enabled: bool = Field(
        default=True,
        description="Enable security audit logging"
    )


class ObservabilityConfig(ConfigSection):
    """Observability configuration."""
    
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )
    tracing_enabled: bool = Field(
        default=True,
        description="Enable distributed tracing"
    )
    tracing_endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry collector endpoint"
    )
    profiling_enabled: bool = Field(
        default=False,
        description="Enable performance profiling"
    )


class AuraConfig(ConfigSection):
    """Main AURA configuration."""
    
    # Environment
    environment: str = Field(
        default="development",
        description="Deployment environment",
        pattern="^(development|staging|production)$"
    )
    
    # Service info
    service_name: str = Field(
        default="aura-intelligence",
        description="Service name"
    )
    service_version: str = Field(
        default="2.0.0",
        description="Service version"
    )
    
    # Sub-configurations
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    tda: TDAConfig = Field(
        default_factory=TDAConfig,
        description="TDA configuration"
    )
    agents: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent configuration"
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="Observability configuration"
    )
    
    # Feature flags
    feature_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags"
    )
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate and normalize environment."""
        return v.lower()
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    def get_feature(self, flag: str) -> bool:
        """Get feature flag value."""
        return self.feature_flags.get(flag, False)