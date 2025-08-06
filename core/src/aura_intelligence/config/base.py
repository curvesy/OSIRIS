"""
Base Configuration System for AURA Intelligence

This module provides a centralized configuration management system
to eliminate hardcoded values and improve testability.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
from functools import lru_cache


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    # Neo4j settings
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "aura"))
    
    # Redis settings
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    redis_password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    
    # PostgreSQL settings (if needed)
    postgres_uri: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://localhost:5432/aura"))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable must be set")
            

@dataclass
class MessagingConfig:
    """Messaging system configuration."""
    # Kafka settings
    kafka_bootstrap_servers: List[str] = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    )
    kafka_security_protocol: str = field(default_factory=lambda: os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"))
    kafka_sasl_mechanism: Optional[str] = field(default_factory=lambda: os.getenv("KAFKA_SASL_MECHANISM"))
    kafka_sasl_username: Optional[str] = field(default_factory=lambda: os.getenv("KAFKA_SASL_USERNAME"))
    kafka_sasl_password: Optional[str] = field(default_factory=lambda: os.getenv("KAFKA_SASL_PASSWORD"))
    kafka_schema_registry_url: str = field(
        default_factory=lambda: os.getenv("KAFKA_SCHEMA_REGISTRY_URL", "http://localhost:8081")
    )
    
    # NATS settings
    nats_url: str = field(default_factory=lambda: os.getenv("NATS_URL", "nats://localhost:4222"))
    nats_token: Optional[str] = field(default_factory=lambda: os.getenv("NATS_TOKEN"))
    
    # RabbitMQ settings
    rabbitmq_url: str = field(
        default_factory=lambda: os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    )


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    # OpenTelemetry
    otel_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )
    otel_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OTEL_API_KEY"))
    otel_service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "aura-intelligence"))
    
    # Prometheus
    prometheus_port: int = field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090")))
    
    # Jaeger
    jaeger_endpoint: str = field(
        default_factory=lambda: os.getenv("JAEGER_ENDPOINT", "http://localhost:14268")
    )
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))


@dataclass
class ServicesConfig:
    """External services configuration."""
    # TDA Service
    tda_service_url: str = field(default_factory=lambda: os.getenv("TDA_SERVICE_URL", "http://localhost:8080"))
    
    # Temporal
    temporal_host: str = field(default_factory=lambda: os.getenv("TEMPORAL_HOST", "localhost:7233"))
    temporal_namespace: str = field(default_factory=lambda: os.getenv("TEMPORAL_NAMESPACE", "default"))
    
    # LangSmith
    langsmith_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY"))
    langsmith_project: Optional[str] = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT"))
    
    # Mem0
    mem0_base_url: str = field(default_factory=lambda: os.getenv("MEM0_BASE_URL", "http://localhost:8080"))
    mem0_api_key: Optional[str] = field(default_factory=lambda: os.getenv("MEM0_API_KEY"))


@dataclass
class SecurityConfig:
    """Security configuration."""
    # API Keys
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("AURA_API_KEY"))
    jwt_secret: Optional[str] = field(default_factory=lambda: os.getenv("JWT_SECRET"))
    
    # Encryption
    encryption_key: Optional[str] = field(default_factory=lambda: os.getenv("ENCRYPTION_KEY"))
    
    # Password policies
    min_password_length: int = field(default_factory=lambda: int(os.getenv("MIN_PASSWORD_LENGTH", "12")))
    require_special_chars: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_SPECIAL_CHARS", "true").lower() == "true"
    )


@dataclass
class AURAConfig:
    """Main AURA Intelligence configuration."""
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    messaging: MessagingConfig = field(default_factory=MessagingConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> "AURAConfig":
        """Load configuration from file."""
        path = Path(config_path)
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path) as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
            
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AURAConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update environment and debug
        if "environment" in data:
            config.environment = data["environment"]
        if "debug" in data:
            config.debug = data["debug"]
            
        # Update sub-configurations
        if "database" in data:
            config.database = DatabaseConfig(**data["database"])
        if "messaging" in data:
            config.messaging = MessagingConfig(**data["messaging"])
        if "observability" in data:
            config.observability = ObservabilityConfig(**data["observability"])
        if "services" in data:
            config.services = ServicesConfig(**data["services"])
        if "security" in data:
            config.security = SecurityConfig(**data["security"])
            
        # Update feature flags
        if "feature_flags" in data:
            config.feature_flags.update(data["feature_flags"])
            
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required passwords and secrets
        if self.environment == "production":
            if not self.database.neo4j_password:
                issues.append("Neo4j password is required in production")
            if not self.security.jwt_secret:
                issues.append("JWT secret is required in production")
            if not self.security.encryption_key:
                issues.append("Encryption key is required in production")
                
        # Check for default/weak values
        if self.database.neo4j_password == "password":
            issues.append("Neo4j password should not be default value")
            
        # Check service connectivity (optional)
        # This could be extended to actually test connections
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "database": {
                "neo4j_uri": self.database.neo4j_uri,
                "neo4j_user": self.database.neo4j_user,
                "neo4j_database": self.database.neo4j_database,
                "redis_url": self.database.redis_url,
                "redis_db": self.database.redis_db,
                "postgres_uri": self.database.postgres_uri
            },
            "messaging": {
                "kafka_bootstrap_servers": self.messaging.kafka_bootstrap_servers,
                "kafka_security_protocol": self.messaging.kafka_security_protocol,
                "nats_url": self.messaging.nats_url,
                "rabbitmq_url": self.messaging.rabbitmq_url
            },
            "observability": {
                "otel_endpoint": self.observability.otel_endpoint,
                "otel_service_name": self.observability.otel_service_name,
                "prometheus_port": self.observability.prometheus_port,
                "jaeger_endpoint": self.observability.jaeger_endpoint,
                "log_level": self.observability.log_level,
                "log_format": self.observability.log_format
            },
            "services": {
                "tda_service_url": self.services.tda_service_url,
                "temporal_host": self.services.temporal_host,
                "temporal_namespace": self.services.temporal_namespace,
                "mem0_base_url": self.services.mem0_base_url
            },
            "security": {
                "min_password_length": self.security.min_password_length,
                "require_special_chars": self.security.require_special_chars
            },
            "feature_flags": self.feature_flags
        }


# Global configuration instance
_config: Optional[AURAConfig] = None


# Compatibility class for BaseSettings
class BaseSettings:
    """Compatibility class for Pydantic BaseSettings."""
    model_config = {"extra": "allow"}  # Mimics Pydantic config
    
    def __init__(self, **kwargs):
        """Initialize with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, obj: Dict[str, Any]):
        """Mimics Pydantic's model_validate."""
        return cls(**obj)
    
    def model_dump(self) -> Dict[str, Any]:
        """Mimics Pydantic's model_dump."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def dict(self) -> Dict[str, Any]:
        """Backward compatibility for dict() method."""
        return self.model_dump()


# Compatibility enums
class EnvironmentType:
    """Environment type enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class EnhancementLevel:
    """Enhancement level enumeration."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"
    ULTIMATE = "ultimate"


@lru_cache(maxsize=1)
def get_config() -> AURAConfig:
    """Get the global configuration instance."""
    global _config
    
    if _config is None:
        # Try to load from file
        config_path = os.getenv("AURA_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            _config = AURAConfig.from_file(config_path)
        else:
            # Use environment variables
            _config = AURAConfig()
            
        # Validate in production
        if _config.environment == "production":
            issues = _config.validate()
            if issues:
                raise ValueError(f"Configuration validation failed: {', '.join(issues)}")
    
    return _config


def set_config(config: AURAConfig) -> None:
    """Set the global configuration instance (mainly for testing)."""
    global _config
    _config = config
    get_config.cache_clear()


def reset_config() -> None:
    """Reset configuration (mainly for testing)."""
    global _config
    _config = None
    get_config.cache_clear()