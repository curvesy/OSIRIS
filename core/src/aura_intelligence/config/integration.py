"""
Integration configuration for AURA Intelligence.

Defines settings for external service integrations.
"""

from typing import List, Optional

from pydantic import Field

from .base import BaseSettings


class IntegrationSettings(BaseSettings):
    """
    Configuration for external service integrations.
    
    Manages connections to databases, message queues, and other services.
    Environment variables: AURA_INTEGRATION__*
    """
    
    model_config = BaseSettings.model_config.copy()
    model_config.update(env_prefix="AURA_INTEGRATION__")
    """
    Configuration for external service integrations.
    
    Manages connections to databases, message queues, and other services.
    """
    
    # Database configuration
    database_url: str = Field(
        default="sqlite:///aura_intelligence.db",
        description="Database connection URL"
    )
    database_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=20,
        ge=0,
        description="Maximum overflow connections"
    )
    
    # Redis configuration
    redis_enabled: bool = Field(
        default=True,
        description="Enable Redis integration"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_max_connections: int = Field(
        default=50,
        ge=10,
        description="Maximum Redis connections"
    )
    
    # Message queue configuration
    message_queue_type: str = Field(
        default="rabbitmq",
        description="Message queue type (rabbitmq, kafka, redis)"
    )
    message_queue_url: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        description="Message queue connection URL"
    )
    message_queue_exchange: str = Field(
        default="aura_intelligence",
        description="Message queue exchange name"
    )
    
    # Webhook configuration
    webhook_enabled: bool = Field(
        default=True,
        description="Enable webhook integrations"
    )
    webhook_timeout_seconds: int = Field(
        default=30,
        ge=5,
        description="Webhook request timeout"
    )
    webhook_retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Webhook retry attempts"
    )
    
    # External API endpoints
    external_api_endpoints: List[str] = Field(
        default_factory=list,
        description="List of external API endpoints"
    )
    
    # Service discovery
    enable_service_discovery: bool = Field(
        default=False,
        description="Enable service discovery"
    )
    consul_url: Optional[str] = Field(
        default=None,
        description="Consul service discovery URL"
    )
    
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.enable_service_discovery or self.message_queue_type != "redis"