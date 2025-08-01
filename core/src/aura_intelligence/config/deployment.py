"""
Deployment configuration for AURA Intelligence.

Defines settings for different deployment modes (shadow, canary, production).
"""

from typing import List, Optional

from pydantic import Field, field_validator

from .base import BaseSettings


class DeploymentSettings(BaseSettings):
    """
    Deployment configuration for different modes.
    
    Supports shadow mode, canary deployments, and production rollouts.
    Environment variables: AURA_DEPLOYMENT__*
    """
    
    model_config = BaseSettings.model_config.copy()
    model_config.update(env_prefix="AURA_DEPLOYMENT__")
    """
    Deployment configuration for different modes.
    
    Supports shadow mode, canary deployments, and production rollouts.
    """
    
    # Deployment mode
    deployment_mode: str = Field(
        default="shadow",
        description="Deployment mode (shadow, canary, production)"
    )
    
    # Shadow mode configuration
    shadow_enabled: bool = Field(
        default=True,
        description="Enable shadow mode"
    )
    shadow_traffic_percentage: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of traffic to shadow"
    )
    shadow_comparison_enabled: bool = Field(
        default=True,
        description="Enable shadow comparison"
    )
    shadow_logging_enabled: bool = Field(
        default=True,
        description="Enable shadow mode logging"
    )
    
    # Canary deployment configuration
    canary_enabled: bool = Field(
        default=False,
        description="Enable canary deployment"
    )
    canary_traffic_percentage: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Percentage of traffic to canary"
    )
    canary_rollback_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Error rate threshold for canary rollback"
    )
    canary_duration_minutes: int = Field(
        default=60,
        ge=5,
        description="Canary deployment duration in minutes"
    )
    
    # Production configuration
    production_replicas: int = Field(
        default=3,
        ge=1,
        description="Number of production replicas"
    )
    production_auto_scaling: bool = Field(
        default=True,
        description="Enable auto-scaling in production"
    )
    production_min_replicas: int = Field(
        default=2,
        ge=1,
        description="Minimum replicas for auto-scaling"
    )
    production_max_replicas: int = Field(
        default=10,
        ge=1,
        description="Maximum replicas for auto-scaling"
    )
    
    # Container configuration
    container_registry: str = Field(
        default="docker.io",
        description="Container registry URL"
    )
    container_image: str = Field(
        default="aura-intelligence/core",
        description="Container image name"
    )
    container_tag: str = Field(
        default="latest",
        description="Container image tag"
    )
    
    # Resource limits
    cpu_request: str = Field(
        default="500m",
        description="CPU request (Kubernetes format)"
    )
    cpu_limit: str = Field(
        default="2000m",
        description="CPU limit (Kubernetes format)"
    )
    memory_request: str = Field(
        default="512Mi",
        description="Memory request (Kubernetes format)"
    )
    memory_limit: str = Field(
        default="2Gi",
        description="Memory limit (Kubernetes format)"
    )
    
    # Health checks
    liveness_probe_path: str = Field(
        default="/health/live",
        description="Liveness probe path"
    )
    readiness_probe_path: str = Field(
        default="/health/ready",
        description="Readiness probe path"
    )
    startup_probe_path: str = Field(
        default="/health/startup",
        description="Startup probe path"
    )
    
    # Feature flags
    feature_flags: List[str] = Field(
        default_factory=list,
        description="Enabled feature flags"
    )
    
    @field_validator("deployment_mode")
    @classmethod
    def validate_deployment_mode(cls, v: str) -> str:
        """Validate deployment mode."""
        allowed = {"shadow", "canary", "production"}
        if v not in allowed:
            raise ValueError(f"Deployment mode must be one of {allowed}")
        return v
    
    @property
    def is_shadow_mode(self) -> bool:
        """Check if running in shadow mode."""
        return self.deployment_mode == "shadow" and self.shadow_enabled
    
    @property
    def is_canary_mode(self) -> bool:
        """Check if running in canary mode."""
        return self.deployment_mode == "canary" and self.canary_enabled
    
    @property
    def full_image_name(self) -> str:
        """Get full container image name with registry and tag."""
        return f"{self.container_registry}/{self.container_image}:{self.container_tag}"