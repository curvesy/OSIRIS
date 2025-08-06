"""
Observability configuration for AURA Intelligence.

Defines settings for monitoring, metrics, logging, and tracing.
"""

from typing import List, Optional

from pydantic import Field, validator

from .base import BaseSettings


class ObservabilitySettings(BaseSettings):
    """
    Observability configuration for monitoring and metrics.
    
    Supports Prometheus, Grafana, and OpenTelemetry integrations.
    Environment variables: AURA_OBSERVABILITY__*
    """
    
    model_config = BaseSettings.model_config.copy()
    model_config.update(env_prefix="AURA_OBSERVABILITY__")

    
    # Metrics configuration
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Port for metrics endpoint"
    )
    metrics_path: str = Field(
        default="/metrics",
        description="Path for metrics endpoint"
    )
    
    # Prometheus configuration
    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    prometheus_pushgateway_url: Optional[str] = Field(
        default=None,
        description="Prometheus pushgateway URL"
    )
    prometheus_job_name: str = Field(
        default="aura_intelligence",
        description="Prometheus job name"
    )
    
    # Logging configuration
    log_format: str = Field(
        default="json",
        description="Log format (json, text)"
    )
    log_to_file: bool = Field(
        default=True,
        description="Enable file logging"
    )
    log_file_max_size_mb: int = Field(
        default=100,
        ge=10,
        description="Maximum log file size in MB"
    )
    log_file_retention_days: int = Field(
        default=30,
        ge=1,
        description="Log file retention in days"
    )
    
    # Tracing configuration
    enable_tracing: bool = Field(
        default=True,
        description="Enable distributed tracing"
    )
    tracing_backend: str = Field(
        default="jaeger",
        description="Tracing backend (jaeger, zipkin, otlp)"
    )
    tracing_endpoint: Optional[str] = Field(
        default=None,
        description="Tracing collector endpoint"
    )
    tracing_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Tracing sample rate (0.0-1.0)"
    )
    
    # Health check configuration
    enable_health_check: bool = Field(
        default=True,
        description="Enable health check endpoint"
    )
    health_check_port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Port for health check endpoint"
    )
    health_check_path: str = Field(
        default="/health",
        description="Path for health check endpoint"
    )
    
    # Alert configuration
    enable_alerts: bool = Field(
        default=True,
        description="Enable alerting"
    )
    alert_webhook_urls: List[str] = Field(
        default_factory=list,
        description="Webhook URLs for alerts"
    )
    alert_email_addresses: List[str] = Field(
        default_factory=list,
        description="Email addresses for alerts"
    )
    
    # Performance monitoring
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    profiling_interval_seconds: int = Field(
        default=60,
        ge=10,
        description="Profiling interval in seconds"
    )
    
    @validator("log_format")
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        if v not in {"json", "text"}:
            raise ValueError("Log format must be 'json' or 'text'")
        return v
    
    @validator("tracing_backend")
    def validate_tracing_backend(cls, v: str) -> str:
        """Validate tracing backend."""
        allowed = {"jaeger", "zipkin", "otlp", "none"}
        if v not in allowed:
            raise ValueError(f"Tracing backend must be one of {allowed}")
        return v
    
    @property
    def metrics_url(self) -> str:
        """Get full metrics URL."""
        return f"http://0.0.0.0:{self.metrics_port}{self.metrics_path}"
    
    @property
    def health_check_url(self) -> str:
        """Get full health check URL."""
        return f"http://0.0.0.0:{self.health_check_port}{self.health_check_path}"