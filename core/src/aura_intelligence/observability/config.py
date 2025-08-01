"""
🔧 Observability Configuration - Latest 2025 Patterns
Professional configuration management for neural observability system.
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime, timezone


@dataclass
class ObservabilityConfig:
    """
    Latest 2025 observability configuration patterns.
    
    Follows enterprise configuration best practices with:
    - Environment-based defaults
    - Type safety with dataclasses
    - Comprehensive feature flags
    - Production-ready security settings
    """
    
    # === Core Organism Identity ===
    organism_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    organism_generation: int = field(default_factory=lambda: int(os.getenv("ORGANISM_GENERATION", "1")))
    deployment_environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    service_version: str = field(default_factory=lambda: os.getenv("SERVICE_VERSION", "2025.7.27"))
    
    # === LangSmith 2.0 Configuration (Latest July 2025) ===
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "collective-intelligence-neural-v2"))
    langsmith_endpoint: str = field(default_factory=lambda: os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))
    langsmith_enable_streaming: bool = field(default_factory=lambda: os.getenv("LANGSMITH_STREAMING", "true").lower() == "true")
    langsmith_enable_evaluation: bool = field(default_factory=lambda: os.getenv("LANGSMITH_EVALUATION", "true").lower() == "true")
    langsmith_batch_size: int = field(default_factory=lambda: int(os.getenv("LANGSMITH_BATCH_SIZE", "10")))
    
    # === OpenTelemetry Configuration ===
    otel_service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "collective-intelligence-organism"))
    otel_exporter_endpoint: str = field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))
    otel_api_key: str = field(default_factory=lambda: os.getenv("OTEL_API_KEY", ""))
    otel_headers: Dict[str, str] = field(default_factory=lambda: {
        "api-key": os.getenv("OTEL_API_KEY", ""),
        "organism-id": os.getenv("ORGANISM_ID", ""),
    })
    otel_enable_auto_instrumentation: bool = field(default_factory=lambda: os.getenv("OTEL_AUTO_INSTRUMENT", "true").lower() == "true")
    otel_batch_size: int = field(default_factory=lambda: int(os.getenv("OTEL_BATCH_SIZE", "512")))
    otel_export_timeout: int = field(default_factory=lambda: int(os.getenv("OTEL_EXPORT_TIMEOUT", "30000")))
    
    # === Prometheus Configuration ===
    prometheus_port: int = field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "8000")))
    prometheus_enable_multiprocess: bool = field(default_factory=lambda: os.getenv("PROMETHEUS_MULTIPROCESS", "true").lower() == "true")
    prometheus_registry_path: str = field(default_factory=lambda: os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc"))
    
    # === Structured Logging Configuration ===
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))
    log_enable_correlation: bool = field(default_factory=lambda: os.getenv("LOG_CORRELATION", "true").lower() == "true")
    log_enable_crypto_signatures: bool = field(default_factory=lambda: os.getenv("LOG_CRYPTO_SIGNATURES", "true").lower() == "true")
    
    # === Knowledge Graph Configuration ===
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))
    neo4j_enable_memory_consolidation: bool = field(default_factory=lambda: os.getenv("NEO4J_MEMORY_CONSOLIDATION", "true").lower() == "true")
    
    # === Real-time Streaming Configuration ===
    enable_real_time_streaming: bool = field(default_factory=lambda: os.getenv("ENABLE_STREAMING", "false").lower() == "true")
    kafka_bootstrap_servers: str = field(default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    kafka_topic_prefix: str = field(default_factory=lambda: os.getenv("KAFKA_TOPIC_PREFIX", "organism"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    
    # === Advanced Features ===
    enable_cost_tracking: bool = field(default_factory=lambda: os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true")
    enable_anomaly_detection: bool = field(default_factory=lambda: os.getenv("ENABLE_ANOMALY_DETECTION", "true").lower() == "true")
    enable_performance_profiling: bool = field(default_factory=lambda: os.getenv("ENABLE_PERFORMANCE_PROFILING", "true").lower() == "true")
    enable_cryptographic_audit: bool = field(default_factory=lambda: os.getenv("ENABLE_CRYPTO_AUDIT", "true").lower() == "true")
    
    # === Health Monitoring Configuration ===
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("HEALTH_CHECK_INTERVAL", "30")))
    health_score_threshold: float = field(default_factory=lambda: float(os.getenv("HEALTH_SCORE_THRESHOLD", "0.7")))
    enable_auto_recovery: bool = field(default_factory=lambda: os.getenv("ENABLE_AUTO_RECOVERY", "true").lower() == "true")
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        
        # Ensure organism_id is set in headers
        if self.organism_id and "organism-id" not in self.otel_headers:
            self.otel_headers["organism-id"] = self.organism_id
        
        # Validate critical configurations
        if self.deployment_environment == "production":
            self._validate_production_config()
    
    def _validate_production_config(self):
        """Validate configuration for production deployment."""
        
        critical_configs = []
        
        # Check for missing production-critical configurations
        if not self.langsmith_api_key and self.langsmith_enable_streaming:
            critical_configs.append("LANGSMITH_API_KEY required for streaming traces")
        
        if not self.otel_api_key and self.otel_exporter_endpoint != "http://localhost:4317":
            critical_configs.append("OTEL_API_KEY required for external OTLP endpoint")
        
        if self.neo4j_password == "password":
            critical_configs.append("NEO4J_PASSWORD should be changed from default")
        
        if critical_configs:
            import warnings
            warnings.warn(
                f"Production configuration issues detected: {'; '.join(critical_configs)}",
                UserWarning
            )
    
    def get_resource_attributes(self) -> Dict[str, str]:
        """Get OpenTelemetry resource attributes."""
        
        return {
            "service.name": self.otel_service_name,
            "service.version": self.service_version,
            "deployment.environment": self.deployment_environment,
            "organism.id": self.organism_id,
            "organism.generation": str(self.organism_generation),
            "organism.type": "self_repairing_digital_organism",
            "architecture.pattern": "bio_inspired_collective_intelligence",
            "observability.version": "2025.7.27",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    def get_langsmith_tags(self) -> List[str]:
        """Get LangSmith tags for trace categorization."""
        
        return [
            f"generation:{self.organism_generation}",
            f"environment:{self.deployment_environment}",
            f"organism:{self.organism_id[:8]}",
            "neural-observability",
            "collective-intelligence",
            "bio-inspired-architecture"
        ]
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled."""
        
        feature_map = {
            "cost_tracking": self.enable_cost_tracking,
            "anomaly_detection": self.enable_anomaly_detection,
            "performance_profiling": self.enable_performance_profiling,
            "cryptographic_audit": self.enable_cryptographic_audit,
            "real_time_streaming": self.enable_real_time_streaming,
            "auto_recovery": self.enable_auto_recovery,
            "langsmith_streaming": self.langsmith_enable_streaming,
            "langsmith_evaluation": self.langsmith_enable_evaluation,
            "otel_auto_instrumentation": self.otel_enable_auto_instrumentation,
            "log_correlation": self.log_enable_correlation,
            "crypto_signatures": self.log_enable_crypto_signatures,
            "memory_consolidation": self.neo4j_enable_memory_consolidation,
        }
        
        return feature_map.get(feature, False)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert configuration to dictionary for serialization."""
        
        # Exclude sensitive information
        sensitive_fields = {
            "langsmith_api_key", "otel_api_key", "neo4j_password"
        }
        
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
            if field.name not in sensitive_fields
        }


# Default configuration instance
default_config = ObservabilityConfig()


def create_config(**overrides) -> ObservabilityConfig:
    """
    Create observability configuration with overrides.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        ObservabilityConfig: Configured instance
    """
    
    return ObservabilityConfig(**overrides)


def create_development_config() -> ObservabilityConfig:
    """Create configuration optimized for development."""
    
    return ObservabilityConfig(
        deployment_environment="development",
        log_level="DEBUG",
        prometheus_enable_multiprocess=False,
        enable_real_time_streaming=False,
        otel_exporter_endpoint="http://localhost:4317",
        langsmith_enable_streaming=False,
        enable_cryptographic_audit=False
    )


def create_production_config() -> ObservabilityConfig:
    """Create configuration optimized for production."""
    
    return ObservabilityConfig(
        deployment_environment="production",
        log_level="INFO",
        prometheus_enable_multiprocess=True,
        enable_real_time_streaming=True,
        langsmith_enable_streaming=True,
        enable_cryptographic_audit=True,
        enable_auto_recovery=True
    )
