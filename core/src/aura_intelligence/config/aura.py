"""
Main AURA Intelligence configuration.

Combines all configuration modules into a single settings class.
"""

from typing import Optional

from pydantic import Field

from .base import BaseSettings
from .agent import AgentSettings
from .memory import MemorySettings
from .api import APISettings
from .observability import ObservabilitySettings
from .integration import IntegrationSettings
from .security import SecuritySettings
from .deployment import DeploymentSettings


class AURASettings(BaseSettings):
    """
    Complete AURA Intelligence configuration.
    
    This is the main configuration class that combines all settings modules.
    All settings can be overridden via environment variables with the prefix AURA_.
    
    Example:
        AURA_ENVIRONMENT=production
        AURA_AGENT__AGENT_COUNT=10
        AURA_API__OPENAI_API_KEY=sk-...
    """
    
    # Sub-configurations
    agent: AgentSettings = Field(
        default_factory=AgentSettings,
        description="Agent configuration"
    )
    memory: MemorySettings = Field(
        default_factory=MemorySettings,
        description="Memory configuration"
    )
    api: APISettings = Field(
        default_factory=APISettings,
        description="API configuration"
    )
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings,
        description="Observability configuration"
    )
    integration: IntegrationSettings = Field(
        default_factory=IntegrationSettings,
        description="Integration configuration"
    )
    security: SecuritySettings = Field(
        default_factory=SecuritySettings,
        description="Security configuration"
    )
    deployment: DeploymentSettings = Field(
        default_factory=DeploymentSettings,
        description="Deployment configuration"
    )
    
    # Global settings
    service_name: str = Field(
        default="aura-intelligence",
        description="Service name for identification"
    )
    version: str = Field(
        default="1.0.0",
        description="Service version"
    )
    
    def validate_configuration(self) -> list[str]:
        """
        Validate the complete configuration.
        
        Returns:
            List of validation warnings or errors
        """
        warnings = []
        
        # Check API keys for production
        if self.is_production:
            if not self.api.has_openai_key and not self.api.has_anthropic_key:
                warnings.append("No AI API keys configured for production")
            
            if self.security.requires_auth_setup:
                warnings.append("Authentication not properly configured for production")
            
            if self.deployment.deployment_mode == "shadow":
                warnings.append("Shadow mode should not be used in production environment")
        
        # Check memory configuration
        if self.memory.requires_api_key and not self.api.pinecone_api_key:
            warnings.append(f"Vector store {self.memory.vector_store_type} requires API key")
        
        # Check observability
        if self.observability.enable_tracing and not self.observability.tracing_endpoint:
            warnings.append("Tracing enabled but no endpoint configured")
        
        return warnings
    
    def get_legacy_config(self) -> dict:
        """
        Get configuration in legacy format for backward compatibility.
        
        Returns:
            Dictionary with legacy configuration structure
        """
        return {
            "environment": self.environment.value,
            "agent_config": self.agent.get_agent_config_dict(),
            "memory_config": self.memory.model_dump(),
            "api_keys": {
                "openai": self.api.get_api_key("openai"),
                "anthropic": self.api.get_api_key("anthropic"),
                "google": self.api.get_api_key("google"),
                "pinecone": self.api.get_api_key("pinecone"),
            },
            "observability": self.observability.model_dump(),
            "security": {
                "enable_auth": self.security.enable_auth,
                "auth_provider": self.security.auth_provider,
            },
            "deployment": {
                "mode": self.deployment.deployment_mode,
                "shadow_enabled": self.deployment.shadow_enabled,
                "canary_enabled": self.deployment.canary_enabled,
            }
        }
    
    @classmethod
    def from_env(cls) -> "AURASettings":
        """
        Create settings from environment variables.
        
        This is the recommended way to create settings in production.
        """
        return cls()
    
    def print_configuration_summary(self) -> None:
        """Print a summary of the configuration."""
        print("🔧 AURA Intelligence Configuration Summary")
        print("=" * 50)
        print(f"Environment: {self.environment.value}")
        print(f"Service: {self.service_name} v{self.version}")
        print(f"Deployment Mode: {self.deployment.deployment_mode}")
        print(f"Agents: {self.agent.agent_count} ({self.agent.enhancement_level.value})")
        print(f"Memory: {self.memory.vector_store_type}")
        print(f"Observability: Metrics={self.observability.enable_metrics}, Tracing={self.observability.enable_tracing}")
        print(f"Security: Auth={self.security.enable_auth} ({self.security.auth_provider})")
        
        warnings = self.validate_configuration()
        if warnings:
            print("\n⚠️  Configuration Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\n✅ Configuration validated successfully")
        print("=" * 50)