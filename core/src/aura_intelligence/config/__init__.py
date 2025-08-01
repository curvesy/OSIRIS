"""
🔧 AURA Intelligence Configuration Module

Modular, type-safe configuration system using Pydantic v2.

IMPORTANT: This module maintains backward compatibility with legacy naming conventions.
The *Settings classes are the modern implementation, while *Config aliases are provided
for backward compatibility with existing code and tests.
"""

from .base import BaseSettings, EnvironmentType, EnhancementLevel
from .agent import AgentSettings
from .memory import MemorySettings
from .api import APISettings
from .observability import ObservabilitySettings
from .integration import IntegrationSettings
from .security import SecuritySettings
from .deployment import DeploymentSettings
from .aura import AURASettings

# Backward compatibility aliases
# These map the modern *Settings classes to the legacy *Config names
# TODO: Migrate all code to use *Settings naming convention
UltimateAURAConfig = AURASettings
AgentConfig = AgentSettings
MemoryConfig = MemorySettings
KnowledgeConfig = AURASettings  # Knowledge is part of main AURA settings
TopologyConfig = AURASettings  # Topology is part of main AURA settings

# Factory functions for configuration
def get_ultimate_config() -> AURASettings:
    """
    Get the ultimate AURA configuration with default settings.
    
    Returns:
        AURASettings: Default ultimate configuration
    """
    config = AURASettings()
    # Set ultimate defaults
    config.agent.enhancement_level = EnhancementLevel.ULTIMATE
    config.agent.agent_count = 10
    config.memory.enable_consciousness = True
    return config

def get_production_config() -> AURASettings:
    """
    Get production-ready AURA configuration.
    
    Returns:
        AURASettings: Production configuration
    """
    config = AURASettings(environment=EnvironmentType.PRODUCTION)
    # Production defaults
    config.agent.enhancement_level = EnhancementLevel.ADVANCED
    config.agent.agent_count = 5
    config.security.enable_auth = True
    config.observability.enable_metrics = True
    config.observability.enable_tracing = True
    config.deployment.deployment_mode = "production"
    return config

def get_enterprise_config() -> AURASettings:
    """
    Get enterprise AURA configuration with all features enabled.
    
    Returns:
        AURASettings: Enterprise configuration
    """
    config = AURASettings(environment=EnvironmentType.PRODUCTION)
    # Enterprise features
    config.agent.enhancement_level = EnhancementLevel.ULTIMATE
    config.agent.agent_count = 20
    config.memory.enable_consciousness = True
    config.memory.enable_mem0 = True
    config.security.enable_auth = True
    config.security.enable_encryption = True
    config.observability.enable_metrics = True
    config.observability.enable_tracing = True
    config.observability.enable_profiling = True
    config.deployment.deployment_mode = "multi_region"
    config.deployment.canary_enabled = True
    return config

def get_development_config() -> AURASettings:
    """
    Get development AURA configuration for local testing.
    
    Returns:
        AURASettings: Development configuration
    """
    config = AURASettings(environment=EnvironmentType.DEVELOPMENT)
    # Development defaults
    config.agent.enhancement_level = EnhancementLevel.BASIC
    config.agent.agent_count = 3
    config.security.enable_auth = False
    config.observability.enable_metrics = True
    config.observability.enable_tracing = False
    return config

__all__ = [
    # Base classes
    "BaseSettings",
    "EnvironmentType",
    "EnhancementLevel",
    
    # Modern Settings classes
    "AgentSettings",
    "MemorySettings",
    "APISettings",
    "ObservabilitySettings",
    "IntegrationSettings",
    "SecuritySettings",
    "DeploymentSettings",
    "AURASettings",
    
    # Backward compatibility aliases
    "UltimateAURAConfig",
    "AgentConfig",
    "MemoryConfig",
    "KnowledgeConfig",
    "TopologyConfig",
    
    # Factory functions
    "get_ultimate_config",
    "get_production_config",
    "get_enterprise_config",
    "get_development_config",
]