"""
🔧 AURA Intelligence Configuration (Legacy Compatibility)

This module provides backward compatibility for the old configuration system.
New code should use the modular configuration in aura_intelligence.config package.
"""

import warnings
from typing import Dict, Any, Optional

# Import from new modular configuration
from aura_intelligence.config import (
    AURASettings,
    EnvironmentType,
    EnhancementLevel,
    AgentSettings,
    MemorySettings,
    APISettings,
    ObservabilitySettings,
    IntegrationSettings,
    SecuritySettings,
    DeploymentSettings,
)

# Warn about deprecated usage
warnings.warn(
    "Direct import from aura_intelligence.config is deprecated. "
    "Use aura_intelligence.config.* modules instead.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy aliases for backward compatibility
AgentConfig = AgentSettings
MemoryConfig = MemorySettings
APIConfig = APISettings
ObservabilityConfig = ObservabilitySettings
IntegrationConfig = IntegrationSettings
SecurityConfig = SecuritySettings
DeploymentConfig = DeploymentSettings


class Config:
    """Legacy configuration class for backward compatibility."""
    
    def __init__(self):
        """Initialize with new settings system."""
        self._settings = AURASettings.from_env()
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration in legacy format."""
        return self._settings.get_legacy_config()
    
    @property
    def environment(self) -> EnvironmentType:
        """Get environment type."""
        return self._settings.environment
    
    @property
    def agent_config(self) -> AgentSettings:
        """Get agent configuration."""
        return self._settings.agent
    
    @property
    def memory_config(self) -> MemorySettings:
        """Get memory configuration."""
        return self._settings.memory
    
    @property
    def api_config(self) -> APISettings:
        """Get API configuration."""
        return self._settings.api
    
    @property
    def observability_config(self) -> ObservabilitySettings:
        """Get observability configuration."""
        return self._settings.observability
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self._settings.is_production
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self._settings.is_development


# Legacy function for backward compatibility
def get_config() -> Dict[str, Any]:
    """Get configuration in legacy format."""
    warnings.warn(
        "get_config() is deprecated. Use AURASettings.from_env() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return AURASettings.from_env().get_legacy_config()
