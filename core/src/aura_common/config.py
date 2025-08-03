"""
Configuration utilities for AURA Intelligence
"""

import os
from typing import Dict, Any, Optional
from ..aura_intelligence.config.base import get_config as get_aura_config


def is_feature_enabled(feature: str) -> bool:
    """
    Check if a feature flag is enabled.
    
    Args:
        feature: Feature name (e.g., "tda.gpu_acceleration")
        
    Returns:
        True if feature is enabled, False otherwise
    """
    # First check environment variable
    env_key = f"FEATURE_{feature.upper().replace('.', '_')}"
    env_value = os.getenv(env_key)
    
    if env_value is not None:
        return env_value.lower() in ('true', '1', 'yes', 'on')
    
    # Then check configuration
    try:
        config = get_aura_config()
        return config.feature_flags.get(feature, False)
    except:
        # Default to False if configuration not available
        return False


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value by key.
    
    Args:
        key: Configuration key (dot notation supported)
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    try:
        config = get_aura_config()
        
        # Support dot notation
        parts = key.split('.')
        value = config
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    except:
        return default


class FeatureFlagManager:
    """Manage feature flags for AURA Intelligence."""
    
    def __init__(self):
        self._overrides: Dict[str, bool] = {}
    
    def is_enabled(self, feature: str) -> bool:
        """Check if feature is enabled."""
        # Check overrides first
        if feature in self._overrides:
            return self._overrides[feature]
            
        return is_feature_enabled(feature)
    
    def enable(self, feature: str):
        """Enable a feature flag."""
        self._overrides[feature] = True
    
    def disable(self, feature: str):
        """Disable a feature flag."""
        self._overrides[feature] = False
    
    def reset(self, feature: str):
        """Reset feature flag to default."""
        self._overrides.pop(feature, None)
    
    def reset_all(self):
        """Reset all feature flag overrides."""
        self._overrides.clear()


# Global feature flag manager instance
feature_flags = FeatureFlagManager()