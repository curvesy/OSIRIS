"""
⚙️ AURA Configuration Management
Type-safe configuration with validation and feature flags.
"""

from .base import AuraConfig, ConfigSection
from .loaders import ConfigLoader, EnvConfigLoader, FileConfigLoader
from .validators import ConfigValidator, validate_config
from .feature_flags import FeatureFlag, FeatureManager, is_feature_enabled
from .manager import ConfigManager, get_config, get_config_value

__all__ = [
    # Base
    "AuraConfig",
    "ConfigSection",
    
    # Loaders
    "ConfigLoader",
    "EnvConfigLoader",
    "FileConfigLoader",
    
    # Validators
    "ConfigValidator",
    "validate_config",
    
    # Feature Flags
    "FeatureFlag",
    "FeatureManager",
    "is_feature_enabled",
    
    # Manager
    "ConfigManager",
    "get_config",
    "get_config_value",
]