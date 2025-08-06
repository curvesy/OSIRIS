"""
⚙️ Configuration Manager
Singleton pattern for global configuration management.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import json
from threading import Lock

from .base import AuraConfig
from .loaders import ConfigLoader, EnvConfigLoader, FileConfigLoader, CompositeLoader
from .validators import validate_config
from ..logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """
    Thread-safe singleton configuration manager.
    
    Features:
    - Multiple config sources (env, files, defaults)
    - Hot reload support
    - Validation on load
    - Feature flag management
    """
    
    _instance: Optional['ConfigManager'] = None
    _lock: Lock = Lock()
    _config: Optional[AuraConfig] = None
    _loaders: List[ConfigLoader] = []
    
    def __new__(cls) -> 'ConfigManager':
        """Ensure single instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize manager (only runs once)."""
        if not self._initialized:
            self._initialized = True
            self._config = None
            self._loaders = []
            self._config_sources: List[Path] = []
            self._reload_callbacks: List[callable] = []
            
            # Set up default loaders
            self._setup_default_loaders()
    
    def _setup_default_loaders(self) -> None:
        """Set up default configuration loaders."""
        # Environment variables (highest priority)
        self._loaders.append(EnvConfigLoader(prefix="AURA_"))
        
        # Configuration files
        config_paths = [
            Path.home() / ".aura" / "config.json",
            Path("/etc/aura/config.json"),
            Path("./config.json"),
            Path("./aura.config.json"),
        ]
        
        for path in config_paths:
            if path.exists():
                self._loaders.append(FileConfigLoader(path))
                self._config_sources.append(path)
                logger.info(f"Found config file", path=str(path))
    
    def add_loader(self, loader: ConfigLoader) -> None:
        """Add a custom configuration loader."""
        self._loaders.append(loader)
        logger.info(f"Added config loader", loader_type=type(loader).__name__)
    
    def load(self, force_reload: bool = False) -> AuraConfig:
        """
        Load configuration from all sources.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded and validated configuration
        """
        if self._config is not None and not force_reload:
            return self._config
        
        with self._lock:
            logger.info("Loading configuration", sources=len(self._loaders))
            
            # Start with defaults
            config_dict = {}
            
            # Apply each loader in order
            for loader in self._loaders:
                try:
                    loader_config = loader.load()
                    config_dict = self._deep_merge(config_dict, loader_config)
                    logger.debug(
                        "Applied config loader",
                        loader=type(loader).__name__,
                        keys=list(loader_config.keys())
                    )
                except Exception as e:
                    logger.error(
                        "Failed to load config",
                        loader=type(loader).__name__,
                        error=str(e)
                    )
            
            # Create config object
            try:
                self._config = AuraConfig(**config_dict)
                
                # Validate
                validate_config(self._config)
                
                logger.info(
                    "Configuration loaded successfully",
                    environment=self._config.environment,
                    service_name=self._config.service_name,
                    feature_flags=list(self._config.feature_flags.keys())
                )
                
                # Notify callbacks
                for callback in self._reload_callbacks:
                    try:
                        callback(self._config)
                    except Exception as e:
                        logger.error("Reload callback failed", error=str(e))
                
                return self._config
                
            except Exception as e:
                logger.error("Failed to create config", error=str(e))
                raise
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self) -> AuraConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load()
        return self._config
    
    def reload(self) -> AuraConfig:
        """Force reload configuration from all sources."""
        return self.load(force_reload=True)
    
    def on_reload(self, callback: callable) -> None:
        """Register callback for configuration reload."""
        self._reload_callbacks.append(callback)
    
    def update_feature_flag(self, flag: str, enabled: bool) -> None:
        """
        Update a feature flag value.
        
        Args:
            flag: Feature flag name
            enabled: Whether to enable or disable
        """
        if self._config is None:
            self.load()
        
        self._config.feature_flags[flag] = enabled
        logger.info(
            "Feature flag updated",
            flag=flag,
            enabled=enabled
        )
    
    def is_feature_enabled(self, flag: str) -> bool:
        """Check if a feature flag is enabled."""
        if self._config is None:
            self.load()
        return self._config.get_feature(flag)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        if self._config is None:
            self.load()
        return self._config.model_dump()
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """
        Export configuration as JSON.
        
        Args:
            path: Optional path to save JSON
            
        Returns:
            JSON string
        """
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, default=str)
        
        if path:
            path.write_text(json_str)
            logger.info("Configuration saved", path=str(path))
        
        return json_str


# Global instance
_config_manager = ConfigManager()


def get_config() -> AuraConfig:
    """
    Get global configuration instance.
    
    Returns:
        Current configuration
        
    Example:
        ```python
        from aura_common.config import get_config
        
        config = get_config()
        if config.is_production():
            # Production-specific logic
            pass
        ```
    """
    return _config_manager.get()


def reload_config() -> AuraConfig:
    """Force reload configuration."""
    return _config_manager.reload()


def update_feature_flag(flag: str, enabled: bool) -> None:
    """Update a feature flag."""
    _config_manager.update_feature_flag(flag, enabled)


def is_feature_enabled(flag: str) -> bool:
    """Check if a feature is enabled."""
    return _config_manager.is_feature_enabled(flag)


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value by key."""
    config = get_config()
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