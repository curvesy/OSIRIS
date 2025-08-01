"""
⚙️ Configuration Loaders
Load configuration from various sources.
"""

from typing import Dict, Any, Optional, List, Protocol
from pathlib import Path
import os
import json
import yaml
from abc import ABC, abstractmethod

from ..errors import ConfigurationError


class ConfigLoader(Protocol):
    """Protocol for configuration loaders."""
    
    def load(self) -> Dict[str, Any]:
        """Load configuration data."""
        ...


class EnvConfigLoader:
    """
    Load configuration from environment variables.
    
    Features:
    - Prefix-based filtering
    - Nested key support (AURA_TDA_ENGINE -> tda.engine)
    - Type conversion
    """
    
    def __init__(self, prefix: str = "AURA_"):
        """
        Initialize environment loader.
        
        Args:
            prefix: Environment variable prefix
        """
        self.prefix = prefix
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from environment."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.prefix):].lower()
                
                # Convert underscores to dots for nesting
                # AURA_TDA_ENGINE -> tda.engine
                config_key = config_key.replace('_', '.')
                
                # Convert value
                converted_value = self._convert_value(value)
                
                # Set nested value
                self._set_nested(config, config_key, converted_value)
        
        return config
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String
        return value
    
    def _set_nested(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested configuration value."""
        parts = key.split('.')
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value


class FileConfigLoader:
    """
    Load configuration from file.
    
    Supports:
    - JSON files
    - YAML files
    - Automatic format detection
    """
    
    def __init__(self, path: Path):
        """
        Initialize file loader.
        
        Args:
            path: Configuration file path
        """
        self.path = Path(path)
        
        if not self.path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {path}",
                config_file=str(path)
            )
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            content = self.path.read_text()
            
            # Detect format by extension
            if self.path.suffix in ('.json', '.jsonc'):
                return self._load_json(content)
            elif self.path.suffix in ('.yaml', '.yml'):
                return self._load_yaml(content)
            else:
                # Try JSON first, then YAML
                try:
                    return self._load_json(content)
                except Exception:
                    return self._load_yaml(content)
                    
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {self.path}",
                config_file=str(self.path),
                cause=e
            )
    
    def _load_json(self, content: str) -> Dict[str, Any]:
        """Load JSON configuration."""
        # Remove comments for JSONC
        lines = []
        for line in content.split('\n'):
            # Remove single-line comments
            comment_pos = line.find('//')
            if comment_pos >= 0:
                line = line[:comment_pos]
            lines.append(line)
        
        cleaned_content = '\n'.join(lines)
        return json.loads(cleaned_content)
    
    def _load_yaml(self, content: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            import yaml
            return yaml.safe_load(content) or {}
        except ImportError:
            raise ConfigurationError(
                "YAML support requires PyYAML package",
                suggestions=["pip install pyyaml"]
            )


class DictConfigLoader:
    """Load configuration from dictionary."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with dictionary.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def load(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return self.config.copy()


class CompositeLoader:
    """
    Composite loader that combines multiple loaders.
    
    Features:
    - Load from multiple sources
    - Priority-based merging
    - Error handling per loader
    """
    
    def __init__(self, loaders: List[ConfigLoader]):
        """
        Initialize with list of loaders.
        
        Args:
            loaders: List of loaders (applied in order)
        """
        self.loaders = loaders
    
    def load(self) -> Dict[str, Any]:
        """Load and merge from all loaders."""
        config = {}
        
        for loader in self.loaders:
            try:
                loader_config = loader.load()
                config = self._deep_merge(config, loader_config)
            except Exception as e:
                # Log but continue with other loaders
                from ..logging import get_logger
                logger = get_logger(__name__)
                logger.warning(
                    "Loader failed, continuing",
                    loader=type(loader).__name__,
                    error=str(e)
                )
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class RemoteConfigLoader:
    """
    Load configuration from remote source.
    
    Features:
    - HTTP/HTTPS support
    - Authentication
    - Caching
    - Retry logic
    """
    
    def __init__(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        cache_ttl: Optional[int] = None
    ):
        """
        Initialize remote loader.
        
        Args:
            url: Configuration URL
            headers: Optional HTTP headers
            timeout: Request timeout
            cache_ttl: Cache TTL in seconds
        """
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: Optional[float] = None
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from remote source."""
        import time
        import requests
        
        # Check cache
        if self._cache is not None and self.cache_ttl:
            if time.time() - self._cache_time < self.cache_ttl:
                return self._cache.copy()
        
        try:
            response = requests.get(
                self.url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            config = response.json()
            
            # Update cache
            self._cache = config
            self._cache_time = time.time()
            
            return config
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load remote configuration: {self.url}",
                endpoint=self.url,
                cause=e
            )