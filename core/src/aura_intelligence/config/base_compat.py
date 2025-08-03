"""
Compatibility layer for BaseSettings to support both old and new config systems.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import os


class BaseSettings:
    """
    Compatibility class that mimics Pydantic BaseSettings interface
    but uses our dataclass-based configuration internally.
    """
    
    model_config = {"extra": "allow"}  # Mimics Pydantic config
    
    def __init__(self, **kwargs):
        """Initialize with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, obj: Dict[str, Any]):
        """Mimics Pydantic's model_validate."""
        return cls(**obj)
    
    def model_dump(self) -> Dict[str, Any]:
        """Mimics Pydantic's model_dump."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def dict(self) -> Dict[str, Any]:
        """Backward compatibility for dict() method."""
        return self.model_dump()


# Re-export the enums that other modules expect
class EnvironmentType:
    """Environment type enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class EnhancementLevel:
    """Enhancement level enumeration."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


# Import the actual configuration from base.py
from .base import get_config, AURAConfig