"""
Base atomic component for AURA Intelligence
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field

from ..logging import get_logger

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ComponentMetadata:
    """Metadata for atomic components."""
    component_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    component_type: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)


class AtomicComponent(ABC, Generic[T, R]):
    """
    Base class for all atomic components in AURA Intelligence.
    
    Atomic components are:
    - Single responsibility
    - Independently testable
    - Composable
    - Observable
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize atomic component.
        
        Args:
            name: Component name
            config: Component configuration
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.metadata = ComponentMetadata(component_type=self.__class__.__name__)
        self.logger = get_logger(self.name)
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the component."""
        if self._initialized:
            return
            
        self.logger.info(f"Initializing {self.name}")
        await self._initialize()
        self._initialized = True
        self.logger.info(f"Initialized {self.name}")
        
    async def _initialize(self) -> None:
        """Component-specific initialization. Override in subclasses."""
        pass
        
    @abstractmethod
    async def process(self, input_data: T) -> R:
        """
        Process input data and return result.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed result
        """
        pass
        
    async def cleanup(self) -> None:
        """Clean up component resources."""
        if not self._initialized:
            return
            
        self.logger.info(f"Cleaning up {self.name}")
        await self._cleanup()
        self._initialized = False
        self.logger.info(f"Cleaned up {self.name}")
        
    async def _cleanup(self) -> None:
        """Component-specific cleanup. Override in subclasses."""
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata."""
        return {
            "component_id": self.metadata.component_id,
            "component_type": self.metadata.component_type,
            "version": self.metadata.version,
            "created_at": self.metadata.created_at.isoformat(),
            "tags": self.metadata.tags,
            "name": self.name,
            "initialized": self._initialized
        }
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', id='{self.metadata.component_id}')"