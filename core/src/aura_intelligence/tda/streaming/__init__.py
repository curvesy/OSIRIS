"""
🌊 Streaming Multi-Scale TDA Module
Real-time topological data analysis with incremental updates
"""

from typing import Protocol, AsyncIterator, Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from ..models import PersistenceDiagram, TDAMetrics


@dataclass
class TDAStatistics:
    """Statistics for streaming TDA processing"""
    points_processed: int
    current_diagram_size: int
    processing_rate: float  # points per second
    memory_usage_mb: float
    last_update: datetime
    
    
@dataclass
class DiagramUpdate:
    """Represents changes to persistence diagram"""
    added_features: List[tuple]  # New persistent features
    removed_features: List[tuple]  # Features that died
    modified_features: List[tuple]  # Features with changed persistence
    timestamp: datetime
    

@dataclass 
class MultiScaleResult:
    """Results from multi-scale TDA processing"""
    scale: int
    diagram: PersistenceDiagram
    statistics: TDAStatistics
    topological_summary: Dict[str, Any]


class StreamingTDAProcessor(Protocol):
    """Protocol for streaming TDA processors"""
    
    async def process_point(self, point: np.ndarray) -> Optional[DiagramUpdate]:
        """Process single point and return diagram changes"""
        ...
    
    async def process_batch(self, points: np.ndarray) -> Optional[DiagramUpdate]:
        """Process batch of points efficiently"""
        ...
        
    async def get_current_diagram(self) -> PersistenceDiagram:
        """Get current persistence diagram"""
        ...
        
    async def get_statistics(self) -> TDAStatistics:
        """Get processing statistics"""
        ...
        
    async def reset(self) -> None:
        """Reset processor state"""
        ...


class DataWindow(Protocol):
    """Protocol for sliding window data structures"""
    
    def add_point(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Add point and return evicted point if window is full"""
        ...
        
    def add_batch(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Add batch and return evicted points"""
        ...
        
    def get_points(self) -> np.ndarray:
        """Get all points in window"""
        ...
        
    def slide(self, n_points: int) -> np.ndarray:
        """Slide window by n points, return evicted"""
        ...
        
    @property
    def size(self) -> int:
        """Current number of points in window"""
        ...
        
    @property
    def capacity(self) -> int:
        """Maximum window capacity"""
        ...


class StreamAdapter(Protocol):
    """Protocol for adapting various data sources to TDA streams"""
    
    async def to_point_stream(
        self, 
        source: AsyncIterator[Any]
    ) -> AsyncIterator[np.ndarray]:
        """Convert source data to point stream"""
        ...


# Re-export key components
from .windows import SlidingWindow, MultiScaleWindows
from .incremental_persistence import IncrementalPersistence
from .multi_scale import MultiScaleProcessor
from .adapters import KafkaToTDAAdapter, WebSocketToTDAAdapter

__all__ = [
    # Protocols
    'StreamingTDAProcessor',
    'DataWindow', 
    'StreamAdapter',
    
    # Data classes
    'TDAStatistics',
    'DiagramUpdate',
    'MultiScaleResult',
    
    # Implementations
    'SlidingWindow',
    'MultiScaleWindows',
    'IncrementalPersistence',
    'MultiScaleProcessor',
    'KafkaToTDAAdapter',
    'WebSocketToTDAAdapter'
]