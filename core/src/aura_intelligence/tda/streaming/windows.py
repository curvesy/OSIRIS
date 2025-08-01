"""
🪟 Sliding Window Data Structures for Streaming TDA
Memory-efficient circular buffers with multi-scale support
"""

import threading
from typing import Optional, Dict, List
import numpy as np
from dataclasses import dataclass
import psutil
import structlog

from prometheus_client import Gauge, Histogram

logger = structlog.get_logger(__name__)

# Metrics
WINDOW_SIZE = Gauge('tda_window_size', 'Current window size', ['scale'])
WINDOW_MEMORY = Gauge('tda_window_memory_bytes', 'Memory usage per window', ['scale'])
SLIDE_LATENCY = Histogram('tda_window_slide_latency_seconds', 'Window slide operation latency')


@dataclass
class WindowStats:
    """Statistics for a sliding window"""
    total_points_seen: int = 0
    total_slides: int = 0
    current_size: int = 0
    memory_bytes: int = 0


class CircularBuffer:
    """Thread-safe circular buffer for numpy arrays"""
    
    def __init__(self, capacity: int, dim: int):
        self.capacity = capacity
        self.dim = dim
        self.buffer = np.zeros((capacity, dim), dtype=np.float32)
        self.head = 0
        self.size = 0
        self.lock = threading.RLock()
        
    def add(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Add single point, return evicted point if buffer is full"""
        with self.lock:
            evicted = None
            if self.size == self.capacity:
                evicted = self.buffer[self.head].copy()
                
            self.buffer[self.head] = point
            self.head = (self.head + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            
            return evicted
            
    def add_batch(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Add batch of points efficiently"""
        with self.lock:
            n_points = len(points)
            evicted = None
            
            if n_points >= self.capacity:
                # Batch larger than buffer - keep only last capacity points
                evicted = self.get_all()
                self.buffer[:] = points[-self.capacity:]
                self.head = 0
                self.size = self.capacity
            else:
                # Calculate evictions
                n_evict = max(0, self.size + n_points - self.capacity)
                if n_evict > 0:
                    evicted = np.zeros((n_evict, self.dim), dtype=np.float32)
                    for i in range(n_evict):
                        idx = (self.head - self.size + i) % self.capacity
                        evicted[i] = self.buffer[idx]
                
                # Add new points
                for point in points:
                    self.buffer[self.head] = point
                    self.head = (self.head + 1) % self.capacity
                    
                self.size = min(self.size + n_points, self.capacity)
                
            return evicted
            
    def get_all(self) -> np.ndarray:
        """Get all points in order"""
        with self.lock:
            if self.size == 0:
                return np.array([], dtype=np.float32).reshape(0, self.dim)
                
            if self.size < self.capacity:
                return self.buffer[:self.size].copy()
            else:
                # Reorder circular buffer
                return np.concatenate([
                    self.buffer[self.head:],
                    self.buffer[:self.head]
                ])
                
    def clear(self) -> None:
        """Clear buffer"""
        with self.lock:
            self.head = 0
            self.size = 0


class SlidingWindow:
    """Sliding window with configurable slide size"""
    
    def __init__(self, capacity: int, slide_size: int, dim: int = 3):
        self.capacity = capacity
        self.slide_size = slide_size
        self.dim = dim
        self.buffer = CircularBuffer(capacity, dim)
        self.stats = WindowStats()
        self._update_metrics()
        
    def add_point(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Add point and return evicted point if window slides"""
        evicted = self.buffer.add(point)
        self.stats.total_points_seen += 1
        self.stats.current_size = self.buffer.size
        
        # Check if we need to slide
        if self.buffer.size == self.capacity and \
           self.stats.total_points_seen % self.slide_size == 0:
            with SLIDE_LATENCY.time():
                slide_evicted = self.slide(self.slide_size)
                if evicted is None:
                    evicted = slide_evicted
                else:
                    evicted = np.vstack([evicted, slide_evicted])
                    
        self._update_metrics()
        return evicted
        
    def add_batch(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Add batch of points"""
        evicted = self.buffer.add_batch(points)
        self.stats.total_points_seen += len(points)
        self.stats.current_size = self.buffer.size
        self._update_metrics()
        return evicted
        
    def get_points(self) -> np.ndarray:
        """Get all points in window"""
        return self.buffer.get_all()
        
    def slide(self, n_points: int) -> np.ndarray:
        """Manually slide window by n points"""
        points = self.get_points()
        if len(points) <= n_points:
            self.buffer.clear()
            return points
            
        evicted = points[:n_points]
        remaining = points[n_points:]
        
        self.buffer.clear()
        self.buffer.add_batch(remaining)
        self.stats.total_slides += 1
        self.stats.current_size = self.buffer.size
        
        return evicted
        
    @property
    def size(self) -> int:
        """Current number of points"""
        return self.buffer.size
        
    def _update_metrics(self) -> None:
        """Update Prometheus metrics"""
        WINDOW_SIZE.labels(scale=str(self.capacity)).set(self.size)
        
        # Estimate memory usage
        memory_bytes = (
            self.buffer.buffer.nbytes +  # Main buffer
            self.buffer.buffer.itemsize * 100  # Overhead estimate
        )
        self.stats.memory_bytes = memory_bytes
        WINDOW_MEMORY.labels(scale=str(self.capacity)).set(memory_bytes)


class MultiScaleWindows:
    """Manage multiple windows at different time scales"""
    
    def __init__(self, scales: List[int], base_slide_ratio: float = 0.1, dim: int = 3):
        """
        Args:
            scales: List of window sizes (e.g., [100, 1000, 10000])
            base_slide_ratio: Slide size as ratio of window size
            dim: Dimension of points
        """
        self.scales = sorted(scales)
        self.windows: Dict[int, SlidingWindow] = {}
        
        for scale in scales:
            slide_size = max(1, int(scale * base_slide_ratio))
            self.windows[scale] = SlidingWindow(scale, slide_size, dim)
            
        logger.info("Multi-scale windows initialized", scales=scales)
        
    def add_point(self, point: np.ndarray) -> Dict[int, Optional[np.ndarray]]:
        """Add point to all windows, return evictions per scale"""
        evictions = {}
        for scale, window in self.windows.items():
            evictions[scale] = window.add_point(point)
        return evictions
        
    def add_batch(self, points: np.ndarray) -> Dict[int, Optional[np.ndarray]]:
        """Add batch to all windows"""
        evictions = {}
        for scale, window in self.windows.items():
            evictions[scale] = window.add_batch(points)
        return evictions
        
    def get_windows_data(self) -> Dict[int, np.ndarray]:
        """Get data from all windows"""
        return {scale: window.get_points() for scale, window in self.windows.items()}
        
    def get_statistics(self) -> Dict[int, WindowStats]:
        """Get statistics for all windows"""
        return {scale: window.stats for scale, window in self.windows.items()}
        
    def get_total_memory_usage(self) -> int:
        """Get total memory usage across all windows"""
        return sum(window.stats.memory_bytes for window in self.windows.values())