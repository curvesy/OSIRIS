"""
🔄 Incremental Persistence Diagram Updates
Efficient algorithms for real-time persistence computation in streaming settings
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass, field
import heapq
from collections import defaultdict
import time

import structlog
from prometheus_client import Counter, Histogram, Gauge

from ..models import PersistenceDiagram, PersistenceFeature
from .windows import StreamingWindow
from ...observability.tracing import get_tracer

logger = structlog.get_logger(__name__)

# Metrics
UPDATES_PROCESSED = Counter('tda_incremental_updates_total', 'Total incremental updates', ['algorithm'])
UPDATE_LATENCY = Histogram('tda_incremental_update_latency_seconds', 'Update latency', ['algorithm'])
FEATURES_TRACKED = Gauge('tda_features_tracked', 'Number of features being tracked', ['algorithm', 'dimension'])
VINEYARD_SIZE = Gauge('tda_vineyard_size', 'Size of vineyard structure')


@dataclass
class SimplexNode:
    """Node in the simplex tree for incremental updates"""
    vertices: Tuple[int, ...]
    filtration_value: float
    dimension: int
    birth_time: Optional[float] = None
    death_time: Optional[float] = None
    children: Dict[int, 'SimplexNode'] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.vertices)
    
    def __eq__(self, other):
        return self.vertices == other.vertices


class IncrementalVineyardProcessor:
    """
    Vineyard-based incremental persistence computation
    Maintains persistence across sliding window updates
    """
    
    def __init__(self, window_size: int, max_dimension: int = 2):
        self.window_size = window_size
        self.max_dimension = max_dimension
        self.current_diagram: Optional[PersistenceDiagram] = None
        self.vineyard: List[PersistenceDiagram] = []
        self.simplex_tree: Dict[Tuple[int, ...], SimplexNode] = {}
        self.point_indices: Dict[int, int] = {}  # Maps point ID to window index
        self.next_point_id = 0
        self.tracer = get_tracer()
        
    async def add_point(self, window: StreamingWindow, point: np.ndarray) -> PersistenceDiagram:
        """Add a point and update persistence incrementally"""
        async with self.tracer.trace_async_operation(
            "incremental_vineyard_add",
            window_size=self.window_size
        ):
            start_time = time.time()
            
            # Get point ID and add to window
            point_id = self.next_point_id
            self.next_point_id += 1
            
            # Add point to window
            old_point_id = await window.add_point(point)
            
            # Update point indices
            current_index = window.current_index
            self.point_indices[point_id] = current_index
            
            # Remove old point if window is full
            if old_point_id is not None and old_point_id in self.point_indices:
                await self._remove_point(old_point_id)
            
            # Add new simplices
            await self._add_simplices(point_id, point, window)
            
            # Update persistence
            diagram = await self._update_persistence()
            
            # Update metrics
            UPDATE_LATENCY.labels(algorithm="vineyard").observe(time.time() - start_time)
            UPDATES_PROCESSED.labels(algorithm="vineyard").inc()
            VINEYARD_SIZE.set(len(self.vineyard))
            
            return diagram
    
    async def _remove_point(self, point_id: int):
        """Remove a point and its associated simplices"""
        # Find all simplices containing this point
        to_remove = []
        for vertices, node in self.simplex_tree.items():
            if point_id in vertices:
                to_remove.append(vertices)
        
        # Remove simplices
        for vertices in to_remove:
            del self.simplex_tree[vertices]
        
        # Clean up point index
        if point_id in self.point_indices:
            del self.point_indices[point_id]
    
    async def _add_simplices(self, point_id: int, point: np.ndarray, window: StreamingWindow):
        """Add simplices for the new point"""
        # Get all points in window
        points = window.get_all_points()
        point_ids = list(self.point_indices.keys())
        
        # Add 0-simplex (vertex)
        self.simplex_tree[(point_id,)] = SimplexNode(
            vertices=(point_id,),
            filtration_value=0.0,
            dimension=0
        )
        
        # Add higher dimensional simplices
        for i, other_id in enumerate(point_ids[:-1]):  # Exclude the new point
            if other_id == point_id:
                continue
                
            # Calculate distance for 1-simplex (edge)
            other_point = points[self.point_indices[other_id]]
            distance = np.linalg.norm(point - other_point)
            
            # Add edge
            edge = tuple(sorted([point_id, other_id]))
            self.simplex_tree[edge] = SimplexNode(
                vertices=edge,
                filtration_value=distance,
                dimension=1
            )
            
            # Add higher dimensional simplices if within max_dimension
            if self.max_dimension >= 2:
                await self._add_higher_simplices(point_id, other_id, points)
    
    async def _add_higher_simplices(self, new_id: int, other_id: int, points: np.ndarray):
        """Add 2-simplices and higher"""
        # Find common neighbors to form triangles
        new_neighbors = self._get_neighbors(new_id)
        other_neighbors = self._get_neighbors(other_id)
        common_neighbors = new_neighbors.intersection(other_neighbors)
        
        for neighbor_id in common_neighbors:
            # Form 2-simplex (triangle)
            triangle = tuple(sorted([new_id, other_id, neighbor_id]))
            
            # Calculate filtration value (max edge length)
            edges = [
                (new_id, other_id),
                (new_id, neighbor_id),
                (other_id, neighbor_id)
            ]
            
            max_distance = 0.0
            for v1, v2 in edges:
                p1 = points[self.point_indices[v1]]
                p2 = points[self.point_indices[v2]]
                distance = np.linalg.norm(p1 - p2)
                max_distance = max(max_distance, distance)
            
            self.simplex_tree[triangle] = SimplexNode(
                vertices=triangle,
                filtration_value=max_distance,
                dimension=2
            )
    
    def _get_neighbors(self, point_id: int) -> Set[int]:
        """Get all neighbors of a point (connected by edges)"""
        neighbors = set()
        for vertices in self.simplex_tree:
            if len(vertices) == 2 and point_id in vertices:
                for v in vertices:
                    if v != point_id:
                        neighbors.add(v)
        return neighbors
    
    async def _update_persistence(self) -> PersistenceDiagram:
        """Update persistence diagram using vineyard algorithm"""
        # Sort simplices by filtration value and dimension
        sorted_simplices = sorted(
            self.simplex_tree.values(),
            key=lambda s: (s.filtration_value, s.dimension)
        )
        
        # Union-find for connected components
        parent = {}
        birth_time = {}
        features = []
        
        def find(x):
            if x not in parent:
                parent[x] = x
                return x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y, time):
            px, py = find(x), find(y)
            if px != py:
                # The younger component dies
                if birth_time.get(px, 0) < birth_time.get(py, 0):
                    parent[py] = px
                    # Record death of py
                    features.append(PersistenceFeature(
                        dimension=0,
                        birth=birth_time[py],
                        death=time,
                        persistence=time - birth_time[py]
                    ))
                else:
                    parent[px] = py
                    # Record death of px
                    features.append(PersistenceFeature(
                        dimension=0,
                        birth=birth_time[px],
                        death=time,
                        persistence=time - birth_time[px]
                    ))
        
        # Process simplices
        for simplex in sorted_simplices:
            if simplex.dimension == 0:
                # Birth of connected component
                v = simplex.vertices[0]
                if v not in birth_time:
                    birth_time[v] = simplex.filtration_value
            elif simplex.dimension == 1:
                # Edge - may kill a component
                v1, v2 = simplex.vertices
                union(v1, v2, simplex.filtration_value)
        
        # Add infinite features for components that never die
        for v in birth_time:
            if find(v) == v:  # Representative of its component
                features.append(PersistenceFeature(
                    dimension=0,
                    birth=birth_time[v],
                    death=float('inf'),
                    persistence=float('inf')
                ))
        
        # Create diagram
        diagram = PersistenceDiagram(features=features)
        
        # Update tracking metrics
        for dim in range(self.max_dimension + 1):
            count = sum(1 for f in features if f.dimension == dim)
            FEATURES_TRACKED.labels(algorithm="vineyard", dimension=str(dim)).set(count)
        
        # Store in vineyard
        self.vineyard.append(diagram)
        self.current_diagram = diagram
        
        return diagram


class StreamingRipsPersistence:
    """
    Streaming Rips complex with incremental persistence updates
    Optimized for point cloud data streams
    """
    
    def __init__(self, window_size: int, max_radius: float, max_dimension: int = 2):
        self.window_size = window_size
        self.max_radius = max_radius
        self.max_dimension = max_dimension
        self.distance_matrix: Optional[np.ndarray] = None
        self.current_diagram: Optional[PersistenceDiagram] = None
        self.tracer = get_tracer()
        
    async def update(self, window: StreamingWindow) -> PersistenceDiagram:
        """Update persistence diagram with current window state"""
        async with self.tracer.trace_async_operation(
            "streaming_rips_update",
            window_size=self.window_size
        ):
            start_time = time.time()
            
            # Get current points
            points = window.get_all_points()
            n_points = len(points)
            
            if n_points < 2:
                return PersistenceDiagram(features=[])
            
            # Update distance matrix incrementally
            await self._update_distances(points, window)
            
            # Build Rips filtration
            filtration = await self._build_rips_filtration(n_points)
            
            # Compute persistence
            diagram = await self._compute_persistence(filtration)
            
            # Update metrics
            UPDATE_LATENCY.labels(algorithm="streaming_rips").observe(time.time() - start_time)
            UPDATES_PROCESSED.labels(algorithm="streaming_rips").inc()
            
            return diagram
    
    async def _update_distances(self, points: np.ndarray, window: StreamingWindow):
        """Update distance matrix for new points"""
        n_points = len(points)
        
        if self.distance_matrix is None or self.distance_matrix.shape[0] != n_points:
            # Full recomputation needed
            self.distance_matrix = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = np.linalg.norm(points[i] - points[j])
                    self.distance_matrix[i, j] = dist
                    self.distance_matrix[j, i] = dist
        else:
            # Incremental update for new point
            new_idx = window.current_index
            for i in range(n_points):
                if i != new_idx:
                    dist = np.linalg.norm(points[new_idx] - points[i])
                    self.distance_matrix[new_idx, i] = dist
                    self.distance_matrix[i, new_idx] = dist
    
    async def _build_rips_filtration(self, n_points: int) -> List[Tuple[float, int, Tuple[int, ...]]]:
        """Build Rips filtration up to max_radius"""
        filtration = []
        
        # Add vertices
        for i in range(n_points):
            filtration.append((0.0, 0, (i,)))
        
        # Add edges
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if self.distance_matrix[i, j] <= self.max_radius:
                    filtration.append((
                        self.distance_matrix[i, j],
                        1,
                        (i, j)
                    ))
        
        # Add higher dimensional simplices
        if self.max_dimension >= 2:
            # Add triangles
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    for k in range(j + 1, n_points):
                        # Check if all edges exist
                        d_ij = self.distance_matrix[i, j]
                        d_ik = self.distance_matrix[i, k]
                        d_jk = self.distance_matrix[j, k]
                        
                        max_edge = max(d_ij, d_ik, d_jk)
                        if max_edge <= self.max_radius:
                            filtration.append((
                                max_edge,
                                2,
                                (i, j, k)
                            ))
        
        # Sort by filtration value
        filtration.sort(key=lambda x: (x[0], x[1]))
        
        return filtration
    
    async def _compute_persistence(self, filtration: List[Tuple[float, int, Tuple[int, ...]]]) -> PersistenceDiagram:
        """Compute persistence from filtration"""
        features = []
        
        # Simple persistence computation for dimension 0
        # (More sophisticated algorithms would be used in production)
        union_find = {}
        birth_times = {}
        
        def find(x):
            if x not in union_find:
                union_find[x] = x
                birth_times[x] = 0.0
                return x
            if union_find[x] != x:
                union_find[x] = find(union_find[x])
            return union_find[x]
        
        def union(x, y, time):
            px, py = find(x), find(y)
            if px != py:
                # Merge components
                if birth_times[px] < birth_times[py]:
                    union_find[py] = px
                    features.append(PersistenceFeature(
                        dimension=0,
                        birth=birth_times[py],
                        death=time,
                        persistence=time - birth_times[py]
                    ))
                else:
                    union_find[px] = py
                    features.append(PersistenceFeature(
                        dimension=0,
                        birth=birth_times[px],
                        death=time,
                        persistence=time - birth_times[px]
                    ))
        
        # Process filtration
        for value, dim, simplex in filtration:
            if dim == 0:
                find(simplex[0])
            elif dim == 1:
                union(simplex[0], simplex[1], value)
        
        # Add infinite features
        roots = set()
        for v in union_find:
            roots.add(find(v))
        
        for root in roots:
            features.append(PersistenceFeature(
                dimension=0,
                birth=birth_times[root],
                death=float('inf'),
                persistence=float('inf')
            ))
        
        # Update metrics
        for dim in range(self.max_dimension + 1):
            count = sum(1 for f in features if f.dimension == dim)
            FEATURES_TRACKED.labels(algorithm="streaming_rips", dimension=str(dim)).set(count)
        
        return PersistenceDiagram(features=features)