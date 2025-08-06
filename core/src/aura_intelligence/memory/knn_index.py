"""
k-NN Index for High-Speed Vector Search
======================================

A clean abstraction over various k-NN backends (sklearn, faiss, annoy).
Designed for production use with proper error handling and extensibility.
"""

import numpy as np
from typing import List, Tuple, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class KNNConfig:
    """Configuration for k-NN index."""
    metric: str = 'cosine'  # 'cosine', 'euclidean', 'manhattan'
    backend: str = 'sklearn'  # 'sklearn', 'faiss', 'annoy'
    initial_capacity: int = 1000
    
    def __post_init__(self):
        """Validate configuration."""
        valid_metrics = {'cosine', 'euclidean', 'manhattan'}
        valid_backends = {'sklearn', 'faiss', 'annoy'}
        
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {self.metric}. Must be one of {valid_metrics}")
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. Must be one of {valid_backends}")


class VectorIndex(Protocol):
    """Protocol defining the interface for vector indices."""
    
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors with their IDs."""
        ...
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        ...
    
    def __len__(self) -> int:
        """Return number of vectors in index."""
        ...


class BaseKNNIndex(ABC):
    """Abstract base class for k-NN indices."""
    
    def __init__(self, embedding_dim: int, config: KNNConfig):
        self.embedding_dim = embedding_dim
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index."""
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        pass
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")
        if self.config.initial_capacity <= 0:
            raise ValueError(f"Invalid initial capacity: {self.config.initial_capacity}")
    
    def _validate_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Validate input vectors and IDs."""
        if vectors.shape[0] != len(ids):
            raise ValueError(f"Mismatch: {vectors.shape[0]} vectors vs {len(ids)} IDs")
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Wrong dimension: expected {self.embedding_dim}, got {vectors.shape[1]}")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs detected")


class SklearnKNNIndex(BaseKNNIndex):
    """Scikit-learn based k-NN index implementation."""
    
    def __init__(self, embedding_dim: int, config: KNNConfig):
        super().__init__(embedding_dim, config)
        
        # Lazy import to avoid dependency if not used
        from sklearn.neighbors import NearestNeighbors
        
        self._vectors = np.empty((0, self.embedding_dim), dtype=np.float32)
        self._ids: List[str] = []
        self._id_to_idx = {}
        
        # Configure sklearn model
        metric = 'cosine' if config.metric == 'cosine' else config.metric
        self._model = NearestNeighbors(
            n_neighbors=min(10, config.initial_capacity),
            metric=metric,
            algorithm='auto'
        )
        self._is_fitted = False
    
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index."""
        self._validate_vectors(vectors, ids)
        
        # Update mappings
        start_idx = len(self._ids)
        for i, id_ in enumerate(ids):
            if id_ in self._id_to_idx:
                raise ValueError(f"ID already exists: {id_}")
            self._id_to_idx[id_] = start_idx + i
        
        # Append vectors
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._ids.extend(ids)
        self._is_fitted = False
        
        logger.debug(f"Added {len(ids)} vectors. Total: {len(self._ids)}")
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if len(self._ids) == 0:
            return []
        
        # Ensure model is fitted
        if not self._is_fitted:
            self._fit()
        
        # Reshape query if needed
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Limit k to available vectors
        k = min(k, len(self._ids))
        
        # Perform search
        distances, indices = self._model.kneighbors(query, n_neighbors=k)
        
        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self._ids[idx], float(dist)))
        
        return results
    
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._ids)
    
    def _fit(self) -> None:
        """Fit the sklearn model with current vectors."""
        if len(self._vectors) > 0:
            self._model.fit(self._vectors)
            self._is_fitted = True


class KNNIndex:
    """
    Factory class that creates the appropriate k-NN index based on config.
    This is the main interface that users should interact with.
    """
    
    def __init__(self, embedding_dim: int, config: Optional[KNNConfig] = None):
        self.config = config or KNNConfig()
        self._impl = self._create_implementation(embedding_dim)
    
    def _create_implementation(self, embedding_dim: int) -> BaseKNNIndex:
        """Create the appropriate implementation based on config."""
        if self.config.backend == 'sklearn':
            return SklearnKNNIndex(embedding_dim, self.config)
        elif self.config.backend == 'faiss':
            # Future: return FaissKNNIndex(embedding_dim, self.config)
            raise NotImplementedError("Faiss backend not yet implemented")
        elif self.config.backend == 'annoy':
            # Future: return AnnoyKNNIndex(embedding_dim, self.config)
            raise NotImplementedError("Annoy backend not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index."""
        return self._impl.add(vectors, ids)
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        return self._impl.search(query, k)
    
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._impl)