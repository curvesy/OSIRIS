"""
FastRP: Fast Random Projection for Topological Embeddings
========================================================

A clean implementation of FastRP for converting high-dimensional
topological features into dense embeddings suitable for k-NN search.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FastRPConfig:
    """Configuration for FastRP embeddings."""
    embedding_dim: int = 128
    iterations: int = 3
    normalization: str = 'l2'  # 'l2', 'l1', or None
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")
        if self.iterations <= 0:
            raise ValueError(f"Invalid iterations: {self.iterations}")
        if self.normalization not in ['l2', 'l1', None]:
            raise ValueError(f"Invalid normalization: {self.normalization}")


class FastRP:
    """
    Fast Random Projection for creating embeddings from feature vectors.
    
    This implementation focuses on simplicity and correctness over
    micro-optimizations. It's designed to be easily testable and
    extensible.
    """
    
    def __init__(self, input_dim: int, config: Optional[FastRPConfig] = None):
        self.input_dim = input_dim
        self.config = config or FastRPConfig()
        self._projection_matrix = None
        self._initialized = False
        
    def _initialize(self):
        """Initialize the random projection matrix."""
        if self._initialized:
            return
            
        np.random.seed(self.config.random_seed)
        
        # Create random projection matrix
        # Using standard Gaussian for simplicity
        self._projection_matrix = np.random.randn(
            self.input_dim, 
            self.config.embedding_dim
        ).astype(np.float32)
        
        # Normalize columns for stability
        norms = np.linalg.norm(self._projection_matrix, axis=0)
        self._projection_matrix /= norms
        
        self._initialized = True
        logger.debug(f"Initialized FastRP: {self.input_dim} -> {self.config.embedding_dim}")
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform feature vectors into embeddings.
        
        Args:
            features: Array of shape (n_samples, input_dim)
            
        Returns:
            embeddings: Array of shape (n_samples, embedding_dim)
        """
        if not self._initialized:
            self._initialize()
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, "
                f"got {features.shape[1]}"
            )
        
        # Initial projection
        embeddings = features @ self._projection_matrix
        
        # Iterative refinement
        for i in range(self.config.iterations - 1):
            # Simple power iteration for better quality
            embeddings = 0.5 * embeddings + 0.5 * (features @ self._projection_matrix)
            
            # Normalize if configured
            if self.config.normalization == 'l2':
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                embeddings /= norms
            elif self.config.normalization == 'l1':
                sums = np.sum(np.abs(embeddings), axis=1, keepdims=True)
                sums[sums == 0] = 1
                embeddings /= sums
        
        return embeddings.astype(np.float32)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        For FastRP, fitting is just initialization, so this is
        equivalent to transform.
        """
        return self.transform(features)