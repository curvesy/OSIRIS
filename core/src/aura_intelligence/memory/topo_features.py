"""
Topological Feature Extraction
=============================

Extracts numerical features from topological data (persistence diagrams,
Betti numbers) for use in embedding generation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..tda.models import BettiNumbers


@dataclass
class TopologicalFeatures:
    """Container for extracted topological features."""
    betti_features: np.ndarray  # Betti numbers as features
    persistence_features: np.ndarray  # Persistence diagram features
    combined: np.ndarray  # All features concatenated
    
    @property
    def dimension(self) -> int:
        """Total feature dimension."""
        return len(self.combined)


class TopologicalFeatureExtractor:
    """
    Extracts fixed-size feature vectors from topological data.
    
    This is designed to be simple and interpretable, extracting
    standard statistical features that capture the essence of
    the topological structure.
    """
    
    def __init__(self, persistence_bins: int = 10):
        self.persistence_bins = persistence_bins
        self._feature_dim = None
    
    def extract(
        self, 
        betti_numbers: BettiNumbers,
        persistence_diagram: np.ndarray
    ) -> TopologicalFeatures:
        """
        Extract features from topological data.
        
        Args:
            betti_numbers: Topological invariants
            persistence_diagram: Array of (birth, death) pairs
            
        Returns:
            TopologicalFeatures object with all extracted features
        """
        # Extract Betti features
        betti_features = self._extract_betti_features(betti_numbers)
        
        # Extract persistence features
        persistence_features = self._extract_persistence_features(persistence_diagram)
        
        # Combine all features
        combined = np.concatenate([betti_features, persistence_features])
        
        # Cache feature dimension
        if self._feature_dim is None:
            self._feature_dim = len(combined)
        
        return TopologicalFeatures(
            betti_features=betti_features,
            persistence_features=persistence_features,
            combined=combined
        )
    
    def _extract_betti_features(self, betti: BettiNumbers) -> np.ndarray:
        """Extract features from Betti numbers."""
        # Direct Betti numbers
        features = [
            float(betti.b0),
            float(betti.b1),
            float(betti.b2)
        ]
        
        # Derived features
        total_betti = betti.b0 + betti.b1 + betti.b2
        features.extend([
            total_betti,
            betti.b1 / (total_betti + 1e-6),  # Relative b1
            betti.b2 / (total_betti + 1e-6)   # Relative b2
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_persistence_features(self, diagram: np.ndarray) -> np.ndarray:
        """Extract features from persistence diagram."""
        features = []
        
        # Handle empty diagram
        if diagram.size == 0:
            # Return zeros for all features
            return np.zeros(
                4 +  # Basic stats
                self.persistence_bins +  # Histogram
                3,  # Derived features
                dtype=np.float32
            )
        
        # Ensure 2D
        if diagram.ndim == 1:
            diagram = diagram.reshape(-1, 2)
        
        # Calculate persistence values
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistences = deaths - births
        
        # Basic statistics
        features.extend([
            np.mean(persistences),
            np.std(persistences),
            np.min(persistences),
            np.max(persistences)
        ])
        
        # Persistence histogram
        hist, _ = np.histogram(
            persistences,
            bins=self.persistence_bins,
            range=(0, np.max(persistences) + 1e-6)
        )
        features.extend(hist / (len(persistences) + 1e-6))
        
        # Derived features
        features.extend([
            len(diagram),  # Number of features
            np.sum(persistences),  # Total persistence
            np.sum(persistences > np.mean(persistences))  # Significant features
        ])
        
        return np.array(features, dtype=np.float32)
    
    @property
    def feature_dimension(self) -> int:
        """Get the total feature dimension."""
        if self._feature_dim is None:
            # Calculate by extracting from dummy data
            dummy_betti = BettiNumbers(b0=1, b1=0, b2=0)
            dummy_diagram = np.array([[0, 1]])
            features = self.extract(dummy_betti, dummy_diagram)
            self._feature_dim = features.dimension
        return self._feature_dim