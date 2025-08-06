"""
Adaptive Fusion Scorer for Shape Memory V2
=========================================

Implements state-of-the-art fusion scoring that combines:
- FastRP cosine similarity
- Wasserstein topological distance  
- Temporal decay for embedding freshness

Based on research from ACL 2023 and ICML 2024.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for adaptive fusion scoring."""
    # Base weights (sum to 1.0)
    base_alpha: float = 0.7  # FastRP cosine weight
    base_beta: float = 0.3   # Wasserstein weight
    
    # Temporal decay parameters
    tau_hours: float = 168.0  # Decay time constant (1 week)
    min_confidence: float = 0.5  # Minimum confidence for old embeddings
    
    # Wasserstein normalization
    wasserstein_95_percentile: float = 1.0  # Will be learned from data
    
    # Exploration for A/B testing
    exploration_noise: float = 0.01
    
    def __post_init__(self):
        """Validate configuration."""
        if not np.isclose(self.base_alpha + self.base_beta, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {self.base_alpha + self.base_beta}")
        if self.tau_hours <= 0:
            raise ValueError(f"tau_hours must be positive, got {self.tau_hours}")


class AdaptiveFusionScorer:
    """
    Production-grade fusion scorer with dynamic weight adjustment.
    
    Key innovations:
    - Confidence-based weight adjustment
    - Learned Wasserstein normalization
    - Temporal decay handling
    - A/B testing support via exploration
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._wasserstein_history: List[float] = []
        self._initialized = False
        
    def score(
        self,
        fastrp_similarity: float,
        persistence_diagram1: np.ndarray,
        persistence_diagram2: np.ndarray,
        embedding_age_hours: float = 0.0,
        enable_exploration: bool = False
    ) -> Dict[str, float]:
        """
        Compute adaptive fusion score.
        
        Args:
            fastrp_similarity: Cosine similarity from FastRP (0-1)
            persistence_diagram1: Query persistence diagram
            persistence_diagram2: Stored persistence diagram
            embedding_age_hours: Age of stored embedding in hours
            enable_exploration: Add exploration noise for A/B testing
            
        Returns:
            Dictionary with score components and final score
        """
        # Compute Wasserstein distance
        wasserstein_dist = self._compute_wasserstein(
            persistence_diagram1, 
            persistence_diagram2
        )
        
        # Update normalization statistics
        self._update_normalization(wasserstein_dist)
        
        # Normalize Wasserstein to [0, 1]
        wasserstein_norm = self._normalize_wasserstein(wasserstein_dist)
        
        # Compute embedding confidence based on age
        confidence = self._compute_confidence(embedding_age_hours)
        
        # Adjust weights based on confidence
        alpha, beta = self._adjust_weights(confidence)
        
        # Compute base score
        base_score = alpha * fastrp_similarity + beta * (1 - wasserstein_norm)
        
        # Add exploration noise if enabled
        if enable_exploration:
            noise = np.random.normal(0, self.config.exploration_noise)
            final_score = np.clip(base_score + noise, 0, 1)
        else:
            final_score = base_score
            
        return {
            "final_score": final_score,
            "fastrp_similarity": fastrp_similarity,
            "wasserstein_distance": wasserstein_dist,
            "wasserstein_normalized": wasserstein_norm,
            "embedding_confidence": confidence,
            "alpha": alpha,
            "beta": beta,
            "age_hours": embedding_age_hours
        }
    
    def _compute_wasserstein(
        self, 
        diagram1: np.ndarray, 
        diagram2: np.ndarray
    ) -> float:
        """
        Compute Wasserstein distance between persistence diagrams.
        
        Handles edge cases like empty diagrams.
        """
        # Handle empty diagrams
        if diagram1.size == 0 and diagram2.size == 0:
            return 0.0
        elif diagram1.size == 0 or diagram2.size == 0:
            return 1.0
            
        # Ensure 2D arrays
        if diagram1.ndim == 1:
            diagram1 = diagram1.reshape(-1, 2)
        if diagram2.ndim == 1:
            diagram2 = diagram2.reshape(-1, 2)
            
        # Compute persistence values
        persist1 = diagram1[:, 1] - diagram1[:, 0]
        persist2 = diagram2[:, 1] - diagram2[:, 0]
        
        # Use 1-Wasserstein distance
        try:
            return wasserstein_distance(persist1, persist2)
        except Exception as e:
            logger.warning(f"Wasserstein computation failed: {e}")
            return 1.0
    
    def _update_normalization(self, wasserstein_dist: float):
        """Update running statistics for Wasserstein normalization."""
        self._wasserstein_history.append(wasserstein_dist)
        
        # Keep only recent history (last 10k values)
        if len(self._wasserstein_history) > 10000:
            self._wasserstein_history = self._wasserstein_history[-10000:]
            
        # Update 95th percentile if we have enough data
        if len(self._wasserstein_history) >= 100:
            self.config.wasserstein_95_percentile = np.percentile(
                self._wasserstein_history, 95
            )
            self._initialized = True
    
    def _normalize_wasserstein(self, wasserstein_dist: float) -> float:
        """Normalize Wasserstein distance to [0, 1]."""
        if not self._initialized:
            # Use default normalization until we have data
            return min(wasserstein_dist, 1.0)
            
        # Normalize by 95th percentile
        normalized = wasserstein_dist / (self.config.wasserstein_95_percentile + 1e-6)
        return min(normalized, 1.0)
    
    def _compute_confidence(self, age_hours: float) -> float:
        """
        Compute embedding confidence based on age.
        
        Uses exponential decay: confidence = exp(-age/tau)
        """
        if age_hours <= 0:
            return 1.0
            
        confidence = np.exp(-age_hours / self.config.tau_hours)
        return max(confidence, self.config.min_confidence)
    
    def _adjust_weights(self, confidence: float) -> Tuple[float, float]:
        """
        Adjust fusion weights based on embedding confidence.
        
        As embeddings get older (lower confidence), we rely more
        on topological distance and less on FastRP similarity.
        """
        # Scale alpha by confidence
        alpha = self.config.base_alpha * confidence
        
        # Increase beta to compensate
        beta = self.config.base_beta + (1 - confidence) * 0.2
        
        # Renormalize to sum to 1
        total = alpha + beta
        return alpha / total, beta / total
    
    def batch_score(
        self,
        fastrp_similarities: np.ndarray,
        query_diagram: np.ndarray,
        stored_diagrams: List[np.ndarray],
        embedding_ages: np.ndarray,
        enable_exploration: bool = False
    ) -> np.ndarray:
        """
        Efficiently score multiple candidates.
        
        Returns array of fusion scores.
        """
        scores = []
        
        for i, (sim, diagram, age) in enumerate(
            zip(fastrp_similarities, stored_diagrams, embedding_ages)
        ):
            result = self.score(
                sim, query_diagram, diagram, age, enable_exploration
            )
            scores.append(result["final_score"])
            
        return np.array(scores)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scorer statistics for monitoring."""
        return {
            "initialized": self._initialized,
            "wasserstein_samples": len(self._wasserstein_history),
            "wasserstein_95_percentile": self.config.wasserstein_95_percentile,
            "config": {
                "base_alpha": self.config.base_alpha,
                "base_beta": self.config.base_beta,
                "tau_hours": self.config.tau_hours
            }
        }