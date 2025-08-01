"""
🔢 Signature Vectorization

Convert topological signatures to 128-d embeddings for similarity search.
Implements persistence→embedding transformation for DuckDB VSS integration.

Based on partab.md: "persistence→128-d embedding" specification.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
from dataclasses import dataclass

from aura_intelligence.enterprise.data_structures import TopologicalSignature
from aura_intelligence.utils.logger import get_logger


@dataclass
class VectorEmbedding:
    """Vector embedding with metadata."""
    vector: np.ndarray
    dimension: int
    signature_hash: str
    embedding_method: str


class SignatureVectorizer:
    """
    🔢 Topological Signature Vectorizer
    
    Converts TopologicalSignature objects to dense vector embeddings
    for similarity search in DuckDB VSS.
    
    Features:
    - Betti number encoding with normalization
    - Persistence diagram features
    - Hash-based uniqueness features
    - Configurable vector dimensions (default 128)
    """
    
    def __init__(self, vector_dimension: int = 128):
        """Initialize the signature vectorizer."""
        
        self.vector_dimension = vector_dimension
        self.logger = get_logger(__name__)
        
        # Validate dimension
        if vector_dimension < 16 or vector_dimension > 2048:
            raise ValueError("vector_dimension must be between 16 and 2048")
        
        # Feature allocation
        self.betti_features = 3      # betti_0, betti_1, betti_2
        self.anomaly_features = 1    # anomaly_score
        self.hash_features = 16      # hash-based uniqueness
        self.persistence_features = min(32, vector_dimension - 20)  # persistence diagram features
        self.random_features = max(0, vector_dimension - 20 - self.persistence_features)  # remaining
        
        self.logger.info(f"🔢 Signature Vectorizer initialized (dim={vector_dimension})")
    
    async def vectorize_signature(self, signature: TopologicalSignature) -> np.ndarray:
        """
        Convert a topological signature to vector embedding.
        
        Args:
            signature: TopologicalSignature to vectorize
            
        Returns:
            Dense vector embedding of specified dimension
        """
        
        try:
            vector = np.zeros(self.vector_dimension, dtype=np.float32)
            idx = 0
            
            # 1. Betti number features (normalized)
            # Handle both old format (betti_0, betti_1, betti_2) and new format (betti_numbers list)
            if hasattr(signature, 'betti_0'):
                vector[idx] = self._normalize_betti(signature.betti_0, max_val=100)
                vector[idx + 1] = self._normalize_betti(signature.betti_1, max_val=50)
                vector[idx + 2] = self._normalize_betti(signature.betti_2, max_val=20)
            else:
                # New format with betti_numbers list
                betti_nums = signature.betti_numbers + [0, 0, 0]  # Pad with zeros
                vector[idx] = self._normalize_betti(betti_nums[0], max_val=100)
                vector[idx + 1] = self._normalize_betti(betti_nums[1], max_val=50)
                vector[idx + 2] = self._normalize_betti(betti_nums[2], max_val=20)
            idx += self.betti_features
            
            # 2. Anomaly score feature (use consciousness_level if anomaly_score not available)
            if hasattr(signature, 'anomaly_score'):
                vector[idx] = np.clip(signature.anomaly_score, 0.0, 1.0)
            else:
                vector[idx] = np.clip(signature.consciousness_level, 0.0, 1.0)
            idx += self.anomaly_features
            
            # 3. Hash-based uniqueness features
            signature_hash = getattr(signature, 'hash', None) or signature.signature_hash
            hash_features = self._extract_hash_features(signature_hash)
            vector[idx:idx + self.hash_features] = hash_features
            idx += self.hash_features
            
            # 4. Persistence diagram features (if available)
            if hasattr(signature, 'persistence_diagram') and signature.persistence_diagram:
                persistence_features = self._extract_persistence_features(signature.persistence_diagram)
                vector[idx:idx + len(persistence_features)] = persistence_features[:self.persistence_features]
            else:
                # Use Betti-based approximation
                persistence_features = self._approximate_persistence_features(signature)
                vector[idx:idx + len(persistence_features)] = persistence_features
            idx += self.persistence_features
            
            # 5. Random features for diversity (seeded by hash)
            if self.random_features > 0:
                signature_hash = getattr(signature, 'hash', None) or signature.signature_hash
                random_features = self._generate_random_features(signature_hash)
                vector[idx:idx + self.random_features] = random_features
            
            # Normalize vector to unit length for cosine similarity
            vector = self._normalize_vector(vector)
            
            return vector
            
        except Exception as e:
            signature_hash = getattr(signature, 'hash', None) or signature.signature_hash
            self.logger.error(f"❌ Vectorization failed for signature {signature_hash[:8]}...: {e}")
            # Return zero vector as fallback
            return np.zeros(self.vector_dimension, dtype=np.float32)

    def vectorize_signature_sync(self, signature: TopologicalSignature) -> np.ndarray:
        """
        Synchronous version of vectorize_signature for thread pool execution.

        Args:
            signature: TopologicalSignature to vectorize

        Returns:
            Dense vector embedding of specified dimension
        """

        try:
            vector = np.zeros(self.vector_dimension, dtype=np.float32)
            idx = 0

            # 1. Betti number features (normalized)
            # Handle both old format (betti_0, betti_1, betti_2) and new format (betti_numbers list)
            if hasattr(signature, 'betti_0'):
                vector[idx] = self._normalize_betti(signature.betti_0, max_val=100)
                vector[idx + 1] = self._normalize_betti(signature.betti_1, max_val=50)
                vector[idx + 2] = self._normalize_betti(signature.betti_2, max_val=20)
            else:
                # New format with betti_numbers list
                betti_nums = signature.betti_numbers + [0, 0, 0]  # Pad with zeros
                vector[idx] = self._normalize_betti(betti_nums[0], max_val=100)
                vector[idx + 1] = self._normalize_betti(betti_nums[1], max_val=50)
                vector[idx + 2] = self._normalize_betti(betti_nums[2], max_val=20)
            idx += self.betti_features

            # 2. Anomaly score feature (use consciousness_level if anomaly_score not available)
            if hasattr(signature, 'anomaly_score'):
                vector[idx] = np.clip(signature.anomaly_score, 0.0, 1.0)
            else:
                vector[idx] = np.clip(signature.consciousness_level, 0.0, 1.0)
            idx += self.anomaly_features

            # 3. Hash-based uniqueness features
            signature_hash = getattr(signature, 'hash', None) or signature.signature_hash
            hash_features = self._extract_hash_features(signature_hash)
            vector[idx:idx + self.hash_features] = hash_features
            idx += self.hash_features

            # 4. Persistence diagram features (if available)
            if hasattr(signature, 'persistence_diagram') and signature.persistence_diagram:
                persistence_features = self._extract_persistence_features(signature.persistence_diagram)
                vector[idx:idx + len(persistence_features)] = persistence_features[:self.persistence_features]
            else:
                # Use Betti-based approximation
                persistence_features = self._approximate_persistence_features(signature)
                vector[idx:idx + len(persistence_features)] = persistence_features
            idx += self.persistence_features

            # 5. Random features for diversity (seeded by hash)
            if self.random_features > 0:
                random_features = self._generate_random_features(signature_hash)
                vector[idx:idx + self.random_features] = random_features

            # Normalize vector to unit length for cosine similarity
            vector = self._normalize_vector(vector)

            return vector

        except Exception as e:
            self.logger.error(f"❌ Sync vectorization failed for signature {signature.signature_hash[:8]}...: {e}")
            # Return zero vector as fallback
            return np.zeros(self.vector_dimension, dtype=np.float32)
    
    def _normalize_betti(self, betti_value: int, max_val: int = 100) -> float:
        """Normalize Betti number to [0, 1] range."""
        return min(float(betti_value) / max_val, 1.0)
    
    def _extract_hash_features(self, signature_hash: str) -> np.ndarray:
        """Extract binary features from signature hash."""
        
        if not signature_hash:
            return np.zeros(self.hash_features, dtype=np.float32)
        
        # Convert hash to integer
        # Try hex first, fallback to Python hash for non-hex strings
        try:
            hash_int = int(signature_hash[:16], 16) if len(signature_hash) >= 16 else 0
        except ValueError:
            # Fallback to Python's hash function for non-hex strings
            hash_int = abs(hash(signature_hash)) % (2**32)
        
        # Extract binary features
        features = np.zeros(self.hash_features, dtype=np.float32)
        for i in range(self.hash_features):
            features[i] = float((hash_int >> i) & 1) * 0.1  # Scale to [0, 0.1]
        
        return features
    
    def _extract_persistence_features(self, persistence_diagram) -> np.ndarray:
        """Extract features from persistence diagram."""

        features = np.zeros(self.persistence_features, dtype=np.float32)

        if not persistence_diagram:
            return features

        # Handle different persistence diagram formats
        pairs = []
        if isinstance(persistence_diagram, dict):
            # Handle dict format with birth_death_pairs
            if "birth_death_pairs" in persistence_diagram:
                pairs = persistence_diagram["birth_death_pairs"]
        elif isinstance(persistence_diagram, list):
            # Handle list format directly
            pairs = persistence_diagram

        if not pairs:
            return features

        # Sort by persistence (death - birth)
        try:
            sorted_pairs = sorted(pairs, key=lambda x: float(x[1]) - float(x[0]), reverse=True)

            # Extract top persistence pairs
            for i, pair in enumerate(sorted_pairs[:self.persistence_features // 2]):
                if i * 2 + 1 < self.persistence_features:
                    birth, death = float(pair[0]), float(pair[1])
                    features[i * 2] = min(birth, 1.0)  # Normalize birth time
                    features[i * 2 + 1] = min(death - birth, 1.0)  # Normalize persistence
        except (ValueError, TypeError, IndexError):
            # Return zeros if parsing fails
            pass

        return features
    
    def _approximate_persistence_features(self, signature: TopologicalSignature) -> np.ndarray:
        """Approximate persistence features from Betti numbers."""
        
        features = np.zeros(self.persistence_features, dtype=np.float32)
        
        # Use Betti numbers to approximate persistence structure
        # This is a simplified approximation - in production use real persistence diagrams

        # Handle both old format (betti_0, betti_1, betti_2) and new format (betti_numbers list)
        if hasattr(signature, 'betti_0'):
            betti_0, betti_1, betti_2 = signature.betti_0, signature.betti_1, signature.betti_2
        else:
            betti_nums = signature.betti_numbers + [0, 0, 0]  # Pad with zeros
            betti_0, betti_1, betti_2 = betti_nums[0], betti_nums[1], betti_nums[2]

        # Approximate birth times based on Betti numbers
        if betti_0 > 0:
            features[0] = 0.0  # Components born at time 0
            features[1] = min(betti_0 / 10.0, 1.0)  # Persistence approximation

        if betti_1 > 0:
            features[2] = 0.2  # Cycles born later
            features[3] = min(betti_1 / 5.0, 1.0)

        if betti_2 > 0:
            features[4] = 0.4  # Voids born even later
            features[5] = min(betti_2 / 2.0, 1.0)

        # Fill remaining with scaled Betti ratios
        for i in range(6, min(self.persistence_features, 12)):
            if i % 3 == 0:
                features[i] = betti_0 / (betti_0 + betti_1 + betti_2 + 1)
            elif i % 3 == 1:
                features[i] = betti_1 / (betti_0 + betti_1 + betti_2 + 1)
            else:
                features[i] = betti_2 / (betti_0 + betti_1 + betti_2 + 1)
        
        return features
    
    def _generate_random_features(self, signature_hash: str) -> np.ndarray:
        """Generate deterministic random features seeded by hash."""
        
        if not signature_hash:
            return np.zeros(self.random_features, dtype=np.float32)
        
        # Use hash as seed for reproducible randomness
        # Convert hash to integer using simple hash function if not hex
        try:
            hash_int = int(signature_hash[:8], 16) if len(signature_hash) >= 8 else 0
        except ValueError:
            # Fallback to Python's hash function for non-hex strings
            hash_int = abs(hash(signature_hash)) % (2**32)

        np.random.seed(hash_int % (2**32))
        
        # Generate small random values for diversity
        features = np.random.normal(0, 0.01, self.random_features).astype(np.float32)
        
        return features
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length for cosine similarity."""
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        else:
            return vector
    
    async def vectorize_batch(self, signatures: List[TopologicalSignature]) -> List[np.ndarray]:
        """Vectorize a batch of signatures efficiently."""
        
        vectors = []
        for signature in signatures:
            vector = await self.vectorize_signature(signature)
            vectors.append(vector)
        
        return vectors
    
    def compute_similarity(self, vector1: np.ndarray, vector2: np.ndarray, 
                          metric: str = "cosine") -> float:
        """Compute similarity between two vectors."""
        
        if metric == "cosine":
            return float(np.dot(vector1, vector2))  # Vectors are already normalized
        elif metric == "euclidean":
            return float(1.0 / (1.0 + np.linalg.norm(vector1 - vector2)))
        elif metric == "dot_product":
            return float(np.dot(vector1, vector2))
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding configuration."""
        
        return {
            "vector_dimension": self.vector_dimension,
            "feature_allocation": {
                "betti_features": self.betti_features,
                "anomaly_features": self.anomaly_features,
                "hash_features": self.hash_features,
                "persistence_features": self.persistence_features,
                "random_features": self.random_features
            },
            "embedding_method": "topological_signature_v1"
        }
